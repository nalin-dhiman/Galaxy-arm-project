"""Read public scientific data pickles without importing their executable classes.

Only NumPy array/dtype constructors and basic builtin containers are allowed.
Every pandas/scipy/gzbuilder object becomes an inert node holding serialized
arguments/state. No downloaded Python source or arbitrary pickle global runs.
"""
from __future__ import annotations
import builtins
import gzip
import io
import pickle
import numpy as np

class InertNode(dict):
    def __new__(cls,*args,**kwargs):
        obj=super().__new__(cls);obj._constructor_args=args;obj._constructor_kwargs=kwargs
        return obj
    def __init__(self,*args,**kwargs):pass
    def __setstate__(self,state):
        self._serialized_state=state
        if isinstance(state,dict):self.__dict__.update(state)
        elif isinstance(state,tuple):
            for part in state:
                if isinstance(part,dict):self.__dict__.update(part)

_CLASSES={}
class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name in ('_reconstruct','scalar'):
            return getattr(np.core.multiarray,name)
        if module=='pandas.core.indexes.frozen' and name=='FrozenNDArray':
            return np.ndarray
        if module=='numpy' and name in ('ndarray','dtype'):
            return getattr(np,name)
        if module=='builtins' and name in ('slice','set','frozenset','dict','list','tuple','complex'):
            return getattr(builtins,name)
        key=(module,name)
        if key not in _CLASSES:
            _CLASSES[key]=type(name,(InertNode,),{'_source_module':module,'_source_name':name})
        return _CLASSES[key]

def load_data_only(path):
    raw=gzip.decompress(path.read_bytes()) if str(path).endswith('.gz') else path.read_bytes()
    if len(raw)>64*1024*1024:raise ValueError('Serialized data exceeds the inspection size limit')
    return DataOnlyUnpickler(io.BytesIO(raw)).load()
