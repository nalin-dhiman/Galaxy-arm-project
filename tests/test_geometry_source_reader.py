from pathlib import Path
import pickle,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from geometry_source_reader import load_data_only,InertNode


def test_unknown_pickle_global_remains_inert(tmp_path):
    marker=tmp_path/'never_execute'
    class Dangerous:
        def __reduce__(self):
            return (eval,("__import__('pathlib').Path("+repr(str(marker))+").write_text('bad')",))
    p=tmp_path/'legacy.pkl';p.write_bytes(pickle.dumps(Dangerous()))
    obj=load_data_only(p)
    assert isinstance(obj,InertNode)
    assert not marker.exists()


def test_numeric_numpy_arrays_remain_data(tmp_path):
    p=tmp_path/'array.pkl';values=np.arange(20.).reshape(4,5)
    p.write_bytes(pickle.dumps({'coords':values},protocol=4))
    result=load_data_only(p)
    assert np.array_equal(result['coords'],values)
