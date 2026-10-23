"""Error-aware empirical BTFR distances with explicit conditional uncertainty.

The Gaussian vertical likelihood assumes locally flat latent log-speed density;
it is not a selection-corrected population or distance-ladder model. All inputs
must use one consistently defined, independently measured velocity observable.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy.special import logsumexp


def _array(value,name,positive=False):
    a=np.asarray(value,float)
    if not np.isfinite(a).all() or (positive and np.any(a<=0)):
        raise ValueError(name+' must be finite'+(' and positive' if positive else ''))
    return a


@dataclass(frozen=True)
class ErrorAwareBTFR:
    slope: np.ndarray
    intrinsic_scatter_dex: np.ndarray
    intercept_mean: np.ndarray
    intercept_std: np.ndarray
    weights: np.ndarray
    min_calibration_velocity: float
    max_calibration_velocity: float
    calibration_galaxies: int
    velocity_definition: str

    @classmethod
    def fit(cls, velocity,log_mass,error_log_velocity,error_log_mass,
            *, velocity_definition='HI_Vflat_kms',scatter_prior_scale=.3,grid_size=(201,161)):
        v,y,sx,sy=map(lambda q:_array(q[0],q[1],q[2]),[
            (velocity,'velocity',True),(log_mass,'logmass',False),
            (error_log_velocity,'logvelocity errors',False),(error_log_mass,'logmass errors',False)])
        if v.ndim!=1 or any(q.shape!=v.shape for q in (y,sx,sy)) or len(v)<5:
            raise ValueError('At least five one-dimensional matched calibrators required')
        if np.any(sx<0) or np.any(sy<=0) or scatter_prior_scale<=0:
            raise ValueError('Nonnegative speed and positive mass errors/prior scale required')
        x=np.log10(v/100.)
        if np.ptp(x)<.1:raise ValueError('Insufficient independent velocity leverage')
        nb,ns=grid_size
        if min(nb,ns)<21:raise ValueError('Posterior grid too coarse')
        b,s=np.meshgrid(np.linspace(1.5,6.5,nb),np.linspace(0,.8,ns),indexing='ij')
        bf,sf=b.ravel(),s.ravel()
        variance=sy[None,:]**2+bf[:,None]**2*sx[None,:]**2+sf[:,None]**2
        precision=1/variance
        total=precision.sum(axis=1)
        residual=y[None,:]-bf[:,None]*x[None,:]
        a=(precision*residual).sum(axis=1)/total
        logp=-.5*(np.log(variance)+(residual-a[:,None])**2*precision).sum(axis=1)
        logp-=.5*np.log(total)+.5*(sf/scatter_prior_scale)**2
        # Trapezoid integration over both bounded continuous prior coordinates.
        quadrature=np.ones((nb,ns));quadrature[[0,-1],:]*=.5;quadrature[:,[0,-1]]*=.5
        logp+=np.log(quadrature.ravel())
        weights=np.exp(logp-logsumexp(logp))
        return cls(bf,sf,a,1/np.sqrt(total),weights,float(v.min()),float(v.max()),len(v),velocity_definition)

    def summarize(self):
        w=self.weights
        def q(values):
            order=np.argsort(values);cumulative=np.cumsum(w[order]);return np.interp([.05,.5,.95],cumulative,values[order]).tolist()
        return dict(calibration_galaxies=self.calibration_galaxies,velocity_definition=self.velocity_definition,
            slope_5_50_95=q(self.slope),intrinsic_scatter_dex_5_50_95=q(self.intrinsic_scatter_dex),
            intercept_posterior_mean=float(np.sum(w*self.intercept_mean)),
            slope_outer_0p1_prior_mass=float(w[(self.slope<1.6)|(self.slope>6.4)].sum()),
            scatter_above_0p75_prior_mass=float(w[self.intrinsic_scatter_dex>.75].sum()),
            velocity_calibration_range_kms=[self.min_calibration_velocity,self.max_calibration_velocity],
            posterior_scope='Gaussian vertical likelihood; flat latent log-speed and intercept prior; bounded flat slope and half-normal scatter prior; no selection function or shared-anchor covariance fit.')

    def distance_draws(self,velocity,flux_proxy,error_log_velocity,error_log_flux,
                       *,draws=3000,seed=20261023,zero_point_sigma_mag=.05,
                       velocity_definition='HI_Vflat_kms'):
        if velocity_definition!=self.velocity_definition:
            raise ValueError('Velocity observable differs from the calibration definition')
        v=_array(velocity,'velocity',True);c=_array(flux_proxy,'flux proxy',True)
        sx=_array(error_log_velocity,'logspeed error');sc=_array(error_log_flux,'logflux error')
        if np.any(sx<0) or np.any(sc<0) or not np.isfinite(zero_point_sigma_mag) or zero_point_sigma_mag<0:
            raise ValueError('Nonnegative uncertainty inputs required')
        if any(x.ndim for x in (v,c,sx,sc)):raise ValueError('One target per posterior draw call')
        if draws<100:raise ValueError('At least100 posterior draws required')
        rng=np.random.default_rng(seed)
        selected=rng.choice(len(self.weights),size=draws,p=self.weights)
        b=self.slope[selected];s=self.intrinsic_scatter_dex[selected]
        a=rng.normal(self.intercept_mean[selected],self.intercept_std[selected])
        x=rng.normal(float(np.log10(v/100)),float(sx),size=draws)
        logc=rng.normal(float(np.log10(c)),float(sc),size=draws)
        logd=.5*(a+b*x+rng.normal(0,s)-logc)
        # An explicit global distance-scale assumption, not 3000 new anchors.
        logd+=rng.normal(0,zero_point_sigma_mag/5,size=draws)
        return 10**logd

    def predict(self,*args,**kwargs):
        samples=self.distance_draws(*args,**kwargs)
        v=float(args[0] if args else kwargs['velocity'])
        lower,median,upper=np.quantile(samples,[.05,.5,.95])
        outside=not self.min_calibration_velocity<=v<=self.max_calibration_velocity
        return dict(distance_mpc=float(median),model_interval_90_lower_mpc=float(lower),
            model_interval_90_upper_mpc=float(upper),outside_velocity_calibration_range=outside,
            status='abstain_velocity_extrapolation' if outside else 'conditional_kinematic_estimate',
            uncertainty_scope='Model-based posterior predictive interval under stated mass/velocity/flux errors and zero-point assumption; not independently calibrated coverage, and no geometric distance without independent kinematics/flux calibration.')

    def save(self,path):
        path=Path(path)
        if path.suffix!='.npz':
            path=Path(str(path)+'.npz')
        if path.exists():raise FileExistsError('Preserve existing model: '+str(path))
        np.savez_compressed(path,slope=self.slope,scatter=self.intrinsic_scatter_dex,
            a=self.intercept_mean,a_std=self.intercept_std,weights=self.weights,
            metadata=json.dumps(dict(min_v=self.min_calibration_velocity,max_v=self.max_calibration_velocity,
                n=self.calibration_galaxies,velocity_definition=self.velocity_definition)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            m=json.loads(str(d['metadata']));arrays=[_array(d[n],n) for n in ('slope','scatter','a','a_std','weights')]
        if any(a.ndim!=1 or a.shape!=arrays[0].shape for a in arrays) or np.any(arrays[-1]<0) or not np.isclose(arrays[-1].sum(),1):
            raise ValueError('Invalid numerical posterior artifact')
        return cls(*arrays,float(m['min_v']),float(m['max_v']),int(m['n']),m['velocity_definition'])
