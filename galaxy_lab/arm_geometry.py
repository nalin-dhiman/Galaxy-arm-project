"""Conditional angular spiral geometry from independent centerline traces or images.

A log-spiral segment is a model of a luminous ridge, not a standard physical ruler.
The image extractor is a classical matched-template prototype, separate from the
frozen three-band supervised arm-region model. All lengths are angular unless an
external distance is explicitly supplied by a different layer of the application.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.optimize import least_squares, curve_fit
from scipy.spatial import cKDTree
from .filtering import ScienceImage, DiskGeometry, radius_map, masked_gaussian, radial_residual


def deproject_points(points_xy, geometry: DiskGeometry):
    p=np.asarray(points_xy,float)
    if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():
        raise ValueError('Finite Nx2 pixel coordinates required')
    p=p-np.asarray(geometry.center_xy)
    a=np.radians(geometry.pa_deg);c,s=np.cos(a),np.sin(a)
    return np.column_stack([c*p[:,0]+s*p[:,1],(-s*p[:,0]+c*p[:,1])/np.cos(np.radians(geometry.inclination_deg))])


def project_points(points_disk, geometry: DiskGeometry):
    p=np.asarray(points_disk,float);a=np.radians(geometry.pa_deg);c,s=np.cos(a),np.sin(a)
    x=p[:,0];y=p[:,1]*np.cos(np.radians(geometry.inclination_deg))
    return np.column_stack([c*x-s*y,s*x+c*y])+np.asarray(geometry.center_xy)


def geometry_from_builder(center_xy, phi_radians, axis_ratio, inner_arcsec, outer_arcsec):
    """Convert authors' minor-axis/scale-X convention to major-axis/scale-Y."""
    q=float(axis_ratio);pa=np.degrees(float(phi_radians))
    if not 0<q<3:raise ValueError('Invalid reference axis ratio')
    if q<=1:pa+=90
    else:q=1/q
    return DiskGeometry(tuple(np.asarray(center_xy,float)),pa,float(np.degrees(np.arccos(q))),inner_arcsec,outer_arcsec,'independent Galaxy Builder disk fit')


def resample_curve(points, n=128):
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<2 or not np.isfinite(p).all():
        raise ValueError('Finite polyline with at least two points required')
    d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    good=np.r_[True,np.diff(d)>1e-10];p=p[good];d=d[good]
    if len(p)<2 or d[-1]<=0:raise ValueError('Polyline has zero length')
    t=np.linspace(0,d[-1],n)
    return np.column_stack([np.interp(t,d,p[:,i]) for i in range(2)])


def fit_log_spiral(points_xy, geometry: DiskGeometry, *, min_span_deg=20.):
    """Robust log r=a+b theta fit to an ORDERED trace; no pixel-independent CI."""
    p=deproject_points(points_xy,geometry);r=np.linalg.norm(p,axis=1)
    good=r>1.;p=p[good];r=r[good]
    if len(p)<8:raise ValueError('Too few trace points away from the center')
    t=np.unwrap(np.arctan2(p[:,1],p[:,0]));span=float(np.ptp(t))
    if np.degrees(span)<min_span_deg:raise ValueError('Insufficient angular span')
    y=np.log(r);t0=t.mean();x=t-t0
    initial=np.polyfit(x,y,1)
    f=least_squares(lambda z:z[0]+z[1]*x-y,[initial[1],initial[0]],loss='soft_l1',f_scale=.04)
    a,b=f.x
    return {'signed_pitch_deg':float(np.degrees(np.arctan(b))),
            'pitch_deg':float(abs(np.degrees(np.arctan(b)))),
            'angular_span_deg':float(np.degrees(span)),
            'log_radius_residual_rms':float(np.sqrt(np.mean((f.fun)**2))),
            'intercept':float(a),'slope':float(b),'theta_origin':float(t0),'n_points':len(p)}


def trace_bootstrap_pitch(traces, geometry, *, n_bootstrap=300, seed=20261023):
    """Resample entire human drawings; traces need not be unique volunteers."""
    fits=[]
    for trace in traces:
        try:fits.append(fit_log_spiral(trace,geometry)['signed_pitch_deg'])
        except ValueError:pass
    if not fits:return {'trace_count':0,'median_pitch_deg':None,'interval95_deg':None}
    rng=np.random.default_rng(seed);v=np.abs(fits)
    boot=np.median(rng.choice(v,(n_bootstrap,len(v)),replace=True),axis=1)
    return {'trace_count':len(v),'median_pitch_deg':float(np.median(v)),
            'individual_trace_pitch_std_deg':float(np.std(v)),
            'interval95_deg':np.quantile(boot,[.025,.975]).tolist(),
            'resampling_unit':'whole drawing, not independent pixels; author grouping may split one drawing'}


def centerline_metrics(predicted, reference, pixel_scale_arcsec, tolerance_arcsec):
    """Length-uniform point-to-curve distances, including unmatched curve length."""
    def cloud(curves):
        chunks=[]
        for c in curves:
            length=np.linalg.norm(np.diff(c,axis=0),axis=1).sum()*pixel_scale_arcsec
            chunks.append(resample_curve(c,max(8,int(np.ceil(length/.1)))))
        return np.concatenate(chunks) if chunks else np.empty((0,2))
    p,r=cloud(predicted),cloud(reference)
    if len(p) and len(r):
        dp=cKDTree(r).query(p)[0]*pixel_scale_arcsec;dr=cKDTree(p).query(r)[0]*pixel_scale_arcsec
        precision=float(np.mean(dp<=tolerance_arcsec));recall=float(np.mean(dr<=tolerance_arcsec))
        med=float(np.median(np.r_[dp,dr]));p95=float(np.quantile(np.r_[dp,dr],.95))
    else:precision=recall=0.;med=p95=None
    return {'precision':precision,'recall':recall,'f1':float(2*precision*recall/(precision+recall)) if precision+recall else 0.,
            'symmetric_median_arcsec':med,'symmetric_p95_arcsec':p95,
            'predicted_curve_count':len(predicted),'reference_curve_count':len(reference),
            'tolerance_arcsec':float(tolerance_arcsec)}


@dataclass(frozen=True)
class ExtractionConfig:
    smooth_sigma_arcsec: float=.4
    minimum_score: float=1.
    max_segments: int=6
    pitch_min_deg: float=10.
    pitch_max_deg: float=50.
    pitch_step_deg: float=2.5
    theta_count: int=180


def extract_centerlines(image: ScienceImage, geometry: DiskGeometry, config=ExtractionConfig()):
    """Image-only log-spiral template search, conditional on supplied disk geometry.

    Curves span three fixed overlapping radial windows. Their score is clipped
    radial-median-subtracted contrast in units of radial scatter, NOT calibrated
    detection significance. No human arm label or desired arm count is input.
    """
    if config.smooth_sigma_arcsec<=0 or config.minimum_score<=0 or config.max_segments<1:
        raise ValueError('Positive scales/threshold and segment count required')
    data,support=masked_gaussian(image.data,image.valid,config.smooth_sigma_arcsec/image.pixel_scale_arcsec)
    valid=image.valid & (support>.9)
    smooth=ScienceImage(data,image.pixel_scale_arcsec,mask=~valid)
    residual,_=radial_residual(smooth,geometry)
    rr=radius_map(data.shape,geometry,image.pixel_scale_arcsec);bins=np.floor(rr/image.pixel_scale_arcsec).astype(int)
    scale=np.full(bins.max()+1,np.nan)
    for k in range(len(scale)):
        v=residual[valid & (bins==k)]
        if len(v)>=8:scale[k]=1.4826*np.median(np.abs(v-np.median(v)))
    good=np.flatnonzero(np.isfinite(scale) & (scale>0))
    if len(good)<3:return [],{'failure':'insufficient radial scatter samples'}
    scale=np.interp(np.arange(len(scale)),good,scale[good]);floor=np.median(scale[good])*.25
    score=residual/np.maximum(scale[bins],floor)
    score=np.clip(score,-3,8)
    lo=max(geometry.r_inner_arcsec,2*image.pixel_scale_arcsec);hi=geometry.r_outer_arcsec
    theta0=np.linspace(0,2*np.pi,config.theta_count,endpoint=False)
    pitches=np.r_[np.arange(config.pitch_min_deg,config.pitch_max_deg+.01,config.pitch_step_deg),-np.arange(config.pitch_min_deg,config.pitch_max_deg+.01,config.pitch_step_deg)]
    candidates=[]
    for f0,f1 in [(0.,.65),(.2,.85),(.35,1.)]:
        loglo,loghi=np.log(lo)+np.asarray([f0,f1])*np.log(hi/lo)
        rad=np.exp(np.linspace(loglo,loghi,64))/image.pixel_scale_arcsec
        for pitch in pitches:
            theta=theta0[:,None]+np.log(rad/rad[0])[None,:]/np.tan(np.radians(pitch))
            # Reject more than one turn; a log-polar wrap cannot create extra evidence.
            if np.ptp(theta[0])>2*np.pi:continue
            disk=np.stack([rad[None,:]*np.cos(theta),rad[None,:]*np.sin(theta)],axis=-1)
            pix=project_points(disk.reshape(-1,2),geometry).reshape(len(theta0),64,2)
            coords=[pix[:,:,1],pix[:,:,0]]
            val=map_coordinates(score,coords,order=1,mode='constant',cval=-3)
            usable=map_coordinates(valid.astype(float),coords,order=0,mode='constant',cval=0)>0
            frac=usable.mean(axis=1);positive=((val>0)&usable).mean(axis=1)
            strength=np.mean(np.where(usable,val,-3),axis=1)
            for index in np.flatnonzero((strength>=config.minimum_score)&(frac>=.9)&(positive>=.7)):
                candidates.append((float(strength[index]),pix[index],float(pitch)))
    candidates.sort(key=lambda x:-x[0]);accepted=[]
    separation=max(image.psf_fwhm_arcsec or 1.,2*image.pixel_scale_arcsec)
    for strength,curve,pitch in candidates:
        if accepted:
            # Suppress both repeated templates and crossing templates sharing most length.
            nearest=np.min([cKDTree(a['points_xy']).query(curve)[0] for a in accepted],axis=0)*image.pixel_scale_arcsec
            if np.mean(nearest<separation)>.35:continue
        accepted.append({'points_xy':curve,'score':strength,'pitch_deg':abs(pitch),'signed_pitch_deg':pitch})
        if len(accepted)>=config.max_segments:break
    return accepted,{'candidate_count_before_suppression':len(candidates),'valid_fraction':float(valid.mean()),
        'method':'classical fixed-window log-spiral contrast templates','score_is_significance':False,
        'conditional_on_external_geometry':True,'failure':None if accepted else 'no template passed fixed score/support cuts'}


def measure_cross_section_width(image, curve_xy, *, half_width_arcsec=4., psf_fwhm_arcsec=None):
    """Observed sky-plane normal FWHM from a stacked profile; no deprojection.

    Gaussian PSF deconvolution is an approximation. A single normal direction is
    inappropriate for a strongly elongated PSF; failed/unresolved fits are exposed.
    """
    curve=resample_curve(curve_xy,64);tangent=np.gradient(curve,axis=0);norm=np.linalg.norm(tangent,axis=1)
    normal=np.column_stack([-tangent[:,1],tangent[:,0]])/np.maximum(norm[:,None],1e-10)
    offsets=np.linspace(-half_width_arcsec,half_width_arcsec,65)
    sample=curve[:,None,:]+normal[:,None,:]*offsets[None,:,None]/image.pixel_scale_arcsec
    values=map_coordinates(image.data,[sample[:,:,1],sample[:,:,0]],order=1,mode='constant',cval=np.nan)
    ok=map_coordinates(image.valid.astype(float),[sample[:,:,1],sample[:,:,0]],order=0,mode='constant',cval=0)>.5
    values[~ok]=np.nan
    with np.errstate(all='ignore'):
        profile=np.nanmedian(values,axis=0)
    keep=np.isfinite(profile)
    if keep.sum()<40:return {'status':'insufficient_profile','observed_fwhm_arcsec':None,'deconvolved_fwhm_arcsec':None}
    x=offsets[keep];y=profile[keep];baseline=np.median(np.r_[y[:8],y[-8:]]);amp=float(y.max()-baseline)
    if amp<=0:return {'status':'nonpositive_peak','observed_fwhm_arcsec':None,'deconvolved_fwhm_arcsec':None}
    def fun(x,a,mu,sigma,b,c):return a*np.exp(-.5*((x-mu)/sigma)**2)+b+c*x
    try:
        par,_=curve_fit(fun,x,y,p0=[amp,0,max(image.pixel_scale_arcsec,.7),baseline,0],
            bounds=([0,-half_width_arcsec/2,image.pixel_scale_arcsec*.3,-np.inf,-np.inf],
                    [np.inf,half_width_arcsec/2,half_width_arcsec,np.inf,np.inf]),maxfev=5000)
        fwhm=float(par[2]*2.354820045);psf=psf_fwhm_arcsec or image.psf_fwhm_arcsec
        intrinsic=float(np.sqrt(max(0,fwhm**2-psf**2))) if psf else None
        status='unresolved' if psf and fwhm<=1.05*psf else 'fit'
        return {'status':status,'observed_fwhm_arcsec':fwhm,'deconvolved_fwhm_arcsec':intrinsic,
                'center_offset_arcsec':float(par[1]),'profile_rms':float(np.std(y-fun(x,*par))),
                'width_frame':'observed sky-plane normal; not disk intrinsic width',
                'psf_deconvolution':'circular Gaussian approximation' if psf else 'unavailable'}
    except (ValueError,RuntimeError):return {'status':'fit_failed','observed_fwhm_arcsec':None,'deconvolved_fwhm_arcsec':None}
