"""Angular tilted-disk diagnostics from independent emission-line velocities.

This is a fixed-geometry LOS measurement, not a beam-convolved dynamical fit.
No physical distance, mass, photometric redshift or fluid prediction is an input.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DiskGeometry:
    axis_ratio: float
    pa_east_of_north_deg: float
    intrinsic_axis_ratio: float = .2

    @property
    def inclination_deg(self):
        q, q0 = self.axis_ratio, self.intrinsic_axis_ratio
        if not (0 <= q0 < 1 and q0 < q <= 1):
            raise ValueError('Require 0 <= intrinsic thickness < observed axis ratio <= 1')
        return float(np.degrees(np.arccos(np.sqrt((q*q-q0*q0)/(1-q0*q0)))))

    def disk_coordinates(self, east_arcsec, north_arcsec):
        if not np.isfinite(self.pa_east_of_north_deg):
            raise ValueError('Position angle must be finite')
        east, north = np.broadcast_arrays(np.asarray(east_arcsec, float), np.asarray(north_arcsec, float))
        inclination = np.radians(self.inclination_deg)
        if np.sin(inclination) < .05 or np.cos(inclination) < .05:
            raise ValueError('Face-on or edge-on geometry does not identify a stable circular speed')
        pa = np.radians(self.pa_east_of_north_deg)
        major = east*np.sin(pa) + north*np.cos(pa)
        minor = east*np.cos(pa) - north*np.sin(pa)
        radius = np.hypot(major, minor/np.cos(inclination))
        cosine = np.divide(major, radius, out=np.zeros_like(radius), where=radius>0)
        theta = np.arctan2(minor/np.cos(inclination), major)
        return radius, cosine, theta


def annular_los_fit(east, north, velocity, inverse_variance, valid, geometry,
                    *, psf_fwhm_arcsec, ring_width_arcsec=3., min_side_pixels=8):
    """Fit one systemic offset and signed rotation coefficients in angular rings.

    Formal IVAR errors neglect covariance. Sector jackknife and two-side
    differences are diagnostics rather than independently calibrated intervals.
    Radius zero/inner PSF and near-minor-axis pixels are excluded before fitting.
    """
    east, north, velocity, inverse_variance, valid = np.broadcast_arrays(
        np.asarray(east,float), np.asarray(north,float), np.asarray(velocity,float),
        np.asarray(inverse_variance,float), np.asarray(valid,bool))
    if not np.isfinite(psf_fwhm_arcsec) or psf_fwhm_arcsec <= 0:
        raise ValueError('Measured positive PSF FWHM required')
    if ring_width_arcsec < psf_fwhm_arcsec:
        raise ValueError('Angular rings must be at least one PSF FWHM wide')
    radius, cosine, theta = geometry.disk_coordinates(east,north)
    inner = max(3., psf_fwhm_arcsec)
    good = valid & np.isfinite(velocity) & np.isfinite(inverse_variance) & (inverse_variance>0)
    good &= np.isfinite(radius) & (radius>=inner) & (abs(cosine)>=.5)
    bins = np.floor((radius-inner)/ring_width_arcsec).astype(int)
    indices=[]; rows=[]
    for k in sorted(np.unique(bins[good])):
        region=good & (bins==k)
        positive=int(np.sum(region & (cosine>0)));negative=int(np.sum(region & (cosine<0)))
        if min(positive,negative)<min_side_pixels:continue
        indices.append(k)
        rows.append(dict(ring_index=int(k), radius_arcsec=float(np.median(radius[region])),
            positive_side_pixels=positive,negative_side_pixels=negative,pixels=int(region.sum())))
    if len(indices)<2:
        raise ValueError('Fewer than two rings with adequate two-sided sampling')
    fit_good = good & np.isin(bins,indices)
    sin_i=np.sin(np.radians(geometry.inclination_deg))
    ring=bins[fit_good];cos=cosine[fit_good];v=velocity[fit_good];w=inverse_variance[fit_good]
    X=np.column_stack([np.ones(len(v))]+[(ring==k)*sin_i*cos for k in indices])

    def solve(retained):
        a=X[retained]*np.sqrt(w[retained,None]);b=v[retained]*np.sqrt(w[retained])
        if np.linalg.matrix_rank(a)!=len(indices)+1:return None
        coefficients=np.linalg.lstsq(a,b,rcond=None)[0]
        return coefficients,np.linalg.inv(a.T@a)

    answer=solve(np.ones(len(v),bool))
    if answer is None:raise ValueError('Disk design matrix is rank deficient')
    coefficients,covariance=answer
    sign=1. if np.median(coefficients[1:])>=0 else -1.
    prediction=X@coefficients
    residual=v-prediction
    # Leave complete azimuth sectors out; nearby pixels are never counted as
    # new independent objects by this descriptive robustness calculation.
    sectors=np.floor((theta[fit_good]+np.pi)/(2*np.pi)*8).astype(int)%8
    jack=[]
    for sector in range(8):
        if not np.any(sectors==sector):continue
        result=solve(sectors!=sector)
        if result is not None:jack.append(result[0][1:]*sign)
    jack=np.asarray(jack)
    jackerr=np.sqrt((len(jack)-1)/len(jack)*np.sum((jack-jack.mean(axis=0))**2,axis=0)) if len(jack)>=4 else np.full(len(indices),np.nan)
    for j,(k,row) in enumerate(zip(indices,rows)):
        region=ring==k
        speeds=[]
        for side in (cos>0,cos<0):
            use=region & side;xx=sin_i*cos[use]
            speeds.append(float(sign*np.sum(w[use]*xx*(v[use]-coefficients[0]))/np.sum(w[use]*xx**2)))
        row.update(rotation_kms=float(sign*coefficients[j+1]),
            formal_error_kms=float(np.sqrt(max(0,covariance[j+1,j+1]))),
            sector_jackknife_kms=float(jackerr[j]) if np.isfinite(jackerr[j]) else None,
            side1_rotation_kms=speeds[0],side2_rotation_kms=speeds[1],
            side_difference_kms=float(abs(speeds[0]-speeds[1])))
    model=np.full(velocity.shape,np.nan);model[fit_good]=prediction
    return dict(rings=rows, systemic_offset_kms=float(coefficients[0]),
                inclination_deg=geometry.inclination_deg,fit_pixels=int(fit_good.sum()),
                residual_rms_kms=float(np.sqrt(np.mean(residual**2))),
                model=model,fit_mask=fit_good,sector_jackknife_fits=len(jack),
                interpretation='Photometry-fixed projected circular-disk approximation; errors do not include full spatial covariance, beam convolution or noncircular gas support.')


def assess_flat_speed(result, *, half_light_radius_arcsec, quality_flags=0):
    """Necessary optical-flatness gates; never a sufficient HI Vflat calibration."""
    if not np.isfinite(half_light_radius_arcsec) or half_light_radius_arcsec<=0:
        raise ValueError('Positive independently measured angular half-light radius required')
    rings=result['rings']; reasons=[]
    if quality_flags:reasons.append('nonzero_observation_quality_flags')
    if not 30<=result['inclination_deg']<=75:reasons.append('inclination_outside_30_75_deg')
    if len(rings)<3:reasons.append('fewer_than_three_outer_rings')
    outer=rings[-3:]
    speeds=np.asarray([r['rotation_kms'] for r in outer])
    mean=float(np.mean(speeds));spread=float(np.max(abs(speeds-mean))/abs(mean)) if abs(mean)>0 else float('inf')
    extent=float(rings[-1]['radius_arcsec']/(half_light_radius_arcsec/1.678))
    if mean<30:reasons.append('outer_rotation_below_30_kms_or_sign_change')
    if spread>.10:reasons.append('outer_three_rings_not_flat_to_10_percent')
    if extent<3:reasons.append('radial_extent_below_three_inferred_disk_scale_lengths')
    if mean>0 and any(r['side_difference_kms']>.2*mean for r in outer):reasons.append('outer_two_side_asymmetry_above_20_percent')
    return dict(optical_flatness_candidate=not reasons,reasons=reasons,
        outer_mean_rotation_kms=mean,outer_max_fractional_deviation=spread,
        extent_in_inferred_disk_scale_lengths=extent,
        distance_status='withheld_missing_validated_HI_flat_speed_and_independent_3p6um_HI_flux',
        caveat='A photometric half-light radius divided by1.678 is an exponential-disk approximation, not a measured disk decomposition. Optical flatness alone does not identify HI Vflat or validate a BTFR distance.')
