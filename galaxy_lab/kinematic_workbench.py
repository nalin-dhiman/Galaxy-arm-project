"""Read MaNGA H-alpha velocity maps and export angular kinematic diagnostics."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from .spectral_kinematics import DiskGeometry,annular_los_fit,assess_flat_speed


def read_ha_maps(path,ra_deg,dec_deg):
    with fits.open(path) as h:
        ref=h['EMLINE_GVEL'];wcs=WCS(ref.header).celestial
        if not u.Unit(ref.header.get('BUNIT','')).is_equivalent(u.km/u.s):
            raise ValueError('Velocity map must declare km/s-compatible units')
        arrays={};shape=ref.data.shape[1:]
        points=np.array([(x,y) for x in (0,(shape[1]-1)/2,shape[1]-1)
                         for y in (0,(shape[0]-1)/2,shape[0]-1)])
        reference_world=wcs.all_pix2world(points,0)
        if not np.isfinite(reference_world).all():raise ValueError('Nonfinite celestial grid')
        center=SkyCoord(ra_deg*u.deg,dec_deg*u.deg,frame='icrs')
        cx,cy=wcs.world_to_pixel(center)
        if not (-.5<=cx<shape[1]-.5 and -.5<=cy<shape[0]-.5):
            raise ValueError('Supplied sky center is outside the velocity map')
        for name in ('EMLINE_GVEL','EMLINE_GVEL_IVAR','EMLINE_GVEL_MASK','EMLINE_GFLUX','EMLINE_GFLUX_IVAR','EMLINE_GFLUX_MASK'):
            extension=h[name]
            channels={str(v).strip():int(k[1:])-1 for k,v in extension.header.items() if k.startswith('C') and k[1:].isdigit()}
            # MASK/IVAR headers do not always repeat line names; in those cases
            # the science parent's channel ordering is the documented contract.
            parent='EMLINE_GVEL' if 'GVEL' in name else 'EMLINE_GFLUX'
            if 'Ha-6564' not in channels:
                channels={str(v).strip():int(k[1:])-1 for k,v in h[parent].header.items() if k.startswith('C') and k[1:].isdigit()}
            if 'Ha-6564' not in channels:raise ValueError('H-alpha channel missing')
            a=np.asarray(extension.data[channels['Ha-6564']])
            if a.shape!=shape:raise ValueError('Velocity/flux/variance/mask shapes disagree')
            other=WCS(extension.header).celestial
            other_world=other.all_pix2world(points,0)
            if (not np.isfinite(other_world).all() or
                not np.allclose(other_world,reference_world,atol=1e-9,rtol=0) or
                other.wcs.radesys!=wcs.wcs.radesys or
                not np.isclose(other.wcs.equinox,wcs.wcs.equinox,equal_nan=True)):
                raise ValueError('Velocity/flux/variance/mask WCS disagree')
            arrays[name]=a.copy()
        yy,xx=np.indices(shape);sky=wcs.pixel_to_world(xx,yy)
        # Both coordinates use the explicit ICRS frame, including FK5 header
        # transformation, instead of treating small FK5/ICRS offsets as pixels.
        transformed=sky.icrs
        sky=SkyCoord(transformed.ra,transformed.dec,frame='icrs')
        center=SkyCoord(ra_deg*u.deg,dec_deg*u.deg,frame='icrs')
        east,north=center.spherical_offsets_to(sky)
        factor=u.Unit(ref.header['BUNIT']).to(u.km/u.s)
        v=np.asarray(arrays['EMLINE_GVEL'],float)*factor
        ivar=np.asarray(arrays['EMLINE_GVEL_IVAR'],float)/factor**2
        flux=np.asarray(arrays['EMLINE_GFLUX'],float);fi=np.asarray(arrays['EMLINE_GFLUX_IVAR'],float)
        valid=(arrays['EMLINE_GVEL_MASK']==0)&(arrays['EMLINE_GFLUX_MASK']==0)
        valid &= np.isfinite(v)&np.isfinite(ivar)&(ivar>0)&np.isfinite(flux)&np.isfinite(fi)&(fi>0)
        valid &= flux*np.sqrt(np.maximum(0,fi))>=5
        return dict(east=east.arcsec,north=north.arcsec,velocity=v,inverse_variance=ivar,
            valid=valid,flux=flux,wcs=wcs,dapqual=int(h[0].header.get('DAPQUAL',0)),
            input_guess_cz_kms=float(h[0].header.get('SCINPVEL',0)),plateifu=h[0].header.get('PLATEIFU','unknown'))


def analyze_maps(path,output_dir,*,center_ra_deg,center_dec_deg,axis_ratio,pa_deg,
                 half_light_radius_arcsec,psf_fwhm_arcsec,observation_quality_flags=0):
    out=Path(output_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be new or empty')
    data=read_ha_maps(path,center_ra_deg,center_dec_deg)
    geometry=DiskGeometry(axis_ratio,pa_deg)
    result=annular_los_fit(*(data[k] for k in ('east','north','velocity','inverse_variance','valid')),
                          geometry,psf_fwhm_arcsec=psf_fwhm_arcsec)
    quality=int(observation_quality_flags)|data['dapqual']
    gate=assess_flat_speed(result,half_light_radius_arcsec=half_light_radius_arcsec,quality_flags=quality)
    sensitivity=[]
    scenarios=[('pa_minus5',axis_ratio,pa_deg-5,.2),('pa_plus5',axis_ratio,pa_deg+5,.2),
        ('q_minus0p05',axis_ratio-.05,pa_deg,.2),('q_plus0p05',min(.99,axis_ratio+.05),pa_deg,.2),
        ('thickness0p1',axis_ratio,pa_deg,.1),('thickness0p3',axis_ratio,pa_deg,.3)]
    for label,q,pa,q0 in scenarios:
        try:
            alt=annular_los_fit(*(data[k] for k in ('east','north','velocity','inverse_variance','valid')),
                DiskGeometry(q,pa,q0),psf_fwhm_arcsec=psf_fwhm_arcsec)
            sensitivity.append(dict(scenario=label,status='processed',
                outer_mean_rotation_kms=float(np.mean([r['rotation_kms'] for r in alt['rings'][-3:]])),
                inclination_deg=alt['inclination_deg'],rings=len(alt['rings'])))
        except ValueError as exc:sensitivity.append(dict(scenario=label,status='failed',reason=str(exc)))
    model=result.pop('model');fit_mask=result.pop('fit_mask')
    metadata=dict(status='angular_spectral_kinematic_diagnostic',source=str(Path(path).resolve()),
        source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        plateifu=data['plateifu'],center_icrs_deg=[center_ra_deg,center_dec_deg],
        observed_axis_ratio=axis_ratio,photometric_pa_east_of_north_deg=pa_deg,
        assumed_intrinsic_thickness=.2,psf_fwhm_arcsec=psf_fwhm_arcsec,
        quality_flags_combined=quality,input_guess_cz_kms=data['input_guess_cz_kms'],
        valid_ha_spaxels=int(data['valid'].sum()),fit=result,flatness=gate,geometry_sensitivity=sensitivity,
        limitations=['Nonzero DAP velocity or flux masks and Halpha S/N<5 are excluded; this conservative choice can remove valid data.',
            'The DAP map velocities are relative to the input guess redshift. The systemic offset is fitted; no redshift distance is used.',
            'No complete spatial covariance, beam-convolved disk model, gas-pressure/asymmetric-drift correction or independent inclination fit.',
            'Bar/warp/outflow/noncircular motion and erroneous photometric geometry can bias circular-speed coefficients.',
            'Optical ring flatness is a diagnostic and cannot substitute for a calibrated HI velocity observable or absent IR/HI flux.'])
    out.mkdir(parents=True,exist_ok=True)
    import pandas as pd
    pd.DataFrame(result['rings']).to_csv(out/'angular_rotation_curve.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(out/'geometry_sensitivity.csv',index=False)
    (out/'analysis.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    header=data['wcs'].to_header();header['BUNIT']='km/s'
    fits.writeto(out/'los_model.fits',model.astype('f4'),header)
    fits.writeto(out/'los_residual.fits',np.where(fit_mask,data['velocity']-model,np.nan).astype('f4'),header)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/next_galaxy_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(15,3.7),constrained_layout=True)
    lim=float(np.nanpercentile(abs(data['velocity'][data['valid']]),98))
    for ax,img,title in zip(axes[:3],[np.where(data['valid'],data['velocity'],np.nan),model,np.where(fit_mask,data['velocity']-model,np.nan)],['Halpha LOS velocity','Fixed-geometry annular fit','LOS residual']):
        im=ax.imshow(img,origin='lower',cmap='RdBu_r',vmin=-lim,vmax=lim);ax.set_title(title);fig.colorbar(im,ax=ax,label='km/s',shrink=.7)
        ax.set_xticks([]);ax.set_yticks([])
    rr=result['rings'];r=np.array([x['radius_arcsec'] for x in rr]);v=np.array([x['rotation_kms'] for x in rr])
    error=[max(x['formal_error_kms'],x['sector_jackknife_kms'] or 0) for x in rr]
    axes[3].errorbar(r,v,yerr=error,fmt='o',label='Error proxy')
    axes[3].plot(r,[x['side1_rotation_kms'] for x in rr],':',label='Side1');axes[3].plot(r,[x['side2_rotation_kms'] for x in rr],':',label='Side2')
    axes[3].set(xlabel='Deprojected angular radius (arcsec)',ylabel='Circular-disk coefficient (km/s)');axes[3].legend(fontsize=7)
    fig.suptitle(str(data['plateifu'])+': spectral diagnostics; distance withheld')
    fig.savefig(out/'kinematics.png',dpi=150);plt.close(fig)
    from html import escape
    (out/'analysis.html').write_text('<!doctype html><meta charset="utf-8"><title>Spectral kinematics</title><style>body{font:17px/1.5 sans-serif;max-width:1100px;margin:40px auto;padding:20px}img{width:100%}pre{white-space:pre-wrap}</style><h1>Angular spectral kinematics: '+escape(str(data['plateifu']))+'</h1><p>Independent emission-line measurements; physical distance withheld.</p><img src="kinematics.png" alt="Velocity maps and angular rotation curve"><p><a href="angular_rotation_curve.csv">Angular rotation coefficients</a> · <a href="geometry_sensitivity.csv">Geometry sensitivity</a></p><pre>'+escape(json.dumps(metadata,indent=2))+'</pre>')
    return metadata
