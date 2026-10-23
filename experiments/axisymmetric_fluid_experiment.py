#!/usr/bin/env python3
"""Execute axisymmetric local equilibrium validation and conditional disk pilot.

No observed velocity derivative is used. Positive flattened baryon components
are fitted ONLY to the supplied SPARC baryonic-force templates. Their vertical
thickness is a declared sensitivity assumption, not an observed measurement.
The halo's finite computational edge requires confining stresses. Numerical
convergence is evidence for the stated local problem, not a global SIDM halo.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault("MPLCONFIGDIR", "/tmp/galaxy-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls, brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from galaxy_lab.axisymmetric_fluid import AxisymmetricFluidHalo, MiyamotoNagai
from galaxy_lab.fluid import G, IsothermalFluidHalo
from galaxy_lab.coupled_fluid import CoupledSphericalFluidHalo, PlummerBaryons
from experiments.dynamics_experiment import load_curve, fit_curve, MODELS, prediction

OUT = ROOT/"executed_solid/fluid"


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")


def refine_experiment():
    baryons=(MiyamotoNagai(1e10,2.,.4),)
    r=np.geomspace(.2,20.,90)
    mu=np.linspace(0,.95,9)
    R=r[:,None]*np.sqrt(1-mu*mu); z=r[:,None]*mu
    configurations=[(160,8,32,50.),(320,8,40,50.),(640,8,48,50.),
                    (640,16,56,50.),(640,24,72,50.),(640,32,88,50.),
                    (640,24,72,100.),(640,24,72,200.)]
    rows,curves=[],[]
    reference=None
    solutions={}
    for nr,lm,na,outer in configurations:
        h=AxisymmetricFluidHalo(3e7,100.,baryons,outer,nr,lm,na)
        v=h.equatorial_circular_velocity(r)
        density=h.density(R,z)
        residual=h.poisson_density(R,z)/density-1
        label=f"N{nr}_L{lm}_A{na}_outer{outer:g}"
        record=dict(id=label,**h.diagnostics,
                    max_abs_relative_poisson_residual=float(np.max(abs(residual))),
                    median_abs_relative_poisson_residual=float(np.median(abs(residual))))
        solutions[label]=(h,v)
        rows.append(record)
        for ri,vi in zip(r,v):
            curves.append(dict(id=label,radius_kpc=float(ri),velocity_kms=float(vi)))
    reference=solutions["N640_L32_A88_outer50"][1]
    for row in rows:
        v=solutions[row['id']][1]
        row['max_speed_difference_from_N640_L32_outer50_kms']=float(np.max(abs(v-reference)))
    fixed=solutions["N640_L24_A72_outer50"][0]
    iso=IsothermalFluidHalo(3e7,100.)
    zero=AxisymmetricFluidHalo(3e7,100.,(),50.,640,16,56)
    spherical=AxisymmetricFluidHalo(3e7,100.,(MiyamotoNagai(1e10,0.,2.),),50.,640,16,56)
    sphere_reference=CoupledSphericalFluidHalo(3e7,100.,PlummerBaryons(1e10,2.),50.)
    checks=dict(zero_baryon_max_relative_speed_error=float(np.max(abs(zero.equatorial_circular_velocity(r)/iso.circular_velocity(r)-1))),
                spherical_baryon_max_relative_speed_error=float(np.max(abs(spherical.equatorial_circular_velocity(r)/sphere_reference.circular_velocity(r)-1))),
                domain50_vs100_max_speed_difference_kms=float(np.max(abs(solutions['N640_L24_A72_outer50'][1]-solutions['N640_L24_A72_outer100'][1]))),
                domain100_vs200_max_speed_difference_kms=float(np.max(abs(solutions['N640_L24_A72_outer100'][1]-solutions['N640_L24_A72_outer200'][1]))))
    # Isodensity flattening is evaluated from the forward model, not inferred
    # from observed disk geometry. A pressure-supported halo follows potential.
    flattening=[]
    for req in (1.,2.,5.,10.,20.):
        level=float(fixed.density(req,0.))
        zp=brentq(lambda zz:float(fixed.density(0.,zz))-level,0.,min(49.,req*1.5))
        flattening.append(dict(equatorial_radius_kpc=req,polar_radius_kpc=zp,axis_ratio=zp/req))
    scale=3.
    hs=AxisymmetricFluidHalo(3e7/scale**2,100.,(MiyamotoNagai(1e10*scale,2.*scale,.4*scale),),50.*scale,640,24,72)
    checks['distance_rescaling_max_relative_speed_error']=float(np.max(abs(hs.equatorial_circular_velocity(scale*r)/fixed.equatorial_circular_velocity(r)-1)))
    assert checks['zero_baryon_max_relative_speed_error'] < .002
    assert checks['spherical_baryon_max_relative_speed_error'] < .002
    assert checks['distance_rescaling_max_relative_speed_error'] < 1e-10
    assert rows[4]['max_abs_relative_poisson_residual'] < .005
    pd.DataFrame(rows).to_csv(OUT/'grid_multipole_domain_refinement.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'refinement_rotation_profiles.csv',index=False)
    pd.DataFrame(flattening).to_csv(OUT/'known_model_density_flattening.csv',index=False)
    dump(OUT/'closure_checks.json',checks)
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    rr=np.linspace(0,15,180);zz=np.linspace(0,15,180);xx,yy=np.meshgrid(rr,zz)
    rho=fixed.density(xx,yy)
    axes[0].contourf(xx,yy,np.log10(rho),levels=20,cmap='magma')
    axes[0].contour(xx,yy,rho,levels=np.geomspace(rho.min()*2,rho.max()/2,7),colors='white',linewidths=.5)
    axes[0].set(xlabel='Cylindrical R [kpc]',ylabel='Height z [kpc]',title='Dark density responding to a flattened disk',aspect='equal')
    for label,(h,v) in solutions.items():
        axes[1].plot(r,v,label=label)
    axes[1].set(xscale='log',xlabel='R [kpc]',ylabel='Circular speed [km/s]',title='Grid / angular / boundary refinement')
    axes[1].legend(fontsize=6)
    axes[2].plot([a['equatorial_radius_kpc'] for a in flattening],[a['axis_ratio'] for a in flattening],'o-')
    axes[2].set(xlabel='Isodensity equatorial intercept [kpc]',ylabel='Polar / equatorial intercept',ylim=(.4,1.02),title='Halo flattening is predicted conditionally')
    fig.savefig(OUT/'axisymmetric_validation.png',dpi=170);plt.close(fig)
    return dict(checks=checks,refinement=rows,flattening=flattening,
                interpretation='Converged local finite-domain pressure equilibrium; imposed edge stresses, not globally isolated finite halo.')


def template_baryons(a,thickness_fraction=.1):
    """Nonnegative fixed basis fits to baryon-force columns only; never Vobs.

    Use six fixed plane scales per stellar/gas/bulge component. Basis weights
    are nonnegative masses. Vertical thickness is deliberately not fitted.
    Signed outward gas forces cannot be represented by this positive inward
    family; their retained force residual is reported rather than clipped.
    """
    r=a[:,0]
    scales=np.geomspace(max(.1,np.percentile(r,10)*.5),np.max(r)*2,6)
    shapes=np.stack([G*1e10*r*r/(r*r+c*c)**1.5 for c in scales],axis=1)
    baryons=[]; diagnostics=[]; total=np.zeros_like(r)
    parts=dict(stellar_disk=.5*a[:,4]**2,gas_signed=a[:,3]*abs(a[:,3]),stellar_bulge=.7*a[:,5]**2)
    for name,force in parts.items():
        if not np.any(force): continue
        weights,cost=nnls(shapes,force)
        approx=shapes@weights
        total+=approx
        for c,w in zip(scales,weights):
            if w > max(weights.max()*1e-10,1e-10):
                # Bulge spherical; disk/gas thickness is a common declared ratio.
                eta=1. if name=='stellar_bulge' else thickness_fraction
                baryons.append(MiyamotoNagai(float(w*1e10),float(c*(1-eta)),float(c*eta)))
        speed_residual=np.sqrt(np.maximum(approx,0))-np.sign(force)*np.sqrt(abs(force))
        diagnostics.append(dict(component=name,force_rmse_kms2=float(np.sqrt(np.mean((approx-force)**2))),
                                signed_speed_template_rmse_kms=float(np.sqrt(np.mean(speed_residual**2))),
                                negative_force_radii=int(np.sum(force<0)),basis_scales_kpc=scales.tolist(),
                                mass_weights_msun=(weights*1e10).tolist(),
                                thickness_fraction=1. if name=='stellar_bulge' else thickness_fraction))
    original=sum(parts.values())
    return tuple(baryons),dict(components=diagnostics,
        total_force_rmse_kms2=float(np.sqrt(np.mean((total-original)**2))),
        total_baryon_speed_rmse_kms=float(np.sqrt(np.mean((np.sqrt(total)-np.sign(original)*np.sqrt(abs(original)))**2))),
        total_approximated_baryon_mass_msun=float(sum(b.mass_msun for b in baryons)),
        warning='Force-only positive MN approximation. Not unique baryon deprojection; no fitted thickness, maskbits or independent mass uncertainty model.')


def fit_axisymmetric(a,train,baryons,outer,coupled):
    failures=0
    def model(p,r):
        if coupled:
            return AxisymmetricFluidHalo(10.**p[0],10.**p[1],baryons,outer,220,12,40,tolerance=8e-7).equatorial_circular_velocity(r)
        h=IsothermalFluidHalo(10.**p[0],10.**p[1])
        return np.sqrt(h.circular_velocity(r)**2+sum(b.circular_velocity_squared(r) for b in baryons))
    def residual(p):
        nonlocal failures
        try:
            v=model(p,a[train,0])
            if not np.isfinite(v).all(): raise RuntimeError('nonfinite prediction')
            return (v-a[train,1])/a[train,2]
        except RuntimeError:
            failures+=1
            return np.full(train.sum(),1e5)+np.sum(p*p)
    candidates=[]
    seeds=[[7.,2.],[8.,2.],[6.,2.2],[8.5,2.3]]
    # Audit the entire declared rectangle using TRAINING residuals only.
    # This guards against interpreting a missed optimizer basin as physics.
    audit=[]
    for logrho in np.linspace(4.,10.,14):
        for logsigma in np.linspace(1.,2.7,13):
            p=np.array([logrho,logsigma])
            cost=float(np.sum(residual(p)**2))
            if np.isfinite(cost) and cost<1e9:
                audit.append((cost,p))
    seeds.extend(p.tolist() for _,p in sorted(audit,key=lambda row:row[0])[:6])
    for seed in seeds:
        opt=least_squares(residual,seed,bounds=([4.,1.],[10.,2.7]),diff_step=1e-3,
                          max_nfev=160,ftol=2e-7,xtol=2e-7,gtol=2e-7)
        if opt.success and np.sum(opt.fun**2)<1e9:
            candidates.append(opt)
    if not candidates:
        return None,dict(success=False,forward_failures=failures,reason='No converged finite model fit')
    best=min(candidates,key=lambda o:np.sum(o.fun**2))
    pred=model(best.x,a[:,0])
    test=~train
    result=dict(success=True,forward_failures=failures,converged_starts=len(candidates),
                n_local_starts=len(seeds),coarse_audit_points=14*13,
                coarse_audit_min_training_chi2=min((c for c,_ in audit),default=None),
                rho0_msun_kpc3=float(10**best.x[0]),sigma_kms=float(10**best.x[1]),
                training_chi2=float(np.sum(best.fun**2)),
                training_rmse_kms=float(np.sqrt(np.mean((pred[train]-a[train,1])**2))),
                heldout_rmse_kms=float(np.sqrt(np.mean((pred[test]-a[test,1])**2))),
                bound_hit=bool(np.any(np.minimum(best.x-[4.,1.],[10.,2.7]-best.x)<1e-3)))
    return pred,result


def real_disk_pilot():
    windows=pd.read_csv(ROOT/'executed_filters/morphology/arm_windows.csv').set_index('galaxy')
    distances=pd.read_csv(ROOT/'executed_research/images/sparc_metadata.csv').set_index('galaxy')
    records,predictions,templates,sensitivity=[],[],[],[]
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    for ax,g in zip(axes,('NGC2403','NGC3198','NGC5055')):
        a=load_curve(ROOT/f'data/{g}_rotmod.dat')
        kpc_per_arcsec=float(distances.loc[g,'distance_mpc'])*1000*np.pi/(180*3600)
        w=windows.loc[g]
        inside=(a[:,0]>=float(w.inner_radius_arcsec)*kpc_per_arcsec)&(a[:,0]<=float(w.outer_radius_arcsec)*kpc_per_arcsec)
        train=~inside
        baryons,description=template_baryons(a,.1)
        templates.append(dict(galaxy=g,**description))
        outer=3*float(a[:,0].max())
        ax.errorbar(a[:,0],a[:,1],a[:,2],fmt='.',color='gray',label='Independent measured velocities')
        ax.axvspan(float(w.inner_radius_arcsec)*kpc_per_arcsec,float(w.outer_radius_arcsec)*kpc_per_arcsec,alpha=.12,color='teal')
        for coupled in (False,True):
            name='Coupled axisymmetric + MN baryons' if coupled else 'Uncontracted halo + same MN baryons'
            pred,result=fit_axisymmetric(a,train,baryons,outer,coupled)
            row=dict(galaxy=g,model=name,n_train=int(train.sum()),n_test=int(inside.sum()),
                     outer_radius_kpc=outer,distance_mpc_external=float(distances.loc[g,'distance_mpc']),
                     baryon_thickness_fraction=.1,**result)
            records.append(row)
            if pred is None: continue
            ax.plot(a[:,0],pred,label=name)
            for i,r in enumerate(a[:,0]):
                predictions.append(dict(galaxy=g,model=name,radius_kpc=float(r),observed_velocity_kms=float(a[i,1]),
                                        predicted_velocity_kms=float(pred[i]),error_kms=float(a[i,2]),heldout=bool(inside[i])))
            if coupled:
                # Numerical and thickness checks HOLD the trained dark parameters
                # fixed; no choice is selected using the heldout velocities.
                rho,sigma=result['rho0_msun_kpc3'],result['sigma_kms']
                for eta,nr,lm,na,domain in ((.05,440,24,72,outer),(.1,440,24,72,outer),
                                            (.2,440,24,72,outer),(.1,440,24,72,2*outer)):
                    components,_=template_baryons(a,eta)
                    h=AxisymmetricFluidHalo(rho,sigma,components,domain,nr,lm,na,tolerance=8e-7)
                    try:
                        vv=h.equatorial_circular_velocity(a[:,0])
                        # Audit where the actual fitted problem closes, not just
                        # where a friendly synthetic model converges.
                        rad=np.geomspace(max(.2,float(a[:,0].min())),a[:,0].max(),50)
                        mus=np.linspace(0,.95,7)
                        RR=rad[:,None]*np.sqrt(1-mus*mus);ZZ=rad[:,None]*mus
                        resid=h.poisson_density(RR,ZZ)/h.density(RR,ZZ)-1
                        sensitivity.append(dict(galaxy=g,thickness_fraction=eta,
                                                max_speed_change_from_training_grid_kms=float(np.max(abs(vv-pred))),
                                                heldout_rmse_kms=float(np.sqrt(np.mean((vv[inside]-a[inside,1])**2))),
                                                max_abs_relative_poisson_residual=float(np.max(abs(resid))),
                                                median_abs_relative_poisson_residual=float(np.median(abs(resid))),
                                                **h.diagnostics))
                    except RuntimeError as exc:
                        sensitivity.append(dict(galaxy=g,thickness_fraction=eta,success=False,reason=str(exc)))
        # Exact supplied signed baryon-template baseline, separate from the MN
        # approximation control, prevents conflating representation and coupling.
        old=fit_curve(a[train],MODELS[0])
        pp=prediction(a,old['p'],MODELS[0])
        records.append(dict(galaxy=g,model='Uncontracted halo + original signed SPARC baryons',success=bool(old['success']),
                            n_train=int(train.sum()),n_test=int(inside.sum()),
                            heldout_rmse_kms=float(np.sqrt(np.mean((pp[inside]-a[inside,1])**2))),
                            bound_hit=bool(old['bound_hit'])))
        ax.plot(a[:,0],pp,'--',label='Uncontracted + original SPARC baryons')
        ax.set(title=g,xlabel='R [kpc, external catalog distance]',ylabel='Speed [km/s]')
        ax.grid(alpha=.2)
        print(g,json.dumps([r for r in records if r['galaxy']==g]),flush=True)
    axes[0].legend(fontsize=6)
    fig.suptitle('Conditional image-region holdout: trained outside shaded regions; exploratory baryon deprojection')
    fig.savefig(OUT/'real_disk_pilot.png',dpi=170);plt.close(fig)
    pd.DataFrame(records).to_csv(OUT/'real_disk_fits.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'real_disk_predictions.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'real_disk_numerical_thickness_sensitivity.csv',index=False)
    dump(OUT/'baryon_template_approximations.json',templates)
    return dict(fits=records,template_diagnostics=templates,sensitivity=sensitivity,
                interpretation='Exploratory three-galaxy conditional regional interpolation with external D, M/L and force templates. Axisymmetric local equilibrium, approximate baryon deprojection and imposed finite pressure boundary; no fluid microphysics or image-only distance claim.')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--validation-only',action='store_true')
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    summary=dict(validation=refine_experiment())
    if not args.validation_only:
        summary['real_disk_pilot']=real_disk_pilot()
    dump(OUT/'summary.json',summary)
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
