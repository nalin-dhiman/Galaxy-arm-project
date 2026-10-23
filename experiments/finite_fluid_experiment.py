#!/usr/bin/env python3
"""Validate a free-edge equilibrium and execute a fixed three-galaxy pilot.

The previously examined annuli are reused as exploratory regional holdouts,
not a new independent confirmatory sample. No parameters/priors are chosen
using held-out speeds. Public photometric scale lengths inform a population
thickness prior; this is not a resolved measurement of these three scale heights.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/next-galaxy-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq,least_squares

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from galaxy_lab.finite_fluid import FiniteFluidHalo,SphericalFiniteHalo,king_moment
from galaxy_lab.axisymmetric_fluid import AxisymmetricFluidHalo,MiyamotoNagai
from galaxy_lab.fluid import IsothermalFluidHalo
from experiments.axisymmetric_fluid_experiment import template_baryons
from experiments.dynamics_experiment import load_curve
from experiments.btfr_distance_experiment import read_table,SOURCE

OUT=ROOT/'executed_next/fluid'

def dump(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def vertical_first_moment(components,R):
    """Local <|z|> of the analytic MN mixture, integrated over all height."""
    scale=max([b.b_kpc for b in components]+[.1])
    def rho(u):return sum(float(b.density(R,u*scale)) for b in components)
    den=quad(rho,0,np.inf,epsabs=1e-4,epsrel=2e-7,limit=150)[0]
    num=quad(lambda u:rho(u)*u,0,np.inf,epsabs=1e-4,epsrel=2e-7,limit=150)[0]
    return scale*num/den


def photometric_baryons(a,disk_scale_kpc,height_multiplier=1.):
    """Keep plane forces fixed; match stellar vertical moment to a prior.

    Bershady2010 log(hR/hz)=.367log(hR/kpc)+.708, ~25% per-galaxy
    logarithmic scatter. An exponential profile has <|z|>=hz. MN b is NOT hz.
    Fixed bulges stay spherical. Gas thickness remains the old unmeasured .1
    fractional assumption; its independent vertical information is unavailable.
    """
    _,description=template_baryons(a,.1)
    stellar=[p for p in description['components'] if p['component']=='stellar_disk'][0]
    target=float(10**(-.708)*disk_scale_kpc**.633*height_multiplier)
    def components_for(eta):
        return tuple(MiyamotoNagai(float(m),float(c*(1-eta)),float(c*eta))
                     for c,m in zip(stellar['basis_scales_kpc'],stellar['mass_weights_msun']) if m>1.)
    radius=2.2*disk_scale_kpc
    lower,upper=1e-5,.95
    available=(vertical_first_moment(components_for(lower),radius),vertical_first_moment(components_for(upper),radius))
    if not available[0]<target<available[1]:
        raise RuntimeError('Photometric height prior cannot be represented by declared positive MN family')
    eta=float(brentq(lambda x:vertical_first_moment(components_for(x),radius)-target,lower,upper,xtol=1e-9))
    components=[]
    for part in description['components']:
        fraction=eta if part['component']=='stellar_disk' else (1. if part['component']=='stellar_bulge' else .1)
        for c,m in zip(part['basis_scales_kpc'],part['mass_weights_msun']):
            if m>1.:components.append(MiyamotoNagai(float(m),float(c*(1-fraction)),float(c*fraction)))
    record=dict(disk_scale_kpc=float(disk_scale_kpc),height_prior_kpc=target,
                matching_radius_kpc=radius,matched_stellar_first_moment_kpc=vertical_first_moment(components_for(eta),radius),
                stellar_MN_thickness_fraction=eta,prior_log_scatter=.25,height_multiplier=height_multiplier,
                gas_thickness_fraction=.1,baryon_force_diagnostics=description,
                interpretation='Population prior from red/NIR edge-on photometry applied to published 3.6um Rdisk. Not a direct scale-height observation; wavelength, population and MN-deprojection systematics remain.')
    return tuple(components),record


def validate():
    baryons=(MiyamotoNagai(1e10,2.,.4),)
    radii=np.geomspace(.3,20.,60);mu=np.linspace(0,.95,9)
    R=radii[:,None]*np.sqrt(1-mu**2);z=radii[:,None]*mu
    rows=[];solutions={}
    for nr,lm,na,domain in [(200,8,32,250.),(400,16,48,250.),(640,24,72,250.),
                            (960,32,88,250.),(960,32,88,500.)]:
        h=FiniteFluidHalo(3e7,100.,baryons,domain,nr,lm,na,tolerance=1e-9)
        v=h.equatorial_circular_velocity(radii)
        residual=h.poisson_density(R,z)/h.density(R,z)-1
        key=f'N{nr}_L{lm}_D{domain:g}'
        rows.append(dict(id=key,**h.diagnostics,max_relative_poisson_residual=float(np.max(abs(residual)))))
        solutions[key]=(h,v)
    h,v=solutions['N960_L32_D250']
    for row in rows:row['max_speed_difference_from_highest_grid_kms']=float(np.max(abs(solutions[row['id']][1]-v)))
    ref=SphericalFiniteHalo(3e7,100.,5.)
    sphere=FiniteFluidHalo(3e7,100.,(),300.,960,8,32,tolerance=1e-10)
    r=np.geomspace(.2,100.,90)
    relative_speed=float(np.max(abs(sphere.equatorial_circular_velocity(r)/ref.circular_velocity(r)-1)))
    step=1e-4
    pg=(h.pressure(R+step,z)-h.pressure(R-step,z))/(2*step)
    phig=(h.potential_offset(R+step,z)-h.potential_offset(R-step,z))/(2*step)
    hydro=float(np.max(abs(pg/(h.density(R,z)*phig)+1)))
    scale=3.
    hs=FiniteFluidHalo(3e7/scale**2,100.,(MiyamotoNagai(1e10*scale,2*scale,.4*scale),),750.,960,32,88,tolerance=1e-9)
    similarity=float(np.max(abs(hs.equatorial_circular_velocity(radii*scale)/v-1)))
    edges=[]
    for angle in np.linspace(0,1,21):
        edge=brentq(lambda rad:float(h.relative_binding(rad*np.sqrt(1-angle**2),rad*angle)),.001,250.)
        edges.append(dict(mu=float(angle),edge_radius_kpc=edge,
                          density_at_edge=float(h.density(edge*np.sqrt(1-angle**2),edge*angle)),
                          pressure_at_edge=float(h.pressure(edge*np.sqrt(1-angle**2),edge*angle))))
    check=dict(spherical_ODE_max_relative_speed_error=relative_speed,
               spherical_mass_relative_error=abs(sphere.diagnostics['total_halo_mass_msun']/ref.total_mass_msun-1),
               hydrostatic_max_relative_residual=hydro,
               distance_similarity_max_relative_speed_error=similarity,
               doubled_vacuum_domain_max_speed_difference_kms=float(np.max(abs(solutions['N960_L32_D500'][1]-v))),
               doubled_vacuum_domain_relative_mass_change=abs(solutions['N960_L32_D500'][0].diagnostics['total_halo_mass_msun']/h.diagnostics['total_halo_mass_msun']-1))
    assert relative_speed<3e-4 and hydro<1e-5 and similarity<1e-10
    assert rows[3]['max_relative_poisson_residual']<.002
    pd.DataFrame(rows).to_csv(OUT/'numerical_validation.csv',index=False)
    pd.DataFrame(edges).to_csv(OUT/'zero_pressure_surface.csv',index=False)
    dump(OUT/'closure_checks.json',check)
    fig,ax=plt.subplots(1,3,figsize=(15,4.3),constrained_layout=True)
    rr=np.linspace(0,155,210);zz=np.linspace(0,155,210);RR,ZZ=np.meshgrid(rr,zz)
    rho=h.density(RR,ZZ)
    ax[0].contourf(RR,ZZ,np.log10(np.maximum(rho,1)),levels=20,cmap='magma')
    ax[0].contour(RR,ZZ,h.relative_binding(RR,ZZ),levels=[0],colors='cyan')
    ax[0].set(xlabel='R [kpc]',ylabel='z [kpc]',title='Finite halo; cyan zero-pressure surface',aspect='equal')
    rad=np.geomspace(.1,220,300)
    ax[1].plot(rad,h.density(rad,0)/3e7,label='Density / central density')
    ax[1].plot(rad,h.local_dispersion_squared(rad,0)/1e4,label='Local dispersion² / scale²')
    ax[1].set(xscale='log',yscale='log',ylim=(1e-8,2),xlabel='R [kpc]',title='Pressure closure changes near the edge');ax[1].legend(fontsize=8)
    for label,(_,speed) in solutions.items():ax[2].plot(radii,speed-v,label=label)
    ax[2].set(xlabel='R [kpc]',ylabel='Speed difference [km/s]',title='Grid / multipole / vacuum-domain refinement');ax[2].legend(fontsize=6)
    fig.savefig(OUT/'finite_equilibrium_validation.png',dpi=170);plt.close(fig)
    return dict(checks=check,refinement=rows,edge_surface=edges)


def fit_model(a,train,baryons,name,central_binding=5.):
    failures=[]
    def make(p,refined=False,components=baryons):
        rho,scale=10**p[0],10**p[1]
        if name=='finite_coupled':
            return FiniteFluidHalo(rho,scale,components,radial_points=520 if refined else 240,
                                   lmax=24 if refined else 12,angular_points=72 if refined else 40,
                                   tolerance=2e-8 if refined else 5e-7,central_binding=central_binding)
        if name=='old_confined_coupled':
            return AxisymmetricFluidHalo(rho,scale,components,3*a[:,0].max(),
                                         520 if refined else 240,24 if refined else 12,72 if refined else 40,
                                         tolerance=2e-8 if refined else 5e-7)
        return SphericalFiniteHalo(rho,scale,central_binding)
    def predict(p,r,refined=False,components=baryons):
        h=make(p,refined,components)
        if name=='finite_uncoupled':return np.sqrt(h.circular_velocity(r)**2+sum(b.circular_velocity_squared(r) for b in components))
        return h.equatorial_circular_velocity(r)
    def residual(p):
        try:
            v=predict(p,a[train,0])
            if not np.isfinite(v).all():raise RuntimeError('nonfinite forward prediction')
            return (v-a[train,1])/a[train,2]
        except (RuntimeError,ValueError) as exc:
            failures.append(str(exc))
            return np.full(train.sum(),1e5)+float(np.sum(p*p))
    # Entire coarse rectangle evaluated on TRAINING velocities only.
    audit=[]
    for logrho in np.linspace(5.,9.,7):
        for logs in np.linspace(1.2,2.5,7):
            p=np.array([logrho,logs]);cost=float(np.sum(residual(p)**2))
            if cost<1e9:audit.append((cost,p))
    seeds=[p for _,p in sorted(audit,key=lambda x:x[0])[:4]]
    opts=[]
    for p in seeds:
        opt=least_squares(residual,p,bounds=([4.,1.],[10.,2.7]),diff_step=2e-3,
                          max_nfev=110,ftol=2e-6,xtol=2e-6,gtol=2e-6)
        if opt.success and np.sum(opt.fun**2)<1e9:opts.append(opt)
    if not opts:return None,dict(success=False,reason='No converged valid fit',forward_failures=len(failures),failure_reasons=sorted(set(failures)))
    best=min(opts,key=lambda x:np.sum(x.fun**2));pred=predict(best.x,a[:,0])
    row=dict(success=True,rho0_msun_kpc3=float(10**best.x[0]),velocity_scale_kms=float(10**best.x[1]),
             central_binding=central_binding,free_dark_parameters=2,
             training_chi2=float(np.sum(best.fun**2)),
             training_rmse_kms=float(np.sqrt(np.mean((pred[train]-a[train,1])**2))),
             heldout_rmse_kms=float(np.sqrt(np.mean((pred[~train]-a[~train,1])**2))),
             bound_hit=bool(np.any(np.minimum(best.x-[4.,1.],[10.,2.7]-best.x)<.002)),
             coarse_training_grid_points=49,local_starts=len(seeds),converged_starts=len(opts),
             forward_failures=len(failures),failure_reasons=sorted(set(failures)))
    return (best.x,pred,predict,make),row


def real_pilot():
    table=read_table().set_index('galaxy')
    windows=pd.read_csv(ROOT/'executed_filters/morphology/arm_windows.csv').set_index('galaxy')
    records=[];predictions=[];priors=[];sensitivities=[]
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    for ax,g in zip(axes,('NGC2403','NGC3198','NGC5055')):
        a=load_curve(ROOT/f'data/{g}_rotmod.dat');meta=table.loc[g];w=windows.loc[g]
        factor=meta.distance_mpc*1000*np.pi/(180*3600)
        inside=(a[:,0]>=w.inner_radius_arcsec*factor)&(a[:,0]<=w.outer_radius_arcsec*factor);train=~inside
        baryons,prior=photometric_baryons(a,float(meta.disk_scale_kpc));priors.append(dict(galaxy=g,**prior))
        ax.errorbar(a[:,0],a[:,1],a[:,2],fmt='.',color='gray',label='Observed independent speeds')
        ax.axvspan(w.inner_radius_arcsec*factor,w.outer_radius_arcsec*factor,color='teal',alpha=.12)
        for name in ('finite_uncoupled','old_confined_coupled','finite_coupled'):
            fit,record=fit_model(a,train,baryons,name)
            record=dict(galaxy=g,model=name,distance_mpc_external=float(meta.distance_mpc),n_train=int(train.sum()),n_test=int(inside.sum()),**record)
            records.append(record)
            print(json.dumps(record),flush=True)
            if fit is None:continue
            p,pred,predict,make=fit
            ax.plot(a[:,0],pred,label=name)
            for i,r in enumerate(a[:,0]):predictions.append(dict(galaxy=g,model=name,radius_kpc=float(r),observed_velocity_kms=float(a[i,1]),predicted_velocity_kms=float(pred[i]),heldout=bool(inside[i])))
            if name=='finite_coupled':
                model=make(p,True);vv=model.equatorial_circular_velocity(a[:,0])
                record['physical_diagnostics']=model.diagnostics
                record['refinement_max_speed_change_kms']=float(np.max(abs(vv-pred)))
                for mult in (np.exp(-.25),1.,np.exp(.25)):
                    comps,description=photometric_baryons(a,float(meta.disk_scale_kpc),float(mult))
                    hh=FiniteFluidHalo(10**p[0],10**p[1],comps,radial_points=520,lmax=24,angular_points=72,tolerance=2e-8)
                    v2=hh.equatorial_circular_velocity(a[:,0])
                    sensitivities.append(dict(galaxy=g,type='photometric_height_prior',height_multiplier=float(mult),
                        max_speed_change_from_primary_kms=float(np.max(abs(v2-pred))),
                        heldout_rmse_kms=float(np.sqrt(np.mean((v2[inside]-a[inside,1])**2))),
                        thickness_fraction=description['stellar_MN_thickness_fraction'],dark_parameters_held_fixed=True))
                for binding in (3.,7.):
                    hh=FiniteFluidHalo(10**p[0],10**p[1],baryons,radial_points=520,lmax=24,angular_points=72,tolerance=2e-8,central_binding=binding)
                    v2=hh.equatorial_circular_velocity(a[:,0])
                    sensitivities.append(dict(galaxy=g,type='unselected_cutoff_sensitivity',central_binding=binding,
                        max_speed_change_from_primary_kms=float(np.max(abs(v2-pred))),
                        heldout_rmse_kms=float(np.sqrt(np.mean((v2[inside]-a[inside,1])**2))),dark_parameters_held_fixed=True))
        ax.set(title=g,xlabel='R [kpc; supplied external distance]',ylabel='Speed [km/s]');ax.grid(alpha=.2)
    axes[0].legend(fontsize=6)
    fig.suptitle('Exploratory reused image-region holdouts; fixed photometric thickness prior, King W0=5')
    fig.savefig(OUT/'finite_real_galaxy_pilot.png',dpi=170);plt.close(fig)
    pd.DataFrame([{k:v for k,v in row.items() if k!='physical_diagnostics'} for row in records]).to_csv(OUT/'real_galaxy_fits.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'real_galaxy_predictions.csv',index=False)
    pd.DataFrame(sensitivities).to_csv(OUT/'prior_cutoff_sensitivity.csv',index=False)
    dump(OUT/'photometric_thickness_priors.json',priors)
    return dict(fits=records,photometric_thickness_priors=priors,sensitivities=sensitivities,
                interpretation='Exploratory reused three-galaxy conditional interpolation, not fresh validation. Finite closure W0=5 predeclared. Fixed stellar photometric population prior; gas height and M/L uncertain. No viscosity or absolute image-only distance is identified.')


def cutoff_profile():
    """Secondary training-only outer-cutoff profile, retaining every outcome.

    Added after inspecting the primary W0=5 pilot. This checks dependence on an
    explicit unobserved assumption; it is not a new confirmatory test. Only
    training chi2 selects the profiled model. The four-point grid does not define
    a confidence interval or establish a unique physical truncation radius.
    """
    table=read_table().set_index('galaxy')
    windows=pd.read_csv(ROOT/'executed_filters/morphology/arm_windows.csv').set_index('galaxy')
    primary=json.loads((OUT/'summary.json').read_text())['real_galaxy_pilot']['fits']
    records=[];selected=[]
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),constrained_layout=True)
    for ax,g in zip(axes,('NGC2403','NGC3198','NGC5055')):
        a=load_curve(ROOT/f'data/{g}_rotmod.dat');meta=table.loc[g];w=windows.loc[g]
        factor=meta.distance_mpc*1000*np.pi/(180*3600)
        inside=(a[:,0]>=w.inner_radius_arcsec*factor)&(a[:,0]<=w.outer_radius_arcsec*factor);train=~inside
        baryons,_=photometric_baryons(a,float(meta.disk_scale_kpc))
        for binding in (3.,5.,7.,9.):
            if binding==5.:
                result=dict(next(x for x in primary if x['galaxy']==g and x['model']=='finite_coupled'))
                result.pop('physical_diagnostics',None)
            else:
                fit,result=fit_model(a,train,baryons,'finite_coupled',binding)
                if fit is not None:
                    p,pred,_,make=fit;h=make(p,True)
                    refined=h.equatorial_circular_velocity(a[:,0])
                    result['refinement_max_speed_change_kms']=float(np.max(abs(refined-pred)))
                    result['physical_diagnostics']=h.diagnostics
            result.update(galaxy=g,central_binding=binding,posthoc_secondary_analysis=True)
            records.append(result)
            print('Cutoff profile',json.dumps(result),flush=True)
        valid=[r for r in records if r['galaxy']==g and r['success']]
        chosen=min(valid,key=lambda r:r['training_chi2'])
        row=dict(galaxy=g,central_binding_selected_on_training=chosen['central_binding'],
                 training_chi2=chosen['training_chi2'],heldout_rmse_kms=chosen['heldout_rmse_kms'],
                 grid_boundary_selected=chosen['central_binding'] in (3.,9.),
                 interpretation='Cutoff shape selected on training only from four prelisted values. At least three effective dark-model choices; reused test is exploratory.')
        selected.append(row)
        ax.plot([x['central_binding'] for x in valid],[x['training_chi2'] for x in valid],'o-',label='Training chi²')
        ax.set(title=g,xlabel='Central binding W0',ylabel='Training chi²',yscale='log')
        dump(OUT/'cutoff_profile.json',dict(fits=records,training_selected=selected))
    fig.suptitle('Secondary cutoff sensitivity: training chi² selects; no held-out errors used')
    fig.savefig(OUT/'training_cutoff_profile.png',dpi=170);plt.close(fig)
    pd.DataFrame([{k:v for k,v in row.items() if k!='physical_diagnostics'} for row in records]).to_csv(OUT/'cutoff_profile.csv',index=False)
    return dict(fits=records,training_selected=selected,
                interpretation='Posthoc after primary fixed-W0 pilot. All grid outcomes retained. No posterior on outer mass, tidal history, viscosity or image distance.')


def fitted_closure_audit(summary):
    """Check actual fitted states, not just the friendly controlled example."""
    table=read_table().set_index('galaxy');rows=[]
    fits=[r for r in summary['real_galaxy_pilot']['fits'] if r['model']=='finite_coupled']
    fits+=summary.get('secondary_cutoff_profile',{}).get('fits',[])
    seen=set()
    for row in fits:
        if not row['success']:continue
        key=(row['galaxy'],row['central_binding'])
        if key in seen:continue
        seen.add(key)
        a=load_curve(ROOT/f'data/{key[0]}_rotmod.dat')
        baryons,_=photometric_baryons(a,float(table.loc[key[0],'disk_scale_kpc']))
        h=FiniteFluidHalo(row['rho0_msun_kpc3'],row['velocity_scale_kms'],baryons,
                           radial_points=640,lmax=32,angular_points=88,tolerance=1e-9,central_binding=key[1])
        rad=np.geomspace(max(.2,float(a[:,0].min())),float(a[:,0].max()),50)
        mu=np.linspace(0,.95,9);R=rad[:,None]*np.sqrt(1-mu*mu);z=rad[:,None]*mu
        rho=h.density(R,z);valid=rho>row['rho0_msun_kpc3']*1e-8
        residual=(h.poisson_density(R,z)[valid]/rho[valid]-1)
        domain=h.diagnostics['outer_radius_kpc']
        doubled=FiniteFluidHalo(row['rho0_msun_kpc3'],row['velocity_scale_kms'],baryons,2*domain,
                                640,32,88,tolerance=1e-9,central_binding=key[1])
        speed=h.equatorial_circular_velocity(a[:,0]);speed2=doubled.equatorial_circular_velocity(a[:,0])
        rows.append(dict(galaxy=key[0],
                         max_relative_poisson_residual=float(np.max(abs(residual))),
                         valid_poisson_points=int(valid.sum()),total_poisson_points=int(valid.size),
                         density_floor_fraction=1e-8,
                         doubled_vacuum_domain_max_speed_change_kms=float(np.max(abs(speed2-speed))),
                         doubled_vacuum_domain_relative_mass_change=abs(doubled.diagnostics['total_halo_mass_msun']/h.diagnostics['total_halo_mass_msun']-1),
                         **h.diagnostics))
    pd.DataFrame(rows).to_csv(OUT/'fitted_closure_audit.csv',index=False)
    return rows


def previous_reference_baselines():
    old=pd.read_csv(ROOT/'executed_solid/fluid/real_disk_fits.csv')
    return [r for r in old.fillna('').to_dict('records') if r['model'].startswith('Uncontracted')]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validation-only',action='store_true')
    ap.add_argument('--cutoff-profile-only',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.cutoff_profile_only:
        summary=json.loads((OUT/'summary.json').read_text())
        summary['secondary_cutoff_profile']=cutoff_profile()
        summary['fitted_closure_audit']=fitted_closure_audit(summary)
        summary['previous_stage_reference_baselines']=previous_reference_baselines()
        dump(OUT/'summary.json',summary)
        return
    protocol=dict(physical_model='Globally finite lowered-isothermal ergodic distribution in combined self-consistent halo + MN baryon potential',
        central_binding_primary=5.,cutoff_sensitivity=[3.,7.],cutoff_sensitivity_selection='None; dark parameters held fixed, no new winner selected',
        stellar_thickness_prior='hz=10^(-.708) Rdisk^.633 kpc; log scatter .25, MN <|z|> matched at R=2.2 Rdisk',
        gas_thickness='Old .1 MN fractional assumption retained, not constrained by stellar relation',
        real_targets=['NGC2403','NGC3198','NGC5055'],
        holdout='Exactly reused prior image annuli, fit outside and predict inside; previously examined, exploratory only',
        source_table_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        old_artifacts='Read only; no old output or old source module is overwritten')
    dump(OUT/'protocol.json',protocol)
    summary=dict(protocol=protocol,validation=validate())
    dump(OUT/'summary.json',summary)
    if not args.validation_only:
        summary['real_galaxy_pilot']=real_pilot()
        dump(OUT/'summary.json',summary)
        summary['secondary_cutoff_profile']=cutoff_profile()
        summary['fitted_closure_audit']=fitted_closure_audit(summary)
        summary['previous_stage_reference_baselines']=previous_reference_baselines()
    dump(OUT/'summary.json',summary)
    print('Completed finite-equilibrium experiment',flush=True)

if __name__=='__main__':main()
