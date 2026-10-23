#!/usr/bin/env python3
"""Exercise frozen deployed commands on predetermined local examples.

No training, outcome-based example selection, source-data writes, or network.
Use a different --output-dir for a deliberate repeat; evaluated examples and
their command/provenance record are preserved by default.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
from datetime import datetime,timezone
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'executed_next/integration')
    args=parser.parse_args();out=args.output_dir.resolve()
    marker=out/'cli_validation.json'
    if marker.exists() or (out/'image_example').exists():
        raise SystemExit('Preserve existing integration outputs; choose a new --output-dir to repeat')
    out.mkdir(parents=True,exist_ok=True)
    commands=[]
    def run(name,argv):
        result=subprocess.run([sys.executable,'-m','galaxy_lab']+argv,cwd=ROOT,
                              capture_output=True,text=True)
        stdout=out/(name+'.stdout.log');stderr=out/(name+'.stderr.log')
        stdout.write_text(result.stdout);stderr.write_text(result.stderr)
        commands.append(dict(name=name,argv=[sys.executable,'-m','galaxy_lab']+argv,
                             cwd=str(ROOT),returncode=result.returncode,
                             stdout_log=str(stdout),stderr_log=str(stderr)))
        return result
    inputs=dict(image='executed_solid/data/native_images/1-197402_grz_native.fits',
                psf='executed_solid/data/native_images/1-197402_psf.fits',
                marks='executed_solid/data/annotations/gz3d_1-197402_127_14719695.fits.gz',
                display_model='executed_solid/segmentation/arm_model.joblib',
                raw_distance_features_model='executed_solid/segmentation/distance_feature_arm_model.joblib',
                redshift_model='executed_solid/redshift/deployed_model/redshift_model.joblib',
                observed_catalog='executed_solid/data/matched_catalog.csv',
                btfr_model='executed_next/kinematics/models/HI_Vflat_posterior.npz',
                btfr_matched_catalog='executed_next/kinematics/btfr_matched_data.csv')
    image=run('guarded_image',['analyze-guarded','--image',inputs['image'],'--psf',inputs['psf'],
        '--units','nanomaggy/pixel','--reference-marks',inputs['marks'],
        '--arm-model',inputs['display_model'],'--distance-arm-model',inputs['raw_distance_features_model'],
        '--redshift-model',inputs['redshift_model'],'--catalog',inputs['observed_catalog'],
        '--galaxy-id','1-197402','--output-dir',str(out/'image_example'),
        '--survey','legacy-dr9','--population','strong-resolved-spiral'])
    checks=[]
    checks.append(dict(name='guarded_image_command_success',passed=image.returncode==0))
    if image.returncode==0:
        decision=json.loads((out/'image_example/release_decision.json').read_text())
        checks.append(dict(name='declared_in_scope_example_accepted',
                           passed=decision['status']=='conditional_catalog_redshift',decision=decision))
        # Test declared scope logic separately without recomputing any image.
        sys.path.insert(0,str(ROOT))
        from galaxy_lab.release_guard import screen_analysis
        metadata=json.loads((out/'image_example/analysis.json').read_text())
        for label,survey,population in [('unvalidated_survey','other-survey','strong-resolved-spiral'),
                                       ('unvalidated_population','legacy-dr9','arbitrary-galaxy')]:
            result=screen_analysis(metadata,survey=survey,population=population)
            checks.append(dict(name=label,passed=result['status']=='abstain' and result['accepted_redshift'] is None,
                               result=result,method='same frozen analysis metadata; pure screening API, no image rerun'))
    table=pd.read_csv(ROOT/inputs['btfr_matched_catalog'])
    eligible=table[(table.distance_method==5)&(table.vf2019>0)&(table.quality<=2)&(table.inclination_deg>=30)].sort_values('galaxy')
    chosen=eligible.head(2);examples=[]
    for _,row in chosen.iterrows():
        destination=out/('btfr_'+row.galaxy+'.json')
        argv=['btfr-distance','--model',inputs['btfr_model'],'--velocity-kms',repr(float(row.vf2019)),
              '--velocity-error-kms',repr(float(row.e_vf2019)),'--velocity-definition','HI_Vflat_kms',
              '--baryon-flux-proxy',repr(float(row.flux_proxy)),
              '--log-flux-error-dex',repr(float(row.log_flux_error)),'--out',str(destination)]
        result=run('btfr_'+row.galaxy,argv)
        checks.append(dict(name='btfr_'+row.galaxy+'_command_success',passed=result.returncode==0))
        if result.returncode==0:
            response=json.loads(destination.read_text())
            examples.append(dict(galaxy=row.galaxy,observed_velocity_kms=float(row.vf2019),
                observed_velocity_error_kms=float(row.e_vf2019),baryon_flux_proxy_msun_per_mpc2=float(row.flux_proxy),
                log_flux_error_dex=float(row.log_flux_error),source_distance_method=int(row.distance_method),
                reference_distance_mpc=float(row.distance_mpc),response=response))
    (out/'btfr_example.json').write_text(json.dumps(dict(
        selection='Lexicographically first two eligible supernova-method SPARC holdouts; all eligible objects count two. Chosen without inspecting prediction errors.',
        eligible_holdouts=int(len(eligible)),examples=examples,
        interpretation='Frozen calibration applied to independently measured matching HI Vflat and catalog-derived baryon flux. These previously evaluated method holdouts are software examples, not a new accuracy test.'),indent=2)+'\n')
    row=chosen.iloc[0]
    common=['btfr-distance','--model',inputs['btfr_model'],'--velocity-kms',repr(float(row.vf2019)),
            '--velocity-error-kms',repr(float(row.e_vf2019)),'--velocity-definition','HI_Wm50_over2_kms',
            '--baryon-flux-proxy',repr(float(row.flux_proxy)),'--log-flux-error-dex',repr(float(row.log_flux_error))]
    refused=run('wrong_velocity_definition',common)
    checks.append(dict(name='incorrect_velocity_definition_refused',
        passed=refused.returncode!=0 and 'Velocity observable differs' in refused.stderr,
        returncode=refused.returncode))
    outside=common.copy();outside[outside.index('--velocity-kms')+1]='1000.'
    outside[outside.index('--velocity-definition')+1]='HI_Vflat_kms'
    outside+=['--out',str(out/'btfr_outside_range.json')]
    result=run('outside_velocity_range',outside)
    response=json.loads((out/'btfr_outside_range.json').read_text()) if result.returncode==0 else {}
    checks.append(dict(name='out_of_range_accepted_distance_withheld',
        passed=(result.returncode==0 and response.get('status')=='abstain_velocity_extrapolation'
                and response.get('accepted_distance_mpc','missing') is None),response=response))
    record=dict(status='passed' if all(c['passed'] for c in checks) else 'failed',
                executed_at_utc=datetime.now(timezone.utc).isoformat(),
                examples_selected_before_prediction=True,no_retraining=True,no_network=True,
                inputs={name:dict(path=path,sha256=sha(path)) for name,path in inputs.items()},
                commands=commands,checks=checks,
                interpretation='Command behavior and conditional-scope validation only. Passing is not independent distance accuracy, image-only distance identification or viscosity validation.')
    marker.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=record['status'],checks=[dict(name=c['name'],passed=c['passed']) for c in checks],
        guarded_image_decision=decision if image.returncode==0 else None,
        btfr_distances=[dict(galaxy=x['galaxy'],accepted_distance_mpc=x['response']['accepted_distance_mpc']) for x in examples]),indent=2))
    if record['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
