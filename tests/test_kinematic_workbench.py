import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from galaxy_lab.kinematic_workbench import read_ha_maps


def synthetic_file(path,velocity_unit='km/s',bad_wcs=False):
    w=WCS(naxis=2);w.wcs.crpix=[10,10];w.wcs.crval=[120,35];w.wcs.cdelt=[-.5/3600,.5/3600];w.wcs.ctype=['RA---TAN','DEC--TAN']
    header=w.to_header();header['C01']='Ha-6564';header['RADESYS']='FK5';header['EQUINOX']=2000
    data=np.ones((1,20,20),dtype='f4');hdus=[fits.PrimaryHDU()]
    for name in ('EMLINE_GVEL','EMLINE_GVEL_IVAR','EMLINE_GVEL_MASK','EMLINE_GFLUX','EMLINE_GFLUX_IVAR','EMLINE_GFLUX_MASK'):
        h=header.copy();h['BUNIT']=velocity_unit if name=='EMLINE_GVEL' else ''
        if bad_wcs and name=='EMLINE_GFLUX':h['CRPIX1']=11
        a=data*10 if name.endswith(('GFLUX','GVEL')) else data.copy()
        if name.endswith('MASK'):a=np.zeros_like(a,dtype='i4')
        hdus.append(fits.ImageHDU(a,header=h,name=name))
    fits.HDUList(hdus).writeto(path)


def test_fk5_wcs_offsets_have_consistent_icrs_frames(tmp_path):
    p=tmp_path/'maps.fits';synthetic_file(p)
    r=read_ha_maps(p,120,35)
    assert np.isfinite(r['east']).all() and r['valid'].sum()==400
    assert np.max(abs(r['east']))<6


def test_velocity_and_inverse_variance_convert_together(tmp_path):
    p=tmp_path/'maps.fits';synthetic_file(p,'m/s');r=read_ha_maps(p,120,35)
    np.testing.assert_allclose(r['velocity'],.01)
    np.testing.assert_allclose(r['inverse_variance'],1e6)


def test_mismatched_channel_grid_rejected(tmp_path):
    p=tmp_path/'maps.fits';synthetic_file(p,bad_wcs=True)
    with pytest.raises(ValueError,match='WCS'):read_ha_maps(p,120,35)


def test_shear_that_matches_diagonal_is_rejected(tmp_path):
    p=tmp_path/'maps.fits';synthetic_file(p)
    with fits.open(p,mode='update') as h:
        scale=h['EMLINE_GFLUX'].header['CDELT1']
        h['EMLINE_GFLUX'].header['PC1_1']=.9
        h['EMLINE_GFLUX'].header['PC1_2']=.1
    with pytest.raises(ValueError,match='WCS'):read_ha_maps(p,120,35)


def test_unrelated_center_is_rejected(tmp_path):
    p=tmp_path/'maps.fits';synthetic_file(p)
    with pytest.raises(ValueError,match='center'):read_ha_maps(p,130,35)


def test_icrs_without_equinox_is_accepted(tmp_path):
    p=tmp_path/'maps.fits';synthetic_file(p)
    with fits.open(p,mode='update') as h:
        for extension in h[1:]:
            extension.header['RADESYS']='ICRS'
            del extension.header['EQUINOX']
    r=read_ha_maps(p,120,35)
    assert r['valid'].sum()==400
