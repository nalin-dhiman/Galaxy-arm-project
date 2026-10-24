"""Actual Chromium interactions, local-only uploads and export verification."""
from pathlib import Path
from playwright.sync_api import sync_playwright
import json,io,math,os
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'test_output';OUT.mkdir(exist_ok=True)
BASE_URL=os.environ.get('GSL_BASE_URL','http://127.0.0.1:8765/')

def mark(canvas,x,y,width=640,height=640):
    box=canvas.bounding_box()
    canvas.click(position={'x':box['width']*x/width,'y':box['height']*y/height})

def export_json(page,selector,name):
    with page.expect_download() as event:
        page.locator(selector).click()
    path=OUT/name
    event.value.save_as(path)
    return json.loads(path.read_text())

def project(radius,azimuth,inclination=50,angle=33):
    a=math.radians(angle);q=math.cos(math.radians(inclination))
    x=radius*math.cos(azimuth);y=radius*math.sin(azimuth)*q
    return 320+x*math.cos(a)-y*math.sin(a),320-x*math.sin(a)-y*math.cos(a)

def run():
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(BASE_URL,wait_until='networkidle')
        assert not errors,errors
        assert page.locator('#image-title').inner_text()=='Synthetic spiral demo'
        assert 'Analytic spiral' in page.locator('#image-help').inner_text()
        page.screenshot(path=str(OUT/'desktop-image.png'),full_page=True)
        canvas=page.locator('#image-canvas');box=canvas.bounding_box();canvas.click(position={'x':box['width']*.3,'y':box['height']*.4});canvas.click(position={'x':box['width']*.6,'y':box['height']*.4})
        assert '192' in page.locator('#pixel-result').inner_text()
        assert '76.8' in page.locator('#angle-result').inner_text()
        page.locator('#distance').fill('10');assert '3.723' in page.locator('#physical-result').inner_text()
        with page.expect_download() as event:page.locator('#download-measurement').click()
        path=OUT/'measurement.json';event.value.save_as(path);measurement=json.loads(path.read_text());assert measurement['pixelScaleArcsec']==.4 and abs(measurement['pixels']-192)<1e-6
        original=page.evaluate("document.getElementById('image-canvas').toDataURL()")
        page.locator('[data-filter="ridge"]').click();filtered=page.evaluate("document.getElementById('image-canvas').toDataURL()");assert filtered!=original
        with page.expect_download() as event:page.locator('#download-image').click()
        event.value.save_as(OUT/'filtered.png');assert Image.open(OUT/'filtered.png').size==(640,640)
        # Actual click-driven thin-disk geometry on independently calculated points.
        page.locator('[data-measure-mode="pitch"]').click()
        page.locator('#geometry-center').click();mark(canvas,320,320)
        assert page.locator('#geometry-center').get_attribute('aria-pressed')=='false'
        page.locator('#disk-inclination').fill('50');page.locator('#disk-angle').fill('33')
        for j in range(12):
            theta=.2+j*.18;radius=70*math.exp(math.tan(math.radians(20))*(theta-.2))
            mark(canvas,*project(radius,theta))
        pitch=export_json(page,'#download-geometry','manual-pitch.json')
        assert pitch['measurementMode']=='pitch' and pitch['count']==12
        assert abs(pitch['signedPitchDeg']-20)<.6,pitch
        assert pitch['rSquared']>.998 and pitch['rmsLogRadius']<.01
        page.screenshot(path=str(OUT/'desktop-pitch.png'),full_page=True)
        page.locator('#undo-point').click();pitch=export_json(page,'#download-geometry','manual-pitch-undo.json');assert pitch['count']==11
        before=page.evaluate("document.getElementById('image-canvas').toDataURL()")
        page.locator('#show-grid').uncheck();assert page.evaluate("document.getElementById('image-canvas').toDataURL()")!=before
        page.locator('#show-grid').check()
        page.locator('[data-measure-mode="spacing"]').click()
        for radius in [50,100,175]:mark(canvas,*project(radius,.6))
        gaps=export_json(page,'#download-geometry','manual-gaps.json')
        assert gaps['measurementMode']=='spacing' and gaps['count']==3
        for actual,expected in zip(gaps['gapsPixels'],[50,75]):assert abs(actual-expected)<2,(actual,expected)
        for actual,expected in zip(gaps['gapsArcsec'],[20,30]):assert abs(actual-expected)<.8
        assert len(gaps['gapsKpc'])==2 and gaps['suppliedAngularDiameterDistanceMpc']==10
        page.locator('#disk-inclination').fill('81')
        assert 'between 0 and 80' in page.locator('#geometry-result').inner_text()
        page.locator('#download-geometry').click();assert 'valid measurement' in page.locator('#status').inner_text()
        page.locator('#disk-inclination').fill('50')
        # Reset restores functional active states and clears supplied disk geometry.
        page.locator('#demo-button').click()
        assert page.locator('#disk-inclination').input_value()=='0'
        assert page.locator('#disk-angle').input_value()=='0'
        assert page.locator('[data-filter="original"]').get_attribute('aria-pressed')=='true'
        page.locator('[data-measure-mode="segment"]').click()
        page.locator('#tab-halo').click();page.wait_for_timeout(100);assert '138' in page.locator('#halo-edge').inner_text()
        old=page.locator('#halo-edge').inner_text();page.locator('#binding').fill('3');page.locator('#binding').dispatch_event('input');assert page.locator('#halo-edge').inner_text()!=old
        page.locator('[data-profile="pressure"]').click();page.screenshot(path=str(OUT/'desktop-halo.png'),full_page=True)
        with page.expect_download() as event:page.locator('#download-halo').click()
        event.value.save_as(OUT/'halo.csv');assert (OUT/'halo.csv').read_text().startswith('radius_kpc,circular_speed_km_s')
        page.locator('#tab-evidence').click();page.wait_for_timeout(100);assert page.locator('#evidence-count').inner_text()=='57 / 59';assert page.locator('#evidence-ap').inner_text()=='0.689'
        page.locator('#consensus').select_option('5');assert page.locator('#evidence-count').inner_text()=='53 / 59'
        page.locator('#cohort').select_option('low_vote_unknown');assert page.locator('#evidence-ap').inner_text()=='Unknown'
        with page.expect_download() as event:page.locator('#download-evidence').click()
        event.value.save_as(OUT/'evidence.csv');assert (OUT/'evidence.csv').read_text().count('\n')==16
        # A real local upload resets synthetic calibration and retains original coordinates.
        image=Image.new('RGB',(1600,800),(40,80,120));buf=io.BytesIO();image.save(buf,format='PNG');page.locator('#tab-image').click();page.locator('#file-input').set_input_files({'name':'test-local.png','mimeType':'image/png','buffer':buf.getvalue()});page.wait_for_function("document.getElementById('image-title').textContent==='test-local.png'")
        assert page.locator('#pixel-scale').input_value()=='';assert page.locator('#distance').input_value()==''
        assert page.locator('#image-canvas').get_attribute('width')=='1200'
        page.locator('#pixel-scale').fill('1');box=canvas.bounding_box();canvas.click(position={'x':box['width']*.25,'y':box['height']*.5});canvas.click(position={'x':box['width']*.75,'y':box['height']*.5});assert '800' in page.locator('#pixel-result').inner_text();assert '800' in page.locator('#angle-result').inner_text()
        # Keyboard selection and accessible tabs are operational.
        page.locator('#reset-points').click();canvas.focus();page.keyboard.press('Enter');page.keyboard.press('ArrowRight');page.keyboard.press('Enter');assert '2.67' in page.locator('#pixel-result').inner_text()
        page.locator('#tab-image').focus();page.keyboard.press('ArrowRight');assert page.locator('#halo-panel').is_visible()
        mobile=browser.new_page(viewport={'width':390,'height':844},device_scale_factor=1,is_mobile=True,has_touch=True);mobile.on('pageerror',lambda error:errors.append(str(error)));mobile.goto(BASE_URL,wait_until='networkidle');mobile.screenshot(path=str(OUT/'mobile-image.png'),full_page=True)
        assert mobile.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        mobile.locator('#tab-halo').click();mobile.wait_for_timeout(100);mobile.screenshot(path=str(OUT/'mobile-halo.png'),full_page=True);assert mobile.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        mobile.locator('#tab-evidence').click();mobile.wait_for_timeout(100);mobile.screenshot(path=str(OUT/'mobile-evidence.png'),full_page=True);assert mobile.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        # Narrow and tablet breakpoints remain usable without horizontal overflow.
        for width in [360,700,820,1100]:
            mobile.set_viewport_size({'width':width,'height':900})
            for panel in ['image','halo','evidence']:
                mobile.locator('#tab-'+panel).click()
                assert mobile.evaluate('document.documentElement.scrollWidth<=window.innerWidth'),(width,panel)
        assert page.locator('a[href*="paper.pdf"],a[href*="supplementary.pdf"],a[href*="overleaf.zip"]').count()==0
        assert not errors,errors
        (OUT/'browser_results.json').write_text(json.dumps({'passed':True,'base_url':BASE_URL,'desktop_viewport':[1440,1100],'mobile_viewport':[390,844],'extra_widths':[360,700,820,1100],'console_errors':errors,'checks':['two-point calibrated measurement','externalD_Aphysicalsize','PNGfilterchange/export','manual projected log-spiral recovery/export','same-ray disk-plane gap/export','invalid geometry refused','undo and overlay controls','new-image geometry and aria reset','haloODEcontrol/CSVexport','frozenconsensusthreshold','unknowntruth','CSVexport','localuploadcalibrationreset','originalpixelsafterdownsample','keyboardpointsandtabs','responsive overflow','no manuscript download links']},indent=2)+'\n')
        browser.close();print('Browser integration passed: local upload, filters, math, controls, exports, keyboard and mobile.')
if __name__=='__main__':run()
