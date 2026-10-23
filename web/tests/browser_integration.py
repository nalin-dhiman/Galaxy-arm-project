"""Actual Chromium interactions, local-only uploads and export verification."""
from pathlib import Path
from playwright.sync_api import sync_playwright
import json,io
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'test_output';OUT.mkdir(exist_ok=True)

def run():
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto('http://127.0.0.1:8765/',wait_until='networkidle')
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
        mobile=browser.new_page(viewport={'width':390,'height':844},device_scale_factor=1,is_mobile=True,has_touch=True);mobile.on('pageerror',lambda error:errors.append(str(error)));mobile.goto('http://127.0.0.1:8765/',wait_until='networkidle');mobile.screenshot(path=str(OUT/'mobile-image.png'),full_page=True)
        assert mobile.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        mobile.locator('#tab-halo').click();mobile.wait_for_timeout(100);mobile.screenshot(path=str(OUT/'mobile-halo.png'),full_page=True);assert mobile.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        mobile.locator('#tab-evidence').click();mobile.wait_for_timeout(100);mobile.screenshot(path=str(OUT/'mobile-evidence.png'),full_page=True);assert mobile.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        assert not errors,errors
        (OUT/'browser_results.json').write_text(json.dumps({'passed':True,'desktop_viewport':[1440,1100],'mobile_viewport':[390,844],'console_errors':errors,'checks':['two-point calibrated measurement','externalD_Aphysicalsize','PNGfilterchange/export','haloODEcontrol/CSVexport','frozenconsensusthreshold','unknowntruth','CSVexport','localuploadcalibrationreset','originalpixelsafterdownsample','keyboardpointsandtabs','mobileoverflow']},indent=2)+'\n')
        browser.close();print('Browser integration passed: local upload, filters, math, controls, exports, keyboard and mobile.')
if __name__=='__main__':run()
