"""Optional browser QA for scrubbing, playback, clock events and transitions."""
import json,time,io,os,shutil
from pathlib import Path
from datetime import datetime
from urllib.request import urlopen,Request
from PIL import Image
import numpy as np
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
URL=os.environ.get('MAP_PROJECTION_QA_URL','http://127.0.0.1:8080')
def state():return json.load(urlopen(URL+'/api/state',timeout=5))
def post(path,data):return json.load(urlopen(Request(URL+path,json.dumps(data).encode(),{'Content-Type':'application/json'}),timeout=5))
def ready(timeout=30):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        s=state()
        if s['error']:raise AssertionError(s['error'])
        if not s['pending'] and not s['transitioning']:return s
        time.sleep(.02)
    raise AssertionError(state())
original=state()['config']
try:
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=os.environ.get('MAP_PROJECTION_CHROME') or shutil.which('google-chrome') or shutil.which('chromium'),headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1100});errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(URL);page.wait_for_function("typeof state!=='undefined' && state")
        page.wait_for_function("document.getElementById('preview').naturalWidth>0")
        print('Loaded browser',flush=True)
        def click(selector,path):
            with page.expect_response(lambda r:r.url.split('?')[0].endswith(path) and r.request.method=='POST') as response:
                page.locator(selector).click()
            assert response.value.ok,response.value.text()

        assert not page.locator('#sun-fps,#sun-from,#sun-until,#sun-zone,#sun-apply,#sun-sunset').count()
        post('/api/config',{'pattern':'sun','sun':{'time':'2026-10-25T16:00:00+01:00','date_mode':'fixed','terrain':'dem','renderer':'live','playing':False,'enabled':True}});ready()
        page.reload();page.wait_for_function("typeof state!=='undefined' && state")
        box=page.locator('#sun-minute').bounding_box()
        page.mouse.move(box['x']+10+(box['width']-20)*(960/1439),box['y']+box['height']/2)
        page.mouse.down();before=state()['frame_id']
        page.mouse.move(box['x']+10+(box['width']-20)*(1000/1439),box['y']+box['height']/2,steps=5)
        time.sleep(.35)
        assert state()['frame_id']>before,'No frame update during held drag'
        page.mouse.up();ready();anchor=state()['config']['sun']['time'];print('Live scrub',anchor,flush=True)
        click('#play','/api/play');page.wait_for_function("state.config.sun.playing")
        assert page.locator('#play').inner_text()=='Stop Animation'
        assert 'animated' in page.locator('#timeline').get_attribute('class')
        page.evaluate("window.clocks=[];new MutationObserver(()=>clocks.push(document.getElementById('clock').textContent)).observe(document.getElementById('clock'),{childList:true})")
        page.locator('#sun-speed').evaluate("e=>{e.value=1000;e.dispatchEvent(new Event('input',{bubbles:true}));}")
        time.sleep(.5)
        s=state();assert s['config']['sun']['minutes_per_second']==60
        updates=page.evaluate('new Set(clocks).size');assert updates>5,updates
        print('Live speed and presented clock',updates,'distinct minute labels in .5s',flush=True)
        page.locator('#sun-speed').evaluate("e=>e.dispatchEvent(new Event('change',{bubbles:true}))")
        click('#play','/api/stop');ready();s=state()
        assert s['config']['sun']['time']==anchor,(anchor,s['config']['sun']['time'])
        page.wait_for_timeout(100)
        assert page.locator('#clock').inner_text()==anchor[11:16]
        print('Stop eased back to original slider time',flush=True)
        click('#play','/api/play');page.wait_for_function('state.config.sun.playing')
        page.locator('#sun-minute').evaluate("e=>{e.value=600;e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));}")
        ready();s=state();assert not s['config']['sun']['playing'];assert not s['transitioning'];assert s['config']['sun']['time'][11:16]=='10:00'
        print('Scrub cancels playback without return',flush=True)
        click('#sun-now','/api/now');ready();s=state();assert not s['config']['sun']['playing']
        assert abs(datetime.fromisoformat(s['config']['sun']['time']).timestamp()-time.time())<3
        print('Now eased to wall time',s['config']['sun']['time'],flush=True)
        click('#shadow-toggle','/api/config');ready();s=state();assert not s['config']['sun']['enabled']
        raw=np.asarray(Image.open(io.BytesIO(urlopen(URL+'/frame.png').read())))
        assert set(np.unique(raw)).issubset({0,255}) and raw.max()==255
        click('#shadow-toggle','/api/config');ready()
        assert page.locator("#error").is_hidden(),page.locator("#error").inner_text()
        assert not errors,errors
        page.screenshot(path=str(ROOT/'.cache/playback-new-desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(250)
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        page.screenshot(path=str(ROOT/'.cache/playback-new-mobile.png'),full_page=True)
        print('Browser controls, filtered preview, desktop/mobile passed',flush=True)
        browser.close()
finally:
    post('/api/config',original)
