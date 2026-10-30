"""Optional live-browser check of reference selection and rendering transitions."""
import json,os,shutil,time
from urllib.request import urlopen,Request
from playwright.sync_api import sync_playwright
URL=os.environ.get('MAP_PROJECTION_QA_URL','http://127.0.0.1:8085')
def state():return json.load(urlopen(URL+'/api/state',timeout=10))
def post(data):return json.load(urlopen(Request(URL+'/api/config',json.dumps(data).encode(),{'Content-Type':'application/json'}),timeout=10))
def ready():
    end=time.monotonic()+45
    while time.monotonic()<end:
        s=state()
        if s['error']:raise AssertionError(s['error'])
        if not s['pending']:return s
        time.sleep(.05)
    raise AssertionError(state())
original=state()['config']
try:
    post({'pattern':'sun','sun':{'backend':'auto','renderer':'live','terrain':'dem','playing':False,'time':'2026-10-25T16:20:00+01:00','date_mode':'fixed'}});ready()
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=os.environ.get('MAP_PROJECTION_CHROME') or shutil.which('google-chrome'),headless=True)
        page=browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(URL);page.wait_for_function("typeof state!=='undefined' && state")
        page.locator('details').filter(has=page.locator('#sun-backend')).evaluate('e=>e.open=true')
        for method in ['reference','auto','reference']:
            with page.expect_response(lambda r:'/api/config' in r.url and r.request.method=='POST') as response:
                page.locator('#sun-backend').select_option(method)
            assert response.value.ok,response.value.text()
            s=ready();assert s['config']['sun']['backend']==method
            assert s['config']['sun']['terrain']=='dem' and s['config']['sun']['renderer']=='live'
            assert ('bilinear ray reference' in s['timings']['backend'])==(method=='reference')
            page.wait_for_function(f'document.getElementById("bake").disabled==={str(method=="reference").lower()}')
            assert urlopen(URL+'/preview.png',timeout=15).status==200
            assert urlopen(URL+'/frame.png',timeout=15).status==200
        assert not errors,errors
        print('Reference/fast transitions, no approximate reference cache, preview and full-resolution capture passed',flush=True)
        browser.close()
finally:post(original);ready()
