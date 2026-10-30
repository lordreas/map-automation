"""Exercise the deployed layer UI and physical-frame capture, restoring settings."""
import argparse
import copy
import io
import json
from pathlib import Path
import statistics
import time
import urllib.request
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--url',default='http://projector.local:8080');parser.add_argument('--output',type=Path,default=Path('/tmp/projector-layer-check'));args=parser.parse_args();args.output.mkdir(exist_ok=True,parents=True)
 def state():
  with urllib.request.urlopen(args.url+'/api/state',timeout=5) as r:return json.load(r)
 def patch(value):
  request=urllib.request.Request(args.url+'/api/config',json.dumps(value).encode(),{'Content-Type':'application/json'})
  with urllib.request.urlopen(request,timeout=10) as response:return json.load(response)
 def ready(timeout=15):
  end=time.monotonic()+timeout
  while time.monotonic()<end:
   s=state()
   if s['error'] or s['output'].get('error'):raise AssertionError((s['error'],s['output'].get('error')))
   if not s['pending']:return s
   time.sleep(.1)
  raise TimeoutError('Projector layer did not become ready')
 def frame(name):
  ready()
  with urllib.request.urlopen(args.url+'/frame.png',timeout=15) as response:data=response.read()
  (args.output/(name+'.png')).write_bytes(data)
  return np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))
 before=state()['config'];reports=[]
 changed={k:copy.deepcopy(before[k]) for k in ('pattern','sun','layers','track')}
 try:
  with sync_playwright() as p:
   browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox']);page=browser.new_page(viewport={'width':1400,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   page.goto(args.url);page.wait_for_function('typeof state!=="undefined"&&state.config.layers');page.wait_for_function('document.getElementById("preview").naturalWidth>0')
   assert page.locator('#tab-layers').is_visible()
   off={name:{'enabled':False} for name in before['layers']}
   patch({'pattern':'sun','sun':{'enabled':False,'playing':False},'track':{'enabled':False},'layers':off})
   black=frame('black');assert black.max()==0,('All off must be black',black.max())
   page.wait_for_function('!state.config.sun.enabled');page.locator('#layer-contours-enabled').check();ready();contours=frame('contours')
   assert np.count_nonzero(contours)>10000
   page.locator('#layer-contours-enabled').locator('xpath=ancestor::details').evaluate('(d)=>d.open=true')
   page.locator('#layer-contours-single').select_option('true');ready();page.wait_for_timeout(1100)
   start=time.monotonic();first=state()['frame_id']
   commands=page.evaluate('''async()=>{const input=document.getElementById('layer-contours-level');let count=0;const start=performance.now();while(performance.now()-start<2500){input.value=String(500+(count*29)%3500);input.dispatchEvent(new Event('input',{bubbles:true}));count++;await new Promise(r=>setTimeout(r,16));}input.dispatchEvent(new Event('change',{bubbles:true}));return count;}''')
   last=ready();elapsed=time.monotonic()-start;reports.append({'scrub_commands':commands,'seconds':elapsed,'published_fps':(last['frame_id']-first)/elapsed,'display_fps':last['timings'].get('display_fps'),'frame_seconds':last['timings'].get('display_frame_seconds')})
   for name in ('satellite','streets','semantic','rivers','peaks','divides'):
    patch({'layers':{layer:{'enabled':layer==name} for layer in before['layers']}});s=ready();a=frame(name)
    if name=='rivers':
     time.sleep(6.5);later=frame('rivers-complete');assert np.count_nonzero(later)>=np.count_nonzero(a)
    assert a.max()>0,name
    if name in ('satellite','streets','semantic'):assert np.count_nonzero(a.max(axis=2)-a.min(axis=2))>10000,name
    reports.append({'layer':name,'peak_selected':s['layers']['peaks']['selected'],'draw_ms':s['timings'].get('display_frame_seconds',0)*1000})
   patch({'layers':{'satellite':{'enabled':True},'contours':{'enabled':True,'single':False},'rivers':{'enabled':True},'peaks':{'enabled':True}},'track':{'enabled':True}});ready();page.wait_for_timeout(1500);frame('combined');page.screenshot(path=str(args.output/'controls.png'),full_page=True)
   assert not errors,errors
   mobile=browser.new_page(viewport={'width':390,'height':844},is_mobile=True,has_touch=True);mobile.goto(args.url);mobile.wait_for_function('typeof state!=="undefined"&&state.config.layers');assert mobile.locator('#layer-peaks-enabled').count()==1;mobile.screenshot(path=str(args.output/'controls-mobile.png'),full_page=True)
   browser.close()
  print(json.dumps(reports,indent=2))
 finally:
  patch(changed);ready()

if __name__=='__main__':main()
