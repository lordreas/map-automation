"""Validate the deployed controls and measure river HDMI presentations with preview."""
import argparse,copy,io,json,time,urllib.request
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://projector.local:8080');args=p.parse_args()
 def get():
  with urllib.request.urlopen(args.url+'/api/state',timeout=5) as r:return json.load(r)
 def patch(value):
  request=urllib.request.Request(args.url+'/api/config',json.dumps(value).encode(),{'Content-Type':'application/json'})
  with urllib.request.urlopen(request,timeout=10) as r:return json.load(r)
 def ready():
  stop=time.monotonic()+20
  while time.monotonic()<stop:
   s=get()
   if s['error'] or s['output'].get('error'):raise AssertionError((s['error'],s['output'].get('error')))
   if not s['pending']:return s
   time.sleep(.05)
  raise TimeoutError('Output did not settle')
 def image():
  ready()
  with urllib.request.urlopen(args.url+'/preview.png?width=960&height=540',timeout=12) as r:return np.array(Image.open(io.BytesIO(r.read())).convert('RGB'))
 before=get()['config'];restore={k:copy.deepcopy(before[k]) for k in ('pattern','sun','layers','track')};reports={};revision=None
 try:
  with sync_playwright() as p:
   browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox']);page=browser.new_page(viewport={'width':1400,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto(args.url)
   page.wait_for_function('typeof state!=="undefined"&&state.config.sun.normalize!==undefined')
   def only(name):return {n:{'enabled':n==name} for n in before['layers']}
   patch({'pattern':'sun','track':{'enabled':False},'layers':only('white'),'sun':{'time':before['sun']['time'][:10]+'T00:00:00','playing':False,'enabled':True,'shadow_floor':True,'shadow_floor_amount':.1}})
   floor=image();assert 24<=np.percentile(floor[floor>0],95)<=28
   page.locator('#layer-shadow-enabled').locator('xpath=ancestor::details').evaluate('(d)=>d.open=true')
   page.wait_for_function('state.config.sun.shadow_floor===true');page.locator('#shadow-floor-enabled').uncheck();page.wait_for_function('state.config.sun.shadow_floor===false');assert image().max()==0
   reports['night_floor']='10% visible; toggle off is black'
   patch({'sun':{'enabled':False},'layers':only('contours')});patch({'layers':{'contours':{'single':True,'level':2000}}});ready();page.wait_for_function('state.config.layers.contours.single===true')
   page.locator('#layer-contours-single').locator('xpath=ancestor::details').evaluate('(d)=>d.open=true')
   for style in ('above','below','line'):
    page.locator('#layer-contours-fill').select_option(style);page.wait_for_function(f'state.config.layers.contours.fill==={json.dumps(style)}');assert image().max()>200
   reports['elevation_styles']=['line','fill above','fill below']
   patch({'layers':only('satellite')});ready();page.locator('#layer-satellite-month').locator('xpath=ancestor::details').evaluate('(d)=>d.open=true')
   images=[]
   for month in (1,7):
    page.locator('#layer-satellite-month').select_option(str(month));page.wait_for_function(f'state.config.layers.satellite.month==={month}');images.append(image())
   difference=float(np.abs(images[0].astype(float)-images[1]).mean());assert difference>5;reports['january_july_mean_difference']=difference
   patch({'layers':{'satellite':{'month':0}}});ready();latencies=[]
   for enabled in (False,True,False,True):
    start=time.monotonic();patch({'layers':{'satellite':{'enabled':enabled}}});ready();latencies.append(round(time.monotonic()-start,3))
   reports['warm_satellite_toggle_seconds']=latencies
   patch({'layers':only('semantic')});ready();image();latencies=[]
   for enabled in (False,True):
    start=time.monotonic();patch({'layers':{'semantic':{'enabled':enabled}}});ready();latencies.append(round(time.monotonic()-start,3))
   reports['warm_landcover_toggle_seconds']=latencies
   reports['rivers']=[]
   for background in (False,True):
    layers=only('rivers');layers['satellite']['enabled']=background;patch({'layers':layers});ready();page.wait_for_function('document.getElementById("preview").naturalWidth>0')
    data=page.evaluate('''async()=>{const samples=[],start=performance.now();const record=e=>{const d=JSON.parse(e.data);samples.push([performance.now()-start,d.presentation_id]);};events.addEventListener('message',record);await fetch('/api/rivers/replay',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await new Promise(r=>setTimeout(r,6200));events.removeEventListener('message',record);return samples;}''')
    useful=[a for a in data if 1200<a[0]<5600];fps=len({a[1] for a in useful})/((useful[-1][0]-useful[0][0])/1000)
    reports['rivers'].append({'satellite_background':background,'physical_fps_with_preview':round(fps,1)})
   reports['errors']=errors;assert not errors
   reports['output']=get()['output'].get('active_mode');page.screenshot(path='/tmp/layer-polish-controls.png',full_page=True);browser.close()
  print(json.dumps(reports,indent=2))
 finally:patch(restore);ready()

if __name__=='__main__':main()
