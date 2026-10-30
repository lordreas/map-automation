"""Measure changed HDMI frames during a river reveal with the browser preview on."""
import argparse,json,time,urllib.request
from playwright.sync_api import sync_playwright


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://projector.local:8080');args=p.parse_args()
 def get():
  with urllib.request.urlopen(args.url+'/api/state',timeout=5) as r:return json.load(r)
 def patch(value):
  request=urllib.request.Request(args.url+'/api/config',json.dumps(value).encode(),{'Content-Type':'application/json'})
  with urllib.request.urlopen(request,timeout=10) as r:return json.load(r)
 def ready():
  stop=time.monotonic()+25
  while time.monotonic()<stop:
   s=get()
   if s['error'] or s['output'].get('error'):raise AssertionError((s['error'],s['output'].get('error')))
   if not s['pending']:return s
   time.sleep(.05)
  raise TimeoutError('River renderer did not become ready')
 initial=get()
 if not initial['output'].get('connected'):raise SystemExit('Connect HDMI before measuring physical presentation events.')
 before=initial['config'];restore={'layers':before['layers'],'sun':{'enabled':before['sun']['enabled'],'playing':before['sun']['playing']},'track':{'enabled':before['track']['enabled']},'pattern':before['pattern']};reports=[];revision=None
 try:
  with sync_playwright() as p:
   browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox']);page=browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto(args.url)
   page.wait_for_function('typeof state!=="undefined"&&state.config.layers')
   for background in (False,True):
    layers={n:{'enabled':n=='rivers' or (n=='satellite' and background)} for n in before['layers']}
    revision=patch({'pattern':'sun','layers':layers,'sun':{'enabled':False,'playing':False},'track':{'enabled':False}})['revision'];s=ready()
    page.wait_for_function('document.getElementById("preview").naturalWidth>0')
    data=page.evaluate('''async()=>{const samples=[],start=performance.now();const record=e=>{const d=JSON.parse(e.data);samples.push([performance.now()-start,d.presentation_id]);};events.addEventListener('message',record);await fetch('/api/rivers/replay',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await new Promise(r=>setTimeout(r,6200));events.removeEventListener('message',record);return samples;}''')
    useful=[a for a in data if 1200<a[0]<5600];fps=len({a[1] for a in useful})/((useful[-1][0]-useful[0][0])/1000)
    reports.append({'background':'satellite' if background else 'black','physical_fps':round(fps,1),'preview':True,'chromatic_pixels':before['optics']['chromatic_pixels'],'river_thickness':before['layers']['rivers']['river_thickness']})
   if errors:raise AssertionError(errors)
   print(json.dumps(reports,indent=2));browser.close()
 finally:
  # Do not replace a view the user selected independently during this check.
  if revision is not None and get()['revision']==revision:patch(restore);ready()

if __name__=='__main__':main()
