"""Measure physical presentation events while the browser scrubs contours."""
import argparse,json,time,urllib.request
from playwright.sync_api import sync_playwright


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://projector.local:8080');args=p.parse_args()
 def get():
  with urllib.request.urlopen(args.url+'/api/state') as response:return json.load(response)
 def patch(c):
  r=urllib.request.Request(args.url+'/api/config',json.dumps(c).encode(),{'Content-Type':'application/json'})
  with urllib.request.urlopen(r) as response:return json.load(response)
 before=get()['config'];restore={'layers':before['layers'],'sun':{'enabled':before['sun']['enabled'],'playing':before['sun']['playing']},'track':{'enabled':before['track']['enabled']},'pattern':before['pattern']}
 try:
  patch({'pattern':'sun','layers':{n:dict(enabled=n=='contours',**({'single':True} if n=='contours' else {})) for n in before['layers']},'sun':{'enabled':False,'playing':False},'track':{'enabled':False}})
  with sync_playwright() as p:
   b=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox']);page=b.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto(args.url)
   page.wait_for_function('typeof state!=="undefined"&&!state.pending&&document.getElementById("preview").naturalWidth>0')
   result=page.evaluate('''async()=>{
    const input=document.getElementById('layer-contours-level'),samples=[];const start=performance.now();
    const record=e=>{const d=JSON.parse(e.data);samples.push([performance.now()-start,d.presentation_id,d.display_fps]);};events.addEventListener('message',record);
    let commands=0;while(performance.now()-start<6500){input.value=String(2000+1500*Math.sin(commands/45));input.dispatchEvent(new Event('input',{bubbles:true}));commands++;await new Promise(r=>setTimeout(r,16));}
    input.dispatchEvent(new Event('change',{bubbles:true}));events.removeEventListener('message',record);
    return {commands,seconds:(performance.now()-start)/1000,samples};}''')
   useful=[s for s in result.pop('samples') if 1500<s[0]<6200];unique={s[1] for s in useful};result['physical_fps']=len(unique)/((useful[-1][0]-useful[0][0])/1000) if len(useful)>1 else 0;result['preview_enabled']=True;result['chromatic_pixels']=before['optics']['chromatic_pixels'];result['errors']=errors
   print(json.dumps(result,indent=2));b.close()
 finally:patch(restore)

if __name__=='__main__':main()
