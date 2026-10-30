"""Real cache, Engine worker, HTTP previews and native 4K KMS presentation.

Run only while the production display service is stopped. Uses a private config
in memory; never persists changes to the installation settings.
"""
import argparse,copy,json,os,resource,threading,time,urllib.request
from pathlib import Path
import numpy as np

def thermal_snapshot():
 result={}
 try:
  for path in Path('/sys/class/thermal').glob('thermal_zone*'):
   result[(path/'type').read_text().strip()]=int((path/'temp').read_text())/1000
  for path in Path('/sys/devices/system/cpu/cpufreq').glob('policy*'):
   result[path.name+'_max_kHz']=int((path/'scaling_max_freq').read_text())
 except OSError:pass
 return result


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--events',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=12);p.add_argument('--variant',choices=['baseline','bulk','half','atomic','atomic-half'],default='baseline');p.add_argument('--only-preview',action='store_true');p.add_argument('--switch-interval',type=float);p.add_argument('--include-transitions',action='store_true');p.add_argument('--date',default='2026-10-25');p.add_argument('--preview-width',type=int,default=960);a=p.parse_args()
 if a.switch_interval is not None:
  import sys
  sys.setswitchinterval(a.switch_interval)
 os.environ['MAP_PROJECTION_EVENTS']=str(a.events.resolve())
 from map_projection import config as settings
 from map_projection.engine import Engine
 from map_projection.display import Display
 from map_projection.frames import SunFrame
 from map_projection.server import start
 if a.variant=='bulk':
  from .event_variants import install_bulk
  install_bulk()
 elif a.variant=='half':
  from map_projection import gpu_events
  gpu_events.GATHER=gpu_events.GATHER.replace('precision highp float;','precision mediump float;')
 elif a.variant!='baseline':
  from .event_variants import install
  install(half=a.variant=='atomic-half')
 thermal_start=thermal_snapshot()
 root=a.root.resolve();c=settings.load(root/'map_projection/settings.json')
 c['sun'].update(backend='events',grid_width=4096,shadow_size=4096,playing=False,date_mode='fixed',time=f'{a.date}T16:20:00+01:00')
 c['pattern']='sun';c['output'].update(width=3840,height=2160,refresh=60,fullscreen=True)
 # Use installation caches and source paths, so all calibration/DEM identity
 # checks follow the production Engine path.
 dem=root/'elevation_tiles/elevation_43.5_48.4_4.9_16.4_shape_11697x20000_resamp_average_bboxmerge_v2.npy'
 started=time.monotonic();engine=Engine(root/'stl_files/full_map_0.4mm.stl',c,cache=root/'map_projection/.cache',dem_path=dem,gpu=True)
 display=None;server=None;stop=threading.Event();requests={'preview':0,'state':0,'errors':0};results=[]
 try:
  server=start(engine,'127.0.0.1',8081);display=Display(direct=True,gpu=True);display.configure(c['output']);engine.output_status={**display.status,'connected':True}
  deadline=time.monotonic()+180
  while not isinstance(engine.frame,SunFrame):
   if engine.error:raise RuntimeError(engine.error)
   if time.monotonic()>deadline:raise RuntimeError('Engine preparation timeout')
   time.sleep(.01)
  def pump():
   display.poll();frame=engine.frame
   shown=display.present(frame,before_swap=lambda:engine.fulfill_captures(display.capture),preserve_preview=not(engine.config['sun']['playing'] or engine.transition is not None))
   if shown:engine.presented(frame)
   engine.fulfill_captures(display.capture,accept_new=not(engine.config['sun']['playing'] or engine.transition is not None))
   return frame if shown else None
  def pause():
   with engine.events:engine.events.wait_for(lambda:engine.frame is not display.last_frame or engine.error is not None,timeout=.01)
  pump();setup=time.monotonic()-started
  # Instrument submission only; GPU work can finish during SwapWindow. Do not
  # insert glFinish into the measured production loop.
  submission=[0.];draw=display.gpu_renderer.draw
  def timed_draw(*args,**kwargs):
   t=time.monotonic();result=draw(*args,**kwargs);submission[0]=time.monotonic()-t;return result
  display.gpu_renderer.draw=timed_draw
  def client():
   state_at=preview_at=0.
   while not stop.is_set():
    now=time.monotonic()
    for name,url,at,period in [('state','/api/state',state_at,1.),('preview',f'/preview.png?width={a.preview_width}&height={round(a.preview_width*9/16)}',preview_at,.75)]:
     if now<at:continue
     try:
      with urllib.request.urlopen('http://127.0.0.1:8081'+url,timeout=8) as response:response.read()
      requests[name]+=1
     except Exception:requests['errors']+=1
     if name=='state':state_at=now+period
     else:preview_at=now+period
    stop.wait(.01)
  for label,speed,stamp,preview in [('sunset_1_min_per_second',1,'16:20',False),('sunset_15_min_per_second',15,'15:00',False),('day_60_min_per_second',60,'06:00',False),('sunset_with_browser_preview',1,'16:20',True)]:
   if a.only_preview and not preview:continue
   engine.update({'sun':{'playing':False,'time':f'{a.date}T{stamp}:00+01:00','minutes_per_second':speed}},persist=False)
   target=engine.revision
   while engine.processed!=target:
    if engine.error:raise RuntimeError(engine.error)
    pump();time.sleep(.001)
   pump();engine.update({'sun':{'playing':True}},persist=False)
   # Warm up clock, seeks and presentation before collecting intervals.
   warm=time.monotonic()+1
   while time.monotonic()<warm:pump();time.sleep(.001)
   stop.clear();thread=threading.Thread(target=client,daemon=True) if preview else None
   if thread:thread.start()
   times=[];durations=[];submits=[];stamps=[];daylight=[];begin=time.monotonic();end=begin+a.seconds
   while time.monotonic()<end:
    if engine.error:raise RuntimeError(engine.error)
    frame=pump()
    if frame is not None:
     times.append(time.monotonic());durations.append(display.last_duration);submits.append(submission[0]);stamps.append(frame.time);daylight.append(frame.sun[2]>0)
    pause() # Same frame-ready wait as __main__.
   stop.set()
   if thread:
    while thread.is_alive():pump();thread.join(.001)
   intervals=np.diff(times)*1000
   result={'case':label,'minutes_per_second':speed,'preview':preview,'frames':len(times),'elapsed_seconds':times[-1]-times[0],'fps':(len(times)-1)/(times[-1]-times[0]),'interval_median_ms':float(np.median(intervals)),'interval_p95_ms':float(np.percentile(intervals,95)),'interval_max_ms':float(np.max(intervals)),'intervals_over_25ms_fraction':float(np.mean(intervals>25)),'present_median_ms':float(np.median(durations))*1000,'submission_median_ms':float(np.median(submits))*1000,'first_time':stamps[0],'last_time':stamps[-1]}
   day_intervals=intervals[np.asarray(daylight[:-1]) & np.asarray(daylight[1:])]
   result['daylight_fps']=float(1000/np.mean(day_intervals)) if len(day_intervals) else None
   result['daylight_interval_p95_ms']=float(np.percentile(day_intervals,95)) if len(day_intervals) else None
   result['daylight_frames']=int(np.count_nonzero(daylight))
   results.append(result);print(json.dumps(result),flush=True)
  transitions=[]
  if a.include_transitions:
   stop.clear();thread=threading.Thread(target=client,daemon=True);thread.start()
   for kind in ['now','stop_return']:
    for trial in range(3):
     engine.update({'sun':{'playing':False,'date_mode':'fixed','time':f'{a.date}T16:20:00+01:00','minutes_per_second':60}},persist=False)
     target=engine.revision
     while engine.processed!=target:pump();time.sleep(.001)
     pump()
     if kind=='stop_return':
      engine.update({'sun':{'playing':True}},persist=False);until=time.monotonic()+1.5
      while time.monotonic()<until:pump();time.sleep(.001)
      engine.stop_animation()
     else:engine.now()
     times=[];stamp=[];begin=time.monotonic()
     while time.monotonic()-begin<5:
      if engine.error:raise RuntimeError(engine.error)
      frame=pump()
      if frame is not None:times.append(time.monotonic());stamp.append(frame.time)
      if engine.transition is None and engine.last_presented is engine.frame:break
      pause()
     intervals=np.diff(times)*1000
     row={'kind':kind,'trial':trial,'frames':len(times),'duration_seconds':time.monotonic()-begin,'interval_p95_ms':float(np.percentile(intervals,95)) if len(intervals) else None,'interval_max_ms':float(np.max(intervals)) if len(intervals) else None,'first_time':stamp[0] if stamp else None,'last_time':stamp[-1] if stamp else None}
     transitions.append(row);print(json.dumps(row),flush=True)
   stop.set()
   while thread.is_alive():pump();thread.join(.001)
  result={'synthetic':False,'thermal_start':thermal_start,'thermal_end':thermal_snapshot(),'date':a.date,'preview_width':a.preview_width,'switch_interval':a.switch_interval,'variant':a.variant,'cache':str(a.events),'setup_seconds':setup,'output':display.status,'terrain_grid':engine.timings.get('terrain_grid'),'calibration':c['corners'],'cases':results,'transitions':transitions,'http_requests':requests,'peak_rss_KiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'includes':['Engine worker','solar position','real daily cache updates','4096 terrain','current calibration','3840x2160 shading','SDL KMS/DRM HDMI swap at 60Hz','HTTP server',f'{a.preview_width}x{round(a.preview_width*9/16)} preview requests in final case'],'excludes':['remote browser painting/network latency','physical optical verification']}
  a.output.write_text(json.dumps(result,indent=2))
 finally:
  stop.set();engine.close()
  if display:display.close()
  if server:server.shutdown();server.server_close()

if __name__=='__main__':main()
