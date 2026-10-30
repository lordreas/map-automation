import copy
from datetime import datetime, timedelta
import hashlib
import io
import json
from pathlib import Path
import struct
import threading
import time
import queue
import os
from datetime import timezone
from zoneinfo import ZoneInfo
import uuid

import numpy as np
from PIL import Image, ImageDraw

from . import config as settings
from .geometry import apply, geo_to_xy, projector, sun_camera, unproject, xy_to_geo
from .native import Native
from .solar import position
from .terrain import Terrain
from .frames import SunFrame,SunScene
from .playback import PlaybackClock,TimeTransition


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


class Engine:
    def __init__(self, mesh, config, config_path=None, cache=None, dem_path=None,gpu=False,playback_origin=None):
        self.mesh=Path(mesh).resolve()
        self.config_path=config_path
        self.cache=Path(cache or settings.ROOT / ".cache")
        self.cache.mkdir(parents=True,exist_ok=True)
        self.native=Native()
        stat=self.mesh.stat()
        with self.mesh.open("rb") as f:
            header=f.read(84)
        if len(header)!=84:raise ValueError("Truncated STL header.")
        self.triangles=struct.unpack_from("<I",header,80)[0]
        if not self.triangles or stat.st_size!=84+self.triangles*50:
            raise ValueError("A complete binary STL is required (ASCII STL is not supported).")
        self.mesh_key=digest([str(self.mesh),stat.st_size,stat.st_mtime_ns])
        metadata=self.cache / (self.mesh_key+".json")
        if metadata.exists():
            self.bounds=np.array(json.loads(metadata.read_text())["bounds"],np.float64)
        else:
            print(f"Scanning {self.triangles:,} triangles ({stat.st_size/1e6:.1f} MB)...",flush=True)
            self.bounds=self.native.scan(self.mesh)
            metadata.write_text(json.dumps({"bounds":self.bounds.tolist()}))
        self.config=settings.validate(copy.deepcopy(config))
        self.terrain=Terrain(self.native,self.mesh,self.bounds,self.cache,self.mesh_key,dem_path)
        projector(self.config,self.bounds)
        self.lock=threading.RLock()
        self.events=threading.Condition(self.lock)
        self.instance=uuid.uuid4().hex
        self.event_sequence=0
        self.displayed_time=None
        self.last_presented=None
        self.wake=threading.Event()
        self.stopped=threading.Event()
        self.revision=1
        self.processed=0
        self.frame=None
        self.frame_id=0
        self.frame_time=None
        self.frame_pattern=None
        self.presented_revision=0
        self._working_revision=None
        self.output_status={"note":"Display is starting."}
        self.status="Starting renderer"
        self.error=None
        self.timings={}
        self.geometry_key=None
        self.depth=self.normals=None
        self.query=None
        self.query_result=None
        self.bake=[]
        self.bake_total=0
        self.gpu_capable=gpu
        self.lighting_key=None
        self.lighting_normals=self.lighting_lookup=self.gpu_scene=None
        self.lighting_work={}
        self.playback_clock=None
        self.transition=None
        self.motion=None
        self.playback_origin=playback_origin
        self.capture_requests=queue.Queue()
        self.pending_captures=[]
        self.preview_cache={}
        self.presentation_id=0;self._present_key=None;self._frame_geometry_key=None;self.live_until=0.
        from .tracks import TrackLayer
        self.track=TrackLayer();self.track.configure(self.config['track'])
        from .layers import LayerStack
        self.layers=LayerStack()
        self.cache_manager=None;self.cache_compiler=None;self.cache_pending=None
        bank=Path(os.environ.get('MAP_PROJECTION_HORIZONS',str(self.cache/'horizons4096-compressed')))
        if os.name=='posix' and (bank/'manifest.json').is_file():
            from .day_cache import DayCompiler,CacheManager
            events=Path(os.environ.get('MAP_PROJECTION_EVENTS',str(self.cache/'events')))
            self.cache_compiler=DayCompiler(bank,events,self.config['sun']['latitude'],self.config['sun']['longitude'])
            self.cache_manager=CacheManager(self.cache_compiler,self.cache_ready)
        self.worker=threading.Thread(target=self.run,name="terrain-renderer",daemon=True)
        self.worker.start()
        self.wake.set()

    def snapshot(self):
        with self.lock:
            c=copy.deepcopy(self.config)
            _,_,_,corners=projector(c,self.bounds)
            _,sun=position((self.displayed_time or self.frame_time or c["sun"]["time"]) if c["pattern"]=="sun" else c["sun"]["time"],c["sun"]["latitude"],c["sun"]["longitude"])
            return {"instance":self.instance,"config":c,"revision":self.revision,"rendered_revision":self.processed,"frame_id":self.frame_id,
                    "displayed_time":self.displayed_time,"transitioning":self.transition is not None,"motion":self.motion,
                    "pending":self.processed!=self.revision or (isinstance(self.frame,SunFrame) and self.presented_revision!=self.revision),"displayed_pattern":self.frame_pattern,
                    "frame_time":self.frame_time,"status":self.status,"error":self.error,"output":copy.deepcopy(self.output_status),
                    "sun":sun,"corners":corners.tolist(),"timings":dict(self.timings),"query":copy.deepcopy(self.query_result),
                    "mesh":{"name":self.mesh.name,"triangles":self.triangles,"bounds":self.bounds.tolist(),
                            "bytes":self.mesh.stat().st_size,"width_mm":self.bounds[3]-self.bounds[0],"height_mm":self.bounds[4]-self.bounds[1]},
                    "bake":{"remaining":len(self.bake),"total":self.bake_total},
                    "cache_bytes":sum(p.stat().st_size for p in self.cache.glob("*") if p.is_file()),
                    "track":self.track.snapshot(),"layers":self.layers.snapshot(),"date_cache":self.cache_state(),"presentation_id":self.presentation_id}

    def update(self, patch, persist=True):
        with self.lock:
            candidate=settings.merge(self.config,patch)
            if candidate==self.config and self.error is None:return
            if self.playback_clock is not None and 'time' not in patch.get('sun',{}):
                candidate['sun']['time']=self.playback_clock.sample(time.monotonic())[0].isoformat()
            if candidate["output"]["mirror"]!=self.config["output"]["mirror"]:
                if candidate["corners"] is not None:
                    candidate["corners"]=[[candidate["output"]["width"]-p[0],p[1]] for p in candidate["corners"]]
            if "output" in patch:
                candidate["pixel"]["x"]=min(candidate["pixel"]["x"],candidate["output"]["width"]-1)
                candidate["pixel"]["y"]=min(candidate["pixel"]["y"],candidate["output"]["height"]-1)
            settings.validate(candidate)
            projector(candidate,self.bounds)
            if candidate==self.config and self.error is None:return
            if persist and self.config_path:settings.save(self.config_path,candidate)
            was_running=self.playback_clock is not None
            if self.cache_manager and candidate['sun']['time'][:10]!=self.config['sun']['time'][:10]:self.cache_manager.reset_failure()
            if candidate['sun']['playing'] and not self.config['sun']['playing']:
                self.playback_origin=(candidate['sun']['time'],candidate['sun']['date_mode'])
            self.config=candidate
            if 'time' in patch.get('sun',{}) or 'track' in patch or 'optics' in patch or 'layers' in patch:self.live_until=time.monotonic()+.25
            self.track.configure(candidate['track'])
            if candidate['pattern']!='sun' or any(k in patch.get('sun',{}) for k in ('time','playing','date_mode')):
                self.transition=None
                self.motion=None
            self.playback_clock=PlaybackClock(candidate['sun'],time.monotonic()) if was_running and candidate['sun']['playing'] and candidate['pattern']=='sun' else None
            self.revision+=1
            self.error=None
            self.bake=[]
            if candidate["pattern"]!="sun":
                # Calibration/pixel patterns and blackout remain immediate,
                # even while the worker is finishing an expensive mesh pass.
                self.frame=self.render(candidate)
                self.events.notify_all()
                self.frame_id+=1
                self.frame_time=None
                self.frame_pattern=candidate["pattern"]
                self.processed=self.revision
                self.status="Ready"
            else:
                self.status="Preparing sunlight; the previous frame remains visible"
        self.wake.set()

    def set_gpu(self,enabled):
        with self.lock:
            if self.gpu_capable==enabled:return
            self.gpu_capable=enabled;self.revision+=1;self.playback_clock=None
        self.wake.set()

    def now(self):
        with self.lock:
            begin=datetime.fromisoformat(self.displayed_time or self.config['sun']['time'])
            self.update({'pattern':'sun','sun':{'playing':False,'date_mode':'today'}},persist=False)
            self.transition=TimeTransition(begin,self.config['sun']['timezone'])
            self.motion='now'
            self.revision+=1
        self.wake.set()

    def play(self):
        with self.lock:
            self.update({'pattern':'sun','sun':{'playing':True}})

    def stop_animation(self):
        with self.lock:
            if not self.config['sun']['playing']:return
            begin=datetime.fromisoformat(self.displayed_time or self.config['sun']['time'])
            target,date_mode=self.playback_origin or (self.config['sun']['time'],self.config['sun']['date_mode'])
            self.update({'sun':{'playing':False,'date_mode':date_mode}},persist=False)
            self.transition=TimeTransition(begin,self.config['sun']['timezone'],datetime.fromisoformat(target))
            self.motion='return'
            self.revision+=1
        self.wake.set()

    def time_event(self):
        with self.lock:
            return {'instance':self.instance,'sequence':self.event_sequence,'frame_id':self.frame_id,
                    'revision':self.revision,
                    'time':self.displayed_time,'playing':self.config['sun']['playing'],
                    'transitioning':self.transition is not None,'motion':self.motion,'pattern':self.frame_pattern,'presentation_id':self.presentation_id,'display_fps':self.timings.get('display_fps')}

    def publish_time(self,moment):
        self.displayed_time=moment
        self.event_sequence+=1
        self.events.notify_all()

    def presented(self,frame):
        with self.lock:
            overlay=self.overlay()
            river_clock=frame.layers.get('river_clock') if isinstance(frame,SunFrame) and frame.layers else None
            river_motion=(river_clock.generation,river_clock.sample()) if river_clock else None
            visual=(id(frame),self.config['optics']['chromatic_pixels'],(overlay['generation'],overlay['progress'],overlay['color']) if overlay else None,river_motion)
            changed=visual!=self._present_key
            self._present_key=visual
            if changed:self.presentation_id+=1
            self.last_presented=frame
            self.track.presented()
            if river_clock:river_clock.start()
            if isinstance(frame,SunFrame):self.frame_pattern='sun'
            current=frame is self.frame and self.processed==self.revision
            if current:
                self.presented_revision=self.revision
                self.frame_pattern=self.config['pattern']
            if current and self.config['sun']['playing'] and self.playback_clock is None:
                self.playback_clock=PlaybackClock(self.config['sun'],time.monotonic())
            if current and self.transition is not None:self.transition.start(time.monotonic())
            if changed:self.publish_time(frame.time if isinstance(frame,SunFrame) else self.frame_time if frame is self.frame else self.displayed_time)
        self.wake.set()

    def overlay(self):
        if self.config['pattern']!='sun':return None
        state=self.track.state()
        if state and state['key'][0]==self._frame_geometry_key:return state
        return None

    def replay_track(self):
        self.track.replay()
        with self.events:self.events.notify_all()
        self.wake.set()

    def replay_rivers(self):
        self.layers.river_clock.reset()
        with self.events:self.events.notify_all()
        self.wake.set()

    def cache_ready(self):
        with self.events:
            self.revision+=1;self.error=None;self.events.notify_all()
        self.wake.set()

    def cache_state(self):
        status=self.cache_compiler.status() if self.cache_compiler else {'state':'unavailable'}
        return {'pending':bool(self.cache_pending),'requested_date':self.cache_pending,
                'active_date':(self.displayed_time or self.frame_time or '')[:10] or None,'generation':status}

    def cached_time(self,c):
        if c['sun']['backend']!='events' or not self.cache_manager:
            self.cache_pending=None;return c
        from .day_cache import utc_days,complete,DAY
        moment=datetime.fromisoformat(c['sun']['time']);days=utc_days(moment.date().isoformat(),c['sun']['timezone'],c['sun']['latitude'],c['sun']['longitude'])
        active=getattr(self,'daily_events',None)
        missing=[d for d in days if not (active and active.path.name==d) and not complete(self.cache_compiler.root/d)]
        if not missing:
            self.cache_pending=None;return c
        self.cache_pending=moment.date().isoformat();self.cache_manager.request(days)
        fallback=getattr(self,'daily_events',None)
        if fallback is None:
            available=[p for p in self.cache_compiler.root.iterdir() if DAY.fullmatch(p.name) and complete(p)]
            if available:
                from .horizon_cache import Events
                nearest=min(available,key=lambda p:abs((datetime.fromisoformat(p.name)-moment.replace(tzinfo=None,hour=0,minute=0,second=0,microsecond=0)).total_seconds()))
                fallback=Events(nearest)
        result=copy.deepcopy(c)
        if fallback:
            # Preserve the requested clock, but use a UTC day with known events.
            utc=moment.astimezone(timezone.utc);d=datetime.fromisoformat(fallback.meta['start'])
            actual=utc.replace(year=d.year,month=d.month,day=d.day).astimezone(ZoneInfo(c['sun']['timezone']))
            result['sun']['time']=actual.isoformat()
        else:
            # The first cache can be generated while the registered map/GPX
            # remains usable without sunlight.
            result['sun'].update(enabled=False,backend='gpu')
        return result

    @staticmethod
    def compatible_frame(old,new):
        a=copy.deepcopy(old);b=copy.deepcopy(new)
        for item in (a,b):
            for key in ('time','playing','date_mode','minutes_per_second'):item['sun'].pop(key,None)
            # A completed geographic layer frame is still useful while its
            # sliders move; geometry and lighting settings must still agree.
            item.pop('layers',None)
        return a==b

    def fulfill_captures(self,capture,accept_new=True):
        for request,pending in list(self.pending_captures):
            if not pending.ready:continue
            if pending.error:request['error']=pending.error
            else:request['image']=pending.image
            request['event'].set();self.pending_captures.remove((request,pending))
        if not accept_new:return
        requests=[]
        while not self.capture_requests.empty():
            try:requests.append(self.capture_requests.get_nowait())
            except queue.Empty:break
        images={}
        for request in requests:
            key=(request['full'],request['size'])
            try:
                if key not in images:images[key]=capture(*key)
                result=images[key]
                if hasattr(result,'ready'):
                    self.pending_captures.append((request,result));continue
                request['image']=result
            except Exception as exc:request['error']=exc
            if 'image' in request or 'error' in request:request['event'].set()

    def nudge(self, corner, dx, dy, step=1):
        if (type(corner) is not int or corner not in range(4)
            or type(dx) is not int or type(dy) is not int
            or (dx,dy) not in ((1,0),(-1,0),(0,1),(0,-1))
            or type(step) is not int or step not in (1,5,10)):
            raise ValueError("Choose corner 0–3, a cardinal direction and a 1, 5 or 10 px step.")
        with self.lock:
            _,_,_,corners=projector(self.config,self.bounds)
            corners[corner]+=[dx*step,dy*step]
            self.update({"corners":corners.tolist(),"pattern":"calibration","sun":{"playing":False}})

    def request_query(self, query):
        if not isinstance(query,dict):raise ValueError("Query must be an object.")
        if query.get("kind") not in ("pixel","geo","map"):raise ValueError("Query kind must be pixel, geo or map.")
        required={"pixel":("x","y"),"geo":("latitude","longitude"),"map":("x","y")}[query["kind"]]
        for key in required:
            if isinstance(query.get(key),bool) or not isinstance(query.get(key),(int,float)) or not np.isfinite(query[key]):
                raise ValueError(f"Query {key} must be a finite number.")
        with self.lock:
            self.query=copy.deepcopy(query)
            self.query_result={"status":"Finding surface and checking visibility…"}
        self.wake.set()

    def bake_day(self):
        with self.lock:
            if self.config['sun']['backend'] in ('reference','precise','events'):
                raise ValueError('The continuous DEM ray reference renders live; day caching is unavailable in this mode.')
            self.update({'sun':{'playing':False}})
            self.playback_clock=None
            start=datetime.fromisoformat(self.config["sun"]["time"].replace("Z","+00:00")).replace(hour=5,minute=0,second=0,microsecond=0)
            step=self.config["sun"]["step_minutes"]
            self.bake=[(start+timedelta(minutes=m)).isoformat() for m in range(0,16*60+1,step)]
            self.bake_total=len(self.bake)
        self.wake.set()

    def geometry(self,c):
        matrix,inverse,origin,_=projector(c,self.bounds)
        w,h=c["output"]["width"],c["output"]["height"]
        key=digest([self.native.signature,self.mesh_key,matrix.tolist(),w,h])
        if key!=self.geometry_key:
            depth_file=self.cache / (key+"-depth.npy")
            normal_file=self.cache / (key+"-normal.npy")
            self.set_status("Preparing sunlight: intersecting projector pixels with the original STL")
            start=time.monotonic()
            if depth_file.exists() and normal_file.exists():
                depth=np.load(depth_file,mmap_mode="r",allow_pickle=False)
                normals=np.load(normal_file,mmap_mode="r",allow_pickle=False)
                if depth.shape!=(h,w) or normals.shape!=(h,w,3):raise ValueError("Invalid cached projection dimensions.")
            else:
                depth,normals=self.native.raster(self.mesh,self.bounds,matrix,w,h)
                np.save(depth_file,depth);np.save(normal_file,normals)
            self.depth,self.normals=depth,normals
            self.geometry_key=key
            self.timings["projection_seconds"]=round(time.monotonic()-start,3)
        return matrix,inverse,origin,key

    def set_status(self,status):
        with self.lock:
            if self._working_revision is None or self._working_revision==self.revision:
                self.status=status

    def sun_frame(self,c,persist=False):
        if persist and c['sun']['backend'] in ('reference','precise','events'):
            raise ValueError('The DEM ray reference cannot use the approximate CPU day cache.')
        c=self.cached_time(c)
        matrix,inverse,origin,key=self.geometry(c)
        try:self.track.prepare(c,key,self.depth,inverse,origin,self.bounds)
        except Exception as exc:
            with self.track.lock:self.track.error=str(exc);self.track.vertices=None
        s=c["sun"]
        sun,angles=position(s["time"],s["latitude"],s["longitude"])
        terrain=map_size=terrain_size=None
        if s["renderer"]=="live":
            terrain,map_size,terrain_size=self.terrain.get(c,self.set_status)
            if s['terrain']=='dem':
                lighting_key=digest([key,self.terrain.key])
                if lighting_key!=self.lighting_key:
                    self.set_status('Preparing fixed terrain coordinates and normals')
                    self.lighting_normals,self.lighting_lookup=self.native.lighting_scene(self.depth,inverse,origin,terrain,map_size,terrain_size)
                    self.lighting_work={};self.gpu_scene=None;self.lighting_key=lighting_key
                if s['backend'] in ('reference','precise','events') and not self.gpu_capable:
                    raise ValueError('The DEM ray reference requires the GPU display.')
                gpu=self.gpu_capable and s['backend']!='cpu' and not persist
                if gpu:
                    if self.gpu_scene is None:
                        lookup=self.lighting_lookup;gh,gw=terrain.shape
                        uv=np.empty((*lookup.shape,2),np.uint16)
                        safe=np.maximum(lookup,0);uv[:,:,0]=safe%gw;uv[:,:,1]=safe//gw;uv[lookup<0]=65535
                        self.gpu_scene=SunScene(lighting_key,terrain,self.lighting_normals,uv,terrain_size)
                    bias=s['bias_mm']*terrain_size[0]/map_size[0]
                    if s['backend']=='events':
                        import os
                        from datetime import timezone
                        from .horizon_cache import Events,identity
                        date=datetime.fromisoformat(s['time']).astimezone(timezone.utc).date().isoformat()
                        path=Path(os.environ.get('MAP_PROJECTION_EVENTS',str(self.cache/'events')))/date
                        # A local day's midnight can belong to the preceding
                        # UTC date. Night needs no visibility events; reuse the
                        # local day's bank until daylight enters its UTC date.
                        if sun[2]<=0 and not (path/'manifest.json').exists():
                            local_date=datetime.fromisoformat(s['time']).date().isoformat()
                            local_path=path.parent/local_date
                            if not (local_path/'manifest.json').exists() and self.cache_compiler:
                                from .day_cache import utc_days
                                local_path=path.parent/utc_days(local_date,s['timezone'],s['latitude'],s['longitude'])[0]
                            path=local_path
                        event_signature=(path,self.terrain.key,bias,s['latitude'],s['longitude'])
                        if getattr(self,'daily_event_signature',None)!=event_signature:
                            events=self.cache_compiler.load(path) if self.cache_compiler else Events(path)
                            if events.meta['horizon_key']!=identity(terrain,terrain_size,bias):raise ValueError('Daily event cache geometry/bias does not match terrain')
                            if events.meta['latitude']!=s['latitude'] or events.meta['longitude']!=s['longitude']:raise ValueError('Daily event cache solar location does not match installation')
                            self.daily_events=events;self.daily_event_signature=event_signature
                        self.gpu_scene=SunScene(lighting_key+str(path),terrain,self.lighting_normals,self.gpu_scene.uv,terrain_size,self.daily_events)

                    self.timings['backend']='GPU · cached horizon events' if s['backend']=='events' else 'GPU · conservative beams + exact rays' if s['backend']=='precise' else 'GPU · bilinear ray reference' if s['backend']=='reference' else 'GPU · sun-aligned scans'
                    self.timings['terrain_grid']=list(reversed(terrain.shape))
                    return SunFrame(self.gpu_scene,sun,s['ambient'],bias,s['time'],s['enabled'],s['backend'] if s['backend'] in ('reference','precise','events') else 'fast')
        if not s['enabled']:
            if self.lighting_lookup is not None and s['terrain']=='dem' and s['renderer']=='live':
                return np.repeat(np.where(self.lighting_lookup>=0,255,0).astype(np.uint8)[:,:,None],3,axis=2)
            return np.repeat(np.where(np.isfinite(self.depth),255,0).astype(np.uint8)[:,:,None],3,axis=2)
        frame_key=digest(["sun-v3",key,sun.tolist(),s["shadow_size"],s["ambient"],s["bias_mm"],s["renderer"],self.terrain.key if s["renderer"]=="live" else None])
        frame_path=self.cache / (frame_key+".png")
        if frame_path.exists() and not s['playing']:
            self.set_status("Replaying cached sun frame")
            with Image.open(frame_path) as image:return np.array(image.convert("RGB"))
        size=s["shadow_size"]
        start=time.monotonic()
        if s["renderer"]=="live":
            self.set_status("Calculating live terrain shadows")
            dem=s["terrain"]=="dem"
            # Physical-print bias (mm) is converted to metres for geographic DEM.
            bias=s["bias_mm"]*(terrain_size[0]/map_size[0] if dem else 1)
            if dem:
                rgb=self.native.fast_live(terrain,self.lighting_normals,self.lighting_lookup,terrain_size,sun,s['ambient'],bias,self.lighting_work)
                self.timings['backend']='CPU · optimized terrain sweep'
            else:
                rgb=self.native.live(self.depth,self.normals,inverse,origin,terrain,map_size,terrain_size,sun,s["ambient"],bias,dem)
                self.timings['backend']='CPU · printed STL'
            self.timings["shadow_seconds"]=round(time.monotonic()-start,3)
            self.timings["terrain_grid"]=list(reversed(terrain.shape))
            if persist:Image.fromarray(rgb).convert("L").save(frame_path,compress_level=1)
            return rgb
        sun_matrix,texel=sun_camera(sun,self.bounds,size)
        if angles["altitude"]>0:
            self.set_status("Computing terrain cast shadows from the original STL")
            shadow,_=self.native.raster(self.mesh,self.bounds,sun_matrix,size,size,perspective=False)
        else:
            shadow=np.full((size,size),np.inf,np.float32)
        rgb=self.native.shade(self.depth,self.normals,inverse,origin,sun_matrix,shadow,sun,s["ambient"],s["bias_mm"])
        self.timings["shadow_seconds"]=round(time.monotonic()-start,3)
        self.timings["shadow_texel_mm"]=round(texel,3)
        if persist or not s['playing']:Image.fromarray(rgb).convert("L").save(frame_path,compress_level=1)
        return rgb

    def render(self,c):
        w,h=c["output"]["width"],c["output"]["height"]
        pattern=c["pattern"]
        if pattern=="sun":
            frame=self.sun_frame(c)
            from .lighting import daily_peak,options,apply as expose
            reference_key=(getattr(getattr(self,'terrain',None),'key',None),c['sun']['time'][:10],c['sun']['latitude'],c['sun']['longitude'],c['sun']['timezone'],c['sun']['normalize'])
            if getattr(self,'exposure_key',None)!=reference_key:
                normals=frame.scene.normals if isinstance(frame,SunFrame) else self.normals
                self.exposure_peak=daily_peak(normals,c['sun']) if c['sun']['normalize'] and normals is not None else 1.
                self.exposure_key=reference_key
            gain,floor,gamma=options(c['sun'],self.exposure_peak)
            if isinstance(frame,SunFrame):
                from dataclasses import replace
                frame=replace(frame,light_gain=gain,light_floor=floor,light_gamma=gamma)
            elif c['sun']['enabled']:frame=expose(frame,np.isfinite(self.depth),gain,floor,gamma)
            _,inverse,origin,key=self.geometry(c)
            state=self.layers.prepare(c,key,self.depth,inverse,origin,self.bounds)
            neutral=c['layers']['white']['enabled'] and c['layers']['white']['opacity']==1 and not any(v['enabled'] for n,v in c['layers'].items() if n!='white')
            # Preserve the original one-pass sunlight path when the layer stack
            # is exactly its neutral white background.
            if neutral:return frame
            if isinstance(frame,SunFrame):
                from dataclasses import replace
                return replace(frame,layers=state)
            from .layers import compose_cpu
            return compose_cpu(frame,state,c['sun']['enabled'])
        frame=np.zeros((h,w,3),np.uint8)
        if pattern=="white":frame[:]=255
        elif pattern=="pixel":frame[c["pixel"]["y"],c["pixel"]["x"]]=255
        elif pattern in ("checkerboard","columns","rows"):
            x=np.arange(w,dtype=np.uint16)[None,:]
            y=np.arange(h,dtype=np.uint16)[:,None]
            values=(x+y if pattern=="checkerboard" else x if pattern=="columns" else y)%2*255
            frame[:]=values[:,:,None]
        elif pattern=="chromatic":
            image=Image.fromarray(frame);draw=ImageDraw.Draw(image);cx,cy=w/2,h/2
            for radius in np.linspace(min(w,h)*.07,min(w,h)*.65,7):
                draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline='white',width=2)
            for fy in (.1,.3,.5,.7,.9):
                for fx in (.1,.3,.5,.7,.9):
                    x,y=fx*w,fy*h;draw.line((x-14,y,x+14,y),fill='white',width=2);draw.line((x,y-14,x,y+14),fill='white',width=2)
            frame=np.asarray(image)
        elif pattern=="calibration":
            matrix,_,_,corners=projector(c,self.bounds)
            image=Image.fromarray(frame);draw=ImageDraw.Draw(image)
            draw.fontmode="1"
            mw,mh=self.bounds[3:5]-self.bounds[:2]
            for fraction in np.linspace(0,1,11):
                for points in (((-mw/2+fraction*mw,-mh/2,0),(-mw/2+fraction*mw,mh/2,0)),
                               ((-mw/2,-mh/2+fraction*mh,0),(mw/2,-mh/2+fraction*mh,0))):
                    p=apply(matrix,points)
                    draw.line([tuple(a) for a in p],fill=(35,80,95),width=1)
            draw.line([tuple(a) for a in np.vstack([corners,corners[0]])],fill=(210,240,240),width=1)
            for i,(x,y) in enumerate(corners):
                x,y=round(x),round(y)
                colour=((104,235,179),(255,200,110),(137,180,255),(245,140,175))[i]
                draw.line((x-14,y,x+14,y),fill=colour,width=1)
                draw.line((x,y-14,x,y+14),fill=colour,width=1)
                draw.text((x+8,y+8),str(i+1),fill=colour)
            frame=np.asarray(image)
        return np.ascontiguousarray(frame)

    def solve_query(self,c,query):
        matrix,inverse,origin,_=self.geometry(c)
        if query["kind"]=="pixel":
            u,v=query["x"],query["y"]
            if int(u)!=u or int(v)!=v or not 0<=u<self.depth.shape[1] or not 0<=v<self.depth.shape[0]:
                raise ValueError("Pixel is outside the output raster.")
            u,v=int(u),int(v)
            if not np.isfinite(self.depth[v,u]):return {"status":"This pixel misses the mesh.","visible":False}
            hit=unproject((u,v),self.depth[v,u],inverse,origin)
            return {"status":"Surface intersection","pixel":[u,v],"map_mm":hit.tolist(),**xy_to_geo(*hit[:2],self.bounds,c["geo"])}
        if query["kind"]=="geo":
            x,y=geo_to_xy(query["latitude"],query["longitude"],self.bounds,c["geo"])
        else:x,y=query["x"],query["y"]
        self.set_status("Intersecting target coordinate with the STL")
        hit=self.native.ray(self.mesh,self.bounds,[x,y,self.bounds[5]+100],[0,0,-1])
        if hit is None:return {"status":"Coordinate misses the mesh.","visible":False}
        ray=hit-origin
        first=self.native.ray(self.mesh,self.bounds,origin,ray/np.linalg.norm(ray))
        visible=first is not None and np.linalg.norm(first-hit)<.02
        projected=apply(matrix,hit)[0]
        u,v=np.floor(projected).astype(int)
        inside=0<=u<self.depth.shape[1] and 0<=v<self.depth.shape[0]
        actual=unproject((u,v),self.depth[v,u],inverse,origin) if inside and np.isfinite(self.depth[v,u]) else None
        result={"status":"Target visible" if visible and inside else "Target is occluded or outside the image",
                "visible":bool(visible and inside),"pixel":[int(u),int(v)],"continuous_pixel":projected.tolist(),
                "map_mm":hit.tolist(),"pixel_hit_mm":actual.tolist() if actual is not None else None,
                "pixel_error_mm":float(np.linalg.norm(actual-hit)) if actual is not None else None,
                **xy_to_geo(*hit[:2],self.bounds,c["geo"])}
        return result

    def run(self):
        while not self.stopped.is_set():
            self.wake.wait(.1)
            self.wake.clear()
            with self.lock:
                c,revision=copy.deepcopy(self.config),self.revision
                query,self.query=self.query,None
                bake_time=self.bake[0] if self.bake and revision==self.processed else None
                pending=revision!=self.processed
                transition=self.transition
                active=(c['sun']['playing'] and self.playback_clock is not None) or transition is not None
                animate=c['pattern']=='sun' and active and (not self.gpu_capable or self.last_presented is self.frame)
                finished_transition=False
                if transition is not None:
                    moment,finished_transition=transition.sample(time.monotonic())
                    c['sun']['time']=moment.isoformat()
                if transition is None and c['pattern']=='sun' and c['sun']['playing']:
                    clock=self.playback_clock or PlaybackClock(c['sun'],time.monotonic())
                    moment,finished=clock.sample(time.monotonic())
                    c['sun']['time']=moment.isoformat()
                    if finished:c['sun']['playing']=False
                self._working_revision=revision
            if not pending and not query and not bake_time and not animate:
                # Waiting for the first GPU presentation must not spin.
                continue
            try:
                if query:
                    result=self.solve_query(c,query)
                    with self.lock:
                        if revision!=self.revision:
                            self.query_result={"status":"Configuration changed during query; please query again."}
                        else:
                            self.query_result=result
                            self.status="Ready"
                            if query["kind"]!="pixel" and result.get("visible") and result.get("pixel_hit_mm"):
                                u,v=result["pixel"]
                                self.update({"pattern":"pixel","pixel":{"x":u,"y":v},"sun":{"playing":False}})
                    continue
                if bake_time:
                    c["sun"]["time"]=bake_time
                    self.sun_frame(c,persist=True)
                    with self.lock:
                        if self.bake and self.bake[0]==bake_time:self.bake.pop(0)
                        if revision==self.revision:
                            self.status=f"Caching day: {self.bake_total-len(self.bake)} / {self.bake_total}"
                    continue
                start=time.monotonic()
                frame=self.render(c)
                with self.lock:
                    current=revision==self.revision
                    if not current and not (isinstance(frame,SunFrame) and self.compatible_frame(c,self.config)):continue
                    self.frame=frame;self._frame_geometry_key=self.geometry_key
                    if not current:self.wake.set()
                    self.events.notify_all()
                    self.frame_id+=1
                    self.frame_time=frame.time if isinstance(frame,SunFrame) else c["sun"]["time"] if c["pattern"]=="sun" else None
                    if not isinstance(frame,SunFrame):self.frame_pattern=c["pattern"]
                    self.processed=revision
                    self.status="Generating day cache; showing previous day" if self.cache_pending else "Ready"
                    self.error=None
                    self.timings["frame_seconds"]=round(time.monotonic()-start,3)
                    if current and c['pattern']=='sun':
                        self.config['sun']['time']=c['sun']['time']
                        self.config['sun']['playing']=c['sun']['playing']
                        if c['sun']['playing'] and self.playback_clock is None and not isinstance(frame,SunFrame):
                            self.playback_clock=PlaybackClock(c['sun'],time.monotonic())
                        if not c['sun']['playing']:self.playback_clock=None
                    if finished_transition and self.transition is transition:
                        self.transition=None
                        self.motion=None
                        if self.config_path:settings.save(self.config_path,self.config)
                    if not self.output_status.get('connected'):
                        if self.transition is not None:self.transition.start(time.monotonic())
                        self.publish_time(self.frame_time)
                    # CPU rendering runs immediately again. GPU work waits for
                    # presentation, which supplies natural display backpressure.
                    if active and not isinstance(frame,SunFrame):self.wake.set()
            except Exception as exc:
                with self.lock:
                    if revision!=self.revision:continue
                    self.error=str(exc)
                    self.status="Render failed"
                    self.processed=revision
                    self.bake=[]
                    self.config["sun"]["playing"]=False
                    self.playback_clock=None
                    if query:self.query_result={"status":str(exc),"visible":False}

    def preview(self, full=False, size=(960,540),encoding="PNG"):
        with self.lock:frame=self.frame;frame_id=(getattr(self,'frame_id',0),getattr(self,'presentation_id',0))
        if frame is None:return None
        key=(frame_id,full,size,encoding)
        if isinstance(frame,SunFrame):
            with self.lock:
                if key in self.preview_cache:return self.preview_cache[key]
            request={'full':full,'size':size,'event':threading.Event()}
            self.capture_requests.put(request)
            if not request['event'].wait(5):return None
            if 'error' in request:return None
            frame=request['image']
            if frame is None:return None
        if not isinstance(self.frame,SunFrame) and hasattr(self,'track') and self.overlay():
            from .tracks import compose_track
            if not self.output_status.get('connected'):self.track.presented()
            frame=compose_track(frame,self.overlay())
        if not isinstance(self.frame,SunFrame) and getattr(self,'config',{}).get('optics',{}).get('chromatic_pixels'):
            from .optics import compensate_rgb
            frame=compensate_rgb(frame,self.config['optics']['chromatic_pixels'])
        image=Image.fromarray(frame)
        if not full:image.thumbnail(size,Image.Resampling.LANCZOS,reducing_gap=3.)
        result=io.BytesIO()
        if encoding=='JPEG':
            if image.mode not in ('L','RGB'):image=image.convert('RGB')
            image.save(result,format='JPEG',quality=85,subsampling=2)
        else:image.save(result,format="PNG",compress_level=1)
        data=result.getvalue()
        with self.lock:
            self.preview_cache={k:v for k,v in getattr(self,'preview_cache',{}).items() if k[0]==frame_id}
            self.preview_cache[key]=data
        return data

    def close(self):
        self.stopped.set()
        if self.cache_manager:self.cache_manager.close()
        self.wake.set()
        with self.events:self.events.notify_all()
