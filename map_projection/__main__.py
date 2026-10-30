import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import webbrowser

from . import config as settings
from .engine import Engine
from .server import start


def main():
    parser=argparse.ArgumentParser(description="Native-pixel terrain projection with LAN controls.")
    parser.add_argument("--mesh",type=Path,default=settings.ROOT.parent/"stl_files"/"full_map_0.4mm.stl")
    parser.add_argument("--config",type=Path,default=settings.ROOT/"settings.json")
    parser.add_argument("--dem",type=Path,default=settings.ROOT.parent/"elevation_tiles"/"elevation_43.5_48.4_4.9_16.4_shape_11697x20000_resamp_average_bboxmerge_v2.npy")
    parser.add_argument("--host",default="0.0.0.0")
    parser.add_argument("--port",type=int,default=8080)
    parser.add_argument("--timezone",default=os.environ.get("MAP_PROJECTION_TIMEZONE"),help="Installation IANA timezone, e.g. Europe/Berlin or Asia/Tokyo.")
    parser.add_argument("--headless",action="store_true",help="Run controls and rendering without opening a physical display.")
    parser.add_argument("--direct",action="store_true",help="Require Linux KMS/DRM fullscreen; refuse desktop fallback.")
    backend=parser.add_mutually_exclusive_group()
    backend.add_argument('--gpu',action='store_true',help='Use GLES 3.1 GPU shadows and direct texture presentation (default for --direct).')
    backend.add_argument('--cpu',action='store_true',help='Use the CPU renderer and SDL streaming output.')
    parser.add_argument("--fullscreen",action="store_true")
    parser.add_argument("--display",type=int)
    parser.add_argument("--width",type=int)
    parser.add_argument("--height",type=int)
    parser.add_argument("--refresh",type=int)
    parser.add_argument("--open-browser",action="store_true")
    args=parser.parse_args()
    config=settings.load(args.config,args.timezone)
    if os.environ.pop('MAP_PROJECTION_RESUME_HDMI','0')=='1' and config['pattern']=='sun':
        config['sun']['playing']=True
    if args.cpu:config['sun']['backend']='cpu'
    for key in ("display","width","height","refresh"):
        if getattr(args,key) is not None:config["output"][key]=getattr(args,key)
    config["pixel"]["x"]=min(config["pixel"]["x"],config["output"]["width"]-1)
    config["pixel"]["y"]=min(config["pixel"]["y"],config["output"]["height"]-1)
    if args.fullscreen or args.direct:config["output"]["fullscreen"]=True
    gpu=not args.headless and not args.cpu and (args.gpu or args.direct)
    if gpu:sys.setswitchinterval(.001)
    origin=json.loads(os.environ.pop('MAP_PROJECTION_PLAYBACK_ORIGIN','null'))
    engine=Engine(args.mesh,config,args.config,dem_path=args.dem,gpu=False,playback_origin=origin)
    server=start(engine,args.host,args.port)
    url=f"http://127.0.0.1:{server.server_port}"
    print(f"Map Projection controls: {url}",flush=True)
    print("Map bounds: "+json.dumps(engine.bounds.tolist()),flush=True)
    print(f"Process ID: {os.getpid()}",flush=True)
    if args.open_browser:webbrowser.open(url)
    from .service import Notifier
    notifier=Notifier();notifier.ready()
    recovery=None
    display=None
    if args.headless:engine.output_status={"note":"Headless: preview only. No physical output.","pixel_buffer_verified":False,"direct_scanout":False}
    try:
        if not args.headless:
            from .display import Display
            from .hotplug import Monitor,Recovery
            def make_display():return Display(direct=args.direct,gpu=gpu)
            fallback=(lambda:Display(direct=args.direct,gpu=False)) if gpu and not args.gpu else None
            from .service import restart_for_hdmi
            recovery=Recovery(make_display,engine,args.direct,Monitor() if args.direct else None,fallback,
                              (lambda:restart_for_hdmi(engine)) if args.direct else None)
        signal.signal(signal.SIGTERM,lambda *_:engine.close())
        while not engine.stopped.is_set():
            notifier.tick(time.monotonic())
            if recovery:
                with engine.lock:c=dict(engine.config['output'])
                display=recovery.step(time.monotonic(),c)
            if display:
                try:
                    for action in display.poll():
                        if action=='close':engine.close()
                        elif action=='reconnect':recovery.reconnect()
                        elif action=='escape':
                            if args.direct:engine.update({'pattern':'black','sun':{'playing':False}})
                            else:engine.update({'output':{'fullscreen':False}})
                    if display is recovery.display:
                        frame=engine.frame
                        with engine.lock:
                            overlay=engine.overlay()
                            moving=engine.config['sun']['playing'] or engine.transition is not None or bool(overlay and overlay['animating']) or time.monotonic()<engine.live_until
                            if getattr(frame,'layers',None):
                                clock=frame.layers.get('river_clock')
                                moving=moving or bool(clock and clock.sample()<1)
                            compensation=engine.config['optics']['chromatic_pixels']
                        if display.present(frame,before_swap=lambda:engine.fulfill_captures(display.capture),preserve_preview=not moving,overlay=overlay,compensation=compensation,force=not engine.capture_requests.empty()):
                            engine.presented(frame)
                            if display.gpu:
                                with engine.lock:
                                    engine.timings['display_frame_seconds']=round(display.last_duration,4)
                                    engine.timings['display_components']=display.last_components
                                    if display.interval:engine.timings['display_fps']=round(1/display.interval,1)
                        with engine.lock:moving=engine.config['sun']['playing'] or engine.transition is not None
                        engine.fulfill_captures(display.capture,accept_new=not display.gpu)
                except Exception as exc:recovery.failed(exc,time.monotonic())
            if display and display.gpu:
                # Presentation already supplies backpressure. Wake immediately
                # for the worker's next frame instead of polling/sleeping 1 ms.
                river=engine.frame.layers.get('river_clock') if getattr(engine.frame,'layers',None) else None
                track=engine.overlay()
                reveal=bool((river and river.sample()<1) or (track and track['animating']))
                if not reveal:
                    with engine.events:
                        engine.events.wait_for(lambda:engine.frame is not display.last_frame or engine.stopped.is_set() or not engine.capture_requests.empty(),timeout=.01)
            else:time.sleep(.01)
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()
        if recovery:recovery.close()
        server.shutdown()
        server.server_close()


if __name__=="__main__":main()
