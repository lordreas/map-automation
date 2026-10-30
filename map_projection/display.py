"""Native SDL2 streaming texture with explicit 1:1 drawable validation."""
import ctypes as ct
import os
import time
from collections import deque


class Display:
    def __init__(self, direct=False,gpu=False):
        if os.name == "nt":
            # Must precede creation of any window; prevent DPI virtualization.
            ct.windll.user32.SetProcessDpiAwarenessContext(ct.c_void_p(-4))
        if direct:
            os.environ["SDL_VIDEODRIVER"] = "kmsdrm"
        import sdl2
        self.s = sdl2
        self.window = self.renderer = self.texture = None
        self.direct = direct
        self.gpu=gpu
        self.gl_context=self.gpu_renderer=None
        self.config = None
        self.status = {}
        self.last_frame = None
        self.last_present = 0.
        self.present_times=deque()
        self.configured_at=0.
        sdl2.SDL_SetHint(b"SDL_RENDER_SCALE_QUALITY", b"0")
        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
            error=sdl2.SDL_GetError().decode();sdl2.SDL_Quit()
            raise RuntimeError(error)
        # SDL builds differ in spelling (e.g. "KMSDRM" on Radxa's Debian).
        self.driver = sdl2.SDL_GetCurrentVideoDriver().decode().lower()
        if direct and self.driver != "kmsdrm":
            raise RuntimeError("Direct output requires SDL's kmsdrm driver; refusing desktop fallback.")
        if direct or gpu:
            # Embedded PowerVR stacks expose GLES, rather than desktop OpenGL.
            # Set this before KMS/DRM creates the EGL window surface.
            sdl2.SDL_SetHint(b"SDL_RENDER_DRIVER", b"opengles2")
            self.check(sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_PROFILE_MASK, sdl2.SDL_GL_CONTEXT_PROFILE_ES))
            self.check(sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MAJOR_VERSION, 3 if gpu else 2))
            self.check(sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MINOR_VERSION, 1 if gpu else 0))

    def modes(self):
        s=self.s
        result=[]
        for display in range(s.SDL_GetNumVideoDisplays()):
            modes=[]
            for i in range(s.SDL_GetNumDisplayModes(display)):
                mode=s.SDL_DisplayMode()
                if s.SDL_GetDisplayMode(display,i,ct.byref(mode)) == 0:
                    record={"width":mode.w,"height":mode.h,"refresh":mode.refresh_rate}
                    if record not in modes:modes.append(record)
            result.append({"index":display,"name":s.SDL_GetDisplayName(display).decode(),"modes":modes})
        return result

    def check(self, result):
        if result < 0:
            raise RuntimeError(self.s.SDL_GetError().decode())

    def configure(self, config):
        s=self.s
        w,h,display=config["width"],config["height"],config["display"]
        if display >= s.SDL_GetNumVideoDisplays():
            raise ValueError("Selected output display does not exist.")
        if self.direct and not config["fullscreen"]:
            raise ValueError("Direct KMS output requires fullscreen.")
        chosen=None
        if config["fullscreen"]:
            for i in range(s.SDL_GetNumDisplayModes(display)):
                candidate=s.SDL_DisplayMode()
                if s.SDL_GetDisplayMode(display,i,ct.byref(candidate)) == 0 and (candidate.w,candidate.h,candidate.refresh_rate)==(w,h,config["refresh"]):
                    chosen=candidate;break
            if chosen is None:
                raise ValueError(f"Display {display} does not advertise {w} × {h} at {config['refresh']} Hz; no scaling fallback.")
        self.destroy_window()
        flags=s.SDL_WINDOW_SHOWN | (s.SDL_WINDOW_OPENGL if self.direct or self.gpu else 0)
        self.window=s.SDL_CreateWindow(b"Map Projection | native pixels | Esc: windowed", s.SDL_WINDOWPOS_CENTERED_DISPLAY(display),
                                        s.SDL_WINDOWPOS_CENTERED_DISPLAY(display), w,h,flags)
        if not self.window:raise RuntimeError(s.SDL_GetError().decode())
        if chosen is not None:
            self.check(s.SDL_SetWindowDisplayMode(self.window,ct.byref(chosen)))
            self.check(s.SDL_SetWindowFullscreen(self.window,s.SDL_WINDOW_FULLSCREEN))
        if self.direct:
            # SDL acquires DRM master while creating the window. A733 hotplug
            # can disable HDMI while DRM still reports an unchanged active
            # CRTC. Force it off so the first swap re-enables physical output.
            from .drm import reset_scanout
            reset_scanout()
        if self.gpu:
            self.gl_context=s.SDL_GL_CreateContext(self.window)
            if not self.gl_context:raise RuntimeError(s.SDL_GetError().decode())
            self.check(s.SDL_GL_MakeCurrent(self.window,self.gl_context))
            self.check(s.SDL_GL_SetSwapInterval(1))
            rw,rh=ct.c_int(),ct.c_int();s.SDL_GL_GetDrawableSize(self.window,ct.byref(rw),ct.byref(rh))
            if (rw.value,rh.value)!=(w,h):raise RuntimeError('GPU drawable does not match requested native output dimensions.')
            active=s.SDL_DisplayMode();self.check(s.SDL_GetCurrentDisplayMode(display,ct.byref(active)))
            if chosen is not None and (active.w,active.h,active.refresh_rate)!=(w,h,config['refresh']):
                raise RuntimeError('Driver did not activate the requested GPU fullscreen mode.')
            from .gpu import Renderer
            self.gpu_renderer=Renderer();self.config=dict(config);self.last_frame=None
            self.status={'driver':self.driver,'drawable':[w,h],'active_mode':[active.w,active.h,active.refresh_rate],
                'renderer':self.gpu_renderer.renderer,'graphics_api':self.gpu_renderer.version,
                'accelerated':'llvmpipe' not in self.gpu_renderer.renderer.lower(),
                'pixel_buffer_verified':True,'direct_scanout':self.driver=='kmsdrm',
                'projector_pixels_verified':False,'displays':self.modes(),
                'note':'GPU shadows and native pixel output. Check physical alignment with calibration patterns.'}
            s.SDL_ShowCursor(s.SDL_DISABLE if config['fullscreen'] else s.SDL_ENABLE)
            self.configured_at=time.monotonic()
            return
        self.renderer=s.SDL_CreateRenderer(self.window,-1,s.SDL_RENDERER_ACCELERATED|s.SDL_RENDERER_PRESENTVSYNC)
        if not self.renderer:
            self.renderer=s.SDL_CreateRenderer(self.window,-1,s.SDL_RENDERER_SOFTWARE)
        if not self.renderer:raise RuntimeError(s.SDL_GetError().decode())
        rw,rh=ct.c_int(),ct.c_int()
        self.check(s.SDL_GetRendererOutputSize(self.renderer,ct.byref(rw),ct.byref(rh)))
        if (rw.value,rh.value)!=(w,h):
            self.destroy_window()
            raise RuntimeError(f"Drawable is {rw.value} × {rh.value}, requested {w} × {h}. Refusing interpolation.")
        self.texture=s.SDL_CreateTexture(self.renderer,s.SDL_PIXELFORMAT_RGB24,s.SDL_TEXTUREACCESS_STREAMING,w,h)
        if not self.texture:raise RuntimeError(s.SDL_GetError().decode())
        self.check(s.SDL_SetTextureBlendMode(self.texture,s.SDL_BLENDMODE_NONE))
        version=s.SDL_version()
        s.SDL_GetVersion(ct.byref(version))
        # Radxa's SDL 2.0.14 GLES driver crashes in SetTextureScaleMode.
        # Older SDL uses the nearest-neighbour creation hint set in __init__.
        if (version.major,version.minor,version.patch)>=(2,0,16):
            self.check(s.SDL_SetTextureScaleMode(self.texture,s.SDL_ScaleModeNearest))
        active=s.SDL_DisplayMode()
        self.check(s.SDL_GetCurrentDisplayMode(display,ct.byref(active)))
        if chosen is not None and (active.w,active.h,active.refresh_rate)!=(w,h,config["refresh"]):
            self.destroy_window()
            raise RuntimeError("Driver did not activate the requested fullscreen mode.")
        self.config=dict(config)
        self.last_frame=None
        info=s.SDL_RendererInfo()
        self.check(s.SDL_GetRendererInfo(self.renderer,ct.byref(info)))
        self.status={"driver":self.driver,"drawable":[rw.value,rh.value],"active_mode":[active.w,active.h,active.refresh_rate],
                     "renderer":info.name.decode(),"accelerated":bool(info.flags&s.SDL_RENDERER_ACCELERATED),
                     "pixel_buffer_verified":True,"direct_scanout":self.driver=="kmsdrm",
                     "projector_pixels_verified":False,"displays":self.modes(),
                     "note":"Buffer = drawable. Projector scaling/overscan must be checked using the pixel patterns."}
        s.SDL_ShowCursor(s.SDL_DISABLE if config["fullscreen"] else s.SDL_ENABLE)
        self.configured_at=time.monotonic()

    def poll(self):
        if self.gpu_renderer is not None:self.gpu_renderer.poll_captures()
        event=self.s.SDL_Event()
        actions=[]
        while self.s.SDL_PollEvent(ct.byref(event)):
            if time.monotonic()-self.configured_at>.5 and event.type==getattr(self.s,'SDL_DISPLAYEVENT',-1) and event.display.event in (
                getattr(self.s,'SDL_DISPLAYEVENT_CONNECTED',-2),getattr(self.s,'SDL_DISPLAYEVENT_DISCONNECTED',-3)):
                actions.append('reconnect')
            if event.type==self.s.SDL_QUIT:actions.append("close")
            if event.type==self.s.SDL_KEYDOWN and event.key.keysym.sym==self.s.SDLK_ESCAPE:actions.append("escape")
        return actions

    def present(self, frame,before_swap=None,preserve_preview=False,overlay=None,compensation=0.,force=False):
        overlay_key=(id(overlay['vertices']),overlay['generation'],overlay['progress'],overlay['color']) if overlay else None
        river_clock=frame.layers.get('river_clock') if getattr(frame,'layers',None) else None
        river_motion=(river_clock.generation,river_clock.sample()) if river_clock else None
        unchanged=frame is self.last_frame and overlay_key==getattr(self,'last_overlay',None) and compensation==getattr(self,'last_compensation',0.) and river_motion==getattr(self,'last_river_motion',None)
        if self.gpu:
            if frame is None or self.gpu_renderer is None or (unchanged and not force):return False
            w,h=self.config['width'],self.config['height']
            if frame.shape!=(h,w,3):return False
            rw,rh=ct.c_int(),ct.c_int();self.s.SDL_GL_GetDrawableSize(self.window,ct.byref(rw),ct.byref(rh))
            if (rw.value,rh.value)!=(w,h):raise RuntimeError('Drawable size changed; GPU output stopped to avoid scaling.')
            from .frames import SunFrame
            started=time.monotonic()
            renderer=self.gpu_renderer;layers=frame.layers if isinstance(frame,SunFrame) else None
            if not unchanged:renderer.snapshot_frame=None
            self.capture_framebuffer=None
            self.capture_source=None
            renderer.color_capture=bool(overlay or compensation or layers)
            folded_chromatic=bool(layers and not overlay and (('screen_elevation' in layers['arrays'] and layers['config']['contours']['enabled'] and not any(v['enabled'] for n,v in layers['config'].items() if n!='contours')) or (layers['config']['rivers']['enabled'] and 'screen_rivers' in layers['arrays'])))
            target=renderer.chromatic_target(w,h) if compensation and not folded_chromatic else 0
            if isinstance(frame,SunFrame):
                if layers:
                    if renderer.layer_renderer is None:
                        from .gpu_layers import LayerRenderer
                        renderer.layer_renderer=LayerRenderer(renderer)
                    lighting=renderer.layer_renderer.target(w,h)
                    needs_light=any(layers['config'][n]['enabled'] for n in ('white','satellite','semantic'))
                    if frame.enabled and needs_light and renderer.layer_renderer.lighting_frame is not frame:
                        renderer.draw(frame,target=lighting);renderer.layer_renderer.lighting_frame=frame
                    else:
                        from OpenGL.GLES3 import glBindVertexArray
                        renderer.ow,renderer.oh=w,h;renderer.last_frame=frame
                        glBindVertexArray(renderer.vao)
                    renderer.layer_renderer.draw(layers,target,shading=frame.enabled,compensation=compensation if folded_chromatic else 0.)
                    rivers=renderer.layer_renderer.river_renderer
                    if layers['config']['rivers']['enabled'] and rivers is not None:
                        if not overlay and (not compensation or folded_chromatic):self.capture_source=rivers
                        renderer.timings.update(rivers.timings)
                else:renderer.draw(frame,target=target)
            else:renderer.draw_rgb(frame,target=target)
            base_done=time.monotonic()
            if overlay:renderer.draw_track(overlay,target)
            track_done=time.monotonic()
            if compensation and not folded_chromatic:renderer.compensate(compensation)
            chromatic_done=time.monotonic()
            if preserve_preview and self.capture_source is None and (isinstance(frame,SunFrame) or renderer.color_capture):renderer.snapshot_output(output_framebuffer=self.capture_framebuffer or 0,width=w,height=h)
            self.capturing_output=True
            try:
                if before_swap:before_swap()
            finally:self.capturing_output=False
            capture_done=time.monotonic()
            self.s.SDL_GL_SwapWindow(self.window)
            swapped=time.monotonic()
            self.last_components={name:round(seconds,5) for name,seconds in [('base',base_done-started),('track',track_done-base_done),('chromatic',chromatic_done-track_done),('capture',capture_done-chromatic_done),('swap',swapped-capture_done)]}
            self.last_duration=time.monotonic()-started
            now=time.monotonic()
            self.present_times.append(now)
            while self.present_times and self.present_times[0]<now-2:self.present_times.popleft()
            self.interval=(now-self.present_times[0])/(len(self.present_times)-1) if len(self.present_times)>1 else None
            self.last_present=now
            self.last_frame=frame;self.last_overlay=overlay_key;self.last_compensation=compensation
            self.last_river_motion=river_motion
            return True
        if frame is None or not self.texture:return
        if not hasattr(frame,'ctypes'):return False  # A CPU fallback is still preparing its first frame.
        s=self.s
        w,h=self.config["width"],self.config["height"]
        if frame.shape!=(h,w,3):return
        # Re-check actual drawable, e.g. moving a desktop window between monitors.
        rw,rh=ct.c_int(),ct.c_int()
        self.check(s.SDL_GetRendererOutputSize(self.renderer,ct.byref(rw),ct.byref(rh)))
        if (rw.value,rh.value)!=(w,h):raise RuntimeError("Drawable size changed; output stopped to avoid scaling.")
        # HDMI keeps scanning out at the selected refresh without resubmitting
        # an unchanged frame. Avoid needless GPU work and heat while idle.
        if unchanged and not force:return
        original=frame
        if overlay:
            from .tracks import compose_track
            frame=compose_track(frame,overlay)
        if compensation:
            from .optics import compensate_rgb
            frame=compensate_rgb(frame,compensation)
        self.check(s.SDL_UpdateTexture(self.texture,None,frame.ctypes.data,w*3))
        # Explicit 1:1 rectangles avoid any inherited viewport/scale transform.
        rect=s.SDL_Rect(0,0,w,h)
        self.check(s.SDL_RenderSetViewport(self.renderer,ct.byref(rect)))
        self.check(s.SDL_RenderSetScale(self.renderer,1.,1.))
        self.check(s.SDL_SetRenderDrawColor(self.renderer,0,0,0,255))
        self.check(s.SDL_RenderClear(self.renderer))
        self.check(s.SDL_RenderCopy(self.renderer,self.texture,ct.byref(rect),ct.byref(rect)))
        s.SDL_RenderPresent(self.renderer)
        self.last_frame=original;self.last_overlay=overlay_key;self.last_compensation=compensation
        elapsed=time.monotonic()-self.last_present
        if elapsed<1/self.config["refresh"]:time.sleep(1/self.config["refresh"]-elapsed)
        self.last_present=time.monotonic()
        return True

    def capture(self,full,size):
        source=getattr(self,'capture_framebuffer',None)
        if source is None and getattr(self,'capturing_output',False):source=0
        # Hardware linear blits downsample the normalized HDMI buffer much
        # faster than repeated integer texture fetches on this PowerVR driver.
        packed_source=getattr(self,'capture_source',None) if full else None
        return self.gpu_renderer.capture(full,size,asynchronous=True,output_framebuffer=source,source_renderer=packed_source) if self.gpu_renderer is not None else self.last_frame

    def destroy_window(self):
        if self.gl_context:
            self.s.SDL_GL_MakeCurrent(self.window,self.gl_context)
            if self.gpu_renderer:self.gpu_renderer.close();self.gpu_renderer=None
            self.s.SDL_GL_DeleteContext(self.gl_context);self.gl_context=None
        for name,deleter in (("texture",self.s.SDL_DestroyTexture),("renderer",self.s.SDL_DestroyRenderer),("window",self.s.SDL_DestroyWindow)):
            value=getattr(self,name)
            if value:deleter(value)
            setattr(self,name,None)

    def close(self):
        self.destroy_window()
        self.s.SDL_Quit()
