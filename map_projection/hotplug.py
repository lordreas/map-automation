"""DRM connector detection, including short unplug/replug uevents."""
from pathlib import Path
import hashlib
import os
import socket


class Monitor:
    def __init__(self,root='/sys/class/drm',card=None,events=True):
        self.root=Path(root);self.card=str(card if card is not None else os.environ.get('SDL_KMSDRM_DEVICE_INDEX','0'))
        self.last=None;self.serial=0;self.socket=None
        if events:
            try:
                self.socket=socket.socket(socket.AF_NETLINK,socket.SOCK_DGRAM,15)
                self.socket.bind((0,1));self.socket.setblocking(False)
            except (OSError,AttributeError):
                if self.socket:self.socket.close()
                self.socket=None

    def poll(self):
        if self.socket:
            while True:
                try:event=self.socket.recv(16384)
                except BlockingIOError:break
                except OSError:break
                if b'SUBSYSTEM=drm' in event and b'HOTPLUG=1' in event:self.serial+=1
        connectors=[]
        paths=sorted(self.root.glob(f'card{self.card}-HDMI-*'))
        if not paths:paths=sorted(self.root.glob(f'card{self.card}-DP-*'))
        for path in paths:
            try:
                status=(path/'status').read_text().strip()
                edid=(path/'edid').read_bytes() if (path/'edid').exists() else b''
                modes=(path/'modes').read_text() if (path/'modes').exists() else ''
                connectors.append((path.name,status,hashlib.sha256(edid).hexdigest(),modes))
            except OSError:continue
        signature=(tuple(connectors),self.serial)
        changed=self.last is not None and signature!=self.last
        self.last=signature
        return {'known':bool(connectors),'connected':any(c[1]=='connected' for c in connectors),
                'changed':changed,'connectors':[{'name':c[0],'status':c[1]} for c in connectors]}

    def close(self):
        if self.socket:self.socket.close();self.socket=None


class Recovery:
    """Retry initialization/configuration without stopping the web server."""
    def __init__(self,make_display,engine,direct=False,monitor=None,fallback=None,restart=None):
        self.make_display=make_display;self.engine=engine;self.direct=direct;self.monitor=monitor
        self.display=None;self.next_attempt=0.;self.next_poll=0.;self.connected=True
        self.fallback=fallback
        self.restart=restart

    def reset(self):
        display,self.display=self.display,None
        if display:
            try:display.close()
            except Exception:pass  # Driver may already have destroyed the old surface.
        self.engine.set_gpu(False)

    def reconnect(self):
        if self.restart:self.restart()
        else:self.reset();self.next_attempt=0

    def step(self,now,config):
        if self.monitor and now>=self.next_poll:
            connection=self.monitor.poll();self.next_poll=now+.25
            self.connected=not connection['known'] or connection['connected']
            if connection['changed']:
                if self.display is not None and self.restart:
                    # A fresh process clears SDL 2.0.14's cached atomic mode
                    # state. Rebuilding windows alone can report success while
                    # leaving the physical A733 HDMI transmitter disabled.
                    self.restart()
                    return None
                self.reset();self.next_attempt=0
            if not self.connected:
                self.reset()
                self.engine.output_status={'note':'HDMI disconnected. Reconnect the cable to restore projection.',
                    'connected':False,'recovering':True,'pixel_buffer_verified':False,'direct_scanout':False,
                    'connectors':connection['connectors']}
        if not self.connected:return None
        needs_config=self.display is None or self.display.config!=config
        if needs_config and now>=self.next_attempt:
            try:
                if self.display is None:self.display=self.make_display()
                self.display.configure(config)
                self.engine.set_gpu(self.display.gpu)
                self.engine.output_status=dict(self.display.status,connected=True,recovering=False)
            except Exception as exc:
                if self.display is not None and self.display.gpu and self.fallback:
                    self.reset()
                    try:
                        self.display=self.fallback();self.display.configure(config)
                        self.engine.output_status=dict(self.display.status,gpu_fallback=str(exc),connected=True,recovering=False)
                    except Exception as fallback_error:self.failed(fallback_error,now)
                else:self.failed(exc,now)
        return self.display if self.display and self.display.config==config else None

    def failed(self,error,now):
        self.reset();self.next_attempt=now+1
        self.engine.output_status={'note':'Reopening the display automatically. Web controls remain available.',
            'error':str(error),'recovering':True,'connected':self.connected,
            'pixel_buffer_verified':False,'direct_scanout':False}

    def close(self):
        self.reset()
        if self.monitor:self.monitor.close()
