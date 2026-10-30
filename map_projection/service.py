"""Optional systemd readiness/watchdog; no dependency outside systemd."""
import os
import socket
import sys
import json
from pathlib import Path


class Notifier:
    def __init__(self):
        self.address=os.environ.get('NOTIFY_SOCKET')
        if self.address and self.address.startswith('@'):self.address='\0'+self.address[1:]
        self.next_ping=0.
        self.interval=min(1.,max(.1,int(os.environ.get('WATCHDOG_USEC','3000000'))/3e6))

    def send(self,message):
        if not self.address:return
        try:
            with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as channel:
                channel.settimeout(.1);channel.sendto(message.encode(),self.address)
        except OSError:pass

    def ready(self):self.send('READY=1\nSTATUS=Projection controls available; display reconnects automatically')

    def tick(self,now):
        if now>=self.next_ping:
            self.send('WATCHDOG=1');self.next_ping=now+self.interval


def restart_for_hdmi(engine):
    """Replace vendor graphics state without running its unsafe teardown path."""
    from . import config as settings
    with engine.lock:
        config=engine.snapshot()['config']
        if engine.config_path:settings.save(engine.config_path,config)
    environment=dict(os.environ)
    environment['MAP_PROJECTION_RESUME_HDMI']='1' if config['sun']['playing'] else '0'
    environment['MAP_PROJECTION_PLAYBACK_ORIGIN']=json.dumps(engine.playback_origin)
    print('HDMI changed: replacing the app process to reset SDL/PowerVR scanout.',flush=True)
    # SDL can reuse descriptor zero after dropping the old KMS device. Ensure
    # no stale DRM master survives exec, including a device accidentally at fd 0.
    if not os.isatty(0):
        try:fd=os.open('/dev/tty',os.O_RDWR)
        except OSError:fd=os.open(os.devnull,os.O_RDWR)
        os.dup2(fd,0)
        if fd!=0:os.close(fd)
    for entry in Path('/proc/self/fd').iterdir():
        fd=int(entry.name)
        if fd>2:
            try:os.close(fd)
            except OSError:pass
    os.execve(sys.executable,[sys.executable,'-m','map_projection',*sys.argv[1:]],environment)
