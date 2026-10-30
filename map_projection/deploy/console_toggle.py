#!/usr/bin/env python3
"""Toggle native projection/text VT with a released Ctrl+Alt chord.

Read keyboard evdev events without grabbing devices or recording characters.
Other Ctrl+Alt shortcuts cancel the chord. New USB keyboards are discovered.
"""
import fcntl
import glob
import json
import os
from pathlib import Path
import selectors
import struct
import subprocess
import time
import urllib.request

CTRL = {29, 97}
ALT = {56}  # Do not treat AltGr as Alt.
EVENT = struct.Struct('@llHHi')


class Chord:
    def __init__(self):
        self.keys = set()
        self.armed = False
        self.cancelled = False

    def event(self, code, value):
        if value == 2:
            return False
        if value:
            self.keys.add(code)
            if code not in CTRL | ALT:
                self.cancelled = True
        else:
            self.keys.discard(code)
        if self.keys & CTRL and self.keys & ALT:
            self.armed = True
            if self.keys - (CTRL | ALT):
                self.cancelled = True
        if not self.keys & (CTRL | ALT):
            fire = self.armed and not self.cancelled
            self.armed = self.cancelled = False
            return fire
        return False


def run(*args):
    subprocess.run(args, check=True, timeout=35)


def toggle():
    active = subprocess.run(['systemctl', 'is-active', '--quiet', 'map-projection.service']).returncode == 0
    if active:
        # Preserve the displayed time rather than returning to the animation's
        # original start after restarting the graphics process.
        try:
            with urllib.request.urlopen('http://127.0.0.1:8080/api/state', timeout=2) as response:
                state = json.load(response)
            config = state['config']
            if state.get('displayed_time'):
                config['sun']['time'] = state['displayed_time']
            # Console work pauses playback; resume is an explicit UI action.
            config['sun']['playing'] = False
            target = Path('/home/map/map-automation/map_projection/settings.json')
            temporary = target.with_suffix('.console.tmp')
            temporary.write_text(json.dumps(config, indent=2))
            owner = target.stat()
            os.chown(temporary, owner.st_uid, owner.st_gid)
            temporary.chmod(0o600)
            temporary.replace(target)
        except Exception as exc:
            print('Could not save display state:', exc, flush=True)
        run('systemctl', 'stop', 'map-projection.service')
        run('systemctl', 'start', 'getty@tty2.service')
        run('chvt', '2')
        print('HDMI handed to text console on VT2.', flush=True)
    else:
        run('chvt', '1')
        run('systemctl', 'start', '--no-block', 'map-projection.service')
        print('HDMI handed to projector on VT1.', flush=True)


def main():
    selector = selectors.DefaultSelector()
    devices = {}
    last_toggle = 0.
    while True:
        for path in glob.glob('/dev/input/event*'):
            if path in devices:
                continue
            fd = None
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                bits = bytearray(96)
                # EVIOCGBIT(EV_KEY, 96), shared by arm64 and x86_64 Linux.
                fcntl.ioctl(fd, 0x80604521, bits, True)
                if not all(bits[key // 8] & (1 << (key % 8)) for key in [29, 56]):
                    os.close(fd)
                    continue
                devices[path] = fd
                selector.register(fd, selectors.EVENT_READ, (path, Chord()))
            except OSError:
                if fd is not None:
                    os.close(fd)
        for selected, _ in selector.select(timeout=1):
            path, chord = selected.data
            try:
                data = os.read(selected.fd, EVENT.size * 64)
                if not data:
                    raise OSError('Keyboard disconnected')
                for _, _, kind, code, value in EVENT.iter_unpack(data):
                    if kind == 0 and code == 3:  # SYN_DROPPED: never use stale modifiers.
                        chord.__init__()
                    elif kind == 1 and chord.event(code, value):
                        now = time.monotonic()
                        if now - last_toggle >= .5:
                            last_toggle = now
                            try:
                                toggle()
                            except Exception as exc:
                                print('Console toggle failed:', exc, flush=True)
            except (OSError, ValueError):
                selector.unregister(selected.fd)
                os.close(selected.fd)
                devices.pop(path, None)


if __name__ == '__main__':
    main()
