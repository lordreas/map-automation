"""Force a fresh modeset on the DRM device owned by the direct-output process."""
import ctypes as C
import os
from pathlib import Path


class Resources(C.Structure):
    _fields_=[('count_fbs',C.c_int),('fbs',C.POINTER(C.c_uint32)),
              ('count_crtcs',C.c_int),('crtcs',C.POINTER(C.c_uint32)),
              ('count_connectors',C.c_int),('connectors',C.POINTER(C.c_uint32)),
              ('count_encoders',C.c_int),('encoders',C.POINTER(C.c_uint32)),
              ('min_width',C.c_uint32),('max_width',C.c_uint32),
              ('min_height',C.c_uint32),('max_height',C.c_uint32)]


def reset_scanout():
    # This app owns the selected card exclusively in --direct mode. Reuse the
    # actual SDL master fd: opening the device again would not grant DRM master.
    card='/dev/dri/card'+os.environ.get('SDL_KMSDRM_DEVICE_INDEX','0')
    library=C.CDLL('libdrm.so.2',use_errno=True)
    library.drmIsMaster.argtypes=[C.c_int];library.drmIsMaster.restype=C.c_int
    library.drmModeGetResources.argtypes=[C.c_int]
    library.drmModeGetResources.restype=C.POINTER(Resources)
    library.drmModeFreeResources.argtypes=[C.POINTER(Resources)]
    library.drmModeFreeResources.restype=None
    library.drmModeSetCrtc.argtypes=[C.c_int,C.c_uint32,C.c_uint32,C.c_uint32,C.c_uint32,
                                   C.POINTER(C.c_uint32),C.c_int,C.c_void_p]
    library.drmModeSetCrtc.restype=C.c_int
    for entry in Path('/proc/self/fd').iterdir():
        try:
            if os.readlink(entry)!=card:continue
        except OSError:continue
        fd=int(entry.name)
        if library.drmIsMaster(fd)!=1:continue
        resources=library.drmModeGetResources(fd)
        if not resources:raise OSError(C.get_errno(),'Cannot read DRM resources')
        try:
            crtcs=[resources.contents.crtcs[i] for i in range(resources.contents.count_crtcs)]
            for crtc in crtcs:
                if library.drmModeSetCrtc(fd,crtc,0,0,0,None,0,None):
                    raise OSError(C.get_errno(),'Cannot disable stale DRM scanout')
            return crtcs
        finally:library.drmModeFreeResources(resources)
    raise RuntimeError('SDL does not own the selected DRM card; cannot reset scanout.')
