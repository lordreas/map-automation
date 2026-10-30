"""ctypes boundary for the portable C99 renderer (also builds with ARM gcc)."""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
SIGNATURE = hashlib.sha256(b''.join((ROOT/"native"/name).read_bytes() for name in ('renderer.c','live.c'))).hexdigest()[:12]


def build():
    source = ROOT / "native" / "renderer.c"
    suffix = ".dll" if os.name == "nt" else ".dylib" if sys.platform == "darwin" else ".so"
    target = ROOT / ".cache" / ("renderer-" + SIGNATURE + suffix)
    target.parent.mkdir(exist_ok=True)
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler:
        command = [compiler]
    else:
        try:
            import ziglang  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install gcc (Linux), or pip install ziglang==0.14.1 (Windows).") from exc
        command = [sys.executable, "-m", "ziglang", "cc"]
    command += ["-O3", "-std=c99", "-shared"]
    if sys.platform.startswith('linux'):command += ['-fopenmp']
    if os.name != "nt":
        command += ["-fPIC"]
    command += [str(source), "-o", str(target), "-lm"]
    print("Building bounded-memory STL renderer...", flush=True)
    subprocess.run(command, check=True, cwd=ROOT)
    return target


class Native:
    def __init__(self):
        self.lib = ct.CDLL(str(build()))
        self.signature = SIGNATURE
        f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        i8 = np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
        u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        self.lib.scan_stl.argtypes = [ct.c_char_p, f64]
        self.lib.raster_stl.argtypes = [ct.c_char_p, f64, f64, ct.c_int, ct.c_int, ct.c_int, f32, ct.c_void_p]
        self.lib.shade.argtypes = [ct.c_int, ct.c_int, f32, i8, f64, f64, f64, ct.c_int, f32, f64, ct.c_double, ct.c_double, u8]
        self.lib.shade.restype = None
        self.lib.ray_stl.argtypes = [ct.c_char_p, f64, f64, f64, f64]
        self.lib.horizon.argtypes = [f32,ct.c_int,ct.c_int,ct.c_double,ct.c_double,f64,f32]
        self.lib.horizon.restype = None
        self.lib.shade_grid.argtypes = [ct.c_int,ct.c_int,f32,i8,f64,f64,f32,f32,ct.c_int,ct.c_int,
                                       ct.c_double,ct.c_double,ct.c_double,ct.c_double,f64,ct.c_double,ct.c_double,ct.c_int,u8]
        self.lib.shade_grid.restype = None
        i32=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
        self.lib.prepare_normals.argtypes=[ct.c_int,ct.c_int,f32,ct.c_double,ct.c_double,f32]
        self.lib.prepare_lookup.argtypes=[ct.c_int,ct.c_int,f32,f64,f64,ct.c_double,ct.c_double,ct.c_int,ct.c_int,i32]
        self.lib.transpose_f32.argtypes=[f32,ct.c_int,ct.c_int,f32]
        self.lib.horizon_rows_fast.argtypes=[f32,ct.c_int,ct.c_int,ct.c_double,ct.c_double,f64,f32]
        self.lib.shade_cells.argtypes=[ct.c_int,f32,f32,f32,f64,ct.c_double,ct.c_double,u8]
        self.lib.gather.argtypes=[ct.c_int,i32,u8,u8]

    def lighting_scene(self,depth,inverse,origin,z,map_size,terrain_size):
        gh,gw=z.shape;oh,ow=depth.shape
        normals=np.empty((gh,gw,3),np.float32)
        self.lib.prepare_normals(gw,gh,z,*terrain_size,normals)
        lookup=np.empty((oh,ow),np.int32)
        self.lib.prepare_lookup(ow,oh,depth,np.ascontiguousarray(inverse),origin,*map_size,gw,gh,lookup)
        return normals,lookup

    def fast_live(self,z,normals,lookup,terrain_size,sun,ambient,bias,work):
        gh,gw=z.shape
        if not work:
            work.update(h=np.empty_like(z),zt=np.ascontiguousarray(z.T),ht=np.empty((gw,gh),np.float32),
                        light=np.empty(z.shape,np.uint8))
        if abs(sun[0]/(terrain_size[0]/gw))>=abs(sun[1]/(terrain_size[1]/gh)):
            st=np.array([-sun[1],-sun[0],sun[2]])
            self.lib.horizon_rows_fast(work['zt'],gh,gw,terrain_size[1]/gh,terrain_size[0]/gw,st,work['ht'])
            self.lib.transpose_f32(work['ht'],gh,gw,work['h'])
        else:self.lib.horizon_rows_fast(z,gw,gh,terrain_size[0]/gw,terrain_size[1]/gh,sun,work['h'])
        self.lib.shade_cells(z.size,z,work['h'],normals,sun,ambient,bias,work['light'])
        frame=np.empty((*lookup.shape,3),np.uint8)
        self.lib.gather(lookup.size,lookup,work['light'],frame)
        return frame

    @staticmethod
    def path(path):
        # C runtime fopen on Windows uses the system codepage.
        return str(path).encode("mbcs" if os.name == "nt" else "utf-8")

    def scan(self, path):
        bounds = np.empty(6, np.float64)
        result = self.lib.scan_stl(self.path(path), bounds)
        if result:
            raise ValueError(f"Cannot scan binary STL: error {result} (missing, truncated, or nonfinite mesh).")
        return bounds

    def raster(self, path, bounds, matrix, width, height, perspective=True):
        depth = np.empty((height, width), np.float32)
        normals = np.empty((height, width, 3), np.int8) if perspective else None
        result = self.lib.raster_stl(self.path(path), bounds, np.ascontiguousarray(matrix), width, height,
                                     int(perspective), depth, normals.ctypes.data if normals is not None else None)
        if result:
            raise RuntimeError(f"STL read failed during rendering: {result}")
        return depth, normals

    def shade(self, depth, normals, inverse, origin, sun_matrix, shadow, sun, ambient, bias):
        height, width = depth.shape
        rgb = np.empty((height, width, 3), np.uint8)
        self.lib.shade(width, height, depth, normals, np.ascontiguousarray(inverse), origin,
                       np.ascontiguousarray(sun_matrix), shadow.shape[0], shadow, sun, ambient, bias, rgb)
        return rgb

    def ray(self, path, bounds, origin, direction):
        hit = np.empty(3, np.float64)
        result = self.lib.ray_stl(self.path(path), bounds, np.asarray(origin, np.float64), np.asarray(direction, np.float64), hit)
        if result < 0:
            raise RuntimeError(f"STL read failed during point query: {result}")
        return hit if result == 0 else None

    def live(self, depth, normals, inverse, origin, terrain, map_size, terrain_size, sun, ambient, bias, dem=False):
        gh,gw=terrain.shape
        horizon=np.empty_like(terrain)
        self.lib.horizon(terrain,gw,gh,terrain_size[0]/gw,terrain_size[1]/gh,sun,horizon)
        h,w=depth.shape
        rgb=np.empty((h,w,3),np.uint8)
        self.lib.shade_grid(w,h,depth,normals,np.ascontiguousarray(inverse),origin,terrain,horizon,gw,gh,
                            *map_size,*terrain_size,sun,ambient,bias,int(dem),rgb)
        return rgb
