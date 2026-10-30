"""Verify SDL drawable dimensions and RGB readback at full HD on this laptop."""
import ctypes as ct
import numpy as np
from map_projection.display import Display

d=Display()
try:
    d.configure({"width":1920,"height":1080,"refresh":60,"display":0,"fullscreen":False,"mirror":False})
    s=d.s
    expected=np.zeros((1080,1920,3),np.uint8)
    expected[517,913]=255
    actual=np.empty_like(expected)
    d.check(s.SDL_UpdateTexture(d.texture,None,expected.ctypes.data,1920*3))
    d.check(s.SDL_RenderCopy(d.renderer,d.texture,None,None))
    d.check(s.SDL_RenderReadPixels(d.renderer,None,s.SDL_PIXELFORMAT_RGB24,actual.ctypes.data,1920*3))
    np.testing.assert_array_equal(actual,expected)
    print("FHD SDL RGB readback: exactly one white pixel at (913, 517), all others black.")
    print(d.status)
finally:d.close()
