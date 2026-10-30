"""CPU fallback for the GPU's green-channel radial magnification correction."""
import math
import numpy as np
from PIL import Image


def compensate_rgb(frame,pixels):
    if not pixels:return frame
    image=Image.fromarray(frame).convert('RGB');w,h=image.size
    k=pixels/math.hypot(w/2,h/2)
    green=image.getchannel('G').transform((w,h),Image.Transform.AFFINE,(1+k,0,-k*w/2,0,1+k,-k*h/2),Image.Resampling.BILINEAR)
    r,_,b=image.split();return np.asarray(Image.merge('RGB',(r,green,b))).copy()
