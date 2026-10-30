"""Independent float64 oracle for a continuous, bilinear DEM surface.

Samples live at integer (x, y). Each unit square is one bilinear patch. The
oracle visits every crossed patch and maximizes terrain minus ray analytically,
including an interior quadratic maximum. No shear, mipmaps or marching steps.
"""
import math
import numpy as np


def bilinear_blocked(z, x, y, direction, rise, bias):
    h,w=z.shape;dx,dy=map(float,direction);origin=np.array([x,y],float)
    start=float(z[y,x])+bias;t=0.
    if max(abs(dx),abs(dy))<1e-14:return False
    for _ in range(w+h+4):
        p=origin+np.array([dx,dy])*t
        # Choose the patch on the outgoing side of an exact grid boundary.
        cell=np.floor(p+np.array([dx,dy])*1e-9).astype(int);cx,cy=cell
        if abs(dx)<1e-14:cx=min(cx,w-2)
        if abs(dy)<1e-14:cy=min(cy,h-2)
        cell=np.array([cx,cy])
        if cx<0 or cy<0 or cx>=w-1 or cy>=h-1:return False
        end=min((cx+(dx>=0)-x)/dx if abs(dx)>1e-14 else math.inf,
                (cy+(dy>=0)-y)/dy if abs(dy)>1e-14 else math.inf)
        span=max(0.,end-t);u,v=p-cell
        z00,z10,z01,z11=map(float,(z[cy,cx],z[cy,cx+1],z[cy+1,cx],z[cy+1,cx+1]))
        a=z10-z00;b=z01-z00;c=z11-z10-z01+z00
        q0=z00+a*u+b*v+c*u*v-start-rise*t
        q1=a*dx+b*dy+c*(u*dy+v*dx)-rise;q2=c*dx*dy
        maximum=max(q0,q0+q1*span+q2*span*span)
        if q2<0:
            vertex=-q1/(2*q2)
            if 0<vertex<span:maximum=max(maximum,q0+q1*vertex+q2*vertex*vertex)
        if maximum>1e-3:return True
        t=end
    raise RuntimeError('Bilinear oracle did not exit the DEM')
