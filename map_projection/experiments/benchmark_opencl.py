"""Offscreen PowerVR probe: compute stages, transfers and fresh CPU shadows.
Does not change the application's display or renderer.
"""
import ctypes as C,json,time,statistics
from pathlib import Path
import numpy as np
from map_projection.native import Native
root=Path(__file__).resolve().parent
cl=C.CDLL('/usr/lib/libPVROCL.so.1');P=C.c_void_p;I=C.c_int;U=C.c_uint;S=C.c_size_t;F=C.c_float

def bind(name,ret,args):
 f=getattr(cl,name);f.restype=ret;f.argtypes=args;return f

def ok(code):
 if code:raise RuntimeError('OpenCL error '+str(code))
getplatform=bind('clGetPlatformIDs',I,[U,C.POINTER(P),C.POINTER(U)])
getdevice=bind('clGetDeviceIDs',I,[P,C.c_ulong,U,C.POINTER(P),C.POINTER(U)])
info=bind('clGetDeviceInfo',I,[P,U,S,P,C.POINTER(S)])
platform=P();device=P();ok(getplatform(1,C.byref(platform),None));ok(getdevice(platform,4,1,C.byref(device),None))
buf=C.create_string_buffer(512);ok(info(device,0x102B,512,buf,None));print('GPU: '+buf.value.decode(),flush=True)
context=bind('clCreateContext',P,[P,U,C.POINTER(P),P,P,C.POINTER(I)])
queuefn=bind('clCreateCommandQueue',P,[P,P,C.c_ulong,C.POINTER(I)])
err=I();ctx=context(None,1,C.byref(device),None,None,C.byref(err));ok(err.value);q=queuefn(ctx,device,0,C.byref(err));ok(err.value)
createbuf=bind('clCreateBuffer',P,[P,C.c_ulong,S,P,C.POINTER(I)])
write=bind('clEnqueueWriteBuffer',I,[P,P,U,S,S,P,U,P,P]);read=bind('clEnqueueReadBuffer',I,[P,P,U,S,S,P,U,P,P])
programfn=bind('clCreateProgramWithSource',P,[P,U,C.POINTER(C.c_char_p),P,C.POINTER(I)])
build=bind('clBuildProgram',I,[P,U,C.POINTER(P),C.c_char_p,P,P])
buildinfo=bind('clGetProgramBuildInfo',I,[P,P,U,S,P,P])
kernelfn=bind('clCreateKernel',P,[P,C.c_char_p,C.POINTER(I)])
arg=bind('clSetKernelArg',I,[P,U,S,P]);run=bind('clEnqueueNDRangeKernel',I,[P,P,U,P,C.POINTER(S),P,U,P,P]);finish=bind('clFinish',I,[P])
source=b'''
__kernel void lighting(__global const float*z,__global const float*h,__global const float*n,
 float sx,float sy,float sz,float ambient,float bias,int count,__global uchar*out) {
 int p=get_global_id(0);if(p>=count)return;
 float diffuse=fmax(0.0f,n[3*p]*sx+n[3*p+1]*sy+n[3*p+2]*sz);
 if(h[p]>z[p]+bias)diffuse=0;
 out[p]=sz>0?convert_uchar_sat(floor(255*fmin(1.0f,ambient+(1-ambient)*diffuse)+0.5f)):0;
}
__kernel void gather(__global const int*lookup,__global const uchar*light,int count,__global uchar*out){
 int p=get_global_id(0);if(p>=count)return;
 uchar value=lookup[p]<0?0:light[lookup[p]];vstore3((uchar3)(value),p,out);
}
'''
src=C.c_char_p(source);program=programfn(ctx,1,C.byref(src),None,C.byref(err));ok(err.value)
code=build(program,1,C.byref(device),b'',None,None)
if code:
 log=C.create_string_buffer(16384);buildinfo(program,device,0x1183,len(log),log,None);raise RuntimeError(log.value.decode())
k=kernelfn(program,b'lighting',C.byref(err));ok(err.value);g=kernelfn(program,b'gather',C.byref(err));ok(err.value)
data=np.load(root/'gpu-input.npz');z=np.ascontiguousarray(data['z']);h=np.empty_like(z);normals=np.ascontiguousarray(data['normals']);lookup=np.ascontiguousarray(data['lookup']);rgb=np.empty((*lookup.shape,3),np.uint8);light=np.empty_like(z,dtype=np.uint8)
ts=data['terrain_size'];sun=data['sun'];ambient=float(data['ambient']);bias=float(data['bias']);gh,gw=z.shape
native=Native()
def buffer(a):
 b=createbuf(ctx,1|32,a.nbytes,a.ctypes.data,C.byref(err));ok(err.value);return P(b)
zb,hb,nb,lb,ib,ob=map(buffer,[z,h,normals,light,lookup,rgb])
def setarg(kernel,index,value):ok(arg(kernel,index,C.sizeof(value),C.byref(value)))
for i,v in enumerate([zb,hb,nb,F(sun[0]),F(sun[1]),F(sun[2]),F(ambient),F(bias),I(z.size),lb]):setarg(k,i,v)
for i,v in enumerate([ib,lb,I(lookup.size),ob]):setarg(g,i,v)
def launch(kernel,count):
 size=S(count);ok(run(q,kernel,1,None,C.byref(size),None,0,None,None));ok(finish(q))
launch(k,z.size);launch(g,lookup.size)
results=[]
for trial in range(6):
 s=sun.copy();s[0]+=(trial-2)*.001;s/=np.linalg.norm(s)
 for i in range(3):setarg(k,i+3,F(s[i]))
 t=time.perf_counter()
 if abs(s[0]/(ts[0]/gw))>=abs(s[1]/(ts[1]/gh)):
  zt=np.ascontiguousarray(z.T);ht=np.empty_like(zt);st=np.array([-s[1],-s[0],s[2]])
  native.lib.horizon(zt,gh,gw,ts[1]/gh,ts[0]/gw,st,ht);h[:]=ht.T
 else:native.lib.horizon(z,gw,gh,ts[0]/gw,ts[1]/gh,s,h)
 a=time.perf_counter();ok(write(q,hb,1,0,h.nbytes,h.ctypes.data,0,None,None));b=time.perf_counter()
 launch(k,z.size);c=time.perf_counter();launch(g,lookup.size);d=time.perf_counter()
 ok(read(q,ob,1,0,rgb.nbytes,rgb.ctypes.data,0,None,None));end=time.perf_counter()
 row=dict(zip(['cpu_shadow_ms','upload_ms','gpu_lighting_ms','gpu_gather_ms','readback_ms','total_ms'],[round(v*1000,3) for v in [a-t,b-a,c-b,d-c,end-d,end-t]]))
 results.append(row);print(json.dumps(row),flush=True)
print('MEDIANS '+json.dumps({key:round(statistics.median(r[key] for r in results[1:]),3) for key in results[0]}),flush=True)
(root/'gpu-results.json').write_text(json.dumps(results,indent=2)+'\n')
