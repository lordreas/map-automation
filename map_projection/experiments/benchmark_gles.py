"""Offscreen GLES benchmark. No window, scanout or production code changes."""
import os
os.environ['PYOPENGL_PLATFORM']='egl';os.environ['EGL_PLATFORM']='surfaceless'
import ctypes as C,json,time,statistics
from pathlib import Path
import numpy as np
from OpenGL.EGL import *
from OpenGL.GLES3 import *
from map_projection.native import Native
root=Path(__file__).resolve().parent
D=eglGetDisplay(EGL_DEFAULT_DISPLAY);major=EGLint();minor=EGLint();eglInitialize(D,major,minor)
attrs=(EGLint*13)(EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,0x40,EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE)
configs=(EGLConfig*1)();count=EGLint();eglChooseConfig(D,attrs,configs,1,count)
assert count.value
eglBindAPI(EGL_OPENGL_ES_API)
ctx=eglCreateContext(D,configs[0],EGL_NO_CONTEXT,(EGLint*3)(EGL_CONTEXT_CLIENT_VERSION,3,EGL_NONE))
surf=eglCreatePbufferSurface(D,configs[0],(EGLint*5)(EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE))
eglMakeCurrent(D,surf,surf,ctx)
print('RENDERER',C.string_at(glGetString(GL_RENDERER)).decode(),C.string_at(glGetString(GL_VERSION)).decode(),flush=True)
data=np.load(root/'gpu-input.npz');z=np.ascontiguousarray(data['z']);h=np.empty_like(z);normals=np.ascontiguousarray(data['normals']);lookup=np.ascontiguousarray(data['lookup']);ts=data['terrain_size'];sun=data['sun'];ambient=float(data['ambient']);bias=float(data['bias']);gh,gw=z.shape;oh,ow=lookup.shape
native=Native()
vertex='''#version 300 es
const vec2 p[3]=vec2[3](vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.));
void main(){gl_Position=vec4(p[gl_VertexID],0.,1.);}
'''
lighting='''#version 300 es
precision highp float;
uniform sampler2D terrain;uniform sampler2D horizon;uniform sampler2D normals;
uniform vec3 sun;uniform float ambient;uniform float bias;
out vec4 color;
void main(){ivec2 p=ivec2(gl_FragCoord.xy);float z=texelFetch(terrain,p,0).r;
 float diffuse=max(0.,dot(texelFetch(normals,p,0).rgb,sun));
 if(texelFetch(horizon,p,0).r>z+bias)diffuse=0.;
 float value=sun.z>0.?min(1.,ambient+(1.-ambient)*diffuse):0.;color=vec4(value,value,value,1.);}
'''
presentation='''#version 300 es
precision highp float;precision highp int;
uniform highp usampler2D lookup;
uniform sampler2D terrain;uniform sampler2D horizon;uniform sampler2D normals;
uniform vec3 sun;uniform float ambient;uniform float bias;
out vec4 color;
void main(){uvec2 cell=texelFetch(lookup,ivec2(gl_FragCoord.xy),0).rg;
 if(cell.r==65535u){color=vec4(0.,0.,0.,1.);return;}
 ivec2 p=ivec2(cell);float z=texelFetch(terrain,p,0).r;
 float diffuse=max(0.,dot(texelFetch(normals,p,0).rgb,sun));
 if(texelFetch(horizon,p,0).r>z+bias)diffuse=0.;
 float v=sun.z>0.?min(1.,ambient+(1.-ambient)*diffuse):0.;color=vec4(v,v,v,1.);}
'''
def program(fragment):
 shaders=[]
 for kind,source in [(GL_VERTEX_SHADER,vertex),(GL_FRAGMENT_SHADER,fragment)]:
  s=glCreateShader(kind);glShaderSource(s,source);glCompileShader(s)
  if not glGetShaderiv(s,GL_COMPILE_STATUS):raise RuntimeError(glGetShaderInfoLog(s))
  shaders.append(s)
 p=glCreateProgram()
 for s in shaders:glAttachShader(p,s)
 glLinkProgram(p)
 if not glGetProgramiv(p,GL_LINK_STATUS):raise RuntimeError(glGetProgramInfoLog(p))
 return p
def texture(unit,array,internal,fmt,typ,width=None,height=None):
 t=glGenTextures(1);glActiveTexture(GL_TEXTURE0+unit);glBindTexture(GL_TEXTURE_2D,t)
 for key,val in [(GL_TEXTURE_MIN_FILTER,GL_NEAREST),(GL_TEXTURE_MAG_FILTER,GL_NEAREST),(GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE),(GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE)]:glTexParameteri(GL_TEXTURE_2D,key,val)
 if array is not None:height,width=array.shape[:2]
 glTexImage2D(GL_TEXTURE_2D,0,internal,width,height,0,fmt,typ,None if array is None else array.ctypes.data_as(C.c_void_p))
 return t
def framebuffer(tex):
 fb=glGenFramebuffers(1);glBindFramebuffer(GL_FRAMEBUFFER,fb);glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tex,0)
 assert glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE
 return fb
terrain_tex=texture(0,z,GL_R32F,GL_RED,GL_FLOAT)
horizon_tex=texture(1,h,GL_R32F,GL_RED,GL_FLOAT)
normal_tex=texture(2,normals,GL_RGB32F,GL_RGB,GL_FLOAT)
uvmap=np.stack((np.maximum(lookup,0)%gw,np.maximum(lookup,0)//gw),axis=-1).astype(np.uint16)
uvmap[lookup<0]=65535
lookup_tex=texture(3,uvmap,GL_RG16UI,GL_RG_INTEGER,GL_UNSIGNED_SHORT)
light_tex=texture(4,None,GL_R8,GL_RED,GL_UNSIGNED_BYTE,gw,gh)
out_tex=texture(5,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,ow,oh)
lfb,ofb=framebuffer(light_tex),framebuffer(out_tex)
lp,pp=program(lighting),program(presentation)
glUseProgram(lp)
for name,unit in [('terrain',0),('horizon',1),('normals',2)]:glUniform1i(glGetUniformLocation(lp,name),unit)
glUniform1f(glGetUniformLocation(lp,'ambient'),ambient);glUniform1f(glGetUniformLocation(lp,'bias'),bias)
glUseProgram(pp);glUniform1i(glGetUniformLocation(pp,'lookup'),3);glUniform1i(glGetUniformLocation(pp,'light'),4);glUniform1i(glGetUniformLocation(pp,'grid_width'),gw)
glUseProgram(pp)
for name,unit in [('terrain',0),('horizon',1),('normals',2)]:glUniform1i(glGetUniformLocation(pp,name),unit)
glUniform1f(glGetUniformLocation(pp,'ambient'),ambient);glUniform1f(glGetUniformLocation(pp,'bias'),bias)
fast=C.CDLL(str(root/'fast_dem.so'))
f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
fast.transpose_f32.argtypes=[f32,C.c_int,C.c_int,f32]
fast.horizon_rows_fast.argtypes=[f32,C.c_int,C.c_int,C.c_double,C.c_double,f64,f32]
zt=np.empty((gw,gh),np.float32);ht=np.empty_like(zt);fast.transpose_f32(z,gw,gh,zt)
vao=glGenVertexArrays(1);glBindVertexArray(vao);glDisable(GL_DITHER)
results=[]
for trial in range(6):
 s=sun.copy();s[0]+=(trial-2)*.001;s/=np.linalg.norm(s)
 t=time.perf_counter()
 if abs(s[0]/(ts[0]/gw))>=abs(s[1]/(ts[1]/gh)):
  st=np.array([-s[1],-s[0],s[2]])
  fast.horizon_rows_fast(zt,gh,gw,ts[1]/gh,ts[0]/gw,st,ht);fast.transpose_f32(ht,gh,gw,h)
 else:fast.horizon_rows_fast(z,gw,gh,ts[0]/gw,ts[1]/gh,s,h)
 a=time.perf_counter();glActiveTexture(GL_TEXTURE1);glBindTexture(GL_TEXTURE_2D,horizon_tex)
 glTexSubImage2D(GL_TEXTURE_2D,0,0,0,gw,gh,GL_RED,GL_FLOAT,h.ctypes.data_as(C.c_void_p));glFinish();b=time.perf_counter()
 c=b
 glUseProgram(pp);glUniform3f(glGetUniformLocation(pp,'sun'),*s);glBindFramebuffer(GL_FRAMEBUFFER,ofb);glViewport(0,0,ow,oh);glDrawArrays(GL_TRIANGLES,0,3);glFinish();end=time.perf_counter()
 row=dict(zip(['cpu_shadow_ms','upload_ms','gpu_lighting_ms','gpu_4k_ms','total_ms'],[round(v*1000,3) for v in [a-t,b-a,c-b,end-c,end-t]]))
 results.append(row);print(json.dumps(row),flush=True)
print('MEDIANS '+json.dumps({key:round(statistics.median(r[key] for r in results[1:]),3) for key in results[0]}),flush=True)
(root/'gles-combined-results.json').write_text(json.dumps(results,indent=2)+'\n')
# Numerical readback against a CPU calculation, outside timing.
rgb=np.empty((oh,ow,4),np.uint8);glReadPixels(0,0,ow,oh,GL_RGBA,GL_UNSIGNED_BYTE,rgb.ctypes.data_as(C.c_void_p))
diffuse=np.maximum(0,np.einsum('ijk,k->ij',normals,s));diffuse[h>z+bias]=0
light=np.floor(255*np.minimum(1,ambient+(1-ambient)*diffuse)+.5).astype(np.uint8)
expected=np.where(lookup<0,0,light.reshape(-1)[np.maximum(lookup,0)])
print('MAX_GREY_DIFFERENCE',int(np.max(np.abs(rgb[:,:,0].astype(np.int16)-expected.astype(np.int16)))),flush=True)
