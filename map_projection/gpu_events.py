"""Incremental event replay: touch changed cells, then shade directly at 4K."""
from datetime import datetime,timezone
import ctypes as C
import numpy as np
from OpenGL.GLES3 import *
from OpenGL.GLES3.VERSION.GLES3_3_1 import glDispatchCompute,glMemoryBarrier,glBindImageTexture
DENSE_EVENT_THRESHOLD=500000
UPDATE='''#version 310 es
precision highp float;precision highp int;
layout(local_size_x=64) in;
layout(r32ui,binding=1) writeonly uniform highp uimage2D mask;
layout(std430,binding=3) readonly buffer Changes{uvec2 changes[];};uniform uint count;uniform uint begin;uniform int grid_width;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=count)return;uvec2 c=changes[i+begin];imageStore(mask,ivec2(int(c.x)%grid_width,int(c.x)/grid_width),uvec4(c.y));}
'''
GATHER='''#version 300 es
precision highp float;precision highp int;
uniform highp usampler2D lookup;uniform highp usampler2D mask;uniform sampler2D normals;
uniform int flip_y;uniform int grid_only;uniform float ambient;uniform vec3 sun;
out vec4 color;
void main(){ivec2 pixel=ivec2(gl_FragCoord.xy);uvec2 p;
 if(grid_only!=0)p=uvec2(pixel);else{if(flip_y!=0)pixel.y=textureSize(lookup,0).y-1-pixel.y;p=texelFetch(lookup,pixel,0).rg;}
 if(p.x==65535u){color=vec4(0.,0.,0.,1.);return;}
 uint value=texelFetch(mask,ivec2(p),0).r;vec3 normal=vec3(value&1023u,(value>>10)&1023u,(value>>20)&1023u)*(2./1023.)-1.;float lit=float(value>>30)/3.;
 float v=sun.z<=0.?0.:ambient+(1.-ambient)*lit*max(0.,dot(normal,sun));color=vec4(v,v,v,1.);}
'''
PREVIEW='''#version 300 es
precision highp float;precision highp int;
uniform highp usampler2D lookup;uniform highp usampler2D mask;
uniform vec2 target_size;uniform float ambient;uniform vec3 sun;
out vec4 color;
float sample_light(ivec2 pixel){
 uvec2 p=texelFetch(lookup,pixel,0).rg;if(p.x==65535u||sun.z<=0.)return 0.;
 uint value=texelFetch(mask,ivec2(p),0).r;
 vec3 normal=vec3(value&1023u,(value>>10)&1023u,(value>>20)&1023u)*(2./1023.)-1.;
 return ambient+(1.-ambient)*(float(value>>30)/3.)*max(0.,dot(normal,sun));
}
void main(){
 vec2 scale=vec2(textureSize(lookup,0))/target_size;
 vec2 origin=floor(gl_FragCoord.xy)*scale;float sum=0.;
 for(int y=0;y<SAMPLES_Y;y++)for(int x=0;x<SAMPLES_X;x++){
  ivec2 pixel=ivec2(origin+(vec2(x,y)+.5)*scale/vec2(float(SAMPLES_X),float(SAMPLES_Y)));
  sum+=sample_light(pixel);
 }
 float v=sum/float(SAMPLES_X*SAMPLES_Y);color=vec4(v,v,v,1.);
}
'''
class EventRenderer:
 def __init__(self,r):
  from .gpu import program,compute
  self.r=r;self.program=program(GATHER);self.update=compute(UPDATE)
  for p in (self.program,self.update):r.locations[p]={}
  self.key=None;self.textures=[];self.position=0;self.preview_programs={}
 def prepare(self,frame):
  from .gpu import texture
  e=frame.scene.events;r=self.r
  if e is None:raise ValueError('Daily event data missing')
  if self.key==str(e.path):return
  self.release();h,w=frame.scene.z.shape;limit=int(glGetIntegerv(GL_MAX_TEXTURE_SIZE));self.width=limit
  self.times=np.load(e.path/'sorted-times.npy',mmap_mode='r');self.cells=np.load(e.path/'sorted-cells.npy',mmap_mode='r')
  self.words=np.empty((h,w),np.uint32)
  for row in range(0,h,128):
   packed=np.rint((np.asarray(frame.scene.normals[row:row+128])*.5+.5)*1023).astype(np.uint32)
   self.words[row:row+128]=packed[:,:,0]|(packed[:,:,1]<<10)|(packed[:,:,2]<<20)
  self.words=self.words.reshape(-1);self.words|=np.asarray(e.initial,np.uint32)*np.uint32(0xc0000000)
  self.textures=[texture(10,self.words.reshape(h,w),GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT,immutable=True)]
  self.buffer=glGenBuffers(1)
  self.key=str(e.path);self.position=0
 def draw(self,frame,target=0,physical=True):
  self.prepare(frame);r=self.r;e=frame.scene.events
  moment=datetime.fromisoformat(frame.time).astimezone(timezone.utc);seconds=(moment-datetime.fromisoformat(e.meta['start'])).total_seconds()
  if frame.sun[2]<=0:pos=self.position
  else:
   if not 0<=seconds<86400:raise ValueError('Event cache date mismatch')
   pos=int(np.searchsorted(self.times,np.uint32(round(seconds*1000)),side='right'))
  begin=min(pos,self.position);count=abs(pos-self.position)
  for unit,tex in zip((10,),self.textures):glActiveTexture(GL_TEXTURE0+unit);glBindTexture(GL_TEXTURE_2D,tex)
  if count>=DENSE_EVENT_THRESHOLD:
   np.bitwise_xor.at(self.words,np.asarray(self.cells[begin:begin+count]),np.uint32(0xc0000000))
   glMemoryBarrier(0x100)
   glTexSubImage2D(GL_TEXTURE_2D,0,0,0,r.gw,r.gh,GL_RED_INTEGER,GL_UNSIGNED_INT,self.words.ctypes.data_as(C.c_void_p))
  elif count:
   glMemoryBarrier(0x20)
   selected=np.asarray(self.cells[begin:begin+count]);unique,counts=np.unique(selected,return_counts=True)
   unique=unique[(counts&1)!=0];self.words[unique]^=np.uint32(0xc0000000)
   payload=np.ascontiguousarray(np.stack((unique,self.words[unique]),axis=-1),np.uint32)
   if len(payload):
    p=self.update;glUseProgram(p);glBindBuffer(0x90D2,self.buffer);glBindBufferBase(0x90D2,3,self.buffer)
    glBufferData(0x90D2,payload.nbytes,payload.ctypes.data_as(C.c_void_p),GL_STREAM_DRAW)
    glBindImageTexture(1,self.textures[0],0,False,0,GL_WRITE_ONLY,GL_R32UI)
    r.integer(p,'grid_width',r.gw)
    for first in range(0,len(payload),65535*64):
     size=min(65535*64,len(payload)-first);glUniform1ui(r.location(p,'begin'),first);glUniform1ui(r.location(p,'count'),size);glDispatchCompute((size+63)//64,1,1)
    glMemoryBarrier(0x28)
  self.position=pos;self.frame=frame;self.render(target,physical)
 def render(self,target,physical,grid=False):
  r=self.r;p=self.program;frame=self.frame;glUseProgram(p)
  for name,v in [('lookup',4),('normals',2),('mask',10),('flip_y',physical),('grid_only',grid)]:r.integer(p,name,v)
  r.real(p,'ambient',frame.ambient);glUniform3f(r.location(p,'sun'),*frame.sun)
  glBindFramebuffer(GL_FRAMEBUFFER,target);glViewport(0,0,r.gw if grid else r.ow,r.gh if grid else r.oh);glDrawArrays(GL_TRIANGLES,0,3)
 def preview(self,target,width,height):
  # Compile constant sample counts: no full-grid materialization and no dynamic
  # 8x8 loop for a thumbnail that normally needs only 4x4 area samples.
  from .gpu import program
  r=self.r;key=(max(1,min(8,int(np.ceil(r.ow/width)))),max(1,min(8,int(np.ceil(r.oh/height)))))
  if key not in self.preview_programs:
   source=PREVIEW.replace('SAMPLES_X',str(key[0])).replace('SAMPLES_Y',str(key[1]))
   p=program(source);self.preview_programs[key]=p;r.locations[p]={}
  p=self.preview_programs[key];glUseProgram(p);r.integer(p,'lookup',4);r.integer(p,'mask',10)
  glActiveTexture(GL_TEXTURE10);glBindTexture(GL_TEXTURE_2D,self.textures[0])
  r.real(p,'ambient',self.frame.ambient);glUniform3f(r.location(p,'sun'),*self.frame.sun);glUniform2f(r.location(p,'target_size'),width,height)
  glBindFramebuffer(GL_FRAMEBUFFER,target);glViewport(0,0,width,height);glDrawArrays(GL_TRIANGLES,0,3)
 def release(self):
  if self.textures:glDeleteTextures(len(self.textures),self.textures)
  if getattr(self,'buffer',None) is not None:glDeleteBuffers(1,[self.buffer]);self.buffer=None
  self.textures=[];self.key=None
 def close(self):
  self.release();glDeleteProgram(self.program);glDeleteProgram(self.update)
  for p in self.preview_programs.values():glDeleteProgram(p)
  self.preview_programs.clear()
