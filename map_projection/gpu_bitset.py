"""Exact millisecond event replay with a tiny visibility bitset and C updates."""
from datetime import datetime,timezone
import ctypes as C
import numpy as np
from .lighting import GLSL
from OpenGL.GLES3 import *
COMMON='''#version 300 es
precision highp float;precision highp int;
uniform sampler2D normals;uniform highp usampler2D lookup;uniform highp usampler2D bits;
uniform int grid_width;uniform int mask_width;uniform int flip_y;uniform int enabled;uniform vec3 sun;uniform float ambient;
out vec4 color;
'''+GLSL+'''
float light(ivec2 pixel){vec4 n=texelFetch(normals,pixel,0);if(n.a==0.)return 0.;if(enabled==0)return 1.;if(sun.z<=0.)return exposure(0.);
 uvec2 p=texelFetch(lookup,pixel,0).rg;uint index=p.y*uint(grid_width)+p.x;uint word=index>>5;
 uint value=texelFetch(bits,ivec2(int(word&1023u),int(word>>10)),0).r;
 float lit=float((value>>(index&31u))&1u);return exposure(ambient+(1.-ambient)*lit*max(0.,dot(n.rgb*2.-1.,sun)));}
'''
GATHER=COMMON+'''
void main(){ivec2 p=ivec2(gl_FragCoord.xy);if(flip_y!=0)p.y=textureSize(normals,0).y-1-p.y;float v=light(p);color=vec4(v,v,v,1.);}
'''
PREVIEW=COMMON+'''
uniform vec2 target_size;
void main(){vec2 scale=vec2(textureSize(normals,0))/target_size;vec2 origin=floor(gl_FragCoord.xy)*scale;float sum=0.;
for(int y=0;y<SY;y++)for(int x=0;x<SX;x++)sum+=light(ivec2(origin+(vec2(x,y)+.5)*scale/vec2(float(SX),float(SY))));
float v=sum/float(SX*SY);color=vec4(v,v,v,1.);}
'''

def timeline(times):
 starts=[];previous=None
 for base in range(0,len(times),1<<20):
  block=np.asarray(times[base:base+(1<<20)])
  if (previous is not None and block[0]<previous) or np.any(block[1:]<block[:-1]):raise ValueError('Unsorted event timeline')
  if previous is None or block[0]!=previous:starts.append(np.array([base],np.uint32))
  starts.append((np.flatnonzero(block[1:]!=block[:-1])+base+1).astype(np.uint32));previous=block[-1]
 if not starts:return np.empty(0,np.uint32),np.empty(0,np.uint32)
 starts=np.concatenate(starts);return np.array(times[starts],np.uint32),np.concatenate((starts[1:],np.array([len(times)],np.uint32)))

class BitsetRenderer:
 supports_disabled=True
 def __init__(self,r):
  from .gpu import program
  self.r=r;self.program=program(GATHER);r.locations[self.program]={};self.key=None;self.textures=[];self.preview_programs={}
 def prepare(self,frame):
  from .gpu import texture
  from .native_bits import Bits
  key=(str(frame.scene.events.path),frame.scene.key)
  if key==self.key:return
  self.release();e=frame.scene.events;h,w=frame.scene.z.shape;n=h*w
  self.cells=np.array(np.load(e.path/'sorted-cells.npy',mmap_mode='r'),np.uint32)
  if int(self.cells.max(initial=0))>=n:raise ValueError('Event cache references an invalid terrain cell')
  times=np.load(e.path/'sorted-times.npy',mmap_mode='r');self.times,self.ends=timeline(times);del times
  nwords=(n+31)//32;self.mask_width=1024;self.mask_height=(nwords+1023)//1024
  self.words=np.zeros(self.mask_height*1024,np.uint32);bytes_=np.packbits(np.asarray(e.initial),bitorder='little');self.words.view(np.uint8)[:len(bytes_)]=bytes_
  self.initial_words=self.words.copy();self.final_words=np.zeros_like(self.words)
  final=np.asarray(e.initial)^((np.diff(e.offsets)&1).astype(np.uint8))
  bytes_=np.packbits(final,bitorder='little');self.final_words.view(np.uint8)[:len(bytes_)]=bytes_
  self.native=Bits(len(self.words));normal_words=np.empty(n,np.uint32)
  for base in range(0,n,65536):
   normal=np.asarray(frame.scene.normals).reshape(-1,3)[base:base+65536];encoded=np.rint((normal*.5+.5)*1023).astype(np.uint32)
   normal_words[base:base+len(normal)]=encoded[:,0]|(encoded[:,1]<<10)|(encoded[:,2]<<20)|np.uint32(0xc0000000)
  uv=frame.scene.uv;oh,ow=uv.shape[:2];projected=np.empty((oh,ow),np.uint32)
  for row in range(0,oh,64):
   block=uv[row:row+64];valid=block[:,:,0]!=65535
   ix=np.minimum(block[:,:,1],h-1).astype(np.int64)*w+np.minimum(block[:,:,0],w-1)
   projected[row:row+64]=normal_words[ix];projected[row:row+64][~valid]=0
  self.textures=[texture(10,projected,GL_RGB10_A2,GL_RGBA,GL_UNSIGNED_INT_2_10_10_10_REV),texture(11,self.words.reshape(self.mask_height,1024),GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT)]
  self.position=0;self.key=key
 def uniforms(self,p,physical):
  r=self.r
  for unit,tex in zip((10,11),self.textures):glActiveTexture(GL_TEXTURE0+unit);glBindTexture(GL_TEXTURE_2D,tex)
  for name,value in [('normals',10),('lookup',4),('bits',11),('grid_width',r.gw),('mask_width',self.mask_width),('flip_y',physical),('enabled',self.frame.enabled)]:r.integer(p,name,value)
  r.real(p,'ambient',self.frame.ambient);glUniform3f(r.location(p,'sun'),*self.frame.sun)
  for name,value in [('light_gain',self.frame.light_gain),('light_floor',self.frame.light_floor),('light_gamma',self.frame.light_gamma)]:r.real(p,name,value)
 def draw(self,frame,target=0,physical=True):
  self.prepare(frame);self.frame=frame;e=frame.scene.events
  seconds=(datetime.fromisoformat(frame.time).astimezone(timezone.utc)-datetime.fromisoformat(e.meta['start'])).total_seconds()
  if frame.sun[2]>0:
   if not 0<=seconds<86400:raise ValueError('Event cache date mismatch')
   group=int(np.searchsorted(self.times,np.uint32(round(seconds*1000)),side='right'));position=int(self.ends[group-1]) if group else 0
   begin=min(position,self.position);count=abs(position-self.position)
   dirty=count!=0
   if position<count:
    np.copyto(self.words,self.initial_words);begin=0;count=position
   if len(self.cells)-position<count:
    np.copyto(self.words,self.final_words);begin=position;count=len(self.cells)-position
   if count:
    self.native.flip(self.words,self.cells[begin:begin+count])
   if dirty:
    glActiveTexture(GL_TEXTURE11);glBindTexture(GL_TEXTURE_2D,self.textures[1])
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,self.mask_width,self.mask_height,GL_RED_INTEGER,GL_UNSIGNED_INT,self.words.ctypes.data_as(C.c_void_p))
   self.position=position
  r=self.r;p=self.program;glUseProgram(p);self.uniforms(p,physical);glBindFramebuffer(GL_FRAMEBUFFER,target);glViewport(0,0,r.ow,r.oh);glDrawArrays(GL_TRIANGLES,0,3)
 def preview(self,target,width,height):
  from .gpu import program
  r=self.r;key=(max(1,min(8,int(np.ceil(r.ow/width)))),max(1,min(8,int(np.ceil(r.oh/height)))))
  if key not in self.preview_programs:
   p=program(PREVIEW.replace('SX',str(key[0])).replace('SY',str(key[1])));self.preview_programs[key]=p;r.locations[p]={}
  p=self.preview_programs[key];glUseProgram(p);self.uniforms(p,False);glUniform2f(r.location(p,'target_size'),width,height)
  glBindFramebuffer(GL_FRAMEBUFFER,target);glViewport(0,0,width,height);glDrawArrays(GL_TRIANGLES,0,3)
 def release(self):
  if self.textures:glDeleteTextures(len(self.textures),self.textures)
  self.textures=[];self.key=None;self.cells=self.words=self.native=self.initial_words=self.final_words=self.times=self.ends=None
 def close(self):
  self.release();glDeleteProgram(self.program)
  for p in self.preview_programs.values():glDeleteProgram(p)
