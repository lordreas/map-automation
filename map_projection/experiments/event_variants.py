"""Opt-in replay experiments; never enabled by the production application."""
from datetime import datetime,timezone
import ctypes as C
import numpy as np
from OpenGL.GLES3 import *
from OpenGL.GLES3.VERSION.GLES3_3_1 import glDispatchCompute,glMemoryBarrier,glBindImageTexture
from map_projection import gpu_events

ATOMIC='''#version 320 es
precision highp float;precision highp int;
layout(local_size_x=64) in;
layout(r32ui,binding=1) coherent uniform highp uimage2D mask;
layout(std430,binding=3) readonly buffer Changes{uint cells[];};
uniform uint count;uniform uint begin;uniform int grid_width;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=count)return;uint c=cells[i+begin];
imageAtomicXor(mask,ivec2(int(c)%grid_width,int(c)/grid_width),0xc0000000u);}
'''

def install(half=False):
 base=gpu_events.EventRenderer
 class AtomicEvents(base):
  def __init__(self,r):
   super().__init__(r)
   from map_projection.gpu import compute,program
   glDeleteProgram(self.update);self.update=compute(ATOMIC);r.locations[self.update]={};self.uploaded=None
   if half:
    glDeleteProgram(self.program);self.program=program(gpu_events.GATHER.replace('precision highp float;','precision mediump float;'));r.locations[self.program]={}
  def prepare(self,frame):
   super().prepare(frame)
   if self.uploaded==self.key:return
   glBindBuffer(0x90D2,self.buffer)
   glBufferData(0x90D2,self.cells.nbytes,self.cells.ctypes.data_as(C.c_void_p),GL_STATIC_DRAW)
   self.uploaded=self.key
  def draw(self,frame,target=0,physical=True):
   self.prepare(frame);r=self.r;e=frame.scene.events
   seconds=(datetime.fromisoformat(frame.time).astimezone(timezone.utc)-datetime.fromisoformat(e.meta['start'])).total_seconds()
   if not 0<=seconds<86400:raise ValueError('Event cache date mismatch')
   pos=int(np.searchsorted(self.times,np.uint32(round(seconds*1000)),side='right'));begin=min(pos,self.position);count=abs(pos-self.position)
   glActiveTexture(GL_TEXTURE10);glBindTexture(GL_TEXTURE_2D,self.textures[0])
   if count:
    glMemoryBarrier(0x20);p=self.update;glUseProgram(p);glBindBufferBase(0x90D2,3,self.buffer)
    glBindImageTexture(1,self.textures[0],0,False,0,GL_READ_WRITE,GL_R32UI);r.integer(p,'grid_width',r.gw)
    for first in range(0,count,65535*64):
     size=min(65535*64,count-first);glUniform1ui(r.location(p,'begin'),begin+first);glUniform1ui(r.location(p,'count'),size);glDispatchCompute((size+63)//64,1,1)
    glMemoryBarrier(0x28)
   self.position=pos;self.frame=frame;self.render(target,physical)
  def release(self):
   super().release();self.uploaded=None
 gpu_events.EventRenderer=AtomicEvents

def install_bulk(threshold=500000):
 """Use a contiguous texture upload for dense changes; scatter only sparse ones."""
 base=gpu_events.EventRenderer
 class BulkEvents(base):
  def draw(self,frame,target=0,physical=True):
   self.prepare(frame);e=frame.scene.events
   seconds=(datetime.fromisoformat(frame.time).astimezone(timezone.utc)-datetime.fromisoformat(e.meta['start'])).total_seconds()
   if not 0<=seconds<86400:raise ValueError('Event cache date mismatch')
   pos=int(np.searchsorted(self.times,np.uint32(round(seconds*1000)),side='right'));begin=min(pos,self.position);count=abs(pos-self.position)
   if count<threshold:return super().draw(frame,target,physical)
   selected=np.asarray(self.cells[begin:begin+count]);np.bitwise_xor.at(self.words,selected,np.uint32(0xc0000000))
   glMemoryBarrier(0x100);glActiveTexture(GL_TEXTURE10);glBindTexture(GL_TEXTURE_2D,self.textures[0])
   glTexSubImage2D(GL_TEXTURE_2D,0,0,0,self.r.gw,self.r.gh,GL_RED_INTEGER,GL_UNSIGNED_INT,self.words.ctypes.data_as(C.c_void_p))
   self.position=pos;self.frame=frame;self.render(target,physical)
 gpu_events.EventRenderer=BulkEvents
