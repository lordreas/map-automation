"""Projected river fields, sparse GPU tiles, and a reusable static background."""
import ctypes as C
import numpy as np
from OpenGL.GLES3 import *
from .gpu import program,texture,framebuffer

VERTEX='''#version 300 es
precision highp float;
layout(location=0) in vec4 tile;
uniform vec2 output_size;uniform float thickness;uniform float progress;uniform float chromatic;
const vec2 corner[6]=vec2[6](vec2(0,0),vec2(1,0),vec2(1,1),vec2(0,0),vec2(1,1),vec2(0,1));
void main(){if(tile.z>thickness*.5+1.+abs(chromatic)*1.5||tile.w>progress){gl_Position=vec4(2,2,0,1);return;}
 vec2 pixel=tile.xy+corner[gl_VertexID]*16.;gl_Position=vec4(pixel.x/output_size.x*2.-1.,1.-pixel.y/output_size.y*2.,0,1);}
'''
FRAGMENT='''#version 300 es
precision highp float;precision highp int;
uniform sampler2D field;uniform sampler2D background;
uniform float thickness;uniform float progress;uniform float chromatic;uniform vec3 ink;uniform int empty_background;
out vec4 color;
float coverage(vec3 river){float width=mix(1.,thickness,river.g);
 return (1.-smoothstep(max(0.,width*.5-.65),width*.5+.65,river.r))*(1.-smoothstep(progress-.005,progress,river.b));}
void main(){ivec2 p=ivec2(gl_FragCoord.xy);ivec2 loc=ivec2(p.x,textureSize(field,0).y-1-p.y);
 float a=coverage(texelFetch(field,loc,0).rgb);float g=a;
 if(chromatic!=0.){vec2 size=vec2(textureSize(field,0));vec2 q=gl_FragCoord.xy+(gl_FragCoord.xy-size*.5)*chromatic/length(size*.5);
  g=coverage(texture(field,vec2(q.x,size.y-q.y)/size).rgb);}
 if(max(a,g)<.001)discard;
 vec3 base=empty_background!=0?vec3(0.):texelFetch(background,p,0).rgb;color=vec4(mix(base,ink,vec3(a,g,a)),1.);
}
'''


class Rivers:
 def __init__(self,parent):
  self.parent=parent;self.program=program(FRAGMENT,VERTEX);self.locations={};self.vao=glGenVertexArrays(1);self.buffer=glGenBuffers(1)
  self.field=None;self.tex=None;self.base_tex=None;self.base_fb=None;self.size=None;self.base_key=None;self.count=0

 def loc(self,name):
  if name not in self.locations:self.locations[name]=glGetUniformLocation(self.program,name)
  return self.locations[name]

 def prepare(self,a):
  if self.field is a:return
  if self.tex is not None:glDeleteTextures(1,[self.tex])
  self.tex=texture(9,a,GL_RGB16F,GL_RGB,GL_HALF_FLOAT);self.field=a
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
  h,w=a.shape[:2];nh,nw=(h+15)//16,(w+15)//16
  padded=np.pad(a,((0,nh*16-h),(0,nw*16-w),(0,0)),mode='edge').reshape(nh,16,nw,16,3)
  bounds=padded[:,:,:,:,0].min(axis=(1,3));phase=padded[:,:,:,:,2].min(axis=(1,3))
  def halo(value):
   x=np.pad(value,2,mode='edge');return np.minimum.reduce([x[y:y+nh,z:z+nw] for y in range(5) for z in range(5)])
  y,x=np.mgrid[:nh,:nw];tiles=np.ascontiguousarray(np.stack((x*16,y*16,bounds,halo(phase)),axis=2).reshape(-1,4),np.float32)
  self.count=len(tiles);glBindVertexArray(self.vao);glBindBuffer(GL_ARRAY_BUFFER,self.buffer);glBufferData(GL_ARRAY_BUFFER,tiles.nbytes,tiles,GL_STATIC_DRAW)
  glEnableVertexAttribArray(0);glVertexAttribPointer(0,4,GL_FLOAT,False,16,C.c_void_p(0));glVertexAttribDivisor(0,1)

 def draw(self,state,target,shading,compensation):
  a=state['arrays']['screen_rivers'];self.prepare(a);h,w=a.shape[:2];parent=self.parent;owner=parent.owner
  if self.size!=(w,h):
   if self.base_fb is not None:glDeleteFramebuffers(1,[self.base_fb]);glDeleteTextures(1,[self.base_tex])
   self.base_tex=texture(13,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,w,h);self.base_fb=framebuffer(self.base_tex);self.size=(w,h);self.base_key=None
  cfg=state['config'];only_rivers=not any(v['enabled'] for n,v in cfg.items() if n!='rivers')
  key=(state['key'],compensation) if only_rivers else (id(state),compensation,shading)
  if key!=self.base_key:
   if only_rivers:
    glBindFramebuffer(GL_FRAMEBUFFER,self.base_fb);glViewport(0,0,w,h);glClearColor(0,0,0,1);glClear(GL_COLOR_BUFFER_BIT)
   else:
    base=dict(state);base['arrays']={n:v for n,v in state['arrays'].items() if n not in ('rivers','screen_rivers')};base['config']={n:(dict(v,enabled=False) if n=='rivers' else v) for n,v in cfg.items()}
    destination=owner.chromatic_target(w,h) if compensation and owner else self.base_fb
    glBindVertexArray(owner.vao if owner else self.vao)
    parent.draw(base,destination,shading=shading)
    if compensation and owner:owner.compensate(compensation,self.base_fb)
   self.base_key=key
  # Reuse the complete static map instead of recomputing all geographic layers
  # and lighting for each reveal frame. No CPU image copy or framebuffer readback.
  if only_rivers:
   glBindFramebuffer(GL_FRAMEBUFFER,target);glClearColor(0,0,0,1);glClear(GL_COLOR_BUFFER_BIT)
  else:
   glBindFramebuffer(GL_READ_FRAMEBUFFER,self.base_fb);glBindFramebuffer(GL_DRAW_FRAMEBUFFER,target)
   glBlitFramebuffer(0,0,w,h,0,0,w,h,GL_COLOR_BUFFER_BIT,GL_NEAREST)
  glBindFramebuffer(GL_FRAMEBUFFER,target);glViewport(0,0,w,h);glBindVertexArray(self.vao);glUseProgram(self.program)
  for unit,name,tex in [(9,'field',self.tex),(13,'background',self.base_tex)]:
   glActiveTexture(GL_TEXTURE0+unit);glBindTexture(GL_TEXTURE_2D,tex);glUniform1i(self.loc(name),unit)
  glUniform2f(self.loc('output_size'),w,h);glUniform1f(self.loc('thickness'),cfg['rivers']['river_thickness'])
  clock=state.get('river_clock');glUniform1f(self.loc('progress'),(clock.sample() if clock else 1)*1.005);glUniform1f(self.loc('chromatic'),compensation)
  glUniform1i(self.loc('empty_background'),only_rivers)
  ink=cfg['rivers']['color'];glUniform3f(self.loc('ink'),*[int(ink[i:i+2],16)/255 for i in (1,3,5)])
  glDrawArraysInstanced(GL_TRIANGLES,0,6,self.count)

 def close(self):
  if self.tex is not None:glDeleteTextures(1,[self.tex])
  if self.base_fb is not None:glDeleteFramebuffers(1,[self.base_fb]);glDeleteTextures(1,[self.base_tex])
  glDeleteBuffers(1,[self.buffer]);glDeleteVertexArrays(1,[self.vao]);glDeleteProgram(self.program)
