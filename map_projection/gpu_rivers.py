"""Native pixel events replayed by GLES compute; only the moving front is updated."""
import numpy as np
from OpenGL.GLES3 import *
from OpenGL.GLES3.VERSION.GLES3_3_1 import glDispatchCompute,glMemoryBarrier,glBindImageTexture
from .gpu import compute,program,texture,framebuffer

UPDATE='''#version 310 es
precision highp float;precision highp int;
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Events{uvec2 events[];};
layout(r32ui,binding=0) coherent uniform highp uimage2D image;
uniform sampler2D background;uniform vec3 ink;
uniform float progress;uniform uint begin;uniform uint count;uniform int channel;uniform int empty_background;
void main(){uint j=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*65535u*64u;if(j>=count)return;
 uvec2 event=events[begin+j];ivec2 pixel=ivec2(int(event.x&4095u),int(event.x>>12));
 float phase=float(event.y&65535u)/65535.;float coverage=float((event.y>>16)&255u)/255.;
 float alpha=coverage*(1.-smoothstep(progress-.005,progress,phase));
 vec3 base=empty_background!=0?vec3(0.):texelFetch(background,pixel,0).rgb;
 uint value=packUnorm4x8(vec4(mix(base,ink,alpha),1.));uint previous=imageLoad(image,pixel).r;
 uint mask=channel==0?0x00ff00ffu:0x0000ff00u;
 imageStore(image,pixel,uvec4((previous&~mask)|(value&mask)|0xff000000u,0,0,0));}
'''
INITIALIZE='''#version 310 es
precision highp float;precision highp int;
layout(local_size_x=16,local_size_y=16) in;
layout(r32ui,binding=0) writeonly uniform highp uimage2D image;
uniform sampler2D background;uniform int empty_background;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy);if(any(greaterThanEqual(p,imageSize(image))))return;
 uint value=empty_background!=0?0xff000000u:packUnorm4x8(texelFetch(background,p,0));imageStore(image,p,uvec4(value,0,0,0));}
'''
PRESENT='''#version 310 es
precision highp float;precision highp int;
uniform highp usampler2D image;out vec4 color;
void main(){color=unpackUnorm4x8(texelFetch(image,ivec2(gl_FragCoord.xy),0).r);}
'''
PREVIEW='''#version 310 es
precision highp float;precision highp int;
uniform highp usampler2D image;uniform vec2 target_size;out vec4 color;
void main(){ivec2 size=textureSize(image,0);vec2 scale=vec2(size)/target_size;vec2 origin=floor(gl_FragCoord.xy)*scale;vec4 sum=vec4(0.);
 for(int y=0;y<SY;y++)for(int x=0;x<SX;x++){
  ivec2 p=clamp(ivec2(origin+(vec2(x,y)+.5)*scale/vec2(float(SX),float(SY))),ivec2(0),size-1);p.y=size.y-1-p.y;
  sum+=unpackUnorm4x8(texelFetch(image,p,0).r);}
 color=sum/float(SX*SY);}
'''


class Rivers:
 def __init__(self,parent):
  self.parent=parent;self.program=compute(UPDATE);self.initialize=compute(INITIALIZE);self.present=program(PRESENT);self.locations={};self.previews={}
  self.vao=glGenVertexArrays(1);self.buffers=[];self.events=None;self.field=None
  self.base_tex=self.base_fb=self.image_tex=self.image_fb=None
  self.size=None;self.base_key=None;self.base_state=None;self.style_key=None;self.progress=None
  self.presentation_framebuffer=None;self.timings={}

 def loc(self,name,p=None):
  p=p or self.program;key=(p,name)
  if key not in self.locations:self.locations[key]=glGetUniformLocation(p,name)
  return self.locations[key]

 def prepare(self,state,thickness,compensation):
  field=state['arrays']['screen_rivers'];events=state.get('river_events')
  if self.events and self.field is field and self.events['thickness']==thickness and self.events['compensation']==compensation:return
  if not events or events['field'] is not field or events['thickness']!=thickness or events['compensation']!=compensation:
   from .river_events import compile_events
   events=compile_events(field,thickness,compensation)
  if self.buffers:glDeleteBuffers(len(self.buffers),self.buffers)
  self.buffers=[]
  for array,phases in events['channels']:
   buffer=glGenBuffers(1);glBindBuffer(0x90D2,buffer);glBufferData(0x90D2,max(8,array.nbytes),array if array.nbytes else None,GL_STATIC_DRAW);self.buffers.append(buffer)
  self.events=events;self.field=field;self.progress=None;self.style_key=None

 def allocate(self,w,h):
  if self.size==(w,h):return
  self.release_images()
  self.base_tex=texture(13,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,w,h);self.base_fb=framebuffer(self.base_tex)
  # ES permits read/write images for R32UI, but requires RGBA8 images to be
  # readonly or writeonly. Pack RGBA8 losslessly into one integer per pixel.
  self.image_tex=texture(14,None,GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT,w,h,immutable=True);self.image_fb=None
  self.size=(w,h);self.base_key=None;self.progress=None

 def draw(self,state,target,shading,compensation):
  cfg=state['config'];thickness=cfg['rivers']['river_thickness'];self.prepare(state,thickness,compensation)
  h,w=self.field.shape[:2];self.allocate(w,h);parent=self.parent;owner=parent.owner
  only_rivers=not any(v['enabled'] for n,v in cfg.items() if n!='rivers')
  active=tuple((n,tuple(sorted(v.items()))) for n,v in cfg.items() if n!='rivers' and v['enabled'])
  assets=tuple((n,id(a)) for n,a in state['arrays'].items() if n not in ('rivers','screen_rivers'))
  lighting=id(parent.lighting_frame) if shading and any(cfg[n]['enabled'] for n in ('white','satellite','semantic')) else None
  key=(state['key'],compensation,active,assets,lighting)
  reset=key!=self.base_key
  if reset:
   if not only_rivers:
    base=dict(state);base['arrays']={n:v for n,v in state['arrays'].items() if n not in ('rivers','screen_rivers')};base['config']={n:(dict(v,enabled=False) if n=='rivers' else v) for n,v in cfg.items()}
    destination=owner.chromatic_target(w,h) if compensation and owner else self.base_fb
    glBindVertexArray(owner.vao if owner else self.vao);parent.draw(base,destination,shading=shading)
    if compensation and owner:owner.compensate(compensation,self.base_fb)
   self.base_key=key;self.base_state=state
  clock=state.get('river_clock');progress=(clock.sample() if clock else 1.)*1.005
  style=(cfg['rivers']['color'],thickness,compensation)
  reset=reset or style!=self.style_key or self.progress is None or progress<self.progress
  if reset:
   # Initialize through image stores, not a framebuffer fast clear. PowerVR
   # otherwise retains stale fast-clear metadata for the first partial updates.
   glBindFramebuffer(GL_FRAMEBUFFER,0);glUseProgram(self.initialize)
   glActiveTexture(GL_TEXTURE13);glBindTexture(GL_TEXTURE_2D,self.base_tex);glUniform1i(self.loc('background',self.initialize),13)
   glUniform1i(self.loc('empty_background',self.initialize),only_rivers)
   glBindImageTexture(0,self.image_tex,0,False,0,GL_WRITE_ONLY,GL_R32UI)
   glDispatchCompute((w+15)//16,(h+15)//16,1);glMemoryBarrier(0xffffffff)
  points=0
  if reset or progress!=self.progress:
   upper=np.uint16(min(65535,max(0,int(np.floor(progress*65535)))))
   lower=np.uint16(min(65535,max(0,int(np.floor(((self.progress or 0)-.005)*65535)))))
   glUseProgram(self.program);glActiveTexture(GL_TEXTURE13);glBindTexture(GL_TEXTURE_2D,self.base_tex);glUniform1i(self.loc('background'),13)
   glUniform1f(self.loc('progress'),progress);glUniform1i(self.loc('empty_background'),only_rivers)
   ink=style[0];glUniform3f(self.loc('ink'),*[int(ink[i:i+2],16)/255 for i in (1,3,5)])
   glMemoryBarrier(0x20)  # Previous presentation/preview reads precede these writes.
   glBindImageTexture(0,self.image_tex,0,False,0,GL_READ_WRITE,GL_R32UI)
   for channel,buffer in enumerate(self.buffers):
    phases=self.events['channels'][channel][1]
    begin=0 if reset else int(np.searchsorted(phases,lower,side='left'))
    end=int(np.searchsorted(phases,upper,side='right'))
    if end<=begin:continue
    glUniform1ui(self.loc('begin'),begin);glUniform1ui(self.loc('count'),end-begin);glUniform1i(self.loc('channel'),channel)
    glBindBufferBase(0x90D2,0,buffer);groups=(end-begin+63)//64;glDispatchCompute(min(groups,65535),(groups+65534)//65535,1)
    # The two channels can touch the same pixel. Serialize their image updates
    # so preserving the other channel cannot lose a preceding write.
    glMemoryBarrier(0x20);points+=end-begin
   glMemoryBarrier(0xffffffff);self.progress=progress;self.style_key=style
  glBindFramebuffer(GL_FRAMEBUFFER,target);glViewport(0,0,w,h);glBindVertexArray(self.vao);glUseProgram(self.present)
  glActiveTexture(GL_TEXTURE14);glBindTexture(GL_TEXTURE_2D,self.image_tex);glUniform1i(self.loc('image',self.present),14);glDrawArrays(GL_TRIANGLES,0,3)
  self.timings={'river_pixels_updated':points,'river_pixel_events':sum(len(a) for a,_ in self.events['channels'])}

 def preview(self,target,width,height):
  w,h=self.size;key=(max(1,min(8,int(np.ceil(w/width)))),max(1,min(8,int(np.ceil(h/height)))))
  if key not in self.previews:self.previews[key]=program(PREVIEW.replace('SX',str(key[0])).replace('SY',str(key[1])))
  p=self.previews[key];glBindFramebuffer(GL_FRAMEBUFFER,target);glViewport(0,0,width,height);glBindVertexArray(self.vao);glUseProgram(p)
  glActiveTexture(GL_TEXTURE14);glBindTexture(GL_TEXTURE_2D,self.image_tex);glUniform1i(self.loc('image',p),14);glUniform2f(self.loc('target_size',p),width,height);glDrawArrays(GL_TRIANGLES,0,3)

 def release_images(self):
  for tex,fb in ((self.base_tex,self.base_fb),(self.image_tex,self.image_fb)):
   if fb is not None:glDeleteFramebuffers(1,[fb])
   if tex is not None:glDeleteTextures(1,[tex])
  self.base_tex=self.base_fb=self.image_tex=self.image_fb=None

 def close(self):
  self.release_images()
  if self.buffers:glDeleteBuffers(len(self.buffers),self.buffers)
  glDeleteVertexArrays(1,[self.vao])
  for p in [self.program,self.initialize,self.present,*self.previews.values()]:glDeleteProgram(p)
