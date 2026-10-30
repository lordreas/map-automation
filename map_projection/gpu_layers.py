"""Geographic compositing with conservative sparse contour tiles."""
import ctypes as C
import numpy as np
from OpenGL.GLES3 import *
from .gpu import program,texture,framebuffer

FRAGMENT='''#version 300 es
precision highp float;precision highp int;
uniform sampler2D lighting;uniform highp usampler2D coordinates;
uniform sampler2D elevation;uniform sampler2D rivers;uniform sampler2D peaks;
uniform sampler2D satellite;uniform sampler2D streets;uniform sampler2D semantic;
uniform sampler2D divides;uniform int show_divides;uniform float divide_width;
uniform vec3 divide_ink;uniform float river_thickness;uniform float river_progress;
uniform vec4 opacity;uniform ivec3 features;uniform vec3 widths;
uniform float contour_interval;uniform float contour_level;uniform int single_contour;
uniform vec3 contour_ink;uniform vec3 river_ink;uniform vec3 peak_ink;
uniform vec3 road_ink;uniform float road_width;uniform int shading;
uniform int projected_height;
uniform int contour_fill;uniform float satellite_brightness;
out vec4 color;
float height_at(vec2 uv){
 vec2 p=uv*vec2(textureSize(elevation,0))-.5;ivec2 a=ivec2(floor(p));vec2 t=fract(p);ivec2 hi=textureSize(elevation,0)-1;
 float v00=texelFetch(elevation,clamp(a,ivec2(0),hi),0).r;
 float v10=texelFetch(elevation,clamp(a+ivec2(1,0),ivec2(0),hi),0).r;
 float v01=texelFetch(elevation,clamp(a+ivec2(0,1),ivec2(0),hi),0).r;
 float v11=texelFetch(elevation,clamp(a+ivec2(1,1),ivec2(0),hi),0).r;
 return mix(mix(v00,v10,t.x),mix(v01,v11,t.x),t.y);
}
float mark(sampler2D field,vec2 uv,float width){
 vec2 size=vec2(textureSize(field,0));vec2 dx=dFdx(uv)*size,dy=dFdy(uv)*size;
 float scale=max(sqrt((dot(dx,dx)+dot(dy,dy))*.5),.02);
 float distance=texture(field,uv).r/scale;
 return 1.-smoothstep(max(0.,width*.5-.65),width*.5+.65,distance);
}
void main(){
 ivec2 pixel=ivec2(gl_FragCoord.xy);ivec2 loc=ivec2(pixel.x,textureSize(coordinates,0).y-1-pixel.y);
 uvec2 address=texelFetch(coordinates,loc,0).rg;
 // Derivatives are evaluated before the silhouette branch to keep edge AA valid.
 vec2 uv=vec2(address)/65534.;
 float z=features.x!=0?(projected_height!=0?texelFetch(elevation,loc,0).r:height_at(uv)):0.;
 float derivative=max(fwidth(z),.01);
 vec3 river=features.y!=0?texture(rivers,uv).rgb:vec3(0.);
 float river_alpha=features.y!=0?mark(rivers,uv,mix(1.,river_thickness,river.g))*(1.-smoothstep(river_progress-.005,river_progress,river.b)):0.;
 float peak_alpha=features.z!=0?mark(peaks,uv,widths.z):0.;
 vec2 road=opacity.z>0.?texture(streets,uv).rg:vec2(0.);
 float road_alpha=opacity.z>0.?mark(streets,uv,mix(1.,road_width,road.g))*opacity.z:0.;
 float divide_alpha=show_divides!=0?mark(divides,uv,divide_width):0.;
 if(address.x==65535u){color=vec4(0.,0.,0.,1.);return;}
 vec3 base=vec3(opacity.x);
 if(opacity.y>0.)base=mix(base,clamp(texture(satellite,uv).rgb*satellite_brightness,0.,1.),opacity.y);
 if(opacity.w>0.)base=mix(base,texture(semantic,uv).rgb,opacity.w);
 if(shading!=0)base*=texelFetch(lighting,pixel,0).r;
 base=mix(base,road_ink,road_alpha);
 if(features.x!=0){
  float d=single_contour!=0?abs(z-contour_level):abs(mod(z+contour_interval*.5,contour_interval)-contour_interval*.5);
  float alpha=(1.-smoothstep(max(0.,widths.x*.5-.6),widths.x*.5+.6,d/derivative))*step(.05,abs(z));
  if(single_contour!=0&&contour_fill!=0){alpha=smoothstep(contour_level-derivative*.6,contour_level+derivative*.6,z);if(contour_fill==2)alpha=1.-alpha;}
  base=mix(base,contour_ink,alpha);
 }
 base=mix(base,river_ink,river_alpha);base=mix(base,divide_ink,divide_alpha);base=mix(base,peak_ink,peak_alpha);color=vec4(base,1.);
}
'''

CONTOURS='''#version 300 es
precision highp float;precision highp int;
uniform sampler2D heights;uniform float spacing;uniform float level;uniform float width;
uniform int single_line;uniform int fill_mode;uniform vec3 ink;uniform float chromatic;out vec4 color;
float coverage(float z,float slope){
 if(single_line!=0&&fill_mode!=0){float alpha=smoothstep(level-max(slope,.01)*.6,level+max(slope,.01)*.6,z);return (fill_mode==2?1.-alpha:alpha)*step(-500.,z);}
 float d=single_line!=0?abs(z-level):abs(mod(z+spacing*.5,spacing)-spacing*.5);
 return (1.-smoothstep(max(0.,width*.5-.6),width*.5+.6,d/max(slope,.01)))*step(.05,abs(z))*step(-500.,z);
}
void main(){ivec2 p=ivec2(gl_FragCoord.xy);p.y=textureSize(heights,0).y-1-p.y;
 float z=texelFetch(heights,p,0).r;float a=coverage(z,fwidth(z));float g=a;
 if(chromatic!=0.){
  vec2 size=vec2(textureSize(heights,0));vec2 q=gl_FragCoord.xy+(gl_FragCoord.xy-size*.5)*chromatic/length(size*.5);
  float green_height=texture(heights,vec2(q.x,size.y-q.y)/size).r;
  g=coverage(green_height,fwidth(green_height));
 }
 color=vec4(ink.r*a,ink.g*g,ink.b*a,1.);}
'''

CONTOUR_VERTEX='''#version 300 es
precision highp float;
layout(location=0) in vec4 tile;
uniform vec2 output_size;uniform float spacing;uniform float level;uniform int single_line;uniform int fill_mode;
const vec2 corner[6]=vec2[6](vec2(0,0),vec2(1,0),vec2(1,1),vec2(0,0),vec2(1,1),vec2(0,1));
void main(){
 bool valid=tile.w>=-500.&&(tile.z<-.05||tile.w>.05);
 bool intersects=single_line!=0?(level>=tile.z&&level<=tile.w):(ceil(tile.z/spacing)*spacing<=tile.w);
 if(single_line!=0&&fill_mode!=0){valid=tile.w>=-500.;intersects=fill_mode==1?tile.w>=level:tile.z<=level;}
 if(!valid||!intersects){gl_Position=vec4(2.,2.,0.,1.);return;}
 vec2 pixel=tile.xy+corner[gl_VertexID]*16.;
 gl_Position=vec4(pixel.x/output_size.x*2.-1.,1.-pixel.y/output_size.y*2.,0.,1.);
}
'''


class LayerRenderer:
 def __init__(self,owner=None):
  self.owner=owner;self.river_renderer=None;self.lighting_frame=None
  self.program=program(FRAGMENT);self.locations={};self.textures={};self.arrays={};self.uv_key=None;self.size=None;self.light_tex=None;self.light_fb=None
  self.contour_program=program(CONTOURS,CONTOUR_VERTEX);self.contour_array=None;self.contour_tex=None
  self.contour_vao=glGenVertexArrays(1);self.contour_buffer=glGenBuffers(1);self.contour_count=0
  # GLES requires every active sampler type to have a distinct bound unit.
  self.fallback=texture(15,np.zeros((1,1),np.float16),GL_R16F,GL_RED,GL_HALF_FLOAT)

 def location(self,name):
  if name not in self.locations:self.locations[name]=glGetUniformLocation(self.program,name)
  return self.locations[name]

 def target(self,width,height):
  if self.size!=(width,height):
   if self.light_tex is not None:glDeleteFramebuffers(1,[self.light_fb]);glDeleteTextures(1,[self.light_tex])
   self.light_tex=texture(14,None,GL_R8,GL_RED,GL_UNSIGNED_BYTE,width,height);self.light_fb=framebuffer(self.light_tex);self.size=(width,height)
  return self.light_fb

 def draw(self,state,target,shading=True,compensation=0.):
  cfg=state['config'];arrays=state['arrays']
  if cfg['rivers']['enabled'] and 'screen_rivers' in arrays:
   if self.river_renderer is None:
    from .gpu_rivers import Rivers
    self.river_renderer=Rivers(self)
   return self.river_renderer.draw(state,target,shading,compensation)
  if 'screen_elevation' in arrays and cfg['contours']['enabled'] and not any(v['enabled'] for n,v in cfg.items() if n!='contours'):
   return self.draw_contours(state,target,compensation)
  glUseProgram(self.program)
  if state['key']!=self.uv_key:
   if 'coordinates' in self.textures:glDeleteTextures(1,[self.textures.pop('coordinates')])
   self.textures['coordinates']=texture(1,state['uv'],GL_RG16UI,GL_RG_INTEGER,GL_UNSIGNED_SHORT);self.uv_key=state['key']
  names=('elevation','rivers','peaks','satellite','streets','semantic','divides')
  for unit,name in enumerate(names,2):
   a=arrays.get('screen_elevation',arrays.get(name)) if name=='elevation' else arrays.get(name)
   # Keep these two GPU images resident while disabled, making toggles cheap.
   if a is None and name in ('satellite','semantic'):continue
   if self.arrays.get(name) is not a:
    if name in self.textures:glDeleteTextures(1,[self.textures.pop(name)])
    if a is not None:
     if a.ndim==3 and a.dtype==np.float16:internal,fmt,typ=(GL_RG16F if a.shape[2]==2 else GL_RGB16F),(GL_RG if a.shape[2]==2 else GL_RGB),GL_HALF_FLOAT
     elif a.ndim==3:internal,fmt,typ=GL_RGB8,GL_RGB,GL_UNSIGNED_BYTE
     elif a.dtype==np.float16:internal,fmt,typ=GL_R16F,GL_RED,GL_HALF_FLOAT
     else:internal,fmt,typ=GL_R32F,GL_RED,GL_FLOAT
     self.textures[name]=texture(unit,a,internal,fmt,typ)
     if name!='elevation':
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
    self.arrays[name]=a
  for unit,name in enumerate(('lighting','coordinates')+names):
   tex=self.light_tex if name=='lighting' else self.textures.get(name,self.fallback)
   glActiveTexture(GL_TEXTURE0+unit);glBindTexture(GL_TEXTURE_2D,tex);glUniform1i(self.location(name),unit)
  glUniform4f(self.location('opacity'),*[cfg[n]['opacity'] if cfg[n]['enabled'] and (n=='white' or n in arrays) else 0. for n in ('white','satellite','streets','semantic')])
  glUniform3i(self.location('features'),int('elevation' in arrays),int('rivers' in arrays),int('peaks' in arrays))
  glUniform3f(self.location('widths'),cfg['contours']['width'],cfg['rivers']['river_thickness'],cfg['peaks']['width'])
  glUniform1f(self.location('river_thickness'),cfg['rivers']['river_thickness'])
  clock=state.get('river_clock');glUniform1f(self.location('river_progress'),(clock.sample() if clock else 1.)*1.005)
  glUniform1i(self.location('show_divides'),'divides' in arrays);glUniform1f(self.location('divide_width'),cfg['divides']['width'])
  glUniform1i(self.location('shading'),shading);glUniform1f(self.location('road_width'),cfg['streets']['width'])
  glUniform1i(self.location('projected_height'),'screen_elevation' in arrays)
  glUniform1i(self.location('contour_fill'),('line','above','below').index(cfg['contours']['fill']))
  glUniform1f(self.location('satellite_brightness'),cfg['satellite']['brightness'])
  glUniform1f(self.location('contour_interval'),cfg['contours']['interval']);glUniform1f(self.location('contour_level'),cfg['contours']['level']);glUniform1i(self.location('single_contour'),cfg['contours']['single'])
  for name,uniform in [('contours','contour_ink'),('rivers','river_ink'),('peaks','peak_ink'),('streets','road_ink'),('divides','divide_ink')]:
   ink=cfg[name]['color'];glUniform3f(self.location(uniform),*[int(ink[i:i+2],16)/255 for i in (1,3,5)])
  glBindFramebuffer(GL_FRAMEBUFFER,target);glViewport(0,0,*self.size);glDrawArrays(GL_TRIANGLES,0,3)

 def draw_contours(self,state,target,compensation=0.):
  a=state['arrays']['screen_elevation'];p=self.contour_program;c=state['config']['contours']
  if self.contour_array is not a:
   if self.contour_tex is not None:glDeleteTextures(1,[self.contour_tex])
   self.contour_tex=texture(2,a,GL_R16F,GL_RED,GL_HALF_FLOAT);self.contour_array=a
   glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
   h,w=a.shape;nh,nw=(h+15)//16,(w+15)//16
   padded=np.pad(a.astype(np.float32),((0,nh*16-h),(0,nw*16-w)),mode='edge').reshape(nh,16,nw,16)
   low=padded.min(axis=(1,3));high=padded.max(axis=(1,3))
   # Two neighbouring tiles cover derivatives, the widest stroke, and the
   # maximum chromatic displacement without clipping the compensated green.
   lo=np.pad(low,2,mode='edge');hi=np.pad(high,2,mode='edge')
   low=np.minimum.reduce([lo[y:y+nh,x:x+nw] for y in range(5) for x in range(5)])
   high=np.maximum.reduce([hi[y:y+nh,x:x+nw] for y in range(5) for x in range(5)])
   y,x=np.mgrid[:nh,:nw];tiles=np.ascontiguousarray(np.stack((x*16,y*16,low,high),axis=2).reshape(-1,4),np.float32)
   self.contour_count=len(tiles);glBindVertexArray(self.contour_vao);glBindBuffer(GL_ARRAY_BUFFER,self.contour_buffer);glBufferData(GL_ARRAY_BUFFER,tiles.nbytes,tiles,GL_STATIC_DRAW)
   glEnableVertexAttribArray(0);glVertexAttribPointer(0,4,GL_FLOAT,False,16,C.c_void_p(0));glVertexAttribDivisor(0,1)
  glUseProgram(p);glActiveTexture(GL_TEXTURE2);glBindTexture(GL_TEXTURE_2D,self.contour_tex)
  glUniform1i(glGetUniformLocation(p,'heights'),2)
  for name,value in [('spacing',c['interval']),('level',c['level']),('width',c['width'])]:glUniform1f(glGetUniformLocation(p,name),value)
  glUniform1i(glGetUniformLocation(p,'single_line'),c['single']);ink=c['color'];glUniform3f(glGetUniformLocation(p,'ink'),*[int(ink[i:i+2],16)/255 for i in (1,3,5)])
  glUniform1f(glGetUniformLocation(p,'chromatic'),compensation)
  glUniform1i(glGetUniformLocation(p,'fill_mode'),('line','above','below').index(c['fill']))
  glUniform2f(glGetUniformLocation(p,'output_size'),*self.size)
  glBindFramebuffer(GL_FRAMEBUFFER,target);glViewport(0,0,*self.size);glClearColor(0,0,0,1);glClear(GL_COLOR_BUFFER_BIT)
  glBindVertexArray(self.contour_vao);glDrawArraysInstanced(GL_TRIANGLES,0,6,self.contour_count)

 def close(self):
  if self.river_renderer is not None:self.river_renderer.close()
  if self.contour_tex is not None:glDeleteTextures(1,[self.contour_tex])
  glDeleteBuffers(1,[self.contour_buffer]);glDeleteVertexArrays(1,[self.contour_vao])
  glDeleteProgram(self.contour_program)
  if self.light_fb is not None:glDeleteFramebuffers(1,[self.light_fb]);glDeleteTextures(1,[self.light_tex])
  glDeleteTextures(len(self.textures)+1,list(self.textures.values())+[self.fallback]);glDeleteProgram(self.program)
