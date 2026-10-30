"""All-pixel bilinear ray reference, deliberately independent of the fast sweep."""
import ctypes as C
import numpy as np
from OpenGL.GLES3 import *
from OpenGL.GLES3.VERSION.GLES3_3_1 import glDispatchCompute,glMemoryBarrier,glBindImageTexture

RAYS='''#version 310 es
precision highp float;precision highp int;
layout(local_size_x=8,local_size_y=8) in;
layout(std430,binding=2) buffer Diagnostics{uint failures;};
layout(rgba8,binding=1) writeonly uniform highp image2D result;
uniform highp sampler2D terrain;uniform highp sampler2D hierarchy;uniform highp sampler2D normals;
uniform ivec2 size;uniform int levels;uniform vec2 direction;
uniform float rise;uniform float bias;uniform float ambient;uniform vec3 sun;
int blocked(ivec2 pixel){
 vec2 origin=vec2(pixel);float start=texelFetch(terrain,pixel,0).r+bias;
 float t=0.;int level=0;
 // Negative rays start in the patch before the sample vertex.
 ivec2 tile=pixel-ivec2(lessThan(direction,vec2(0.)));
 if(abs(direction.x)<1.e-8)tile.x=min(tile.x,size.x-2);
 if(abs(direction.y)<1.e-8)tile.y=min(tile.y,size.y-2);
 for(int iteration=0;iteration<32768;iteration++){
  vec2 p=origin+direction*t;
  if(any(lessThan(tile,ivec2(0)))||any(greaterThanEqual(tile,size-1)))return 0;
  ivec2 cell=tile>>level;float span=exp2(float(level));
  vec2 boundary=clamp((vec2(cell)+step(vec2(0.),direction))*span,vec2(0.),vec2(size-1));
  vec2 ends=vec2(1.e25);
  if(abs(direction.x)>1.e-8)ends.x=(boundary.x-origin.x)/direction.x;
  if(abs(direction.y)>1.e-8)ends.y=(boundary.y-origin.y)/direction.y;
  float end=max(t,min(ends.x,ends.y));
  float top=texelFetch(hierarchy,cell,level).r;
  if(top>start+rise*t+.001){
   if(level>0){level--;continue;}
   vec2 uv=p-vec2(tile);float length=end-t;
   float z00=texelFetch(terrain,tile,0).r,z10=texelFetch(terrain,tile+ivec2(1,0),0).r;
   float z01=texelFetch(terrain,tile+ivec2(0,1),0).r,z11=texelFetch(terrain,tile+ivec2(1,1),0).r;
   float a=z10-z00,b=z01-z00,c=z11-z10-z01+z00;
   float q0=z00+a*uv.x+b*uv.y+c*uv.x*uv.y-start-rise*t;
   float q1=a*direction.x+b*direction.y+c*(uv.x*direction.y+uv.y*direction.x)-rise;
   float q2=c*direction.x*direction.y;
   float maximum=max(q0,q0+q1*length+q2*length*length);
   if(q2<0.){float vertex=-q1/(2.*q2);if(vertex>0.&&vertex<length)maximum=max(maximum,q0+q1*vertex+q2*vertex*vertex);}
   if(maximum>.001)return 1;
  }
  // Keep the crossed integer boundary exact. Floating point roundoff must
  // never select a patch behind the ray or require a gap in the ray.
  tile=ivec2(floor(origin+direction*end));
  for(int axis=0;axis<2;axis++){
   float d=direction[axis];if(abs(d)<1.e-8)continue;
   int step_axis=d>0.?1:-1;
   float exit=(float(tile[axis]+(d>0.?1:0))-origin[axis])/d;
   float entry=(float(tile[axis]+(d<0.?1:0))-origin[axis])/d;
   if(exit<=end)tile[axis]+=step_axis;
   else if(entry>end)tile[axis]-=step_axis;
  }
  if(abs(direction.x)<1.e-8)tile.x=min(tile.x,size.x-2);
  if(abs(direction.y)<1.e-8)tile.y=min(tile.y,size.y-2);
  t=end;level=min(level+1,levels);
 }
 // Report a failed traversal to the CPU; never silently call it a shadow.
 return -1;
}
void main(){
 ivec2 pixel=ivec2(gl_GlobalInvocationID.xy);if(any(greaterThanEqual(pixel,size)))return;
 int occluded=blocked(pixel);if(occluded<0)atomicAdd(failures,1u);
 float lit=occluded==0?1.:0.;float diffuse=max(0.,dot(texelFetch(normals,pixel,0).rgb,sun));
 float value=ambient+(1.-ambient)*lit*diffuse;
 imageStore(result,pixel,vec4(value,value,value,1.));
}
'''

class Reference:
    def __init__(self,renderer):
        from .gpu import compute
        self.renderer=renderer;self.program=compute(RAYS);renderer.locations[self.program]={}
        self.texture=None;self.key=None
        self.diagnostics=glGenBuffers(1);glBindBuffer(0x90D2,self.diagnostics)
        glBufferData(0x90D2,4,None,GL_DYNAMIC_READ)

    def prepare(self,scene):
        if self.key==scene.key:return
        from .gpu import texture
        self.release();z=scene.z;h,w=z.shape
        size=1<<(max(w,h)-1).bit_length();self.levels=size.bit_length()-1
        pyramid=np.full((size,size),-1.e20,np.float32)
        # Bounds on continuous patches, not the discontinuous height columns.
        pyramid[:h-1,:w-1]=np.maximum.reduce((z[:-1,:-1],z[1:,:-1],z[:-1,1:],z[1:,1:]))
        self.texture=texture(9,pyramid,GL_R32F,GL_RED,GL_FLOAT)
        for level in range(1,self.levels+1):
            pyramid=np.ascontiguousarray(pyramid.reshape(pyramid.shape[0]//2,2,pyramid.shape[1]//2,2).max(axis=(1,3)))
            glTexImage2D(GL_TEXTURE_2D,level,GL_R32F,pyramid.shape[1],pyramid.shape[0],0,GL_RED,GL_FLOAT,pyramid.ctypes.data_as(C.c_void_p))
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,self.levels)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST_MIPMAP_NEAREST)
        self.key=scene.key

    def draw(self,frame,sx,sy):
        self.prepare(frame.scene);r=self.renderer;p=self.program
        glActiveTexture(GL_TEXTURE9);glBindTexture(GL_TEXTURE_2D,self.texture)
        glMemoryBarrier(0x628);glUseProgram(p)
        glBindBuffer(0x90D2,self.diagnostics);glBindBufferBase(0x90D2,2,self.diagnostics)
        zero=np.zeros(1,np.uint32);glBufferSubData(0x90D2,0,4,zero.ctypes.data_as(C.c_void_p))
        for name,unit in [('terrain',0),('normals',2),('hierarchy',9)]:r.integer(p,name,unit)
        r.integer(p,'levels',self.levels);glUniform2i(r.location(p,'size'),r.gw,r.gh)
        dominant=max(abs(sx),abs(sy));glUniform2f(r.location(p,'direction'),sx/dominant,sy/dominant)
        for name,value in [('rise',frame.sun[2]/dominant),('bias',frame.bias),('ambient',frame.ambient)]:r.real(p,name,value)
        glUniform3f(r.location(p,'sun'),*frame.sun)
        glBindImageTexture(1,r.filtered_tex,0,False,0,GL_WRITE_ONLY,GL_RGBA8)
        glDispatchCompute((r.gw+7)//8,(r.gh+7)//8,1);glMemoryBarrier(0x2288)
        # Four diagnostic bytes, not an image readback. A reference must expose
        # traversal failures instead of masking numerical errors with darkness.
        glBindBuffer(0x90D2,self.diagnostics)
        pointer=glMapBufferRange(0x90D2,0,4,GL_MAP_READ_BIT)
        if not pointer:raise RuntimeError('Cannot read DEM ray diagnostics')
        try:failures=int.from_bytes(C.string_at(pointer,4),'little')
        finally:glUnmapBuffer(0x90D2)
        if failures:raise RuntimeError(f'DEM reference traversal failed for {failures} rays')

    def release(self):
        if self.texture is not None:glDeleteTextures(1,[self.texture])
        self.texture=None;self.key=None

    def close(self):
        self.release();glDeleteProgram(self.program);glDeleteBuffers(1,[self.diagnostics])
