"""Conservative beam sweeps + cached occluders + compacted exact DEM rays.

Only static geometry is preprocessed. Every visibility shortcut is revalidated
against the current sun; neither time samples nor approximate shadow masks are
cached. The independently selectable reference renderer remains the oracle.
"""
import ctypes as C
import numpy as np
from OpenGL.GLES3 import *
from OpenGL.GLES3.VERSION.GLES3_3_1 import glDispatchCompute,glDispatchComputeIndirect,glMemoryBarrier,glBindImageTexture
from .gpu_reference import RAYS,Reference

BLOCK=8
BOUNDS='''#version 310 es
precision highp float;precision highp int;
layout(local_size_x=64) in;
layout(rgba32f,binding=0) writeonly uniform highp image2D bounds_image;
uniform sampler2D blocks;uniform int horizontal;uniform int positive;
uniform int count;uniform int across;uniform int lines;
uniform float offset;uniform float rise;uniform float qmin;uniform float extent;uniform float axis_extent;
vec2 heights(int axis,int cross_axis){
 if(cross_axis<0||cross_axis>=across)return vec2(-1.e20);
 return texelFetch(blocks,horizontal!=0?ivec2(axis,cross_axis):ivec2(cross_axis,axis),0).rg;
}
void main(){
 int q=int(gl_GlobalInvocationID.x);if(q>=lines)return;
 float lower=-1.e20,upper=-1.e20;
 for(int k=0;k<count;k++){
  int axis=positive!=0?count-1-k:k;
  float a=float(q)+qmin-offset*float(k),b=a-offset;
  // Enclose the entire width-one beam through this axis slab. The padding
  // covers roundoff at an exactly integral corridor boundary.
  int first=int(floor(min(a,b)-.002)),last=int(floor(max(a,b)+1.002));
  float top=-1.e20;
  for(int j=0;j<5;j++){int v=first+j;if(v<=last)top=max(top,heights(axis,v).y);}
  // Keep the receiver slab's maximum separate: its ray rise starts at
  // the receiver, not the slab's downstream boundary.
  imageStore(bounds_image,ivec2(q,k),vec4(lower,upper,top,0.));
  upper=max(upper,top+rise*float(k+1));
  float axis_position=positive!=0?float(count-k):float(k);
  if(a>=.002&&a+1.002<=extent&&axis_position<=axis_extent){
   int first_low=int(floor(a-.002));float floor_z=1.e20;
   for(int j=0;j<3;j++){int v=first_low+j;if(float(v)<=floor(a+1.002))floor_z=min(floor_z,heights(axis,v).x);}
   lower=max(lower,floor_z+rise*float(k));
  }
 }
}
'''
# A cached patch is a geometric witness, not cached visibility. Intersect the
# current ray with that patch's rectangle and solve its bilinear quadratic.
WITNESS='''
bool witness(ivec2 pixel,uint code){
 if((code&0x80000000u)==0u)return false;
 ivec2 tile=ivec2(int(code&65535u),int((code>>16)&32767u));
 vec2 origin=vec2(pixel),lo=vec2(tile),hi=lo+1.;float begin=0.,end=1.e25;
 for(int a=0;a<2;a++){
  if(abs(direction[a])<1.e-8){if(origin[a]<lo[a]||origin[a]>hi[a])return false;}
  else{float t0=(lo[a]-origin[a])/direction[a],t1=(hi[a]-origin[a])/direction[a];begin=max(begin,min(t0,t1));end=min(end,max(t0,t1));}
 }
 if(end<=begin)return false;
 vec2 uv=origin+direction*begin-lo;float length=end-begin;
 float z00=texelFetch(terrain,tile,0).r,z10=texelFetch(terrain,tile+ivec2(1,0),0).r;
 float z01=texelFetch(terrain,tile+ivec2(0,1),0).r,z11=texelFetch(terrain,tile+ivec2(1,1),0).r;
 float a=z10-z00,b=z01-z00,c=z11-z10-z01+z00;
 float q0=z00+a*uv.x+b*uv.y+c*uv.x*uv.y-texelFetch(terrain,pixel,0).r-bias-rise*begin;
 float q1=a*direction.x+b*direction.y+c*(uv.x*direction.y+uv.y*direction.x)-rise,q2=c*direction.x*direction.y;
 float maximum=max(q0,q0+q1*length+q2*length*length);
 if(q2<0.){float v=-q1/(2.*q2);if(v>0.&&v<length)maximum=max(maximum,q0+q1*v+q2*v*v);}
 // Conservative margin makes a witness near a grazing intersection fall
 // back to the reference traversal, rather than creating a false shadow.
 return maximum>.05;
}
'''
CLASSIFY='''#version 310 es
precision highp float;precision highp int;
uniform highp sampler2D terrain;uniform highp sampler2D normals;uniform highp sampler2D beam_bounds;
uniform highp usampler2D witnesses;
uniform vec2 direction;uniform float rise;uniform float bias;uniform float ambient;uniform vec3 sun;
uniform int horizontal;uniform int positive;uniform int count;uniform int block_size;
uniform float offset;uniform float qmin;uniform float margin;
layout(std430,binding=1) buffer Candidates{uint count_rays;uint groups_x;uint groups_y;uint groups_z;uint cells[];};
out vec4 color;
'''+WITNESS+'''
void main(){
 ivec2 pixel=ivec2(gl_FragCoord.xy);float diffuse=max(0.,dot(texelFetch(normals,pixel,0).rgb,sun));
 float value=ambient;color=vec4(value,value,value,1.);if(diffuse<=0.||ambient>=1.)return;
 float axis=float(horizontal!=0?pixel.x:pixel.y)/float(block_size);
 float k=positive!=0?float(count)-axis:axis;
 int slab=positive!=0?count-1-int(floor(axis)):int(floor(axis));
 float across=float(horizontal!=0?pixel.y:pixel.x)/float(block_size);
 int q=int(floor(across+offset*k-qmin));
 vec3 bounds=texelFetch(beam_bounds,ivec2(q,slab),0).rgb;
 vec2 limits=bounds.rg-rise*float(block_size)*k;limits.y=max(limits.y,bounds.b);
 float z=texelFetch(terrain,pixel,0).r+bias;
 if(limits.y<z-margin){value=ambient+(1.-ambient)*diffuse;color=vec4(value,value,value,1.);return;}
 if(limits.x>z+margin)return;
 if(witness(pixel,texelFetch(witnesses,pixel,0).r))return;
 uint tag=bounds.g-rise*float(block_size)*k<z-margin?0x80000000u:0u;
 uint i=atomicAdd(count_rays,1u);cells[i]=uint(pixel.x)|(uint(pixel.y)<<16)|tag;
 if((i&63u)==0u){atomicMax(groups_x,min(i/64u+1u,65535u));atomicMax(groups_y,i/(65535u*64u)+1u);}
}
'''
# Reuse exactly the validated intersection/traversal implementation. Only its
# dispatch, blocker recording, and a conservatively certified local stop
# distance differ from reference mode.
TRACE=RAYS[:RAYS.index('void main(){')].replace('layout(local_size_x=8,local_size_y=8)', 'layout(local_size_x=64)')
TRACE=TRACE.replace('int blocked(ivec2 pixel){', '''float trace_limit=1.e25;uint found=0u;
int blocked(ivec2 pixel){''').replace('if(maximum>.001)return 1;', 'if(maximum>.001){found=0x80000000u|uint(tile.x)|(uint(tile.y)<<16);return 1;}')
TRACE=TRACE.replace('  vec2 p=origin+direction*t;', '  if(t>=trace_limit)return 0;\n  vec2 p=origin+direction*t;')
TRACE=TRACE.replace('float end=max(t,min(ends.x,ends.y));', 'float end=min(trace_limit,max(t,min(ends.x,ends.y)));')
TRACE+='''
layout(std430,binding=1) readonly buffer Candidates{uint count_rays;uint groups_x;uint groups_y;uint groups_z;uint cells[];};
layout(r32ui,binding=2) writeonly uniform highp uimage2D witnesses_image;
uniform int horizontal;uniform int positive;uniform int block_size;
void main(){
 uint i=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*65535u*64u;if(i>=count_rays)return;
 ivec2 pixel=ivec2(int(cells[i]&65535u),int((cells[i]>>16)&32767u));
 if((cells[i]&0x80000000u)!=0u){
  int axis=horizontal!=0?pixel.x:pixel.y;
  trace_limit=positive!=0?float((axis/block_size+1)*block_size-axis):float(axis%block_size);
 }
 int occluded=blocked(pixel);if(occluded<0)atomicAdd(failures,1u);
 float diffuse=max(0.,dot(texelFetch(normals,pixel,0).rgb,sun));
 float value=ambient+(1.-ambient)*(occluded==0?diffuse:0.);
 imageStore(result,pixel,vec4(value,value,value,1.));imageStore(witnesses_image,pixel,uvec4(found));
}
'''

class Precise:
    def __init__(self,renderer):
        from .gpu import compute,program
        self.r=renderer;self.reference=Reference(renderer)
        self.bounds=compute(BOUNDS);self.classify=program(CLASSIFY);self.trace=compute(TRACE)
        for p in (self.bounds,self.classify,self.trace):renderer.locations[p]={}
        self.key=None;self.textures=[];self.fence=None

    def prepare(self,scene):
        if self.key==scene.key:return
        from .gpu import texture
        self.release();self.reference.prepare(scene)
        z=scene.z;h,w=z.shape;self.bw=(w+BLOCK-1)//BLOCK;self.bh=(h+BLOCK-1)//BLOCK
        # Inclusive vertices bound all bilinear patches in each block.
        pad=np.full((self.bh*BLOCK+1,self.bw*BLOCK+1),-1.e20,np.float32);pad[:h,:w]=z
        maximum=np.maximum.reduce((pad[:-1,:-1],pad[1:,:-1],pad[:-1,1:],pad[1:,1:]))
        minimum=np.minimum.reduce((pad[:-1,:-1],pad[1:,:-1],pad[:-1,1:],pad[1:,1:]))
        blocks=np.stack((minimum.reshape(self.bh,BLOCK,self.bw,BLOCK).min(axis=(1,3)),
                         maximum.reshape(self.bh,BLOCK,self.bw,BLOCK).max(axis=(1,3))),axis=-1)
        self.textures=[texture(12,np.ascontiguousarray(blocks),GL_RG32F,GL_RG,GL_FLOAT),
            texture(10,None,GL_RGBA32F,GL_RGBA,GL_FLOAT,self.bw+self.bh+8,max(self.bw,self.bh),immutable=True),
            texture(11,np.zeros((h,w),np.uint32),GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT,immutable=True)]
        # Cumulative diagnostic errors are checked after a nonblocking fence.
        glBindBuffer(0x90D2,self.reference.diagnostics);zero=np.zeros(1,np.uint32)
        glBufferSubData(0x90D2,0,4,zero.ctypes.data_as(C.c_void_p));self.key=scene.key

    def check(self,wait=False):
        if self.fence is None:return
        flags=GL_SYNC_FLUSH_COMMANDS_BIT if wait else 0
        status=glClientWaitSync(self.fence,flags,1000000000 if wait else 0)
        if status==GL_TIMEOUT_EXPIRED:return
        if status==GL_WAIT_FAILED:raise RuntimeError('Precise DEM diagnostic fence failed')
        glDeleteSync(self.fence);self.fence=None
        glBindBuffer(0x90D2,self.reference.diagnostics)
        pointer=glMapBufferRange(0x90D2,0,4,GL_MAP_READ_BIT)
        if not pointer:raise RuntimeError('Cannot read precise DEM diagnostics')
        try:failures=int.from_bytes(C.string_at(pointer,4),'little')
        finally:glUnmapBuffer(0x90D2)
        if failures:raise RuntimeError(f'Precise DEM traversal exhausted for {failures} rays')

    def draw(self,frame,sx,sy):
        self.prepare(frame.scene);self.check();r=self.r
        horizontal=abs(sx)>=abs(sy);dominant=sx if horizontal else sy;positive=dominant>0
        count=self.bw if horizontal else self.bh;across=self.bh if horizontal else self.bw
        offset=(sy if horizontal else sx)/abs(dominant);rise=frame.sun[2]/abs(dominant)
        qmin=float(np.floor(min(0.,offset*count)))-2;lines=int(np.ceil(across+abs(offset)*count))+6
        glMemoryBarrier(0x628)
        for unit,tex in zip((12,10,11),self.textures):glActiveTexture(GL_TEXTURE0+unit);glBindTexture(GL_TEXTURE_2D,tex)
        glActiveTexture(GL_TEXTURE9);glBindTexture(GL_TEXTURE_2D,self.reference.texture)
        p=self.bounds;glUseProgram(p)
        for name,v in [('blocks',12),('horizontal',horizontal),('positive',positive),('count',count),('across',across),('lines',lines)]:r.integer(p,name,v)
        for name,v in [('offset',offset),('rise',rise*BLOCK),('qmin',qmin),('extent',((r.gh if horizontal else r.gw)-1)/BLOCK),('axis_extent',((r.gw if horizontal else r.gh)-1)/BLOCK)]:r.real(p,name,v)
        glBindImageTexture(0,self.textures[1],0,False,0,GL_WRITE_ONLY,GL_RGBA32F)
        glDispatchCompute((lines+63)//64,1,1);glMemoryBarrier(0x28)
        glBindBuffer(0x90D2,r.candidates);glBindBufferBase(0x90D2,1,r.candidates)
        counters=np.array([0,0,1,1],np.uint32);glBufferSubData(0x90D2,0,16,counters.ctypes.data_as(C.c_void_p))
        for p in (self.classify,self.trace):
            glUseProgram(p)
            for name,v in [('terrain',0),('normals',2),('hierarchy',9),('beam_bounds',10),('witnesses',11),('horizontal',horizontal),('positive',positive),('count',count),('block_size',BLOCK),('levels',self.reference.levels)]:r.integer(p,name,v)
            for name,v in [('rise',rise),('offset',offset),('qmin',qmin),('ambient',frame.ambient),('bias',frame.bias),('margin',.05+1.e-5*(rise*BLOCK*count+10000.))]:r.real(p,name,v)
            glUniform2f(r.location(p,'direction'),sx/abs(dominant),sy/abs(dominant))
            glUniform2i(r.location(p,'size'),r.gw,r.gh);glUniform3f(r.location(p,'sun'),*frame.sun)
        glBindFramebuffer(GL_FRAMEBUFFER,r.filtered_fb);glViewport(0,0,r.gw,r.gh)
        glUseProgram(self.classify);glDrawArrays(GL_TRIANGLES,0,3);glMemoryBarrier(0x2460)
        glBindBufferBase(0x90D2,2,self.reference.diagnostics)
        glBindImageTexture(1,r.filtered_tex,0,False,0,GL_WRITE_ONLY,GL_RGBA8)
        glBindImageTexture(2,self.textures[2],0,False,0,GL_WRITE_ONLY,GL_R32UI)
        glUseProgram(self.trace);glBindBuffer(0x90EE,r.candidates);glDispatchComputeIndirect(4);glMemoryBarrier(0x2288)
        if self.fence is None:self.fence=glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0)

    def release(self):
        if self.fence is not None:glDeleteSync(self.fence);self.fence=None
        if self.textures:glDeleteTextures(len(self.textures),self.textures)
        self.textures=[];self.key=None;self.reference.release()

    def close(self):
        self.release();self.reference.close()
        for p in (self.bounds,self.classify,self.trace):glDeleteProgram(p)
