"""GPU-resident geographic DEM shadows, shading and native pixel output.

GLES 3.1, one GPU lane per sun-aligned scan, texture-resident horizons. All
methods run on the owning display thread. No per-frame full-size CPU readback.
"""
import os,sys,time,ctypes as C
if sys.platform.startswith('linux'):os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import numpy as np
from OpenGL.GLES3 import *
from OpenGL.GLES3.VERSION.GLES3_3_1 import glDispatchCompute,glDispatchComputeIndirect,glMemoryBarrier,glBindImageTexture
from .frames import SunFrame
from .lighting import GLSL

SCAN='''#version 310 es
precision highp float;precision highp int;
layout(local_size_x=64) in;
layout(r32f,binding=0) writeonly uniform highp image2D horizon_image;
uniform highp sampler2D terrain;
uniform int count;uniform int across;uniform int start;uniform int step_axis;uniform int lines;
uniform float offset;uniform float rise;uniform float qmin;
void main(){
 int q=int(gl_GlobalInvocationID.x);if(q>=lines)return;
 float a0=float(q)+qmin;float m=-1.e20;
 for(int k=0;k<count;k++){
  imageStore(horizon_image,ivec2(q,k),vec4(m-rise*float(k)));
  float a=a0-offset*float(k);float z=-1.e20;
  if(a>=0.&&a<=float(across-1)){
   int lo=int(floor(a)),hi=min(lo+1,across-1);int u=start+step_axis*k;
   z=mix(texelFetch(terrain,ivec2(u,lo),0).r,texelFetch(terrain,ivec2(u,hi),0).r,fract(a));
  }
  m=max(m,z+rise*float(k));
 }
}
'''
SHADE='''#version 310 es
precision highp float;precision highp int;
uniform highp sampler2D horizon_texture;
uniform highp sampler2D terrain;uniform highp sampler2D normals;
uniform int horizontal;uniform int positive;uniform int count;
uniform int lines;uniform int serial;uniform int overhead;
uniform float offset;uniform float qmin;uniform float ambient;uniform float bias;
uniform int mask_only;uniform vec3 sun;
out vec4 color;
void main(){
 ivec2 p=ivec2(gl_FragCoord.xy);int axis=horizontal!=0?p.x:p.y;
 int k=positive!=0?count-1-axis:axis;float across=float(horizontal!=0?p.y:p.x);
 float q=across+offset*float(k)-qmin;int lo=int(floor(q));
 int i0=serial!=0?k*lines+lo:lo*count+k;
 int i1=serial!=0?i0+1:i0+count;
 float h=mix(texelFetch(horizon_texture,ivec2(lo,k),0).r,texelFetch(horizon_texture,ivec2(lo+1,k),0).r,fract(q));
 float z=texelFetch(terrain,p,0).r;float lit=overhead!=0?1.:(h>z+bias?0.:1.);
 float diffuse=max(0.,dot(texelFetch(normals,p,0).rgb,sun))*lit;
 float v=mask_only!=0?lit:ambient+(1.-ambient)*diffuse;
 color=sun.z>0.?vec4(v,v,v,1.):vec4(0.,0.,0.,1.);
}
'''
STABILIZE="""#version 310 es
precision highp float;precision highp int;
uniform sampler2D light;uniform float ambient;
layout(std430,binding=1) buffer Candidates{uint count;uint groups_x;uint groups_y;uint groups_z;uint cells[];};
out vec4 color;
void main(){
 ivec2 pixel=ivec2(gl_FragCoord.xy),size=textureSize(light,0);
 float value=texelFetch(light,pixel,0).r;color=vec4(value,value,value,1.);
 float threshold=ambient+1.5/255.;if(value<=threshold)return;
 int lit=0;
 for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
  ivec2 q=pixel+ivec2(x,y);
  if(all(greaterThanEqual(q,ivec2(0)))&&all(lessThan(q,size))&&texelFetch(light,q,0).r>threshold){
   lit++;if(lit>4)return;
  }
 }
 uint i=atomicAdd(count,1u);cells[i]=uint(pixel.x)|(uint(pixel.y)<<16);
 if((i&63u)==0u){atomicMax(groups_x,min(i/64u+1u,65535u));atomicMax(groups_y,i/(65535u*64u)+1u);}
}
"""
REFINE="""#version 310 es
precision highp float;precision highp int;
layout(local_size_x=64) in;
layout(std430,binding=1) readonly buffer Candidates{uint count;uint groups_x;uint groups_y;uint groups_z;uint cells[];};
layout(rgba8,binding=1) writeonly uniform highp image2D filtered;
uniform highp sampler2D hierarchy;
uniform int levels;uniform ivec2 size;uniform float ambient;uniform float bias;
uniform vec2 direction;uniform float rise;
void main(){
 uint i=gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*65535u*64u;if(i>=count)return;
 ivec2 pixel=ivec2(int(cells[i]&65535u),int(cells[i]>>16));
 // Compact candidates on the GPU so long rays never stall whole fragment tiles.
 float start=texelFetch(hierarchy,pixel,0).r+bias;
 vec2 origin=vec2(pixel)+.5;float t=0.;int level=0;
 for(int iteration=0;iteration<8192;iteration++){
  vec2 p=origin+direction*t;
  if(any(lessThan(p,vec2(0.)))||any(greaterThanEqual(p,vec2(size))))return;
  ivec2 cell=ivec2(p)>>level;float top=texelFetch(hierarchy,cell,level).r;
  if(top>start+rise*t){
   if(level==0){imageStore(filtered,pixel,vec4(ambient,ambient,ambient,1.));return;}
   level--;continue;
  }
  float span=exp2(float(level));vec2 boundary=(vec2(cell)+step(vec2(0.),direction))*span;
  vec2 distance=vec2(1.e25);
  if(abs(direction.x)>1.e-8)distance.x=(boundary.x-p.x)/direction.x;
  if(abs(direction.y)>1.e-8)distance.y=(boundary.y-p.y)/direction.y;
  t+=max(0.,min(distance.x,distance.y))+.001;level=min(level+1,levels);
 }
 // Fail closed on a safety-limit hit; never display a bright error marker.
 imageStore(filtered,pixel,vec4(ambient,ambient,ambient,1.));
}
"""

GATHER='''#version 300 es
precision highp float;precision highp int;
uniform highp usampler2D lookup;uniform sampler2D light;
uniform int flip_y;out vec4 color;
'''+GLSL+'''
void main(){ivec2 pixel=ivec2(gl_FragCoord.xy);if(flip_y!=0)pixel.y=textureSize(lookup,0).y-1-pixel.y;
 uvec2 p=texelFetch(lookup,pixel,0).rg;
 float v=p.r==65535u?0.:exposure(texelFetch(light,ivec2(p),0).r);color=vec4(v,v,v,1.);}
'''


def compute(source):
    shader=glCreateShader(0x91B9);glShaderSource(shader,source);glCompileShader(shader)
    if not glGetShaderiv(shader,GL_COMPILE_STATUS):raise RuntimeError(glGetShaderInfoLog(shader).decode())
    result=glCreateProgram();glAttachShader(result,shader);glLinkProgram(result);glDeleteShader(shader)
    if not glGetProgramiv(result,GL_LINK_STATUS):raise RuntimeError(glGetProgramInfoLog(result).decode())
    return result


def program(fragment,vertex_source=None):
    vertex=vertex_source or '''#version 300 es
    const vec2 p[3]=vec2[3](vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.));
    void main(){gl_Position=vec4(p[gl_VertexID],0.,1.);}
    '''
    if fragment.startswith('#version 310 es'):
        vertex=vertex.replace('#version 300 es','#version 310 es')
    shaders=[]
    for kind,source in [(GL_VERTEX_SHADER,vertex),(GL_FRAGMENT_SHADER,fragment)]:
        shader=glCreateShader(kind);glShaderSource(shader,source);glCompileShader(shader)
        if not glGetShaderiv(shader,GL_COMPILE_STATUS):raise RuntimeError(glGetShaderInfoLog(shader))
        shaders.append(shader)
    result=glCreateProgram()
    for shader in shaders:glAttachShader(result,shader)
    glLinkProgram(result)
    if not glGetProgramiv(result,GL_LINK_STATUS):raise RuntimeError(glGetProgramInfoLog(result))
    for shader in shaders:glDeleteShader(shader)
    return result


def texture(unit,array,internal,fmt,typ,width=None,height=None,immutable=False):
    result=glGenTextures(1);glActiveTexture(GL_TEXTURE0+unit);glBindTexture(GL_TEXTURE_2D,result)
    for key,value in [(GL_TEXTURE_MIN_FILTER,GL_NEAREST),(GL_TEXTURE_MAG_FILTER,GL_NEAREST),
                      (GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE),(GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE)]:
        glTexParameteri(GL_TEXTURE_2D,key,value)
    if array is not None:height,width=array.shape[:2]
    if immutable:
        glTexStorage2D(GL_TEXTURE_2D,1,internal,width,height)
        if array is not None:glTexSubImage2D(GL_TEXTURE_2D,0,0,0,width,height,fmt,typ,array.ctypes.data_as(C.c_void_p))
    else:
        glTexImage2D(GL_TEXTURE_2D,0,internal,width,height,0,fmt,typ,
                     None if array is None else array.ctypes.data_as(C.c_void_p))
    return result


def framebuffer(tex):
    result=glGenFramebuffers(1);glBindFramebuffer(GL_FRAMEBUFFER,result)
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tex,0)
    assert glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE
    return result

RGB='''#version 300 es
precision highp float;precision highp int;
uniform sampler2D image;uniform int flip_y;
out vec4 color;
void main(){ivec2 p=ivec2(gl_FragCoord.xy);if(flip_y!=0)p.y=textureSize(image,0).y-1-p.y;
 color=vec4(texelFetch(image,p,0).rgb,1.);}
'''
PREVIEW='''#version 300 es
precision highp float;precision highp int;
uniform highp usampler2D lookup;uniform sampler2D light;uniform vec2 target_size;
out vec4 color;
void main(){
 vec2 scale=vec2(textureSize(lookup,0))/target_size;
 vec2 origin=floor(gl_FragCoord.xy)*scale;
 ivec2 samples=ivec2(min(vec2(8.),ceil(scale)));float sum=0.;
 for(int y=0;y<8;y++)for(int x=0;x<8;x++){
  if(x>=samples.x||y>=samples.y)continue;
  ivec2 pixel=ivec2(origin+(vec2(x,y)+.5)*scale/vec2(samples));
  uvec2 p=texelFetch(lookup,pixel,0).rg;
  sum+=p.r==65535u?0.:texelFetch(light,ivec2(p),0).r;
 }
 float v=sum/float(samples.x*samples.y);color=vec4(v,v,v,1.);
}
'''


CHROMATIC='''#version 300 es
precision highp float;
uniform sampler2D source;uniform vec2 image_size;uniform float strength;
out vec4 color;
void main(){
 vec2 p=gl_FragCoord.xy,center=image_size*.5;
 vec2 q=p+(p-center)*strength/length(center);
 vec4 base=texelFetch(source,ivec2(p),0);
 float green=all(greaterThanEqual(q,vec2(.5)))&&all(lessThanEqual(q,image_size-vec2(.5)))?texture(source,q/image_size).g:0.;
 color=vec4(base.r,green,base.b,1.);
}
'''


class Renderer:
    def __init__(self):
        self.renderer=C.string_at(glGetString(GL_RENDERER)).decode()
        self.version=C.string_at(glGetString(GL_VERSION)).decode()
        # Compiling the compute program verifies the required API on the actual
        # context; a missing GLES 3.1 implementation fails before accepting frames.
        self.scan=compute(SCAN);self.refine=compute(REFINE);self.stabilize=program(STABILIZE)
        self.shade,self.gather,self.rgb,self.preview=map(program,[SHADE,GATHER,RGB,PREVIEW])
        self.vao=glGenVertexArrays(1);glBindVertexArray(self.vao);glDisable(GL_DITHER);glDisable(GL_BLEND);glDisable(GL_DEPTH_TEST)
        glPixelStorei(GL_UNPACK_ALIGNMENT,1);glPixelStorei(GL_PACK_ALIGNMENT,1)
        self.textures=[];self.scene_key=None;self.light_fb=None;self.filtered_tex=self.filtered_fb=None;self.candidates=None;self.rgb_tex=None;self.rgb_size=None
        self.capture_tex=self.capture_fb=None;self.capture_size=None;self.last_frame=None;self.timings={};self.downsample_targets={};self.snapshot_frame=None;self.snapshot_size=None
        self.pending_captures=[]
        self.locations={}
        self.reference=None;self.precise=None;self.event_renderer=None
        self.track_renderer=None;self.color_capture=False;self.ca_program=None;self.ca_tex=self.ca_fb=None;self.ca_size=None;self.snapshot_color=False
        self.layer_renderer=None
        for p in [self.scan,self.refine,self.shade,self.stabilize,self.gather,self.rgb,self.preview]:
            self.locations[p]={}
        glUseProgram(self.shade)
        for name,unit in [('terrain',0),('normals',2),('horizon_texture',5)]:self.integer(self.shade,name,unit)
        self.integer(self.shade,'serial',1);self.integer(self.shade,'mask_only',0)
        for p in [self.gather,self.preview]:
            glUseProgram(p);self.integer(p,'lookup',4);self.integer(p,'light',8)
        glUseProgram(self.stabilize);self.integer(self.stabilize,'light',3)
        glUseProgram(self.refine);self.integer(self.refine,'hierarchy',6)
        glUseProgram(self.rgb);self.integer(self.rgb,'image',7)

    def location(self,p,name):
        locations=self.locations[p]
        if name not in locations:locations[name]=glGetUniformLocation(p,name)
        return locations[name]

    def integer(self,p,name,value):glUniform1i(self.location(p,name),int(value))
    def real(self,p,name,value):glUniform1f(self.location(p,name),float(value))

    def scene(self,scene,method="fast"):
        if (scene.key,method)==self.scene_key:return
        self.release_scene()
        z=scene.z;gh,gw=z.shape;self.gh,self.gw=gh,gw;self.oh,self.ow=scene.uv.shape[:2]
        limit=int(glGetIntegerv(GL_MAX_TEXTURE_SIZE))
        if method=='events' and os.environ.get('MAP_PROJECTION_BITSET','1')=='1':
            # Cached replay needs lookup + normals + a bitset, not DEM copies,
            # shadow buffers, pyramids or ray candidate storage (~166 MB).
            self.textures=[texture(unit,np.zeros((1,1),np.uint8),GL_R8,GL_RED,GL_UNSIGNED_BYTE) for unit in range(4)]
            self.textures.append(texture(4,scene.uv,GL_RG16UI,GL_RG_INTEGER,GL_UNSIGNED_SHORT))
            self.scene_key=(scene.key,method);return
        if gw+gh+3>limit:raise ValueError('Terrain grid exceeds the GPU texture limit. Choose a smaller grid.')
        self.textures=[texture(0,z,GL_R32F,GL_RED,GL_FLOAT),
            texture(1,np.ascontiguousarray(z.T),GL_R32F,GL_RED,GL_FLOAT),
            texture(2,scene.normals if method!="events" else np.zeros((1,1,3),np.float32),GL_RGB32F,GL_RGB,GL_FLOAT),
            texture(3,None,GL_R8,GL_RED,GL_UNSIGNED_BYTE,gw,gh),
            texture(4,scene.uv,GL_RG16UI,GL_RG_INTEGER,GL_UNSIGNED_SHORT),
            texture(5,None,GL_R32F,GL_RED,GL_FLOAT,gw+gh+3 if method=="fast" else 1,max(gw,gh) if method=="fast" else 1,immutable=True)]
        size=(1<<(max(gw,gh)-1).bit_length()) if method=="fast" else 1;self.levels=size.bit_length()-1
        pyramid=np.full((size,size),-1.e20,np.float32);pyramid[:min(gh,size),:min(gw,size)]=z[:size,:size]
        hierarchy=texture(6,pyramid,GL_R32F,GL_RED,GL_FLOAT)
        for level in range(1,self.levels+1):
            pyramid=np.ascontiguousarray(pyramid.reshape(pyramid.shape[0]//2,2,pyramid.shape[1]//2,2).max(axis=(1,3)))
            glTexImage2D(GL_TEXTURE_2D,level,GL_R32F,pyramid.shape[1],pyramid.shape[0],0,GL_RED,GL_FLOAT,pyramid.ctypes.data_as(C.c_void_p))
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,self.levels)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST_MIPMAP_NEAREST)
        self.textures.append(hierarchy)
        self.filtered_tex=texture(8,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,gw,gh,immutable=True)
        self.candidates=glGenBuffers(1);glBindBuffer(0x90D2,self.candidates)
        glBufferData(0x90D2,16+gw*gh*4,None,GL_DYNAMIC_DRAW)
        self.filtered_fb=framebuffer(self.filtered_tex)
        self.light_fb=framebuffer(self.textures[3]);self.scene_key=(scene.key,method)

    def draw(self,frame,target=0,physical=True):
        self.scene(frame.scene,frame.method);gh,gw=self.gh,self.gw
        glBindVertexArray(self.vao)
        # Rebind after a calibration texture or preview was drawn.
        for unit,tex in enumerate(self.textures):glActiveTexture(GL_TEXTURE0+unit);glBindTexture(GL_TEXTURE_2D,tex)
        if frame.method=='events' and (frame.enabled or os.environ.get('MAP_PROJECTION_BITSET','1')=='1'):
            if self.event_renderer is None:
                if os.environ.get('MAP_PROJECTION_BITSET','1')=='1':
                    from .gpu_bitset import BitsetRenderer as EventRenderer
                elif os.environ.get('MAP_PROJECTION_INTERVALS','0')=='1':
                    from .gpu_intervals import IntervalRenderer as EventRenderer
                else:from .gpu_events import EventRenderer
                self.event_renderer=EventRenderer(self)
            started=time.perf_counter();self.event_renderer.draw(frame,target,physical)
            self.last_frame=frame;self.timings={'backend':'GPU · native bitset events' if getattr(self.event_renderer,'supports_disabled',False) else 'GPU · incremental horizon events','gpu_submit_seconds':time.perf_counter()-started};return
        glBindImageTexture(0,self.textures[5],0,False,0,GL_WRITE_ONLY,GL_R32F)
        ts=frame.scene.terrain_size;sun=frame.sun
        sx,sy=sun[0]/(ts[0]/gw),-sun[1]/(ts[1]/gh)
        horizontal=abs(sx)>=abs(sy);dominant=sx if horizontal else sy
        count=gw if horizontal else gh;across=gh if horizontal else gw
        overhead=abs(dominant)<1e-12
        start=time.perf_counter()
        reference=frame.method in ('reference','precise','events')
        if frame.enabled and sun[2]>0 and not overhead and not reference:
            offset=(sy if horizontal else sx)/abs(dominant);rise=sun[2]/abs(dominant)
            qmin=np.floor(min(0,offset*(count-1)));lines=int(np.ceil(across+abs(offset)*(count-1)))+2
            glUseProgram(self.scan)
            for name,value in [('terrain',0 if horizontal else 1),('count',count),('across',across),
                               ('start',count-1 if dominant>0 else 0),('step_axis',-1 if dominant>0 else 1),('lines',lines)]:
                self.integer(self.scan,name,value)
            for name,value in [('offset',offset),('rise',rise),('qmin',qmin)]:self.real(self.scan,name,value)
            # Previous frame fragments must finish reading the horizon before
            # compute overwrites it; the second barrier publishes the new scan.
            glMemoryBarrier(0x20)
            glDispatchCompute((lines+63)//64,1,1);glMemoryBarrier(0x28)
        else:offset=qmin=0.;lines=across+2
        glBindFramebuffer(GL_FRAMEBUFFER,self.light_fb);glViewport(0,0,gw,gh)
        if not frame.enabled or sun[2]<=0:
            value=0. if frame.enabled else 1.
            glClearColor(value,value,value,1);glClear(GL_COLOR_BUFFER_BIT)
        elif not reference or overhead:
            glUseProgram(self.shade)
            for name,value in [('horizontal',horizontal),('positive',dominant>0),('count',count),('lines',lines),('overhead',overhead)]:self.integer(self.shade,name,value)
            for name,value in [('offset',offset),('qmin',qmin),('ambient',frame.ambient),('bias',frame.bias)]:self.real(self.shade,name,value)
            glUniform3f(self.location(self.shade,'sun'),*sun);glDrawArrays(GL_TRIANGLES,0,3)
        light=self.textures[3]
        if frame.enabled and frame.method=='events':
            if self.event_renderer is None:
                from .gpu_events import EventRenderer
                self.event_renderer=EventRenderer(self)
            self.event_renderer.draw(frame);light=self.filtered_tex
        elif frame.enabled and sun[2]>0 and not overhead and frame.method=='precise':
            if self.precise is None:
                from .gpu_precise import Precise
                self.precise=Precise(self)
            self.precise.draw(frame,sx,sy);light=self.filtered_tex
        elif frame.enabled and sun[2]>0 and not overhead and reference:
            if self.reference is None:
                from .gpu_reference import Reference
                self.reference=Reference(self)
            self.reference.draw(frame,sx,sy);light=self.filtered_tex
        elif frame.enabled and sun[2]>0 and not overhead:
            glMemoryBarrier(0x600)  # prior image writes -> framebuffer / counter update
            glBindBuffer(0x90D2,self.candidates);glBindBufferBase(0x90D2,1,self.candidates)
            counters=np.array([0,0,1,1],np.uint32)
            glBufferSubData(0x90D2,0,16,counters.ctypes.data_as(C.c_void_p))
            glBindFramebuffer(GL_FRAMEBUFFER,self.filtered_fb);glViewport(0,0,gw,gh);glUseProgram(self.stabilize)
            self.real(self.stabilize,'ambient',frame.ambient);glDrawArrays(GL_TRIANGLES,0,3)
            glMemoryBarrier(0x2060)  # fragment SSBO -> indirect command, ray list, image stores
            glUseProgram(self.refine)
            glUniform2f(self.location(self.refine,'direction'),sx/abs(dominant),sy/abs(dominant))
            glUniform2i(self.location(self.refine,'size'),gw,gh)
            self.real(self.refine,'rise',sun[2]/abs(dominant));self.real(self.refine,'bias',frame.bias)
            self.real(self.refine,'ambient',frame.ambient);self.integer(self.refine,'levels',self.levels)
            glBindImageTexture(1,self.filtered_tex,0,False,0,GL_WRITE_ONLY,GL_RGBA8)
            glBindBuffer(0x90EE,self.candidates);glDispatchComputeIndirect(4)
            glMemoryBarrier(0x8);light=self.filtered_tex
        glActiveTexture(GL_TEXTURE8);glBindTexture(GL_TEXTURE_2D,light)
        glBindFramebuffer(GL_FRAMEBUFFER,target);glViewport(0,0,self.ow,self.oh)
        glUseProgram(self.gather);self.integer(self.gather,'flip_y',physical)
        for name,value in [('light_gain',frame.light_gain),('light_floor',frame.light_floor),('light_gamma',frame.light_gamma)]:self.real(self.gather,name,value)
        glDrawArrays(GL_TRIANGLES,0,3)
        # No glFinish here: the display swap/fence supplies the completion point.
        self.timings={'gpu_submit_seconds':round(time.perf_counter()-start,6),'backend':'GPU · conservative beams + exact rays' if frame.method=='precise' else 'GPU · bilinear ray reference' if reference else 'GPU · sun-aligned scans'}
        self.last_frame=frame

    def draw_rgb(self,frame,target=0,physical=True):
        h,w=frame.shape[:2];self.oh,self.ow=h,w;glActiveTexture(GL_TEXTURE7)
        if self.rgb_tex is None or self.rgb_size!=(w,h):
            if self.rgb_tex is not None:glDeleteTextures(1,[self.rgb_tex])
            self.rgb_tex=texture(7,frame,GL_RGB8,GL_RGB,GL_UNSIGNED_BYTE);self.rgb_size=(w,h)
        else:
            glBindTexture(GL_TEXTURE_2D,self.rgb_tex);glPixelStorei(GL_UNPACK_ALIGNMENT,1)
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,w,h,GL_RGB,GL_UNSIGNED_BYTE,frame.ctypes.data_as(C.c_void_p))
        glBindFramebuffer(GL_FRAMEBUFFER,target);glViewport(0,0,w,h);glBindVertexArray(self.vao)
        glUseProgram(self.rgb);self.integer(self.rgb,'flip_y',physical);glDrawArrays(GL_TRIANGLES,0,3)
        self.last_frame=frame

    def chromatic_target(self,width,height):
        if self.ca_size!=(width,height):
            if self.ca_fb is not None:glDeleteFramebuffers(1,[self.ca_fb]);glDeleteTextures(1,[self.ca_tex])
            self.ca_tex=texture(12,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,width,height)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
            self.ca_fb=framebuffer(self.ca_tex);self.ca_size=(width,height)
        return self.ca_fb

    def compensate(self,strength,target=0):
        if self.ca_program is None:self.ca_program=program(CHROMATIC);self.locations[self.ca_program]={}
        p=self.ca_program;glUseProgram(p);glBindVertexArray(self.vao)
        glActiveTexture(GL_TEXTURE12);glBindTexture(GL_TEXTURE_2D,self.ca_tex);self.integer(p,'source',12)
        glUniform2f(self.location(p,'image_size'),*self.ca_size);self.real(p,'strength',strength)
        glBindFramebuffer(GL_FRAMEBUFFER,target);glViewport(0,0,*self.ca_size);glDrawArrays(GL_TRIANGLES,0,3)

    def draw_track(self,state,target=0):
        if self.track_renderer is None:
            from .gpu_track import TrackRenderer
            self.track_renderer=TrackRenderer()
        self.track_renderer.draw(state,target,self.ow,self.oh);glBindVertexArray(self.vao)

    def reduction_target(self,w,h,color):
        key=(w,h,bool(color))
        if key not in self.downsample_targets:
            tex=texture(6,None,GL_RGBA8 if color else GL_R8,GL_RGBA if color else GL_RED,GL_UNSIGNED_BYTE,w,h)
            self.downsample_targets[key]=(tex,framebuffer(tex))
        return self.downsample_targets[key][1]

    def snapshot_output(self,output_framebuffer=0,width=None,height=None):
        """Retain a half-resolution paused frame before swap, without readback."""
        width=width or self.ow;height=height or self.oh;w,h=(width+1)//2,(height+1)//2
        target=self.reduction_target(w,h,self.color_capture)
        glBindFramebuffer(GL_READ_FRAMEBUFFER,output_framebuffer);glBindFramebuffer(GL_DRAW_FRAMEBUFFER,target)
        glBlitFramebuffer(0,height,width,0,0,0,w,h,GL_COLOR_BUFFER_BIT,GL_LINEAR)
        glBindFramebuffer(GL_FRAMEBUFFER,0);self.snapshot_frame=self.last_frame;self.snapshot_size=(w,h);self.snapshot_color=self.color_capture

    def capture(self,full=False,size=(960,540),asynchronous=False,output_framebuffer=None,source_renderer=None):
        if not isinstance(self.last_frame,SunFrame) and not self.color_capture:return self.last_frame
        direct_events=isinstance(self.last_frame,SunFrame) and self.last_frame.method=='events' and (self.last_frame.enabled or getattr(self.event_renderer,'supports_disabled',False))
        w,h=self.ow,self.oh
        if not full:
            ratio=min(1,size[0]/w,size[1]/h);w,h=max(1,round(w*ratio)),max(1,round(h*ratio))
        capture_shape=(w,h,self.color_capture)
        if self.capture_size!=capture_shape:
            if self.capture_fb is not None:glDeleteFramebuffers(1,[self.capture_fb]);glDeleteTextures(1,[self.capture_tex])
            self.capture_tex=texture(6,None,GL_RGBA8 if self.color_capture else GL_R8,GL_RGBA if self.color_capture else GL_RED,GL_UNSIGNED_BYTE,w,h)
            self.capture_fb=framebuffer(self.capture_tex);self.capture_size=capture_shape
        cached=not full and self.snapshot_frame is self.last_frame and self.snapshot_size is not None and self.snapshot_color==self.color_capture and w<=self.snapshot_size[0] and h<=self.snapshot_size[1]
        if source_renderer is not None:
            source_renderer.preview(self.capture_fb,w,h)
        elif output_framebuffer is not None or cached:
            # Read the completed native image before SwapWindow, while its
            # contents are defined. Successive <=2x reductions retain fine
            # detail that a single 4x bilinear reduction would skip.
            if cached:
                sw,sh=self.snapshot_size;source=self.reduction_target(sw,sh,self.color_capture);flip=False
            else:sw,sh=self.ow,self.oh;source=output_framebuffer;flip=True
            while True:
                tw,th=max(w,(sw+1)//2),max(h,(sh+1)//2)
                final=(tw,th)==(w,h)
                if final:target=self.capture_fb
                else:
                    target=self.reduction_target(tw,th,self.color_capture)
                glBindFramebuffer(GL_READ_FRAMEBUFFER,source);glBindFramebuffer(GL_DRAW_FRAMEBUFFER,target)
                glBlitFramebuffer(0,sh if flip else 0,sw,0 if flip else sh,0,0,tw,th,GL_COLOR_BUFFER_BIT,GL_LINEAR)
                if final:break
                source=target;sw,sh=tw,th;flip=False
            glBindFramebuffer(GL_FRAMEBUFFER,self.capture_fb)
        elif direct_events:self.event_renderer.preview(self.capture_fb,w,h)
        else:
            glBindFramebuffer(GL_FRAMEBUFFER,self.capture_fb);glViewport(0,0,w,h);glUseProgram(self.preview)
            glUniform2f(self.location(self.preview,'target_size'),w,h);glDrawArrays(GL_TRIANGLES,0,3)
        glPixelStorei(GL_PACK_ALIGNMENT,1)
        if asynchronous:
            buffer=glGenBuffers(1);glBindBuffer(GL_PIXEL_PACK_BUFFER,buffer)
            channels=4 if self.color_capture else 1
            glBufferData(GL_PIXEL_PACK_BUFFER,w*h*channels,None,GL_STREAM_READ)
            glReadPixels(0,0,w,h,GL_RGBA if self.color_capture else GL_RED,GL_UNSIGNED_BYTE,C.c_void_p(0))
            fence=glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);glFlush()
            glBindBuffer(GL_PIXEL_PACK_BUFFER,0)
            result=PendingCapture(buffer,fence,(h,w,4) if self.color_capture else (h,w))
            self.pending_captures.append(result)
        else:
            result=np.empty((h,w,4) if self.color_capture else (h,w),np.uint8)
            glReadPixels(0,0,w,h,GL_RGBA if self.color_capture else GL_RED,GL_UNSIGNED_BYTE,result.ctypes.data_as(C.c_void_p))
        glBindFramebuffer(GL_FRAMEBUFFER,0)
        return result

    def poll_captures(self):
        for capture in list(self.pending_captures):
            status=glClientWaitSync(capture.fence,0,0)
            if status not in (GL_ALREADY_SIGNALED,GL_CONDITION_SATISFIED,GL_WAIT_FAILED):continue
            try:
                if status==GL_WAIT_FAILED:raise RuntimeError('GPU preview fence failed')
                count=int(np.prod(capture.shape));glBindBuffer(GL_PIXEL_PACK_BUFFER,capture.buffer)
                pointer=glMapBufferRange(GL_PIXEL_PACK_BUFFER,0,count,GL_MAP_READ_BIT)
                if not pointer:raise RuntimeError('Cannot map the GPU preview buffer')
                try:capture.image=np.frombuffer(C.string_at(pointer,count),np.uint8).reshape(capture.shape)
                finally:glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
            except Exception as exc:capture.error=exc
            finally:
                glBindBuffer(GL_PIXEL_PACK_BUFFER,0);glDeleteSync(capture.fence)
                glDeleteBuffers(1,[capture.buffer]);capture.ready=True
                self.pending_captures.remove(capture)

    def release_scene(self):
        self.snapshot_frame=None
        if self.event_renderer is not None:self.event_renderer.release()
        if self.precise is not None:self.precise.release()
        if self.reference is not None:self.reference.release()
        if self.candidates is not None:glDeleteBuffers(1,[self.candidates]);self.candidates=None
        if self.filtered_fb is not None:
            glDeleteFramebuffers(1,[self.filtered_fb]);glDeleteTextures(1,[self.filtered_tex])
            self.filtered_fb=self.filtered_tex=None
        if self.light_fb is not None:glDeleteFramebuffers(1,[self.light_fb]);self.light_fb=None
        if self.textures:glDeleteTextures(len(self.textures),self.textures);self.textures=[]
        self.scene_key=None

    def close(self):
        if self.layer_renderer is not None:self.layer_renderer.close()
        for tex,fb in self.downsample_targets.values():glDeleteFramebuffers(1,[fb]);glDeleteTextures(1,[tex])
        self.downsample_targets.clear()
        for capture in self.pending_captures:
            capture.error=RuntimeError('Display was reopened');capture.ready=True
            glDeleteSync(capture.fence);glDeleteBuffers(1,[capture.buffer])
        self.pending_captures=[]
        self.release_scene()
        if self.track_renderer is not None:self.track_renderer.close()
        if self.ca_program is not None:glDeleteProgram(self.ca_program)
        if self.ca_fb is not None:glDeleteFramebuffers(1,[self.ca_fb]);glDeleteTextures(1,[self.ca_tex])
        if self.reference is not None:self.reference.close()
        if self.precise is not None:self.precise.close()
        if self.event_renderer is not None:self.event_renderer.close()
        if self.rgb_tex is not None:glDeleteTextures(1,[self.rgb_tex])
        if self.capture_fb is not None:glDeleteFramebuffers(1,[self.capture_fb]);glDeleteTextures(1,[self.capture_tex])
        for p in [self.scan,self.refine,self.shade,self.stabilize,self.gather,self.rgb,self.preview]:glDeleteProgram(p)
        glDeleteVertexArrays(1,[self.vao])


class PendingCapture:
    def __init__(self,buffer,fence,shape):
        self.buffer=buffer;self.fence=fence;self.shape=shape
        self.ready=False;self.image=None;self.error=None
