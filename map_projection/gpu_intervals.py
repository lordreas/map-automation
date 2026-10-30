"""Static projector-space interval field: time changes never upload visibility.

Two uint32 words hold an octahedral 11+11-bit normal and a single interval
(21-bit timestamps, <=41.2 ms playback quantization). Rare multiple intervals
use an auxiliary table with the original millisecond timestamps unchanged.
Calibration changes rebuild this small GPU field, never the terrain/day cache.
"""
from datetime import datetime,timezone
import os
import numpy as np
from OpenGL.GLES3 import *
from OpenGL.GLES3.VERSION.GLES3_3_1 import glDispatchCompute,glMemoryBarrier,glBindImageTexture

SENTINEL=(1<<21)-1
TICKS=SENTINEL-1
NORMAL_MASK=(1<<22)-1
DIRECT=os.environ.get('MAP_PROJECTION_INTERVAL_DIRECT')=='1'
COMPUTE=DIRECT and os.environ.get('MAP_PROJECTION_INTERVAL_COMPUTE')=='1'
VISIBILITY='''#version 310 es
precision highp float;precision highp int;
layout(local_size_x=64) in;
layout(std430,binding=3) readonly buffer Headers{uint heads[];};
layout(r32ui,binding=1) writeonly uniform highp uimage2D visibility;
uniform highp usampler2D extra;uniform uint milliseconds;uniform uint count;uniform int width;
uvec4 entry(uint offset){return texelFetch(extra,ivec2(int(offset)%width,int(offset)/width),0);}
void main(){uint id=gl_GlobalInvocationID.x;if(id>=count)return;uint head=heads[id];uint n=entry(head).x;bool lit=false;
for(uint j=0u;j<n;j++){uvec4 pair=entry(head+1u+j/2u);uvec2 range=(j&1u)==0u?pair.xy:pair.zw;
if(milliseconds>=range.x&&milliseconds<range.y){lit=true;break;}}
imageStore(visibility,ivec2(int(head)%width,int(head)/width),uvec4(uint(lit)));}
'''
COMMON='''#version 300 es
precision highp float;precision highp int;
uniform highp usampler2D field;uniform highp usampler2D extra;
uniform uint tick;uniform uint milliseconds;uniform int flip_y;
uniform float ambient;uniform vec3 sun;
out vec4 color;
uvec4 exceptional(uint offset){return texelFetch(extra,ivec2(int(offset&1023u),int(offset>>10)),0);}
float light(ivec2 pixel){
 if(sun.z<=0.)return 0.;
 uvec2 word=texelFetch(field,pixel,0).rg;
 uint begin=(word.x>>22)|((word.y&2047u)<<10);uint end=word.y>>11;
 if(begin==2097151u&&end==2097151u)return 0.;
 bool lit=false;
 #if HANDLE_EXCEPTIONS
 if(begin==2097151u){
  uint count=exceptional(end).x;
  for(uint i=0u;i<count;i++){
   uvec4 entry=exceptional(end+1u+i/2u);uvec2 range=(i&1u)==0u?entry.xy:entry.zw;
   if(milliseconds>=range.x&&milliseconds<range.y){lit=true;break;}
  }
 }else
 #endif
 lit=tick>=begin&&tick<end;
 vec2 xy=vec2(word.x&2047u,(word.x>>11)&2047u)*(2./2047.)-1.;
 vec3 normal=vec3(xy,1.-abs(xy.x)-abs(xy.y));
 if(normal.z<0.)normal.xy=(1.-abs(normal.yx))*sign(normal.xy);
 float v=ambient+(1.-ambient)*float(lit)*max(0.,dot(normalize(normal),sun));return v;
}
'''
if DIRECT:
    COMMON='''#version 300 es
precision highp float;precision highp int;
uniform sampler2D normal_data;uniform highp sampler2D time_range;
uniform highp usampler2D extra;uniform uint milliseconds;uniform int flip_y;
uniform highp usampler2D exception_visibility;
uniform float ambient;uniform vec3 sun;out vec4 color;
uvec4 exceptional(uint offset){return texelFetch(extra,ivec2(int(offset&1023u),int(offset>>10)),0);}
float light(ivec2 pixel){
 if(sun.z<=0.)return 0.;vec4 normal=texelFetch(normal_data,pixel,0);if(normal.a==0.)return 0.;
 vec2 range=texelFetch(time_range,pixel,0).rg;bool lit=float(milliseconds)>=range.x&&float(milliseconds)<range.y;
 #if HANDLE_EXCEPTIONS
 if(range.x<0.){
  uint offset=uint(-range.x-1.);uint count=exceptional(offset).x;lit=false;
  #if COMPUTE_VISIBILITY
  lit=texelFetch(exception_visibility,ivec2(int(offset&1023u),int(offset>>10)),0).r!=0u;
  #else
  for(uint i=0u;i<count;i++){
   uvec4 entry=exceptional(offset+1u+i/2u);uvec2 pair=(i&1u)==0u?entry.xy:entry.zw;
   if(milliseconds>=pair.x&&milliseconds<pair.y){lit=true;break;}
  }
  #endif
 }
 #endif
 return ambient+(1.-ambient)*float(lit)*max(0.,dot(normal.rgb*2.-1.,sun));
}
'''
COMMON=COMMON.replace('COMPUTE_VISIBILITY','1' if COMPUTE else '0')
GATHER=COMMON.replace('HANDLE_EXCEPTIONS','1' if COMPUTE else '0')+'''
void main(){ivec2 p=ivec2(gl_FragCoord.xy);if(flip_y!=0)p.y=textureSize(field,0).y-1-p.y;float v=light(p);color=vec4(v,v,v,1.);}
'''
POINT=COMMON.replace('HANDLE_EXCEPTIONS','1')+'''
void main(){ivec2 p=ivec2(gl_FragCoord.xy);if(flip_y!=0)p.y=textureSize(field,0).y-1-p.y;float v=light(p);color=vec4(v,v,v,1.);}
'''
POINT_VERTEX='''#version 300 es
precision highp float;precision highp int;
layout(location=0) in uvec2 pixel;uniform vec2 output_size;uniform int flip_y;
void main(){vec2 p=vec2(pixel);if(flip_y!=0)p.y=output_size.y-1.-p.y;
gl_Position=vec4(((p+.5)/output_size)*2.-1.,0.,1.);gl_PointSize=1.;}
'''
if os.environ.get('MAP_PROJECTION_INTERVAL_RGBA16')=='1':
    def rgba16(source):
        return source.replace('uvec2 word=texelFetch(field,pixel,0).rg;', 'uvec4 raw=texelFetch(field,pixel,0);uvec2 word=uvec2(raw.x|(raw.y<<16),raw.z|(raw.w<<16));')
    GATHER=rgba16(GATHER);POINT=rgba16(POINT);COMMON=rgba16(COMMON)
if os.environ.get('MAP_PROJECTION_INTERVAL_HALF')=='1':
    GATHER=GATHER.replace('precision highp float;','precision mediump float;')
    POINT=POINT.replace('precision highp float;','precision mediump float;')
if os.environ.get('MAP_PROJECTION_INTERVAL_FLAT')=='1':
    GATHER=GATHER.replace('dot(normalize(normal),sun)','1.')
PREVIEW=COMMON.replace('HANDLE_EXCEPTIONS','1')+'''
uniform vec2 target_size;
void main(){vec2 scale=vec2(textureSize(field,0))/target_size;vec2 origin=floor(gl_FragCoord.xy)*scale;float sum=0.;
 for(int y=0;y<SY;y++)for(int x=0;x<SX;x++)sum+=light(ivec2(origin+(vec2(x,y)+.5)*scale/vec2(float(SX),float(SY))));
 float v=sum/float(SX*SY);color=vec4(v,v,v,1.);}
'''
if DIRECT:
    GATHER=GATHER.replace('textureSize(field,0)','textureSize(normal_data,0)')
    POINT=POINT.replace('textureSize(field,0)','textureSize(normal_data,0)')
    PREVIEW=PREVIEW.replace('textureSize(field,0)','textureSize(normal_data,0)')

def ticks(milliseconds):return np.rint(np.asarray(milliseconds,np.float64)*(TICKS/86400000.)).astype(np.uint32)

def pack(scene):
    """Pack fixed data once; no event sorting, CPU masks or copies per frame."""
    e=scene.events;h,w=scene.z.shape;n=h*w
    cells=np.empty((n,3 if DIRECT else 2),np.uint32);extra=[];extra_size=0;max_normal_error=0.
    for base in range(0,n,65536):
        end=min(n,base+65536);normal=np.asarray(scene.normals).reshape(-1,3)[base:end]
        if DIRECT:
            encoded=np.clip(np.rint((normal*.5+.5)*1023),0,1023).astype(np.uint32)
            codes=encoded[:,0]|(encoded[:,1]<<10)|(encoded[:,2]<<20)|np.uint32(0xc0000000)
        else:
            xy=normal[:,:2]/np.maximum(np.abs(normal).sum(axis=1,keepdims=True),1e-30)
            encoded=np.clip(np.rint((xy*.5+.5)*2047),0,2047).astype(np.uint32)
            codes=encoded[:,0]|(encoded[:,1]<<11)
        encode_time=(lambda v:np.asarray(v,np.uint32)) if DIRECT else ticks
        starts=np.zeros(end-base,np.uint32);ends=np.zeros_like(starts)
        counts=np.diff(e.offsets[base:end+1]);initial=np.asarray(e.initial[base:end],bool)
        ends[(counts==0)&initial]=86400000 if DIRECT else SENTINEL
        two=np.flatnonzero((counts==2)&~initial)
        starts[two]=encode_time(e.times[e.offsets[base+two]])
        ends[two]=encode_time(e.times[e.offsets[base+two]+1])
        one=np.flatnonzero(counts==1);offset=e.offsets[base+one]
        starts[one]=np.where(initial[one],0,encode_time(e.times[offset]))
        ends[one]=np.where(initial[one],encode_time(e.times[offset]),86400000 if DIRECT else SENTINEL)
        unusual=np.flatnonzero((counts>2)|((counts==2)&initial))
        for i in unusual:
            cell=base+int(i);state=bool(initial[i]);previous=0;intervals=[]
            for t in e.times[e.offsets[cell]:e.offsets[cell+1]]:
                if state:intervals.append((previous,int(t)))
                previous=int(t);state=not state
            if state:intervals.append((previous,86400000))
            starts[i]=SENTINEL;ends[i]=extra_size
            block=np.zeros((1+(len(intervals)+1)//2,4),np.uint32);block[0,0]=len(intervals)
            for j,(a,b) in enumerate(intervals):block[1+j//2,(j%2)*2:(j%2)*2+2]=(a,b)
            extra.append(block);extra_size+=len(block)
        if extra_size>=SENTINEL:raise ValueError('Too many exceptional intervals for packed GPU field')
        if DIRECT:
            cells[base:end,0]=codes
            ranges=np.column_stack((starts,ends)).astype(np.float32);ranges[unusual,0]=-(ends[unusual].astype(np.float32)+1);ranges[unusual,1]=0
            cells[base:end,1:]=ranges.view(np.uint32)
            restored=encoded.astype(np.float32)*(2/1023)-1
        else:
            cells[base:end,0]=codes|((starts&1023)<<22)
            cells[base:end,1]=(starts>>10)|(ends<<11)
            decoded=encoded.astype(np.float32)*(2/2047)-1
            restored=np.column_stack((decoded,1-np.abs(decoded).sum(axis=1)));restored/=np.linalg.norm(restored,axis=1,keepdims=True)
        max_normal_error=max(max_normal_error,float(np.linalg.norm(restored-normal,axis=1).max(initial=0)))
    aux=np.concatenate(extra,axis=0) if extra else np.zeros((1,4),np.uint32)
    uv=scene.uv;oh,ow=uv.shape[:2];output=np.empty((oh,ow,cells.shape[1]),np.uint32)
    invalid=np.zeros(3,np.uint32) if DIRECT else np.array([np.uint32(1023<<22),np.uint32(0xffffffff)],np.uint32)
    for row in range(0,oh,64):
        block=uv[row:row+64];valid=block[:,:,0]!=65535
        ix=np.minimum(block[:,:,1],h-1).astype(np.int64)*w+np.minimum(block[:,:,0],w-1)
        output[row:row+64]=cells[ix];output[row:row+64][~valid]=invalid
    return output,aux,{'exception_texels':len(aux),'max_normal_vector_error':max_normal_error,'timestamp_quantization_ms':8 if DIRECT else 86400000/TICKS}

class IntervalRenderer:
    def __init__(self,r):
        from .gpu import program,compute
        self.r=r;self.program=program(GATHER);self.point_program=program(POINT,POINT_VERTEX)
        r.locations[self.program]={};r.locations[self.point_program]={};self.key=None;self.textures=[];self.preview_programs={};self.point_vao=self.point_buffer=None;self.point_count=0
        self.update=compute(VISIBILITY) if COMPUTE else None
        if self.update:r.locations[self.update]={}
        self.header_buffer=None;self.header_count=0
    def prepare(self,frame):
        from .gpu import texture
        key=(str(frame.scene.events.path),frame.scene.key)
        if self.key==key:return
        self.release();field,aux,self.stats=pack(frame.scene);limit=int(glGetIntegerv(GL_MAX_TEXTURE_SIZE));width=min(1024,limit);height=(len(aux)+width-1)//width
        if height>limit:raise ValueError('Exceptional interval table exceeds GPU texture limit')
        data=np.zeros((height*width,4),np.uint32);data[:len(aux)]=aux
        if DIRECT:
            self.textures=[texture(11,np.ascontiguousarray(field[:,:,0]),GL_RGB10_A2,GL_RGBA,GL_UNSIGNED_INT_2_10_10_10_REV)]
        elif os.environ.get('MAP_PROJECTION_INTERVAL_RGBA16')=='1':
            self.textures=[texture(11,field.view(np.uint16).reshape(*field.shape[:2],4),GL_RGBA16UI,GL_RGBA_INTEGER,GL_UNSIGNED_SHORT)]
        else:self.textures=[texture(11,field,GL_RG32UI,GL_RG_INTEGER,GL_UNSIGNED_INT)]
        self.textures.append(texture(12,data.reshape(height,width,4),GL_RGBA32UI,GL_RGBA_INTEGER,GL_UNSIGNED_INT))
        if DIRECT:self.textures.append(texture(13,np.ascontiguousarray(field[:,:,1:]).view(np.float32),GL_RG32F,GL_RG,GL_FLOAT))
        if COMPUTE:
            import ctypes as C
            self.textures.append(texture(14,np.zeros((height,width),np.uint32),GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT,immutable=True))
            headers=[];position=0
            while position<len(aux) and aux[position,0]:headers.append(position);position+=1+(int(aux[position,0])+1)//2
            values=np.array(headers,np.uint32);self.header_count=len(values);self.header_buffer=glGenBuffers(1)
            glBindBuffer(0x90D2,self.header_buffer);glBufferData(0x90D2,values.nbytes,values.ctypes.data_as(C.c_void_p),GL_STATIC_DRAW)
        positions=[]
        for row in range(0,field.shape[0],64):
            block=field[row:row+64];begin=(block[:,:,0]>>22)|((block[:,:,1]&2047)<<10);end=block[:,:,1]>>11
            mask=block[:,:,1:].copy().view(np.float32)[:,:,0]<0 if DIRECT else (begin==SENTINEL)&(end!=SENTINEL)
            yy,xx=np.nonzero(mask)
            if len(yy):positions.append(np.column_stack((xx,yy+row)).astype(np.uint16))
        if positions and not COMPUTE:
            import ctypes as C
            vertices=np.concatenate(positions);self.point_count=len(vertices);self.point_vao=glGenVertexArrays(1);self.point_buffer=glGenBuffers(1)
            glBindVertexArray(self.point_vao);glBindBuffer(GL_ARRAY_BUFFER,self.point_buffer);glBufferData(GL_ARRAY_BUFFER,vertices.nbytes,vertices.ctypes.data_as(C.c_void_p),GL_STATIC_DRAW)
            glVertexAttribIPointer(0,2,GL_UNSIGNED_SHORT,4,C.c_void_p(0));glEnableVertexAttribArray(0);glBindVertexArray(self.r.vao)
        self.key=key
    def uniforms(self,p,physical):
        r=self.r;frame=self.frame
        for unit,tex in zip((11,12,13,14),self.textures):glActiveTexture(GL_TEXTURE0+unit);glBindTexture(GL_TEXTURE_2D,tex)
        r.integer(p,'field',11);r.integer(p,'extra',12);r.integer(p,'flip_y',physical);r.real(p,'ambient',frame.ambient);glUniform3f(r.location(p,'sun'),*frame.sun)
        if DIRECT:r.integer(p,'normal_data',11);r.integer(p,'time_range',13)
        if COMPUTE:r.integer(p,'exception_visibility',14)
        glUniform1ui(r.location(p,'tick'),self.tick);glUniform1ui(r.location(p,'milliseconds'),self.milliseconds)
    def draw(self,frame,target=0,physical=True):
        self.prepare(frame);self.frame=frame;e=frame.scene.events
        seconds=(datetime.fromisoformat(frame.time).astimezone(timezone.utc)-datetime.fromisoformat(e.meta['start'])).total_seconds()
        if frame.sun[2]>0 and not 0<=seconds<86400:raise ValueError('Event cache date mismatch')
        seconds=max(0,min(86400,seconds));self.tick=int(round(seconds*TICKS/86400));self.milliseconds=int(round(seconds*1000))
        if COMPUTE and self.header_count and frame.sun[2]>0:
            p=self.update;glUseProgram(p);self.uniforms(p,False);self.r.integer(p,'width',1024)
            glUniform1ui(self.r.location(p,'count'),self.header_count);glBindBufferBase(0x90D2,3,self.header_buffer)
            glBindImageTexture(1,self.textures[3],0,False,0,GL_WRITE_ONLY,GL_R32UI);glDispatchCompute((self.header_count+63)//64,1,1);glMemoryBarrier(0x28)
        r=self.r;p=self.program;glUseProgram(p);self.uniforms(p,physical);glBindFramebuffer(GL_FRAMEBUFFER,target);glViewport(0,0,r.ow,r.oh);glDrawArrays(GL_TRIANGLES,0,3)
        if self.point_count and frame.sun[2]>0 and os.environ.get('MAP_PROJECTION_INTERVAL_NO_POINTS')!='1':
            p=self.point_program;glUseProgram(p);self.uniforms(p,physical);glUniform2f(r.location(p,'output_size'),r.ow,r.oh)
            glBindVertexArray(self.point_vao);glDrawArrays(GL_POINTS,0,self.point_count);glBindVertexArray(r.vao)
    def preview(self,target,width,height):
        from .gpu import program
        r=self.r;key=(max(1,min(8,int(np.ceil(r.ow/width)))),max(1,min(8,int(np.ceil(r.oh/height)))))
        if key not in self.preview_programs:
            p=program(PREVIEW.replace('SX',str(key[0])).replace('SY',str(key[1])));self.preview_programs[key]=p;r.locations[p]={}
        p=self.preview_programs[key];glUseProgram(p);self.uniforms(p,False);glUniform2f(r.location(p,'target_size'),width,height)
        glBindFramebuffer(GL_FRAMEBUFFER,target);glViewport(0,0,width,height);glDrawArrays(GL_TRIANGLES,0,3)
    def release(self):
        if self.textures:glDeleteTextures(len(self.textures),self.textures)
        self.textures=[];self.key=None
        if self.point_buffer is not None:glDeleteBuffers(1,[self.point_buffer]);self.point_buffer=None
        if self.point_vao is not None:glDeleteVertexArrays(1,[self.point_vao]);self.point_vao=None
        if self.header_buffer is not None:glDeleteBuffers(1,[self.header_buffer]);self.header_buffer=None
        self.point_count=0
    def close(self):
        self.release();glDeleteProgram(self.program);glDeleteProgram(self.point_program)
        if self.update:glDeleteProgram(self.update)
        for p in self.preview_programs.values():glDeleteProgram(p)
        self.preview_programs.clear()
