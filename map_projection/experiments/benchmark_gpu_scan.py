"""GPU sun-aligned parallel prefix maxima; an alternative to independent rays.

A sheared terrain grid makes the sunlight rays parallel to buffer rows. Each
workgroup scans a complete ray in parallel. This is a different interpolation
approximation from both the production sweep and the column-DDA ray tracer.
"""
import argparse
import ctypes as C
import json
import time
from pathlib import Path
import numpy as np
if __package__:
    from .gpu_context import *
else:
    from gpu_context import *
from OpenGL.GLES3.VERSION.GLES3_3_1 import glDispatchCompute,glMemoryBarrier,glBindImageTexture
from map_projection.native import Native
from map_projection.solar import position

COMPUTE='''#version 310 es
precision highp float;precision highp int;
layout(local_size_x=256) in;
layout(std430,binding=0) writeonly buffer Horizon{float horizon[];};
uniform sampler2D terrain;
uniform int count;uniform int across;uniform int start;uniform int step_axis;
uniform float offset;uniform float rise;uniform float qmin;
shared float maxima[256];
void main(){
 int lane=int(gl_LocalInvocationID.x);int q=int(gl_WorkGroupID.x);
 float a0=float(q)+qmin;float values[8];float m=-1.e20;
 for(int i=0;i<8;i++){
  // Exclusive prefix: the receiver's resampled height must not occlude itself.
  values[i]=m;
  int k=lane*8+i;float a=a0-offset*float(k);float z=-1.e20;
  if(k<count&&a>=0.&&a<=float(across-1)){
   int lo=int(floor(a)),hi=min(lo+1,across-1);int u=start+step_axis*k;
   z=mix(texelFetch(terrain,ivec2(u,lo),0).r,texelFetch(terrain,ivec2(u,hi),0).r,fract(a));
  }
  m=max(m,z+rise*float(k));
 }
 maxima[lane]=m;barrier();
 for(int jump=1;jump<256;jump*=2){
  float previous=lane>=jump?maxima[lane-jump]:-1.e20;
  barrier();maxima[lane]=max(maxima[lane],previous);barrier();
 }
 float previous=lane>0?maxima[lane-1]:-1.e20;
 for(int i=0;i<8;i++){
  int k=lane*8+i;
  if(k<count)horizon[q*count+k]=max(previous,values[i])-rise*float(k);
 }
}
'''
SERIAL='''#version 310 es
precision highp float;precision highp int;
layout(local_size_x=64) in;
layout(std430,binding=0) writeonly buffer Horizon{float horizon[];};
uniform sampler2D terrain;
uniform int count;uniform int across;uniform int start;uniform int step_axis;uniform int lines;
uniform float offset;uniform float rise;uniform float qmin;
void main(){
 int q=int(gl_GlobalInvocationID.x);if(q>=lines)return;
 float a0=float(q)+qmin;float m=-1.e20;
 for(int k=0;k<count;k++){
  horizon[k*lines+q]=m-rise*float(k);
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
layout(std430,binding=0) readonly buffer Horizon{float horizon[];};
uniform sampler2D terrain;uniform sampler2D normals;
uniform int horizontal;uniform int positive;uniform int count;
uniform int lines;uniform int serial;
uniform float offset;uniform float qmin;uniform float ambient;uniform float bias;
uniform int mask_only;uniform vec3 sun;
out vec4 color;
void main(){
 ivec2 p=ivec2(gl_FragCoord.xy);int axis=horizontal!=0?p.x:p.y;
 int k=positive!=0?count-1-axis:axis;float across=float(horizontal!=0?p.y:p.x);
 float q=across+offset*float(k)-qmin;int lo=int(floor(q));
 int i0=serial!=0?k*lines+lo:lo*count+k;
 int i1=serial!=0?i0+1:i0+count;
 float h=mix(horizon[i0],horizon[i1],fract(q));
 float z=texelFetch(terrain,p,0).r;float lit=h>z+bias?0.:1.;
 float diffuse=max(0.,dot(texelFetch(normals,p,0).rgb,sun))*lit;
 float v=mask_only!=0?lit:ambient+(1.-ambient)*diffuse;
 color=sun.z>0.?vec4(v,v,v,1.):vec4(0.,0.,0.,1.);
}
'''
GATHER='''#version 300 es
precision highp float;precision highp int;
uniform highp usampler2D lookup;uniform sampler2D light;
out vec4 color;
void main(){uvec2 p=texelFetch(lookup,ivec2(gl_FragCoord.xy),0).rg;
 float v=p.r==65535u?0.:texelFetch(light,ivec2(p),0).r;color=vec4(v,v,v,1.);}
'''


def compute_program(serial,texture_output):
    source=SERIAL if serial else COMPUTE
    if texture_output:
        source=source.replace('layout(std430,binding=0) writeonly buffer Horizon{float horizon[];};',
            'layout(r32f,binding=0) writeonly uniform highp image2D horizon_image;')
        source=source.replace('horizon[k*lines+q]=m-rise*float(k);',
            'imageStore(horizon_image,ivec2(q,k),vec4(m-rise*float(k)));')
        source=source.replace('horizon[q*count+k]=max(previous,values[i])-rise*float(k);',
            'imageStore(horizon_image,ivec2(q,k),vec4(max(previous,values[i])-rise*float(k)));')
    shader=glCreateShader(0x91B9);glShaderSource(shader,source);glCompileShader(shader)
    if not glGetShaderiv(shader,GL_COMPILE_STATUS):raise RuntimeError(glGetShaderInfoLog(shader))
    result=glCreateProgram();glAttachShader(result,shader);glLinkProgram(result)
    if not glGetProgramiv(result,GL_LINK_STATUS):raise RuntimeError(glGetProgramInfoLog(result))
    glDeleteShader(shader);return result


def cpu_shear(z,horizontal,start,step,offset,rise,qmin,lines):
    source=z if horizontal else z.T
    across,count=source.shape;q=np.arange(lines)+qmin
    out=np.empty((lines,count),np.float32);running=np.full(lines,-1e20)
    for k in range(count):
        a=q-offset*k;lo=np.clip(np.floor(a).astype(int),0,across-1);hi=np.minimum(lo+1,across-1)
        height=source[lo,start+step*k]*(1-(a-np.floor(a)))+source[hi,start+step*k]*(a-np.floor(a))
        height=np.where((a<0)|(a>across-1),-1e20,height)
        out[:,k]=running-rise
        running=np.maximum(height,running-rise)
    axis=np.arange(count);k=count-1-axis if step<0 else axis
    a=np.arange(across)[:,None];q=a+offset*k[None,:]-qmin;lo=np.floor(q).astype(int);mix=q-lo
    h=out[lo,k[None,:]]*(1-mix)+out[lo+1,k[None,:]]*mix
    return np.ascontiguousarray(h if horizontal else h.T,dtype=np.float32)


def run(data,width,trials,serial,texture_output):
    source=data['z'];original_lookup=data['lookup'];oldh,oldw=source.shape;height=round(width*oldh/oldw)
    yi=np.minimum(((np.arange(height)+.5)*oldh/height).astype(int),oldh-1)
    xi=np.minimum(((np.arange(width)+.5)*oldw/width).astype(int),oldw-1)
    z=np.ascontiguousarray(source[np.ix_(yi,xi)]);gh,gw=z.shape;oh,ow=original_lookup.shape;ts=data['terrain_size']
    gx=np.maximum(original_lookup,0)%oldw*width//oldw;gy=np.maximum(original_lookup,0)//oldw*height//oldh
    lookup=np.where(original_lookup<0,-1,gy*width+gx).astype(np.int32)
    dy,dx=np.gradient(z,ts[1]/gh,ts[0]/gw);normals=np.stack((-dx,dy,np.ones_like(z)),axis=-1)
    normals=np.ascontiguousarray(normals/np.linalg.norm(normals,axis=-1,keepdims=True),dtype=np.float32)
    zt=texture(0,z,GL_R32F,GL_RED,GL_FLOAT);tt=texture(1,np.ascontiguousarray(z.T),GL_R32F,GL_RED,GL_FLOAT)
    nt=texture(2,normals,GL_RGB32F,GL_RGB,GL_FLOAT)
    lt=texture(3,None,GL_R8,GL_RED,GL_UNSIGNED_BYTE,gw,gh);lfb=framebuffer(lt)
    uv=np.stack((np.maximum(lookup,0)%gw,np.maximum(lookup,0)//gw),axis=-1).astype(np.uint16);uv[lookup<0]=65535
    ut=texture(4,uv,GL_RG16UI,GL_RG_INTEGER,GL_UNSIGNED_SHORT)
    ot=texture(5,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,ow,oh);ofb=framebuffer(ot)
    buffer=None
    if not texture_output:
        buffer=glGenBuffers(1);glBindBuffer(0x90D2,buffer)
        glBufferData(0x90D2,4*max(gw,gh)*(gw+gh+3),None,GL_DYNAMIC_DRAW);glBindBufferBase(0x90D2,0,buffer)
    ht=None;shade_source=SHADE
    if texture_output:
        ht=texture(6,None,GL_R32F,GL_RED,GL_FLOAT,gw+gh+3,max(gw,gh),immutable=True)
        glBindImageTexture(0,ht,0,False,0,GL_WRITE_ONLY,GL_R32F)
        shade_source=shade_source.replace('layout(std430,binding=0) readonly buffer Horizon{float horizon[];};',
            'uniform sampler2D horizon_texture;')
        shade_source=shade_source.replace('float h=mix(horizon[i0],horizon[i1],fract(q));',
            'float h=mix(texelFetch(horizon_texture,ivec2(lo,k),0).r,texelFetch(horizon_texture,ivec2(lo+1,k),0).r,fract(q));')
    cp,sp,gp=compute_program(serial,texture_output),program(shade_source),program(GATHER)
    glUseProgram(sp)
    for name,unit in [('terrain',0),('normals',2)]:glUniform1i(glGetUniformLocation(sp,name),unit)
    for name in ['ambient','bias']:glUniform1f(glGetUniformLocation(sp,name),float(data[name]))
    glUniform1i(glGetUniformLocation(sp,'serial'),int(serial))
    glUniform1i(glGetUniformLocation(sp,'horizon_texture'),6)
    glUseProgram(gp)
    for name,unit in [('lookup',4),('light',3)]:glUniform1i(glGetUniformLocation(gp,name),unit)
    native=Native();timings=[];validation=[]
    for hour in ['09','12','16','18']:
        sun,_=position(f'2026-10-24T{hour}:00:00+02:00',45.95,10.65)
        sx,sy=sun[0]/(ts[0]/gw),-sun[1]/(ts[1]/gh);horizontal=abs(sx)>=abs(sy)
        dominant=sx if horizontal else sy;count=gw if horizontal else gh;across=gh if horizontal else gw
        step=-1 if dominant>0 else 1;start=count-1 if dominant>0 else 0
        offset=(sy if horizontal else sx)/abs(dominant);rise=sun[2]/abs(dominant)
        qmin=np.floor(min(0,offset*(count-1)));lines=int(np.ceil(across+abs(offset)*(count-1)))+2
        glUseProgram(cp);glUniform1i(glGetUniformLocation(cp,'terrain'),0 if horizontal else 1)
        for name,value in [('count',count),('across',across),('start',start),('step_axis',step),('lines',lines)]:glUniform1i(glGetUniformLocation(cp,name),value)
        for name,value in [('offset',offset),('rise',rise),('qmin',qmin)]:glUniform1f(glGetUniformLocation(cp,name),value)
        glUseProgram(sp)
        for name,value in [('horizontal',horizontal),('positive',dominant>0),('count',count),('lines',lines)]:glUniform1i(glGetUniformLocation(sp,name),int(value))
        for name,value in [('offset',offset),('qmin',qmin)]:glUniform1f(glGetUniformLocation(sp,name),value)
        glUniform3f(glGetUniformLocation(sp,'sun'),*sun);glUniform1i(glGetUniformLocation(sp,'mask_only'),0)
        first_timing=len(timings)
        for trial in range(trials+1):
            t=time.perf_counter();glUseProgram(cp);glDispatchCompute((lines+63)//64 if serial else lines,1,1);glMemoryBarrier(0x28 if texture_output else 0x2000);glFinish();a=time.perf_counter()
            glUseProgram(sp);glBindFramebuffer(GL_FRAMEBUFFER,lfb);glViewport(0,0,gw,gh);glDrawArrays(GL_TRIANGLES,0,3);glFinish();b=time.perf_counter()
            glUseProgram(gp);glBindFramebuffer(GL_FRAMEBUFFER,ofb);glViewport(0,0,ow,oh);glDrawArrays(GL_TRIANGLES,0,3);glFinish();end=time.perf_counter()
            if trial:timings.append([(a-t)*1000,(b-a)*1000,(end-b)*1000,(end-t)*1000])
        h=cpu_shear(z,horizontal,start,step,offset,rise,qmin,lines)
        glUseProgram(sp);glUniform1i(glGetUniformLocation(sp,'mask_only'),1)
        glBindFramebuffer(GL_FRAMEBUFFER,lfb);glViewport(0,0,gw,gh);glDrawArrays(GL_TRIANGLES,0,3)
        mask=np.empty((gh,gw),np.uint8);glReadPixels(0,0,gw,gh,GL_RED,GL_UNSIGNED_BYTE,mask.ctypes.data_as(C.c_void_p))
        expected=h>z+float(data['bias']);production=np.empty_like(z)
        native.lib.horizon(z,gw,gh,ts[0]/gw,ts[1]/gh,sun,production)
        current=production>z+float(data['bias'])
        mismatch=(mask==0)!=expected
        margin=np.abs(h-z-float(data['bias']))
        # Numerical shading and the full 4K mapping are checked outside timing.
        glUniform1i(glGetUniformLocation(sp,'mask_only'),0);glDrawArrays(GL_TRIANGLES,0,3)
        light=np.empty((gh,gw),np.uint8);glReadPixels(0,0,gw,gh,GL_RED,GL_UNSIGNED_BYTE,light.ctypes.data_as(C.c_void_p))
        diffuse=np.maximum(0,np.einsum('ijk,k->ij',normals,sun))*(mask!=0)
        cpu_light=np.floor(255*(float(data['ambient'])+(1-float(data['ambient']))*diffuse)+.5).astype(np.uint8)
        glBindFramebuffer(GL_FRAMEBUFFER,ofb)
        rgba=np.empty((oh,ow,4),np.uint8);glReadPixels(0,0,ow,oh,GL_RGBA,GL_UNSIGNED_BYTE,rgba.ctypes.data_as(C.c_void_p))
        gathered=np.where(lookup<0,0,light.ravel()[np.maximum(lookup,0)])
        validation.append({'hour':hour,'cpu_shear_mask_mismatches':int(np.count_nonzero(mismatch)),
            'max_oracle_mismatch_margin_m':round(float(margin[mismatch].max()),6) if mismatch.any() else 0,
            'max_lighting_byte_difference':int(np.max(np.abs(light.astype(np.int16)-cpu_light.astype(np.int16)))),
            'max_gather_byte_difference':int(np.max(np.abs(rgba[:,:,0].astype(np.int16)-gathered.astype(np.int16)))),
            'median_total_ms':round(float(np.median(np.array(timings[first_timing:])[:,-1])),3),
            'production_mask_difference_percent':round(float(np.mean((mask==0)!=current)*100),4),
            'cast_shadow_pixels':int(np.count_nonzero(mask==0))})
    result={'grid':[gw,gh],'output':[ow,oh],'method':'gpu_serial_sheared_scan' if serial else 'gpu_parallel_sheared_scan','frames':len(timings),
            'horizon_storage':'texture' if texture_output else 'ssbo',
            'medians_ms':dict(zip(['gpu_shadow','gpu_lighting','gpu_4k_gather','total'],np.median(timings,axis=0).round(3).tolist())),
            'p95_total_ms':round(float(np.percentile(np.array(timings)[:,-1],95)),3),'validation':validation}
    print(json.dumps(result),flush=True)
    glBindFramebuffer(GL_FRAMEBUFFER,0);glDeleteFramebuffers(2,[lfb,ofb]);glDeleteTextures(6,[zt,tt,nt,lt,ut,ot])
    if buffer is not None:glDeleteBuffers(1,[buffer])
    if ht is not None:glDeleteTextures(1,[ht])
    for p in [cp,sp,gp]:glDeleteProgram(p)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--grids',nargs='+',type=int,default=[512,1024,2048]);parser.add_argument('--trials',type=int,default=5)
    parser.add_argument('--serial',action='store_true',help='One GPU lane scans each sun ray; coalesced output, no workgroup barriers.')
    parser.add_argument('--texture',action='store_true',help='Store GPU horizons in a texture rather than an SSBO.')
    parser.add_argument('--output',type=Path,default=Path('gpu-scan-results.json'));args=parser.parse_args()
    egl=context();data=np.load(args.input)
    results=[run(data,width,args.trials,args.serial,args.texture) for width in args.grids]
    args.output.write_text(json.dumps({'renderer':egl[3],'results':results},indent=2)+'\n')


if __name__=='__main__':main()
