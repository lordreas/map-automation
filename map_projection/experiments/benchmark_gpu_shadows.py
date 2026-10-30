"""Fresh GPU heightfield shadow rays + 4K output, no CPU shadow uploads.

Run in a separate directory on the board with --input gpu-input.npz.
Ray geometry is piecewise-constant heightfield columns, not the STL or the
production interpolated horizon. An independent cell-by-cell CPU DDA checks it.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import statistics
import time
import numpy as np
if __package__:
    from .gpu_context import *
else:
    from gpu_context import *

SHADOW='''#version 300 es
precision highp float;precision highp int;
uniform sampler2D terrain;uniform sampler2D normals;
uniform ivec2 size;uniform int levels;uniform int method;uniform int mask_only;
uniform vec2 direction;uniform float rise;uniform float bias;uniform vec3 sun;uniform float ambient;
out vec4 color;
void main(){
 ivec2 pixel=ivec2(gl_FragCoord.xy);
 float start=texelFetch(terrain,pixel,0).r+bias;
 vec2 origin=vec2(pixel)+.5;
 float t=0.;bool blocked=false;bool completed=false;int level=method==0?0:levels;
 if(sun.z<=0.){color=vec4(0.,0.,0.,1.);return;}
 if(max(abs(direction.x),abs(direction.y))<.00001){completed=true;}
 for(int iteration=0;iteration<8192;iteration++){
  if(completed)break;
  vec2 p=origin+direction*t;
  if(any(lessThan(p,vec2(0.)))||any(greaterThanEqual(p,vec2(size)))){completed=true;break;}
  ivec2 cell=ivec2(p)>>level;
  float top=texelFetch(terrain,cell,level).r;
  float ray=start+rise*t;
  if(top>ray){
   if(level==0){blocked=true;completed=true;break;}
   level--;continue;
  }
  float span=exp2(float(level));
  vec2 boundary=(vec2(cell)+step(vec2(0.),direction))*span;
  vec2 distance=vec2(1.e25);
  if(abs(direction.x)>1.e-8)distance.x=(boundary.x-p.x)/direction.x;
  if(abs(direction.y)>1.e-8)distance.y=(boundary.y-p.y)/direction.y;
  t+=max(0.,min(distance.x,distance.y))+.001;
  if(method!=0)level=min(level+1,levels);
 }
 // A cap hit is marked magenta in RGBA targets; R8=127 identifies it in mask mode.
 if(!completed){color=vec4(.5,0.,.5,1.);return;}
 float lit=blocked?0.:1.;
 float diffuse=max(0.,dot(texelFetch(normals,pixel,0).rgb,sun))*lit;
 float value=mask_only!=0?lit:ambient+(1.-ambient)*diffuse;
 color=vec4(value,value,value,1.);
}
'''
GATHER='''#version 300 es
precision highp float;precision highp int;
uniform highp usampler2D lookup;uniform sampler2D light;
out vec4 color;
void main(){uvec2 p=texelFetch(lookup,ivec2(gl_FragCoord.xy),0).rg;
 float v=p.r==65535u?0.:texelFetch(light,ivec2(p),0).r;
 color=vec4(v,v,v,1.);}
'''


def dda(z,x,y,direction,rise,bias):
    """Slow, independent CPU oracle: visit every cell a ray crosses."""
    gh,gw=z.shape;start=float(z[y,x])+bias;t=0.;ox,oy=x+.5,y+.5
    dx,dy=map(float,direction)
    for _ in range(gw+gh+10):
        px,py=ox+dx*t,oy+dy*t
        if px<0 or py<0 or px>=gw or py>=gh:return False
        cx,cy=int(px),int(py)
        if float(z[cy,cx])>start+rise*t:return True
        tx=(cx+(dx>=0)-px)/dx if abs(dx)>1e-8 else float('inf')
        ty=(cy+(dy>=0)-py)/dy if abs(dy)>1e-8 else float('inf')
        t+=max(0.,min(tx,ty))+.001
    raise RuntimeError('CPU DDA did not terminate')


def run(z,ts,lookup,suns,bias,ambient,methods,trials,validate_count):
    gh,gw=z.shape;oh,ow=lookup.shape
    size=1<<(max(gw,gh)-1).bit_length();levels=size.bit_length()-1
    padded=np.full((size,size),-1e20,np.float32);padded[:gh,:gw]=z
    tex=texture(0,padded,GL_R32F,GL_RED,GL_FLOAT)
    pyramid=padded
    for level in range(1,levels+1):
        pyramid=np.ascontiguousarray(pyramid.reshape(pyramid.shape[0]//2,2,pyramid.shape[1]//2,2).max(axis=(1,3)))
        glTexImage2D(GL_TEXTURE_2D,level,GL_R32F,pyramid.shape[1],pyramid.shape[0],0,GL_RED,GL_FLOAT,pyramid.ctypes.data_as(C.c_void_p))
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,levels)
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST_MIPMAP_NEAREST)
    dy,dx=np.gradient(z,ts[1]/gh,ts[0]/gw)
    normals=np.stack((-dx,dy,np.ones_like(z)),axis=-1)
    normals=np.ascontiguousarray(normals/np.linalg.norm(normals,axis=-1,keepdims=True),dtype=np.float32)
    nt=texture(1,normals,GL_RGB32F,GL_RGB,GL_FLOAT)
    lt=texture(2,None,GL_R8,GL_RED,GL_UNSIGNED_BYTE,gw,gh);lfb=framebuffer(lt)
    uv=np.stack((np.maximum(lookup,0)%gw,np.maximum(lookup,0)//gw),axis=-1).astype(np.uint16);uv[lookup<0]=65535
    ut=texture(3,uv,GL_RG16UI,GL_RG_INTEGER,GL_UNSIGNED_SHORT)
    ot=texture(4,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,ow,oh);ofb=framebuffer(ot)
    sp,gp=program(SHADOW),program(GATHER)
    glUseProgram(sp)
    for name,unit in [('terrain',0),('normals',1)]:glUniform1i(glGetUniformLocation(sp,name),unit)
    glUniform2i(glGetUniformLocation(sp,'size'),gw,gh);glUniform1i(glGetUniformLocation(sp,'levels'),levels)
    glUniform1f(glGetUniformLocation(sp,'bias'),bias);glUniform1f(glGetUniformLocation(sp,'ambient'),ambient)
    glUseProgram(gp)
    for name,unit in [('lookup',3),('light',2)]:glUniform1i(glGetUniformLocation(gp,name),unit)
    results=[];rng=np.random.default_rng(47)
    coords=np.column_stack((rng.integers(0,gw,validate_count),rng.integers(0,gh,validate_count)))
    for method in methods:
        timings=[];errors=0;cap_hits=0;comparisons=0;shadow_pixels=[];max_gray=0
        for si,sun in enumerate(suns):
            raw=np.array([sun[0]/(ts[0]/gw),-sun[1]/(ts[1]/gh)])
            dominant=max(abs(raw));direction=raw/dominant;rise=sun[2]/dominant
            glUseProgram(sp);glUniform2f(glGetUniformLocation(sp,'direction'),*direction)
            glUniform1f(glGetUniformLocation(sp,'rise'),rise);glUniform3f(glGetUniformLocation(sp,'sun'),*sun)
            glUniform1i(glGetUniformLocation(sp,'method'),method)
            glUniform1i(glGetUniformLocation(sp,'mask_only'),0)
            for trial in range(trials+1):
                start=time.perf_counter()
                glUseProgram(sp);glBindFramebuffer(GL_FRAMEBUFFER,lfb);glViewport(0,0,gw,gh)
                glDrawArrays(GL_TRIANGLES,0,3);glFinish();mid=time.perf_counter()
                glUseProgram(gp);glBindFramebuffer(GL_FRAMEBUFFER,ofb);glViewport(0,0,ow,oh)
                glDrawArrays(GL_TRIANGLES,0,3);glFinish();end=time.perf_counter()
                if trial:timings.append([(mid-start)*1000,(end-mid)*1000,(end-start)*1000])
            # Validate mask against CPU DDA, outside measured rendering.
            glUseProgram(sp);glUniform1i(glGetUniformLocation(sp,'mask_only'),1)
            glBindFramebuffer(GL_FRAMEBUFFER,lfb);glViewport(0,0,gw,gh);glDrawArrays(GL_TRIANGLES,0,3)
            mask=np.empty((gh,gw),np.uint8);glReadPixels(0,0,gw,gh,GL_RED,GL_UNSIGNED_BYTE,mask.ctypes.data_as(C.c_void_p))
            cap_hits+=int(np.count_nonzero((mask!=0)&(mask!=255)))
            shadow_pixels.append(int(np.count_nonzero(mask==0)))
            for x,y in coords:
                expected=dda(z,int(x),int(y),direction,rise,bias) if sun[2]>0 else True
                errors+=bool(mask[y,x]==0)!=expected;comparisons+=1
            # Compare 4K gather with exact byte lookup of the rendered light grid.
            glUniform1i(glGetUniformLocation(sp,'mask_only'),0);glDrawArrays(GL_TRIANGLES,0,3)
            light=np.empty((gh,gw),np.uint8);glReadPixels(0,0,gw,gh,GL_RED,GL_UNSIGNED_BYTE,light.ctypes.data_as(C.c_void_p))
            glBindFramebuffer(GL_FRAMEBUFFER,ofb)
            rgba=np.empty((oh,ow,4),np.uint8);glReadPixels(0,0,ow,oh,GL_RGBA,GL_UNSIGNED_BYTE,rgba.ctypes.data_as(C.c_void_p))
            expected=np.where(lookup<0,0,light.ravel()[np.maximum(lookup,0)])
            max_gray=max(max_gray,int(np.max(np.abs(rgba[:,:,0].astype(np.int16)-expected.astype(np.int16)))))
        row={'grid':[gw,gh],'output':[ow,oh],'method':['cell_dda','max_pyramid'][method],
             'medians_ms':dict(zip(['gpu_shadow_and_lighting','gpu_4k_gather','total'],np.median(timings,axis=0).round(3).tolist())),
             'p95_total_ms':round(float(np.percentile(np.array(timings)[:,2],95)),3),
             'frames':len(timings),'oracle_comparisons':comparisons,'oracle_mismatches':int(errors),
             'iteration_cap_pixels':cap_hits,'cast_shadow_pixels':shadow_pixels,'max_gather_byte_difference':max_gray}
        results.append(row);print(json.dumps(row),flush=True)
    glBindFramebuffer(GL_FRAMEBUFFER,0)
    glDeleteFramebuffers(2,[lfb,ofb]);glDeleteTextures(5,[tex,nt,lt,ut,ot]);glDeleteProgram(sp);glDeleteProgram(gp)
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--grids',nargs='+',type=int,default=[512,1024,2048]);parser.add_argument('--trials',type=int,default=5)
    parser.add_argument('--methods',nargs='+',type=int,default=[0,1]);parser.add_argument('--validate',type=int,default=512)
    parser.add_argument('--output',type=Path,default=Path('gpu-shadows-results.json'))
    args=parser.parse_args();egl=context();data=np.load(args.input)
    from map_projection.solar import position
    suns=[position(f'2026-10-24T{hour}:00:00+02:00',45.95,10.65)[0] for hour in ['09','12','16','18']]
    results=[];source=data['z'];original_lookup=data['lookup'];oldh,oldw=source.shape
    for width in args.grids:
        height=round(width*oldh/oldw)
        # Input should be regenerated using the corrected terrain sampler.
        yi=np.minimum(((np.arange(height)+.5)*oldh/height).astype(int),oldh-1)
        xi=np.minimum(((np.arange(width)+.5)*oldw/width).astype(int),oldw-1)
        z=np.ascontiguousarray(source[np.ix_(yi,xi)])
        gx=np.minimum(np.maximum(original_lookup,0)%oldw*width//oldw,width-1)
        gy=np.minimum(np.maximum(original_lookup,0)//oldw*height//oldh,height-1)
        lookup=np.where(original_lookup<0,-1,gy*width+gx).astype(np.int32)
        results.extend(run(z,data['terrain_size'],lookup,suns,float(data['bias']),float(data['ambient']),args.methods,args.trials,args.validate))
    args.output.write_text(json.dumps({'renderer':egl[3],'sun_hours':['09','12','16','18'],'results':results},indent=2)+'\n')


if __name__=='__main__':main()
