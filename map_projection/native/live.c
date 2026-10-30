/* Experimental DEM shading. Static geometry/normal preparation is independent
 * of sun time; every measured frame recomputes shadows and lighting. */
#include <stdint.h>
#include <math.h>
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif
API void threads(int n) {
#ifdef _OPENMP
    omp_set_num_threads(n);
#endif
}
API void prepare_normals(int w,int h,const float*z,double tw,double th,float*n) {
    for(int y=0;y<h;y++)for(int x=0;x<w;x++) {
        int l=x?x-1:x,r=x<w-1?x+1:x,t=y?y-1:y,b=y<h-1?y+1:y;
        double nx=-(z[y*w+r]-z[y*w+l])/((r-l)*tw/w);
        double ny=(z[b*w+x]-z[t*w+x])/((b-t)*th/h);
        double s=1/sqrt(nx*nx+ny*ny+1);size_t p=(size_t)y*w+x;
        n[3*p]=nx*s;n[3*p+1]=ny*s;n[3*p+2]=s;
    }
}
API void prepare_lookup(int w,int h,const float*d,const double*inv,const double*o,
                    double mw,double mh,int gw,int gh,int32_t*lookup) {
    for(int y=0;y<h;y++)for(int x=0;x<w;x++) {
        size_t p=(size_t)y*w+x;double dep=d[p];
        if(!isfinite(dep)){lookup[p]=-1;continue;}
        double px=o[0]+dep*(inv[0]*(x+.5)+inv[1]*(y+.5)+inv[2]);
        double py=o[1]+dep*(inv[3]*(x+.5)+inv[4]*(y+.5)+inv[5]);
        int u=(int)floor((px/mw+.5)*gw),v=(int)floor((.5-py/mh)*gh);
        if(u<0)u=0;if(u>=gw)u=gw-1;if(v<0)v=0;if(v>=gh)v=gh-1;
        lookup[p]=v*gw+u;
    }
}
API void shade_cells(int count,const float*z,const float*h,const float*n,const double*sun,
                 double ambient,double bias,unsigned char*light) {
    #pragma omp parallel for schedule(static) num_threads(2)
    for(int p=0;p<count;p++) {
        double f=fmax(0,n[3*p]*sun[0]+n[3*p+1]*sun[1]+n[3*p+2]*sun[2]);
        if(h[p]>z[p]+bias)f=0;
        light[p]=sun[2]>0?(unsigned char)lround(255*fmin(1,ambient+(1-ambient)*f)):0;
    }
}
API void gather(int count,const int32_t*lookup,const unsigned char*light,unsigned char*rgb) {
    #pragma omp parallel for schedule(static) num_threads(2)
    for(int p=0;p<count;p++) {
        unsigned char c=lookup[p]<0?0:light[lookup[p]];
        rgb[3*p]=rgb[3*p+1]=rgb[3*p+2]=c;
    }
}
/* Experimental float32 SIMD sweep, finite terrain inputs only. */
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fast-math")))
#endif
API void horizon_rows_fast(const float *restrict z,int w,int h,double dx,double dy,
                  const double*sun,float *restrict out) {
    if(sun[2]<=0){for(size_t p=0;p<(size_t)w*h;p++)out[p]=INFINITY;return;}
    double sx=sun[0]/dx,sy=-sun[1]/dy;
    if(fabs(sy)<1e-12){for(size_t p=0;p<(size_t)w*h;p++)out[p]=z[p];return;}
    int first=sy>0?h-1:0,step=sy>0?-1:1;
    double offset=sx/fabs(sy);float rise=(float)(sun[2]/fabs(sy));
    int shift=(int)floor(offset);float frac=(float)(offset-shift);
    int begin=offset<0?1:0,end=offset>0?w-1:w;
    for(int k=0;k<h;k++) {
        int row=first+k*step;
        const float*src=z+(size_t)row*w;float*dst=out+(size_t)row*w;
        if(!k){for(int x=0;x<w;x++)dst[x]=src[x];continue;}
        const float*prev=out+(size_t)(row-step)*w;
        if(begin)dst[0]=src[0];if(end<w)dst[w-1]=src[w-1];
        if(frac==0) {
            #pragma GCC ivdep
            for(int x=begin;x<end;x++)dst[x]=fmaxf(src[x],prev[x+shift]-rise);
        }else {
            #pragma GCC ivdep
            for(int x=begin;x<end;x++) {
                float upstream=prev[x+shift]+(prev[x+shift+1]-prev[x+shift])*frac-rise;
                dst[x]=fmaxf(src[x],upstream);
            }
        }
    }
}
API void transpose_f32(const float *restrict src,int w,int h,float *restrict dst) {
    for(int y0=0;y0<h;y0+=32)for(int x0=0;x0<w;x0+=32)
        for(int y=y0;y<h&&y<y0+32;y++)
            for(int x=x0;x<w&&x<x0+32;x++)dst[(size_t)x*h+y]=src[(size_t)y*w+x];
}
