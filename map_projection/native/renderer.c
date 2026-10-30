/* Bounded-memory binary STL rasterizer. No GPU, antialiasing or texture scaling.
 * Perspective-correct depth at pixel centres is the exact triangle/ray solution.
 * Coordinates: centred X east, Y north (exported STL Y), Z unchanged from STL.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

static double edge(double ax, double ay, double bx, double by, double x, double y) {
    return (bx-ax)*(y-ay)-(by-ay)*(x-ax);
}

API int scan_stl(const char *path, double *bounds) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    unsigned char header[84], buf[50*4096];
    if (fread(header,1,84,f)!=84) { fclose(f); return -2; }
    uint32_t count; memcpy(&count,header+80,4);
    for (int j=0;j<3;j++) { bounds[j]=DBL_MAX; bounds[j+3]=-DBL_MAX; }
    uint32_t done=0;
    while (done<count) {
        size_t n=count-done; if(n>4096)n=4096;
        if(fread(buf,50,n,f)!=n) { fclose(f); return -2; }
        for(size_t i=0;i<n;i++) {
            float v[9]; memcpy(v,buf+i*50+12,36);
            for(int k=0;k<9;k++) {
                if(!isfinite(v[k])) { fclose(f); return -3; }
                int j=k%3;
                if(v[k]<bounds[j])bounds[j]=v[k];
                if(v[k]>bounds[j+3])bounds[j+3]=v[k];
            }
        }
        done+=(uint32_t)n;
    }
    fclose(f); return 0;
}

/* M maps centred map XYZ to homogeneous output XY/depth. Perspective depth is
 * the third homogeneous coordinate. Orthographic depth is affine sun depth.
 * Frontmost triangles win regardless of winding; side walls occlude correctly.
 */
API int raster_stl(const char *path, const double *bounds, const double *m,
                  int width, int height, int perspective, float *depth,
                  int8_t *normals) {
    FILE *f=fopen(path,"rb"); if(!f)return -1;
    unsigned char header[84],buf[50*4096];
    if(fread(header,1,84,f)!=84){fclose(f);return -2;}
    uint32_t count; memcpy(&count,header+80,4);
    double cx=(bounds[0]+bounds[3])*0.5, cy=(bounds[1]+bounds[4])*0.5;
    size_t pixels=(size_t)width*height;
    for(size_t i=0;i<pixels;i++)depth[i]=INFINITY;
    if(normals)memset(normals,0,pixels*3);
    uint32_t done=0;
    while(done<count) {
        size_t n=count-done;if(n>4096)n=4096;
        if(fread(buf,50,n,f)!=n){fclose(f);return -2;}
        for(size_t i=0;i<n;i++) {
            float raw[9];memcpy(raw,buf+i*50+12,36);
            double v[3][3],sx[3],sy[3],sz[3];int skip=0;
            for(int k=0;k<3;k++) {
                v[k][0]=raw[3*k]-cx;v[k][1]=raw[3*k+1]-cy;v[k][2]=raw[3*k+2];
                double q[3];
                for(int r=0;r<3;r++)q[r]=m[4*r]*v[k][0]+m[4*r+1]*v[k][1]+m[4*r+2]*v[k][2]+m[4*r+3];
                if(perspective&&q[2]<=1e-8){skip=1;break;}
                sx[k]=perspective?q[0]/q[2]:q[0];
                sy[k]=perspective?q[1]/q[2]:q[1];sz[k]=q[2];
            }
            if(skip)continue;
            double lo_x=fmin(sx[0],fmin(sx[1],sx[2])),hi_x=fmax(sx[0],fmax(sx[1],sx[2]));
            double lo_y=fmin(sy[0],fmin(sy[1],sy[2])),hi_y=fmax(sy[0],fmax(sy[1],sy[2]));
            if(hi_x<0.5||lo_x>width-0.5||hi_y<0.5||lo_y>height-0.5)continue;
            int x0=(int)fmax(0,ceil(lo_x-0.5)),x1=(int)fmin(width-1,floor(hi_x-0.5));
            int y0=(int)fmax(0,ceil(lo_y-0.5)),y1=(int)fmin(height-1,floor(hi_y-0.5));
            if(x0>x1||y0>y1)continue;
            double area=edge(sx[0],sy[0],sx[1],sy[1],sx[2],sy[2]);
            if(fabs(area)<1e-16)continue;
            int8_t normal[3]={0,0,0};
            if(normals) {
                /* Exported STL is already flipped into north-positive Y. */
                double a[3],b[3],nn[3];for(int k=0;k<3;k++){a[k]=v[1][k]-v[0][k];b[k]=v[2][k]-v[0][k];}
                nn[0]=a[1]*b[2]-a[2]*b[1];nn[1]=a[2]*b[0]-a[0]*b[2];nn[2]=a[0]*b[1]-a[1]*b[0];
                double len=sqrt(nn[0]*nn[0]+nn[1]*nn[1]+nn[2]*nn[2]);
                if(len>1e-16)for(int k=0;k<3;k++)normal[k]=(int8_t)lround(127*nn[k]/len);
            }
            for(int y=y0;y<=y1;y++)for(int x=x0;x<=x1;x++) {
                double a=edge(sx[1],sy[1],sx[2],sy[2],x+0.5,y+0.5)/area;
                double b=edge(sx[2],sy[2],sx[0],sy[0],x+0.5,y+0.5)/area;
                double c=1-a-b;
                if(a < -1e-9 || b < -1e-9 || c < -1e-9)continue;
                double z=perspective?1.0/(a/sz[0]+b/sz[1]+c/sz[2]):a*sz[0]+b*sz[1]+c*sz[2];
                size_t at=(size_t)y*width+x;
                /* Compare at the stored depth precision. Otherwise tiny double
                 * rounding differences can let a later coplanar back face
                 * replace the first surface, especially with ARM FMA. */
                float stored_z=(float)z;
                if(stored_z<depth[at]) {
                    depth[at]=stored_z;
                    if(normals)for(int k=0;k<3;k++)normals[3*at+k]=normal[k];
                }
            }
        }
        done+=(uint32_t)n;
    }
    fclose(f);return 0;
}

API void shade(int width,int height,const float *depth,const int8_t *normals,
               const double *inv,const double *origin,const double *sun_matrix,
               int shadow_size,const float *shadow,const double *sun,
               double ambient,double bias,unsigned char *rgb) {
    for(int y=0;y<height;y++)for(int x=0;x<width;x++) {
        size_t at=(size_t)y*width+x;
        double z=depth[at];
        if(!isfinite(z)||sun[2]<=0){rgb[3*at]=rgb[3*at+1]=rgb[3*at+2]=0;continue;}
        double p[3],q[3];
        for(int k=0;k<3;k++)p[k]=origin[k]+z*(inv[3*k]*(x+0.5)+inv[3*k+1]*(y+0.5)+inv[3*k+2]);
        for(int k=0;k<3;k++)q[k]=sun_matrix[4*k]*p[0]+sun_matrix[4*k+1]*p[1]+sun_matrix[4*k+2]*p[2]+sun_matrix[4*k+3];
        int u=(int)floor(q[0]),v=(int)floor(q[1]);
        double diffuse=fmax(0,(normals[3*at]*sun[0]+normals[3*at+1]*sun[1]+normals[3*at+2]*sun[2])/127.0);
        if(u>=0&&u<shadow_size&&v>=0&&v<shadow_size) {
            float nearest=shadow[(size_t)v*shadow_size+u];
            /* Receiver-plane depth at the shadow texel centre avoids slope
             * acne without blurring the shadow or using a large fixed bias. */
            double denom=0,nu=0,nv=0,su=0,sv=0;
            for(int k=0;k<3;k++) {
                double nn=normals[3*at+k]/127.0;
                denom+=nn*sun_matrix[8+k];nu+=nn*sun_matrix[k];nv+=nn*sun_matrix[4+k];
                su+=sun_matrix[k]*sun_matrix[k];sv+=sun_matrix[4+k]*sun_matrix[4+k];
            }
            double receiver=q[2];
            if(fabs(denom)>.02)receiver-=(nu/su*(u+0.5-q[0])+nv/sv*(v+0.5-q[1]))/denom;
            if(isfinite(nearest)&&receiver>nearest+bias)diffuse=0;
        }
        double value=fmin(1,ambient+(1-ambient)*diffuse);
        unsigned char c=(unsigned char)lround(255*value);
        rgb[3*at]=c;rgb[3*at+1]=c;rgb[3*at+2]=c;
    }
}

/* Exact point query against the original STL, for geographic targets and
 * visibility tests. A scan is slow but uses < 1 MB and runs off the UI thread. */
API int ray_stl(const char *path,const double *bounds,const double *origin,
               const double *dir,double *hit) {
    FILE *f=fopen(path,"rb");if(!f)return -1;
    unsigned char header[84],buf[50*4096];
    if(fread(header,1,84,f)!=84){fclose(f);return -2;}
    uint32_t count;memcpy(&count,header+80,4);
    double cx=(bounds[0]+bounds[3])*0.5,cy=(bounds[1]+bounds[4])*0.5,best=DBL_MAX;
    uint32_t done=0;
    while(done<count) {
        size_t n=count-done;if(n>4096)n=4096;
        if(fread(buf,50,n,f)!=n){fclose(f);return -2;}
        for(size_t i=0;i<n;i++) {
            float raw[9];memcpy(raw,buf+i*50+12,36);
            double v[3][3],e1[3],e2[3],pv[3],tv[3],qv[3];
            for(int k=0;k<3;k++){v[k][0]=raw[k*3]-cx;v[k][1]=raw[k*3+1]-cy;v[k][2]=raw[k*3+2];}
            for(int k=0;k<3;k++){e1[k]=v[1][k]-v[0][k];e2[k]=v[2][k]-v[0][k];tv[k]=origin[k]-v[0][k];}
            pv[0]=dir[1]*e2[2]-dir[2]*e2[1];pv[1]=dir[2]*e2[0]-dir[0]*e2[2];pv[2]=dir[0]*e2[1]-dir[1]*e2[0];
            double det=e1[0]*pv[0]+e1[1]*pv[1]+e1[2]*pv[2];if(fabs(det)<1e-14)continue;
            double u=(tv[0]*pv[0]+tv[1]*pv[1]+tv[2]*pv[2])/det;if(u < -1e-9 || u>1+1e-9)continue;
            qv[0]=tv[1]*e1[2]-tv[2]*e1[1];qv[1]=tv[2]*e1[0]-tv[0]*e1[2];qv[2]=tv[0]*e1[1]-tv[1]*e1[0];
            double vv=(dir[0]*qv[0]+dir[1]*qv[1]+dir[2]*qv[2])/det;if(vv < -1e-9 || u+vv>1+1e-9)continue;
            double t=(e2[0]*qv[0]+e2[1]*qv[1]+e2[2]*qv[2])/det;
            if(t>1e-8&&t<best)best=t;
        }
        done+=(uint32_t)n;
    }
    fclose(f);
    if(best==DBL_MAX)return 1;
    for(int k=0;k<3;k++)hit[k]=origin[k]+best*dir[k];
    return 0;
}

/* Linear-work directional horizon sweep. Heightfield rows run southward.
 * The upstream horizon is linearly interpolated across adjacent scanlines.
 * This is a controlled terrain-grid approximation, not exact STL ray casting.
 * Memory: one float per terrain sample. Work: O(samples), independent of rays.
 */
API void horizon(const float *z,int width,int height,double dx,double dy,
                 const double *sun,float *out) {
    double sx=sun[0]/dx,sy=-sun[1]/dy;
    if(sun[2]<=0){for(size_t i=0;i<(size_t)width*height;i++)out[i]=INFINITY;return;}
    int horizontal=fabs(sx)>=fabs(sy);
    double dominant=horizontal?sx:sy;
    if(fabs(dominant)<1e-12){memcpy(out,z,(size_t)width*height*sizeof(float));return;}
    int count=horizontal?width:height,across=horizontal?height:width;
    int start=dominant>0?count-1:0,step=dominant>0?-1:1;
    double offset=(horizontal?sy:sx)/fabs(dominant),rise=sun[2]/fabs(dominant);
    for(int k=0;k<count;k++) {
        int current=start+k*step,upstream=current-step;
        for(int a=0;a<across;a++) {
            size_t at=horizontal?(size_t)a*width+current:(size_t)current*width+a;
            double upstream_z=-INFINITY;
            double pos=a+offset;
            if(k>0&&pos>=0&&pos<=across-1) {
                int low=(int)floor(pos),high=low+1;if(high>=across)high=low;
                size_t i0=horizontal?(size_t)low*width+upstream:(size_t)upstream*width+low;
                size_t i1=horizontal?(size_t)high*width+upstream:(size_t)upstream*width+high;
                upstream_z=out[i0]+(out[i1]-out[i0])*(pos-low)-rise;
            }
            out[at]=(float)fmax(z[at],upstream_z);
        }
    }
}

API void shade_grid(int width,int height,const float *depth,const int8_t *normals,
                    const double *inv,const double *origin,const float *z,const float *horizon,
                    int gw,int gh,double map_width,double map_height,double terrain_width,
                    double terrain_height,const double *sun,double ambient,double bias,
                    int dem_normals,unsigned char *rgb) {
    for(int y=0;y<height;y++)for(int x=0;x<width;x++) {
        size_t at=(size_t)y*width+x;
        double dep=depth[at];
        if(!isfinite(dep)||sun[2]<=0){rgb[3*at]=rgb[3*at+1]=rgb[3*at+2]=0;continue;}
        double px=origin[0]+dep*(inv[0]*(x+0.5)+inv[1]*(y+0.5)+inv[2]);
        double py=origin[1]+dep*(inv[3]*(x+0.5)+inv[4]*(y+0.5)+inv[5]);
        int u=(int)floor((px/map_width+.5)*gw),v=(int)floor((.5-py/map_height)*gh);
        if(u<0)u=0;if(u>=gw)u=gw-1;if(v<0)v=0;if(v>=gh)v=gh-1;
        size_t cell=(size_t)v*gw+u;
        double diffuse;
        if(dem_normals) {
            int ul=u>0?u-1:u,ur=u<gw-1?u+1:u,vt=v>0?v-1:v,vb=v<gh-1?v+1:v;
            double nx=-(z[(size_t)v*gw+ur]-z[(size_t)v*gw+ul])/((ur-ul)*terrain_width/gw);
            double ny=(z[(size_t)vb*gw+u]-z[(size_t)vt*gw+u])/((vb-vt)*terrain_height/gh);
            diffuse=fmax(0,(nx*sun[0]+ny*sun[1]+sun[2])/sqrt(nx*nx+ny*ny+1));
        } else diffuse=fmax(0,(normals[3*at]*sun[0]+normals[3*at+1]*sun[1]+normals[3*at+2]*sun[2])/127.0);
        if(horizon[cell]>z[cell]+bias)diffuse=0;
        unsigned char value=(unsigned char)lround(255*fmin(1,ambient+(1-ambient)*diffuse));
        rgb[3*at]=rgb[3*at+1]=rgb[3*at+2]=value;
    }
}

#include "live.c"
