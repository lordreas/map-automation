/* Geographic lookup from the first STL hit, and exact squared-distance fields.
 * Separate from the lighting compiler so its existing caches remain valid. */
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
void layer_uv(int w,int h,const float *depth,const double *inv,const double *origin,
              double cx,double cy,double mw,double mh,uint16_t *out) {
 for(int y=0;y<h;y++)for(int x=0;x<w;x++) {
  size_t p=(size_t)y*w+x;double d=depth[p];out[2*p]=out[2*p+1]=65535;
  if(!isfinite(d))continue;
  double px=origin[0]+d*(inv[0]*(x+.5)+inv[1]*(y+.5)+inv[2]);
  double py=origin[1]+d*(inv[3]*(x+.5)+inv[4]*(y+.5)+inv[5]);
  double u=(px+cx)/mw,v=1-(py+cy)/mh;
  if(u<0||v<0||u>1||v>1)continue;
  out[2*p]=(uint16_t)lround(u*65534);out[2*p+1]=(uint16_t)lround(v*65534);
 }
}
static void edt_line(float *f,float *d,int n,int *v,double *z) {
 int k=0;v[0]=0;z[0]=-INFINITY;z[1]=INFINITY;
 for(int q=1;q<n;q++) {
  double s;
  do {int vk=v[k];s=((double)f[q]+(double)q*q-f[vk]-(double)vk*vk)/(2.*(q-vk));if(s>z[k]||k==0)break;k--;}while(1);
  k++;v[k]=q;z[k]=s;z[k+1]=INFINITY;
 }
 k=0;
 for(int q=0;q<n;q++){while(z[k+1]<q)k++;double dx=q-v[k];d[q]=(float)(dx*dx+f[v[k]]);}
}
void layer_heights(int w,int h,const uint16_t *uv,int tw,int th,const float *z,float *out) {
 for(size_t i=0;i<(size_t)w*h;i++) {
  if(uv[2*i]==65535){out[i]=-60000;continue;}
  double x=uv[2*i]/65534.*tw-.5,y=uv[2*i+1]/65534.*th-.5;
  x=fmax(0,fmin(tw-1,x));y=fmax(0,fmin(th-1,y));int x0=(int)x,y0=(int)y,x1=x0+1<tw?x0+1:x0,y1=y0+1<th?y0+1:y0;
  double fx=x-x0,fy=y-y0;
  out[i]=(float)((z[(size_t)y0*tw+x0]*(1-fx)+z[(size_t)y0*tw+x1]*fx)*(1-fy)+(z[(size_t)y1*tw+x0]*(1-fx)+z[(size_t)y1*tw+x1]*fx)*fy);
 }
}
int peak_distance(int w,int h,int count,const float *points,float low,float high,float *out) {
 int n=w>h?w:h;float *f=malloc(n*sizeof(float)),*d=malloc(n*sizeof(float));int *v=malloc(n*sizeof(int));double *z=malloc((n+1)*sizeof(double));
 if(!f||!d||!v||!z){free(f);free(d);free(v);free(z);return -1;}
 for(size_t i=0;i<(size_t)w*h;i++)out[i]=1.e12f;
 int selected=0;
 for(int i=0;i<count;i++)if(points[3*i+2]>=low&&points[3*i+2]<=high) {
  int x=lroundf(points[3*i]),y=lroundf(points[3*i+1]);
  if(x>=0&&x<w&&y>=0&&y<h){out[(size_t)y*w+x]=0;selected++;}
 }
 if(selected) {
  for(int y=0;y<h;y++){for(int x=0;x<w;x++)f[x]=out[(size_t)y*w+x];edt_line(f,d,w,v,z);for(int x=0;x<w;x++)out[(size_t)y*w+x]=d[x];}
  for(int x=0;x<w;x++){for(int y=0;y<h;y++)f[y]=out[(size_t)y*w+x];edt_line(f,d,h,v,z);for(int y=0;y<h;y++)out[(size_t)y*w+x]=sqrtf(d[y]);}
 }else for(size_t i=0;i<(size_t)w*h;i++)out[i]=60000.f;
 free(f);free(d);free(v);free(z);return selected;
}

#include <string.h>
static inline float from_half(uint16_t h) {
 unsigned e=(h>>10)&31,m=h&1023;uint32_t b;
 if(!e)return (h&32768?-1.f:1.f)*ldexpf((float)m,-24);
 b=((uint32_t)(h&32768)<<16)|((uint32_t)(e+112)<<23)|((uint32_t)m<<13);float f;memcpy(&f,&b,4);return f;
}
static inline uint16_t to_half(float x) {
 uint32_t b;memcpy(&b,&x,4);int e=((b>>23)&255)-127+15;uint32_t m=b&0x7fffff;
 if(e<=0)return e<-10?0:(uint16_t)(((m|0x800000)+(1u<<(13-e)))>>(14-e));
 if(e>=31)return 0x7bff;
 uint32_t result=((uint32_t)e<<10)+(m+0x1000)/0x2000;return (uint16_t)(result>0x7bff?0x7bff:result);
}
void layer_rivers(int w,int h,const uint16_t *uv,int tw,int th,const uint16_t *source,uint16_t *out) {
 for(int y=0;y<h;y++)for(int x=0;x<w;x++) {
  size_t p=(size_t)y*w+x;uint16_t *dst=out+3*p;
  if(uv[2*p]==65535){dst[0]=to_half(60000);dst[1]=0;dst[2]=to_half(2);continue;}
  double u=uv[2*p]/65534.*tw-.5,v=uv[2*p+1]/65534.*th-.5;
  u=fmax(0,fmin(tw-1,u));v=fmax(0,fmin(th-1,v));int ix=(int)u,iy=(int)v,jx=ix+1<tw?ix+1:ix,jy=iy+1<th?iy+1:iy;
  float fx=u-ix,fy=v-iy;size_t q[4]={(size_t)iy*tw+ix,(size_t)iy*tw+jx,(size_t)jy*tw+ix,(size_t)jy*tw+jx};
  float weight[4]={(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy},s[3]={0,0,0};
  for(int i=0;i<4;i++)for(int c=0;c<3;c++)s[c]+=weight[i]*from_half(source[3*q[i]+c]);
  int nx=x+1<w?x+1:x,ny=y+1<h?y+1:y;size_t px=(size_t)y*w+nx,py=(size_t)ny*w+x;
  if(uv[2*px]==65535)px=p;if(uv[2*py]==65535)py=p;
  float dx=((int)uv[2*px]-uv[2*p])*tw/65534.f,dy=((int)uv[2*px+1]-uv[2*p+1])*th/65534.f;
  float ex=((int)uv[2*py]-uv[2*p])*tw/65534.f,ey=((int)uv[2*py+1]-uv[2*p+1])*th/65534.f;
  float scale=fmaxf(.05f,sqrtf((dx*dx+dy*dy+ex*ex+ey*ey)*.5f));
  dst[0]=to_half(fminf(60000,s[0]/scale));dst[1]=to_half(s[1]);dst[2]=to_half(s[2]);
 }
}

/* Compile a monotonic reveal into packed pixel events. Playback draws only the
 * newly changed phase band, never scans the full screen or accumulates alpha. */
typedef struct {uint32_t xy, value;} river_event;
static int event_compare(const void *a,const void *b) {
 const river_event *x=a,*y=b;uint32_t px=x->value&65535,py=y->value&65535;
 if(px!=py)return px<py?-1:1;return x->xy<y->xy?-1:x->xy>y->xy;
}
static void event_sort(river_event *events,int count) {
 size_t *offset=calloc(65536,sizeof(size_t));river_event *sorted=malloc((size_t)count*sizeof(river_event));
 if(!offset||(!sorted&&count)){free(offset);free(sorted);qsort(events,(size_t)count,sizeof(*events),event_compare);return;}
 for(int i=0;i<count;i++)offset[events[i].value&65535]++;
 size_t total=0;
 for(int i=0;i<65536;i++){size_t n=offset[i];offset[i]=total;total+=n;}
 for(int i=0;i<count;i++)sorted[offset[events[i].value&65535]++]=events[i];
 if(count)memcpy(events,sorted,(size_t)count*sizeof(*events));free(sorted);free(offset);
}
static inline float river_alpha(float d,float q,float thickness) {
 float width=1+(thickness-1)*q,lo=fmaxf(0,width*.5f-.65f),hi=width*.5f+.65f;
 float t=fminf(1,fmaxf(0,(d-lo)/(hi-lo)));return 1-t*t*(3-2*t);
}
int river_events(int w,int h,const uint16_t *field,float thickness,float chromatic,int green,river_event *out) {
 int count=0;float factor=1+chromatic/sqrtf((float)w*w*.25f+(float)h*h*.25f);
 for(int y=0;y<h;y++)for(int x=0;x<w;x++) {
  size_t p=(size_t)y*w+x;float d=from_half(field[3*p]);
  if(d>thickness*.5f+1+fabsf(chromatic)*1.5f&&!(green&&chromatic!=0&&d>50000))continue;
  float q=from_half(field[3*p+1]),phase=from_half(field[3*p+2]);
  if(green&&chromatic!=0) {
   float u=fminf(w-1,fmaxf(0,(x+.5f-w*.5f)*factor+w*.5f-.5f));
   float v=fminf(h-1,fmaxf(0,(y+.5f-h*.5f)*factor+h*.5f-.5f));
   int ix=(int)u,iy=(int)v,jx=ix+1<w?ix+1:ix,jy=iy+1<h?iy+1:iy;
   float fx=u-ix,fy=v-iy;size_t at[4]={(size_t)iy*w+ix,(size_t)iy*w+jx,(size_t)jy*w+ix,(size_t)jy*w+jx};
   float weights[4]={(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy},values[3]={0,0,0};
   for(int i=0;i<4;i++)for(int c=0;c<3;c++)values[c]+=weights[i]*from_half(field[3*at[i]+c]);
   d=values[0];q=values[1];phase=values[2];
  }
  unsigned alpha=(unsigned)lroundf(river_alpha(d,q,thickness)*255);
  if(!alpha||phase>1.005f)continue;
  if(out){out[count].xy=(uint32_t)x|((uint32_t)(h-1-y)<<12);out[count].value=(uint32_t)lroundf(fminf(1,fmaxf(0,phase))*65535)|((uint32_t)alpha<<16);}
  count++;
 }
 if(out)event_sort(out,count);
 return count;
}
