/* Paint geographic GPX texels through each projector ray's first STL hit.
 * No flat homography or DEM height approximation is used for registration. */
#include <math.h>
#include <stdint.h>
#include <stddef.h>
int project_track(int w,int h,const float *depth,const double *inv,const double *origin,
                  double cx,double cy,double mw,double mh,int tw,int th,
                  const uint8_t *mask,const float *progress,float *vertices) {
 int count=0;
 for(int y=0;y<h;y++)for(int x=0;x<w;x++) {
  size_t p=(size_t)y*w+x;double d=depth[p];if(!isfinite(d))continue;
  double px=origin[0]+d*(inv[0]*(x+.5)+inv[1]*(y+.5)+inv[2]);
  double py=origin[1]+d*(inv[3]*(x+.5)+inv[4]*(y+.5)+inv[5]);
  double u=(px+cx)/mw*tw-.5,v=(1-(py+cy)/mh)*th-.5;
  if(u<0||v<0||u>tw-1||v>th-1)continue;
  int ix=(int)u,iy=(int)v,jx=ix+1<tw?ix+1:ix,jy=iy+1<th?iy+1:iy;
  double fx=u-ix,fy=v-iy;
  size_t q[4]={(size_t)iy*tw+ix,(size_t)iy*tw+jx,(size_t)jy*tw+ix,(size_t)jy*tw+jx};
  double weight[4]={(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy},alpha=0,t=0;
  for(int i=0;i<4;i++){double a=weight[i]*mask[q[i]]/255.;alpha+=a;t+=a*progress[q[i]];}
  if(alpha<.015)continue;
  float *out=vertices+4*(size_t)count++;
  out[0]=2*(x+.5)/w-1;out[1]=1-2*(y+.5)/h;out[2]=t/alpha;out[3]=alpha;
 }
 return count;
}
