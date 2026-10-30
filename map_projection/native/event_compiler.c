/* Standalone daily-event compiler. Python supplies metadata only.
 * All decoding, solar evaluation, visibility bounds, root refinement, CSR
 * assembly, chronological sorting and NPY serialization execute here.
 * Build without fast-math/FMA contraction: reference float32 rounding matters.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <limits.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <dlfcn.h>
#include <omp.h>

typedef struct { double az,el; int lo,hi; float f,left; } Sun;
typedef struct { int lo,hi; float f,left,el; } Query;
typedef struct { int first,last,daylight,nknots; float emin,emax; Sun low,high; int *knots; } Block;
typedef struct { uint32_t *times; size_t size,capacity; } Events;
typedef struct { void *base,*data; size_t bytes; } Mapping;
static int H,W,A,TH,TW,year,doy,cyclic;
static double az_start,az_step,latitude,longitude,step,precision;
static float levels[262144];
static size_t (*zstd_decompress)(void*,size_t,const void*,size_t);
static unsigned (*zstd_error)(size_t);

static void fail(const char *message) { fprintf(stderr,"event compiler: %s (%s)\n",message,strerror(errno)); exit(2); }
static void *allocate(size_t n) { void *p=malloc(n?n:1); if(!p)fail("allocation failed");return p; }
static void *zeroed(size_t n,size_t size) { void *p=calloc(n?n:1,size);if(!p)fail("allocation failed");return p; }
static void path_join(char *out,size_t size,const char *dir,const char *name) {
 if(snprintf(out,size,"%s/%s",dir,name)>=(int)size)fail("path too long");
}
static double clock_seconds(void) { struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9; }
static int leap(int y) {return y%4==0 && (y%100!=0 || y%400==0);}

static void locate(Sun *s) {
 s->lo=s->hi=0;s->f=0;s->left=1;
 if(s->el<=0)return;
 double q=(s->az-az_start)/az_step;
 if(q<0 || q>A-1+(cyclic?1:0))fail("daylight azimuth outside skyline sector");
 s->lo=(int)floor(q);double fraction=q-s->lo;s->f=(float)fraction;s->left=(float)(1.0-fraction);s->hi=s->lo+1;
 if(s->lo>=A)s->lo=0;
 if(s->hi>=A)s->hi=cyclic?0:A-1;
}
static Sun solar(double seconds) {
 /* timedelta uses microsecond rounding. Match its UTC hour decomposition. */
 int y=year,d=doy;int64_t us=(int64_t)nearbyint(seconds*1e6);
 if(us>=86400000000LL){us-=86400000000LL;if(++d>365+leap(y)){++y;d=1;}}
 int hour=(int)(us/3600000000LL),minute=(int)(us/60000000%60),second=(int)(us/1000000%60);
 double h=hour+minute/60.0+(second+(us%1000000)/1e6)/3600.0;
 const double pi=3.14159265358979323846;
 double g=2*pi/(365+leap(y))*(d-1+(h-12)/24);
 double eq=229.18*(.000075+.001868*cos(g)-.032077*sin(g)-.014615*cos(2*g)-.040849*sin(2*g));
 double dec=.006918-.399912*cos(g)+.070257*sin(g)-.006758*cos(2*g)+.000907*sin(2*g)-.002697*cos(3*g)+.00148*sin(3*g);
 double phase=fmod((h*60+eq+4*longitude)/4,360);if(phase<0)phase+=360;
 double angle=(phase-180)*(pi/180);
 double lat=latitude*(pi/180),east=-cos(dec)*sin(angle);
 double north=cos(lat)*sin(dec)-sin(lat)*cos(dec)*cos(angle);
 double up=sin(lat)*sin(dec)+cos(lat)*cos(dec)*cos(angle);
 Sun s={0};s.az=atan2(east,north)*(180/pi);if(s.az<0)s.az+=360;
 s.el=asin(fmax(-1,fmin(1,up)))*(180/pi);locate(&s);return s;
}
static float skyline(const float *p,const Sun *s) {
 float left=p[s->lo]*s->left,right=p[s->hi]*s->f;
 return left+right;
}
static int visible(const float *p,const Sun *s) {return s->el>0 && (float)s->el>skyline(p,s);}
static void append(Events *e,uint32_t t) {
 if(e->size==e->capacity){e->capacity=e->capacity?e->capacity*2:1024;e->times=realloc(e->times,e->capacity*4);if(!e->times)fail("event allocation");}
 e->times[e->size++]=t;
}

static FILE *npy_open(const char *path,size_t *offset,char **header) {
 FILE *f=fopen(path,"rb");if(!f)fail(path);unsigned char prefix[8],len[4]={0};
 if(fread(prefix,1,8,f)!=8 || memcmp(prefix,"\x93NUMPY",6))fail("invalid NPY magic");
 int bytes=prefix[6]==1?2:prefix[6]==2?4:0;if(!bytes || fread(len,1,bytes,f)!=(size_t)bytes)fail("unsupported NPY header");
 uint32_t n=len[0]|((uint32_t)len[1]<<8)|((uint32_t)len[2]<<16)|((uint32_t)len[3]<<24);
 if(n>1024*1024)fail("oversized NPY header");
 char *h=allocate(n+1);if(fread(h,1,n,f)!=n)fail("truncated NPY header");h[n]=0;*offset=8+bytes+n;*header=h;return f;
}
static size_t npy_header(FILE *f,const char *dtype,size_t n) {
 char dict[256];int len=snprintf(dict,sizeof(dict),"{'descr': '%s', 'fortran_order': False, 'shape': (%zu,), }",dtype,n);
 size_t pad=(64-(10+(size_t)len+1)%64)%64;uint16_t size=(uint16_t)(len+pad+1);
 unsigned char length[2]={(unsigned char)size,(unsigned char)(size>>8)};
 if(fwrite("\x93NUMPY\x01\x00",1,8,f)!=8 || fwrite(length,1,2,f)!=2 || fwrite(dict,1,len,f)!=(size_t)len)fail("NPY header write");
 for(size_t i=0;i<pad;i++)fputc(' ',f);
 fputc('\n',f);return 10+size;
}
static Mapping npy_map(const char *dir,const char *name,const char *dtype,size_t n,size_t item) {
 char path[4096];path_join(path,sizeof(path),dir,name);FILE *f=fopen(path,"wb+");if(!f)fail(path);
 size_t offset=npy_header(f,dtype,n);if(fflush(f))fail("NPY header flush");size_t bytes=offset+n*item;
 if(ftruncate(fileno(f),(off_t)bytes))fail("NPY allocation");
 void *base=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fileno(f),0);
 if(base==MAP_FAILED)fail("NPY mapping");
 fclose(f);Mapping m={base,(char*)base+offset,bytes};return m;
}
static void finish_map(Mapping *m) {if(msync(m->base,m->bytes,MS_SYNC))fail("NPY flush");munmap(m->base,m->bytes);}

static void build_levels(void) {
 for(uint32_t code=0;code<262144;code++){
  float low=0,high=89.999f;
  for(int bit=17;bit>=0;bit--){float mid=(low+high)*.5f;if(code&(1u<<bit))low=mid;else high=mid;}
  levels[code]=(low+high)*.5f;
 }
}
static float *decode_tile(int fd,const uint64_t *entry,int h,int w) {
 size_t cells=(size_t)h*w,n=(size_t)A*cells,bytes=n*4;
 unsigned char *packed=allocate(entry[1]),*raw=allocate(bytes);
 size_t got=0;while(got<entry[1]){ssize_t r=pread(fd,packed+got,entry[1]-got,(off_t)(entry[0]+got));if(r<=0)fail("truncated skyline archive");got+=(size_t)r;}
 size_t result=zstd_decompress(raw,bytes,packed,entry[1]);free(packed);
 if(zstd_error(result)||result!=bytes)fail("corrupt Zstandard tile");
 uint32_t *u=allocate(bytes);
 for(size_t i=0;i<n;i++){uint32_t v=raw[i]|((uint32_t)raw[n+i]<<8)|((uint32_t)raw[2*n+i]<<16)|((uint32_t)raw[3*n+i]<<24);u[i]=(v>>1)^(0u-(v&1));}
 free(raw);
 for(int a=0;a<A;a++)for(int y=0;y<h;y++){size_t base=(size_t)a*cells+(size_t)y*w;for(int x=1;x<w;x++)u[base+x]+=u[base+x-1];}
 for(int a=1;a<A;a++)for(size_t c=0;c<cells;c++)u[(size_t)a*cells+c]+=u[(size_t)(a-1)*cells+c];
 float *p=allocate(bytes);if(entry[2]>1)fail("unknown skyline predictor");
 /* Blocked transpose makes each receiver's complete profile contiguous. */
 for(size_t cb=0;cb<cells;cb+=16)for(int ab=0;ab<A;ab+=32)
  for(size_t c=cb;c<cells && c<cb+16;c++)for(int a=ab;a<A && a<ab+32;a++){
   uint32_t value=u[(size_t)a*cells+c];float v;
   if(entry[2]){if(value>=262144)fail("invalid elevation code");v=levels[value];}else memcpy(&v,&value,4);
   p[c*(size_t)A+a]=v;
  }
 free(u);return p;
}
static float *raw_tile(const float *raw,int y,int x,int h,int w) {
 float *p=allocate((size_t)A*h*w*4);
 for(int cy=0;cy<h;cy++)for(int cx=0;cx<w;cx++)for(int ab=0;ab<A;ab+=32)
  for(int a=ab;a<A && a<ab+32;a++)p[((size_t)cy*w+cx)*A+a]=raw[((size_t)a*H+y+cy)*W+x+cx];
 return p;
}

int main(int argc,char **argv) {
 if(argc!=18){fprintf(stderr,"usage: event_compiler source output H W A tileH tileW azStart azStep year doy lat lon step precision threads compressed\n");return 2;}
 const char *source=argv[1],*out=argv[2];H=atoi(argv[3]);W=atoi(argv[4]);A=atoi(argv[5]);TH=atoi(argv[6]);TW=atoi(argv[7]);
 az_start=atof(argv[8]);az_step=atof(argv[9]);year=atoi(argv[10]);doy=atoi(argv[11]);latitude=atof(argv[12]);longitude=atof(argv[13]);step=atof(argv[14]);precision=atof(argv[15]);int threads=atoi(argv[16]),compressed=atoi(argv[17]);
 if(H<1||W<1||A<2||TH<1||TW<1||(uint64_t)H*W>=UINT32_MAX || !isfinite(step)||!isfinite(precision)||precision<=0||precision>step||step>86400||step<.01||threads<1||threads>64||az_step<=0)fail("invalid dimensions or parameters");
 cyclic=fabs(A*az_step-360)<1e-8;double started=clock_seconds();size_t cells=(size_t)H*W;
 omp_set_num_threads(threads);int cols=(W+TW-1)/TW,rows=(H+TH-1)/TH,tiles=cols*rows;
 int m=(int)ceil(86400/step)+1;Sun *sun=allocate((size_t)m*sizeof(Sun));double *times=allocate((size_t)m*sizeof(double));
 for(int i=0;i<m;i++){times[i]=fmin(86400,i*step);sun[i]=solar(times[i]);}
 /* At the default precision every bisection query lies on the same 1/32-s
  * grid. Evaluate its solar model once, rather than once per receiver/event.
  * This is exact temporal reuse, not interpolated solar positions. */
 int divisions=1;while(step/divisions>precision && divisions<256)divisions*=2;
 size_t query_count=step==1.0 && divisions<=32?(size_t)86400*divisions+1:0;
 Query *queries=query_count?allocate(query_count*sizeof(Query)):NULL;
 #pragma omp parallel for schedule(static)
 for(size_t i=0;i<query_count;i++){Sun s=solar((double)i/divisions);queries[i]=(Query){s.lo,s.hi,s.f,s.left,(float)s.el};}
 int span=(int)fmax(1,floor(60/step)),nb=(m-2)/span+1;Block *blocks=zeroed(nb,sizeof(Block));
 for(int b=0;b<nb;b++){
  Block *v=blocks+b;v->first=1+b*span;v->last=v->first+span-1;if(v->last>=m)v->last=m-1;
  double amin=INFINITY,amax=-INFINITY,emin=INFINITY,emax=-INFINITY;
  for(int k=v->first-1;k<=v->last;k++){emin=fmin(emin,sun[k].el);emax=fmax(emax,sun[k].el);if(sun[k].el>0){v->daylight=1;amin=fmin(amin,sun[k].az);amax=fmax(amax,sun[k].az);}}
  v->emin=(float)emin;v->emax=(float)emax;if(!v->daylight)continue;
  v->low.az=amin;v->high.az=amax;v->low.el=v->high.el=1;locate(&v->low);locate(&v->high);
  v->knots=allocate((size_t)(A+1)*sizeof(int));
  for(int a=0;a<A;a++){double az=az_start+a*az_step;if(az>amin&&az<amax)v->knots[v->nknots++]=a;}
 }
 int daylight_blocks=0;
 for(int b=0;b<nb;b++)if(blocks[b].daylight)blocks[daylight_blocks++]=blocks[b];
 nb=daylight_blocks;
 uint32_t *counts=zeroed(cells,4);uint8_t *initial=zeroed(cells,1);Events *events=zeroed(tiles,sizeof(Events));
 int fd=-1;uint64_t *index=NULL;void *raw_base=NULL;size_t raw_bytes=0;const float *raw_data=NULL;
 char path[4096];
 if(compressed){
  void *lib=dlopen("libzstd.so.1",RTLD_NOW);if(!lib)fail("system libzstd missing");
  *(void**)(&zstd_decompress)=dlsym(lib,"ZSTD_decompress");*(void**)(&zstd_error)=dlsym(lib,"ZSTD_isError");if(!zstd_decompress||!zstd_error)fail("libzstd symbols missing");
  size_t off;char *header;path_join(path,sizeof(path),source,"index.npy");FILE *f=npy_open(path,&off,&header);
  size_t nr=0,nc=0;char *shape=strstr(header,"shape");if(!shape||(shape=strchr(shape,'('))==NULL||sscanf(shape+1,"%zu, %zu",&nr,&nc)!=2||nr!=(size_t)tiles||nc!=3||!strstr(header,"<u8"))fail("invalid skyline tile index");
  index=allocate((size_t)tiles*24);if(fread(index,24,tiles,f)!=(size_t)tiles)fail("truncated tile index");free(header);fclose(f);
  path_join(path,sizeof(path),source,"tiles.zst");fd=open(path,O_RDONLY);if(fd<0)fail(path);build_levels();
 }else{
  size_t off;char *header;path_join(path,sizeof(path),source,"horizons.npy");FILE *f=npy_open(path,&off,&header);
  size_t na=0,nh=0,nw=0;char *shape=strstr(header,"shape");if(!shape||(shape=strchr(shape,'('))==NULL||sscanf(shape+1,"%zu, %zu, %zu",&na,&nh,&nw)!=3||na!=(size_t)A||nh!=(size_t)H||nw!=(size_t)W||!strstr(header,"<f4"))fail("invalid raw skyline raster");
  struct stat st;if(fstat(fileno(f),&st))fail("raw file stat");raw_bytes=(size_t)st.st_size;if(raw_bytes<off+(size_t)A*cells*4)fail("truncated raw skyline raster");
  raw_base=mmap(NULL,raw_bytes,PROT_READ,MAP_PRIVATE,fileno(f),0);if(raw_base==MAP_FAILED)fail("raw mapping");raw_data=(const float*)((const char*)raw_base+off);fclose(f);free(header);
 }
 int completed=0;uint64_t checked=0,total=0;
 #pragma omp parallel for schedule(dynamic,1) reduction(+:checked,total)
 for(int tile=0;tile<tiles;tile++){
  int y=(tile/cols)*TH,x=(tile%cols)*TW,h=H-y<TH?H-y:TH,w=W-x<TW?W-x:TW;
  float *profiles=compressed?decode_tile(fd,index+3*(size_t)tile,h,w):raw_tile(raw_data,y,x,h,w);Events *e=events+tile;
  for(int cy=0;cy<h;cy++)for(int cx=0;cx<w;cx++){
   size_t cell=(size_t)(y+cy)*W+x+cx;float *profile=profiles+((size_t)cy*w+cx)*A;
   int previous=visible(profile,sun);initial[cell]=(uint8_t)previous;size_t begin=e->size;
   for(int b=0;b<nb;b++){
    const Block *v=blocks+b;if(!v->daylight)continue;
    float a=skyline(profile,&v->low),z=skyline(profile,&v->high),low=a<z?a:z,high=a>z?a:z;
    for(int j=0;j<v->nknots;j++){z=profile[v->knots[j]];low=low<z?low:z;high=high>z?high:z;}
    /* A conservative margin covering endpoint/interpolation float32 ULPs. */
    low-=(fabsf(low)+1)*0x1p-20f;high+=(fabsf(high)+1)*0x1p-20f;
    if(high<v->emin || low>=v->emax)continue;
    for(int k=v->first;k<=v->last;k++){
     int current=visible(profile,sun+k);checked++;
     if(current!=previous){
      double lo=times[k-1],hi=times[k];
      if(queries){
       int left=0,right=divisions;
       while(right-left>1){int mid=(left+right)/2;const Query *q=queries+(size_t)(k-1)*divisions+mid;float hv=profile[q->lo]*q->left+profile[q->hi]*q->f;
        if((q->el>0 && q->el>hv)==previous)left=mid;else right=mid;}
       lo=times[k-1]+(double)left/divisions;hi=times[k-1]+(double)right/divisions;
      }else while(hi-lo>precision){double mid=(lo+hi)*.5;Sun s=solar(mid);if(visible(profile,&s)==previous)lo=mid;else hi=mid;}
      append(e,(uint32_t)nearbyint((lo+hi)*500));previous=current;
     }
    }
   }
   size_t count=e->size-begin;if(count>UINT32_MAX)fail("too many events for receiver");counts[cell]=(uint32_t)count;
  }
  total+=e->size;free(profiles);int done;
  #pragma omp atomic capture
  done=++completed;
  if(done%((tiles/40)?tiles/40:1)==0 || done==tiles){fprintf(stderr,"native events tiles %d/%d\n",done,tiles);}
 }
 if(fd>=0)close(fd);
 free(index);if(raw_base)munmap(raw_base,raw_bytes);
 if(total>UINT32_MAX)fail("event count exceeds uint32 format");
 Mapping init=npy_map(out,"initial.npy","|u1",cells,1);memcpy(init.data,initial,cells);free(initial);finish_map(&init);
 Mapping offsets=npy_map(out,"offsets.npy","<u4",cells+1,4);uint32_t *off=offsets.data;off[0]=0;
 for(size_t i=0;i<cells;i++)off[i+1]=off[i]+counts[i];
 free(counts);
 Mapping csr=npy_map(out,"times.npy","<u4",total,4);uint32_t *ct=csr.data;
 for(int tile=0;tile<tiles;tile++){
  int y=(tile/cols)*TH,x=(tile%cols)*TW,h=H-y<TH?H-y:TH,w=W-x<TW?W-x:TW;size_t pos=0;
  for(int cy=0;cy<h;cy++)for(int cx=0;cx<w;cx++){size_t cell=(size_t)(y+cy)*W+x+cx;size_t n=off[cell+1]-off[cell];if(n)memcpy(ct+off[cell],events[tile].times+pos,n*4);pos+=n;}
  free(events[tile].times);
 }
 free(events);
 Mapping sorted_times=npy_map(out,"sorted-times.npy","<u4",total,4),sorted_cells=npy_map(out,"sorted-cells.npy","<u4",total,4);
 uint32_t *st=sorted_times.data,*sc=sorted_cells.data;size_t *bins=zeroed(86402,sizeof(size_t)),*cursor=zeroed(86402,sizeof(size_t));
 for(size_t i=0;i<total;i++){if(ct[i]>86400000)fail("event outside UTC day");bins[ct[i]/1000+1]++;}
 for(int i=1;i<86402;i++)bins[i]+=bins[i-1];
 memcpy(cursor,bins,86402*sizeof(size_t));
 for(size_t cell=0;cell<cells;cell++)for(uint32_t j=off[cell];j<off[cell+1];j++){size_t dst=cursor[ct[j]/1000]++;st[dst]=ct[j];sc[dst]=(uint32_t)cell;}
 /* In-place counting permutation within each second. O(events), no large
  * comparison-sort workspace, stable cell order is unnecessary for XOR replay. */
 #pragma omp parallel for schedule(dynamic,16)
 for(int b=0;b<86401;b++){
  size_t begin=bins[b],end=bins[b+1];if(end-begin<2)continue;
  size_t hist[1000]={0},start[1001],next[1000];
  for(size_t i=begin;i<end;i++)hist[st[i]%1000]++;
  start[0]=begin;for(int j=0;j<1000;j++){start[j+1]=start[j]+hist[j];next[j]=start[j];}
  for(int j=0;j<1000;j++)while(next[j]<start[j+1]){
   size_t p=next[j];unsigned key=st[p]%1000;
   if(key==(unsigned)j){next[j]++;continue;}
   size_t q=next[key]++;uint32_t tmp=st[p];st[p]=st[q];st[q]=tmp;tmp=sc[p];sc[p]=sc[q];sc[q]=tmp;
  }
 }
 free(bins);free(cursor);finish_map(&offsets);finish_map(&csr);finish_map(&sorted_times);finish_map(&sorted_cells);
 for(int b=0;b<nb;b++)free(blocks[b].knots);
 free(blocks);free(sun);free(times);free(queries);
 path_join(path,sizeof(path),out,"native-stats.json");FILE *stats=fopen(path,"w");if(!stats)fail(path);
 fprintf(stats,"{\"seconds\":%.9f,\"events\":%llu,\"cells\":%zu,\"threads\":%d,\"evaluated_states\":%llu}\n",clock_seconds()-started,(unsigned long long)total,cells,threads,(unsigned long long)checked);if(fclose(stats))fail("stats flush");
 fprintf(stderr,"native events complete: %llu transitions, %.3f seconds\n",(unsigned long long)total,clock_seconds()-started);return 0;
}
