#include <vector>
#include <cmath>
#include <cstdint>
#include <utility>
extern "C" void simplify(const float* profiles,int n,int width,double tolerance,uint8_t* keep){
 for(int p=0;p<n;p++){
  const float* a=profiles+(size_t)p*width;uint8_t* k=keep+(size_t)p*width;
  k[0]=k[width-1]=1;std::vector<std::pair<int,int>> stack{{0,width-1}};
  while(!stack.empty()){
   auto [lo,hi]=stack.back();stack.pop_back();double slope=(double(a[hi])-a[lo])/(hi-lo),error=0;int split=-1;
   for(int i=lo+1;i<hi;i++){double e=std::abs(double(a[i])-(a[lo]+slope*(i-lo)));if(e>error){error=e;split=i;}}
   if(error>tolerance&&split>=0){k[split]=1;stack.push_back({lo,split});stack.push_back({split,hi});}
  }
 }
}
