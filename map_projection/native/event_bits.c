#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <omp.h>

/* Private thread bitmaps avoid contested atomics and preserve duplicate parity.
 * The only recurring GPU upload is one bit per terrain receiver (1.23 MB).
 */
void flip_event_bits(uint32_t *words,const uint32_t *cells,uint64_t count,
                     uint32_t nwords,int threads,uint32_t *scratch) {
 if(count<16384 || threads<2){
  for(uint64_t i=0;i<count;i++){uint32_t c=cells[i];words[c>>5]^=1u<<(c&31);}
  return;
 }
 memset(scratch,0,(size_t)threads*nwords*4);
 #pragma omp parallel num_threads(threads)
 {
  uint32_t *local=scratch+(size_t)omp_get_thread_num()*nwords;
  #pragma omp for schedule(static)
  for(uint64_t i=0;i<count;i++){uint32_t c=cells[i];local[c>>5]^=1u<<(c&31);}
  #pragma omp for schedule(static)
  for(uint32_t w=0;w<nwords;w++){
   uint32_t value=0;for(int t=0;t<threads;t++)value^=scratch[(size_t)t*nwords+w];words[w]^=value;
  }
 }
}
