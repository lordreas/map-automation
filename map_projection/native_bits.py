"""GIL-free, bounded-memory C bitset updates for visibility replay."""
import ctypes as C,hashlib,os,shutil,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
class Bits:
 def __init__(self,nwords,threads=4):
  source=ROOT/'native/event_bits.c';key=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
  target=ROOT/'.cache'/('event-bits-'+key+'.so');target.parent.mkdir(exist_ok=True)
  if not target.exists():
   compiler=shutil.which('cc') or shutil.which('gcc')
   if not compiler:raise RuntimeError('A C compiler is required for event replay')
   temporary=target.with_name(target.name+'.tmp-'+str(os.getpid()))
   try:
    subprocess.run([compiler,'-O3','-std=c99','-shared','-fPIC','-fopenmp',str(source),'-o',str(temporary)],check=True);temporary.replace(target)
   finally:
    if temporary.exists():temporary.unlink()
  self.lib=C.CDLL(str(target));self.threads=min(threads,os.cpu_count() or 1);self.nwords=nwords
  self.scratch=np.zeros((self.threads,nwords),np.uint32)
  u32=np.ctypeslib.ndpointer(dtype=np.uint32,flags='C_CONTIGUOUS')
  self.lib.flip_event_bits.argtypes=[u32,u32,C.c_uint64,C.c_uint32,C.c_int,u32];self.lib.flip_event_bits.restype=None
 def flip(self,words,cells):self.lib.flip_event_bits(words,cells,len(cells),self.nwords,self.threads,self.scratch)
