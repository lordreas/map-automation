"""Hydrologic topology and phase coordinates for synchronized river reveals."""
import numpy as np


def network_phases(identifiers,downstream,lengths):
 """Return each reach's source/end times in [0,1].

 A single downstream-distance potential defines time over each connected
 network. Short tributaries start later, so every tributary ends exactly when
 the receiving reach begins, independently of its branch length. Edges crossing
 the map boundary remain connected wherever both reaches are in the dataset.
 """
 ids=[int(i) for i in identifiers];lookup={ident:i for i,ident in enumerate(ids)};n=len(ids)
 parent=np.array([lookup.get(int(v),-1) for v in downstream],np.int32)
 lengths=np.maximum(np.asarray(lengths,np.float64),1e-5)
 remaining=np.zeros(n,np.float64);root=np.full(n,-1,np.int32);done=np.zeros(n,bool)
 for start in range(n):
  if done[start]:continue
  path=[];seen=set();i=start
  while i>=0 and not done[i]:
   if i in seen:raise ValueError('River dataset contains a downstream cycle')
   seen.add(i);path.append(i);i=int(parent[i])
  for j in reversed(path):
   p=parent[j]
   if p<0:root[j]=j;remaining[j]=lengths[j]
   else:root[j]=root[p];remaining[j]=remaining[p]+lengths[j]
   done[j]=True
 longest=np.zeros(n,np.float64);np.maximum.at(longest,root,remaining)
 span=longest[root];begin=1-remaining/span;end=begin+lengths/span
 return np.column_stack((begin,end)).astype(np.float32)


def width_weights(discharge):
 q=np.maximum(np.nan_to_num(discharge,nan=0.,posinf=0.,neginf=0.),0.)
 top=max(float(np.percentile(q,99.5)),.1)
 return np.clip(np.log1p(q)/np.log1p(top),0,1).astype(np.float32)
