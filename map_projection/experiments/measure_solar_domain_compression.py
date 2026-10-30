"""Additional visibility-preserving pruning inside an annual solar band.

This is an experiment, not the shipping lossless skyline format. It changes
irrelevant horizon values and stores interval classifications separately.
"""
import json
from pathlib import Path
import numpy as np
from map_projection.skyline_store import Zstd,encode_predicted,elevation_levels
from map_projection.solar import position

def elevation_band(azimuth,latitude=45.95,declination_bound=24.):
    phi=np.deg2rad(latitude);az=np.deg2rad(azimuth);c=np.cos(phi)*np.cos(az);s=np.sin(phi);r=np.hypot(c,s);phase=np.arctan2(c,s);bound=np.sin(np.deg2rad(declination_bound))
    return np.maximum(0,np.rad2deg(np.arcsin(np.clip(-bound/r,-1,1))-phase)),np.rad2deg(np.arcsin(np.clip(bound/r,-1,1))-phase)

def main():
    bank=np.load('map_projection/.cache/horizons4096/horizons.npy',mmap_mode='r');h,w=bank.shape[1:];angles=np.arange(55,305.001,.25);emin,emax=elevation_band(angles)
    # Endpoints bound the monotonic band within each interval; 180 is a knot.
    bottom=np.minimum(emin[:-1],emin[1:]);top=np.maximum(emax[:-1],emax[1:]);codec=Zstd();total=compressed=unused_total=interval_total=clear_total=blocked_total=0
    for yi in range(4):
        y=min(h-16,round((yi+.5)*h/4))
        for xi in range(4):
            x=min(w-256,round((xi+.5)*w/4));a=np.ascontiguousarray(bank[220:1221,y:y+16,x:x+256]);flags=np.zeros((1000,16,256),np.uint8)
            flags[np.maximum(a[:-1],a[1:])<bottom[:,None,None]-.00001]=1
            flags[np.minimum(a[:-1],a[1:])>top[:,None,None]+.00001]=2
            needed=np.zeros(a.shape,bool);needed[:-1]|=flags==0;needed[1:]|=flags==0
            modified=a.copy();modified[~needed]=elevation_levels()[0];data,mode=encode_predicted(modified);assert mode==1
            compressed+=len(codec.compress(data,6))+len(codec.compress(flags.tobytes(),6));total+=a.size
            unused_total+=int((~needed).sum());interval_total+=flags.size;clear_total+=int((flags==1).sum());blocked_total+=int((flags==2).sum())
            # Verify every interval proof using independent dense random points
            # sampled inside its permitted elevation range.
            rng=np.random.default_rng(5)
            for _ in range(10):
                f=float(rng.random());az=angles[:-1]+f*.25;low,high=elevation_band(az);e=low[:,None,None]+(high-low)[:,None,None]*rng.random((1000,1,1));original=(a[:-1]*(1-f)+a[1:]*f)<e
                rebuilt=(modified[:-1]*(1-f)+modified[1:]*f)<e;rebuilt[flags==1]=True;rebuilt[flags==2]=False
                assert np.array_equal(original,rebuilt)
        print('row',yi,'done',flush=True)
    result={'declination_bound_degrees':24,'sea_or_land_independent':True,'sample_values':total,'removable_vertex_fraction':unused_total/total,'always_lit_interval_fraction':clear_total/interval_total,'always_shadowed_interval_fraction':blocked_total/interval_total,'projected_GB':compressed/(total*4)*(1001*h*w*4)/1e9,'qualification':'Visibility equivalent only inside the declared annual sun band, relative to the existing piecewise-linear skyline bank. Not lossless general skyline data.'}
    Path('map_projection/experiments/results/2026-10-25/skyline-solar-domain.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
