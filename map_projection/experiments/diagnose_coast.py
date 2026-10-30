"""Reproduce the bathymetry artefact locally and write a comparison figure."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from map_projection.native import Native
from map_projection.solar import position


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--before',required=True,type=Path)
    parser.add_argument('--after',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();before=np.load(args.before);after=np.load(args.after)
    old,new=before['z'],after['z'];gh,gw=new.shape;ts=after['terrain_size'];native=Native()
    sun,_=position('2026-10-13T12:51:00+02:00',45.95,10.65)
    latitude=np.linspace(48.4,43.5,gh);longitude=np.linspace(4.9,16.4,gw)
    ys=np.flatnonzero((latitude>=43.5)&(latitude<=44.6));xs=np.flatnonzero((longitude>=8)&(longitude<=11))
    extent=(longitude[xs[0]],longitude[xs[-1]],latitude[ys[-1]],latitude[ys[0]])
    fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
    metrics={}
    for column,(label,z) in enumerate([('Before: negative depths kept to -1000 m',old),('After: flat sea before resampling',new)]):
        dy,dx=np.gradient(z,ts[1]/gh,ts[0]/gw)
        normals=np.stack((-dx,dy,np.ones_like(z)),axis=-1);normals/=np.linalg.norm(normals,axis=-1,keepdims=True)
        horizon=np.empty_like(z)
        native.lib.horizon(z,gw,gh,ts[0]/gw,ts[1]/gh,sun,horizon)
        diffuse=np.maximum(0,np.einsum('ijk,k->ij',normals,sun));diffuse[horizon>z+float(after['bias'])]=0
        crop=z[np.ix_(ys,xs)];shade=diffuse[np.ix_(ys,xs)]
        im=axes[0,column].imshow(crop,extent=extent,origin='upper',cmap='terrain',vmin=-1000,vmax=1500,aspect='auto')
        axes[1,column].imshow(shade,extent=extent,origin='upper',cmap='gray',vmin=0,vmax=1,aspect='auto')
        for ax in axes[:,column]:
            ax.set_xlabel('Longitude (east)');ax.set_ylabel('Latitude (north)')
            ax.contour(longitude[xs],latitude[ys],new[np.ix_(ys,xs)],levels=[1],colors=['tab:orange'],linewidths=.55)
        axes[0,column].set_title(label)
        axes[1,column].set_title('Diffuse shading + cast shadows (same sun)')
        metrics[label]={'minimum_m':float(z.min()),'negative_cells':int((z<0).sum()),
                        'coast_max_slope':float(np.max(np.hypot(dx,dy)[np.ix_(ys,xs)]))}
    fig.colorbar(im,ax=axes[0,:],label='Elevation (m)');fig.suptitle('Ligurian / Tuscan coast — orange line: approximate land edge')
    args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=160)
    print(metrics)


if __name__=='__main__':main()
