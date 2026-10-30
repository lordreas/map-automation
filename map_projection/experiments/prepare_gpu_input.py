"""Prepare corrected DEM data locally, retaining an existing exact 4K lookup.

The input is gpu-input.npz from benchmark_cpu.py. Neither this command nor the
GPU benchmarks changes application settings. The DEM must cover config.geo.
"""
import argparse
import copy
from pathlib import Path
import numpy as np
from map_projection.config import DEFAULT,ROOT,load
from map_projection.native import Native
from map_projection.terrain import Terrain


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--config',type=Path,help='Geographic extent of the source DEM; defaults to repository Alps extent.')
    parser.add_argument('--mesh',type=Path,default=ROOT.parent/'stl_files/full_map_0.4mm.stl')
    parser.add_argument('--dem',type=Path,default=ROOT.parent/'elevation_tiles/elevation_43.5_48.4_4.9_16.4_shape_11697x20000_resamp_average_bboxmerge_v2.npy')
    args=parser.parse_args()
    with np.load(args.input,allow_pickle=False) as data:
        original_shape=data['z'].shape
        arrays={key:data[key] for key in ['lookup','bias','ambient']}
    config=load(args.config) if args.config else copy.deepcopy(DEFAULT)
    config['sun'].update(terrain='dem',grid_width=original_shape[1])
    native=Native();bounds=native.scan(args.mesh);stat=args.mesh.stat()
    terrain=Terrain(native,args.mesh,bounds,ROOT/'.cache',
                    f'{args.mesh.resolve()}:{stat.st_size}:{stat.st_mtime_ns}',args.dem)
    z,_,physical=terrain.get(config,print)
    if z.shape!=original_shape:raise ValueError('Mesh extent does not match the saved projector lookup terrain grid.')
    np.savez_compressed(args.output,z=z,terrain_size=np.array(physical),**arrays)
    print(f'Wrote {args.output} ({args.output.stat().st_size/1e6:.1f} MB); terrain {z.shape}, range {z.min():.1f}–{z.max():.1f} m')


if __name__=='__main__':main()
