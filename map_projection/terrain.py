"""Pi-side, bounded-memory preprocessing of STL and geographic DEM sources."""
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
from .prepare_terrain import prepared


def prepare_top(source, destination):
    """Keep upward-facing triangles verbatim. Never weld or decimate the STL.

    Appropriate for the exposed top of this terrain solid. Projection still uses
    the original mesh (including silhouette walls), while this accelerates top
    height extraction. At most ~8 MB of working arrays, independent of file size.
    """
    dtype=np.dtype([("normal","<f4",3),("vertices","<f4",(3,3)),("attribute","<u2")])
    destination=Path(destination)
    temp=destination.with_suffix(".tmp")
    kept=0
    try:
        with open(source,"rb") as src, temp.open("wb") as out:
            header=src.read(84)
            count=struct.unpack_from("<I",header,80)[0]
            out.write(b"map_projection upward-facing source triangles".ljust(80,b"\0"));out.write(struct.pack("<I",0))
            for start in range(0,count,32768):
                chunk=np.fromfile(src,dtype=dtype,count=min(32768,count-start))
                v=chunk["vertices"]
                a,b=v[:,1]-v[:,0],v[:,2]-v[:,0]
                upward=a[:,0]*b[:,1]-a[:,1]*b[:,0]>1e-12
                selected=chunk[upward]
                selected.tofile(out);kept+=len(selected)
            out.seek(80);out.write(struct.pack("<I",kept))
        temp.replace(destination)
    finally:
        if temp.exists():temp.unlink()
    return kept


class Terrain:
    def __init__(self,native,mesh,bounds,cache,mesh_key,dem_path=None):
        self.native,self.mesh,self.bounds,self.cache,self.mesh_key=native,mesh,bounds,cache,mesh_key
        self.dem_path=Path(dem_path) if dem_path else None
        self.key=None
        self.data=None

    def get(self,config,status):
        s,g=config["sun"],config["geo"]
        width=s["grid_width"]
        mw,mh=self.bounds[3:5]-self.bounds[:2]
        height=max(2,round(width*mh/mw))
        dem_signature=None
        source_path=None
        if s["terrain"]=="dem":
            if self.dem_path is None or not self.dem_path.is_file():
                raise ValueError("DEM source not found. Start with --dem path/to/bboxmerge_v2.npy. See README for its geographic extent contract.")
            source_path,policy=prepared(self.dem_path,g)
            stat=source_path.stat()
            dem_signature=[str(source_path.resolve()),stat.st_size,stat.st_mtime_ns,policy]
        key=hashlib.sha256(json.dumps(["terrain-v4-land-mask",self.mesh_key,self.native.signature,s["terrain"],width,height,g,dem_signature],sort_keys=True).encode()).hexdigest()[:24]
        if key!=self.key:
            path=self.cache/(key+"-terrain.npy")
            if path.exists():self.data=np.load(path,mmap_mode="r",allow_pickle=False)
            elif s["terrain"]=="stl":
                status(f"Sampling printed surface at {width} × {height}")
                matrix=np.array([[width/mw,0,0,width/2],[0,-height/mh,0,height/2],[0,0,-1,0]],np.float64)
                # A top-down min-depth pass retains only the exposed surface,
                # even with duplicate/hidden internal triangles. No full-size
                # intermediate STL is needed on the Pi's SD card.
                depths,_=self.native.raster(self.mesh,self.bounds,matrix,width,height,False)
                self.data=np.ascontiguousarray(-depths)
                self.data[~np.isfinite(self.data)]=self.bounds[2]
                np.save(path,self.data)
            else:
                status("Sampling original DEM in bounded row blocks")
                source=np.load(source_path,mmap_mode="r",allow_pickle=False)
                if source.ndim!=2 or not np.issubdtype(source.dtype,np.number):raise ValueError("DEM must be a numeric two-dimensional NPY raster.")
                if list(source.shape)!=policy['shape']:raise ValueError('Prepared land surface dimensions do not match its metadata.')
                # Source NPY is a north-up raster covering exactly config.geo.
                # Pixel-centre bilinear sampling uses only two source rows at once.
                x=(self.bounds[0]+(np.arange(width)+.5)*mw/width)/g["width_mm"]*source.shape[1]-.5
                x=np.clip(x,0,source.shape[1]-1);x0=np.floor(x).astype(int);x1=np.minimum(x0+1,source.shape[1]-1);tx=x-x0
                self.data=np.empty((height,width),np.float32)
                for row in range(height):
                    raw_northing=self.bounds[4]-(row+.5)*mh/height
                    y=np.clip((1-raw_northing/g["height_mm"])*source.shape[0]-.5,0,source.shape[0]-1)
                    y0=int(y);y1=min(y0+1,source.shape[0]-1);ty=y-y0
                    a=source[y0,x0]*(1-tx)+source[y0,x1]*tx
                    b=source[y1,x0]*(1-tx)+source[y1,x1]*tx
                    self.data[row]=a*(1-ty)+b*ty
                np.save(path,self.data)
            self.key=key
        if s["terrain"]=="stl":physical=(float(mw),float(mh))
        else:
            # Local equirectangular metres; the broad Alps extent is explicitly
            # approximated by a single latitude for live geographical shading.
            radius=6371008.8
            physical=(radius*math.radians(g["lon_max"]-g["lon_min"])*math.cos(math.radians((g["lat_min"]+g["lat_max"])/2))*mw/g["width_mm"],
                      radius*math.radians(g["lat_max"]-g["lat_min"])*mh/g["height_mm"])
        return self.data,(float(mw),float(mh)),physical
