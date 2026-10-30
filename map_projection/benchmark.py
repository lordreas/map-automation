"""Run on the actual board to measure its live shadow budget, without a GUI."""
import argparse
import copy
import json
import platform
import statistics
import time

from . import config as settings
from .engine import Engine


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--width",type=int,default=1920)
    parser.add_argument("--height",type=int,default=1080)
    parser.add_argument("--grid",type=int,default=2048)
    parser.add_argument("--frames",type=int,default=5)
    parser.add_argument("--terrain",choices=("stl","dem"),default="stl")
    args=parser.parse_args()
    if not 1<=args.frames<=120:parser.error("--frames must be 1–120")
    config=copy.deepcopy(settings.DEFAULT)
    config["output"].update(width=args.width,height=args.height)
    config["pixel"]={"x":args.width//2,"y":args.height//2}
    config["sun"].update(grid_width=args.grid,terrain=args.terrain)
    engine=Engine(settings.ROOT.parent/"stl_files"/"full_map_0.4mm.stl",config,
                  dem_path=settings.ROOT.parent/"elevation_tiles"/"elevation_43.5_48.4_4.9_16.4_shape_11697x20000_resamp_average_bboxmerge_v2.npy")
    def ready():
        while True:
            state=engine.snapshot()
            if state["error"]:raise RuntimeError(state["error"])
            if state["rendered_revision"]==state["revision"]:return state
            time.sleep(.05)
    try:
        ready()
        timings=[]
        for i in range(args.frames):
            engine.update({"sun":{"time":f"2026-10-13T12:{i//60:02d}:{i%60:02d}+02:00"}},persist=False)
            state=ready();timings.append(state["timings"]["frame_seconds"])
        print(json.dumps({"platform":platform.platform(),"machine":platform.machine(),
                          "output":[args.width,args.height],"terrain":args.terrain,"grid":state["timings"].get("terrain_grid"),
                          "frame_seconds":timings,"median_seconds":statistics.median(timings),
                          "note":"Calculation timing only; does not measure HDMI scanout or presentation."},indent=2))
    finally:engine.close();engine.worker.join(timeout=5)


if __name__=="__main__":main()
