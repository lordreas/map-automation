import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ROOT = Path(__file__).resolve().parent
DEFAULT = {
    "output": {"width": 1920, "height": 1080, "refresh": 60, "display": 0, "fullscreen": False, "mirror": False},
    "projector": {"x": 0., "y": 750., "z": 2064., "roll": 0., "throw_ratio": 1.2, "shift_x": 0., "shift_y": 0.},
    "corners": None,
    "geo": {"lat_min": 43.5, "lat_max": 48.4, "lon_min": 4.9, "lon_max": 16.4,
            "width_mm": 1500., "height_mm": 877.2606049656247},
    "sun": {"time": "2026-10-13T12:00:00+02:00", "timezone": "Europe/Zurich", "latitude": 45.95, "longitude": 10.65,
            "shadow_size": 2048, "ambient": 0., "bias_mm": .08, "step_minutes": 2, "playing": False,
            "renderer": "live", "terrain": "stl", "grid_width": 2048,"backend":"auto",
            "minutes_per_second":15.,"enabled":True,"date_mode":"today",
            "normalize":True,"shadow_floor":True,"shadow_floor_amount":.1},
    "pattern": "sun",
    "pixel": {"x": 960, "y": 540},
    "track": {"enabled": False, "color": None, "width": 4.},
    "layers": {
        "white": {"enabled": True, "opacity": 1.},
        "satellite": {"enabled": False, "opacity": 1., "month": 0, "brightness":1.25},
        "streets": {"enabled": False, "opacity": 1., "width": 4., "color": "#ead8ad"},
        "semantic": {"enabled": False, "opacity": 1.},
        "contours": {"enabled": False, "interval": 200., "level": 2000., "single": False, "fill":"line", "width": 1.5, "color": "#ffffff"},
        "rivers": {"enabled": False, "river_thickness": 5., "color": "#48baff"},
        "divides": {"enabled": False, "width": 1.5, "color": "#cba5ff"},
        "peaks": {"enabled": False, "min_elevation": 2000., "max_elevation": 5000., "width": 5., "color": "#ffcf70"},
    },
    "optics": {"chromatic_pixels": 0.},
}


def merge(base, patch):
    if not isinstance(patch, dict):
        raise ValueError("Configuration must be a JSON object.")
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if key not in result:
            raise ValueError(f"Unknown setting: {key}")
        result[key] = merge(result[key], value) if isinstance(result[key], dict) else value
    return result


def validate(c):
    def number(value, low, high, name, integer=False):
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}.")
        if integer and type(value) is not int:
            raise ValueError(f"{name} must be an integer.")
    number(c["optics"]["chromatic_pixels"],-16,16,"Chromatic compensation")
    o = c["output"]
    number(o["width"], 64, 3840, "Width", True)
    number(o["height"], 64, 2160, "Height", True)
    number(o["refresh"], 24, 120, "Refresh rate", True)
    number(o["display"], 0, 15, "Display", True)
    if any(type(v) is not bool for v in (o["fullscreen"],o["mirror"],c["sun"]["playing"])):
        raise ValueError("Fullscreen, mirror and playing must be booleans.")
    p = c["projector"]
    for key in ("x", "y", "z", "shift_x", "shift_y"):
        number(p[key], -20000, 20000, key)
    number(p["roll"], -180, 180, "Roll")
    number(p["throw_ratio"], .1, 10, "Throw ratio")
    if c["corners"] is not None:
        if not isinstance(c["corners"], list) or len(c["corners"]) != 4:
            raise ValueError("Exactly four corners are required.")
        for point in c["corners"]:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("Each corner needs X and Y coordinates.")
            for value in point:
                number(value, -20000, 20000, "Corner")
    s = c["sun"]
    import re
    track=c['track']
    number(track['width'],.5,24,'Track width')
    if type(track['enabled']) is not bool:raise ValueError('Track visibility must be a boolean.')
    if track['color'] is not None and (not isinstance(track['color'],str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',track['color'])):
        raise ValueError('Track color must be #RRGGBB or automatic.')
    for name,layer in c['layers'].items():
        if type(layer['enabled']) is not bool:raise ValueError(f'{name} visibility must be a boolean.')
        if 'opacity' in layer:number(layer['opacity'],0,1,f'{name} opacity')
        if 'width' in layer:number(layer['width'],.5,24,f'{name} width')
        if 'color' in layer and (not isinstance(layer['color'],str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',layer['color'])):raise ValueError(f'{name} color must be #RRGGBB.')
    contours=c['layers']['contours'];peaks=c['layers']['peaks']
    if contours['fill'] not in ('line','above','below'):raise ValueError('Choose an elevation line or fill above/below.')
    number(c['layers']['satellite']['month'],-1,12,'Satellite month',True)
    number(c['layers']['satellite']['brightness'],.25,3,'Satellite brightness')
    for key in ('normalize','shadow_floor'):
        if type(s[key]) is not bool:raise ValueError('Lighting options must be booleans.')
    number(s['shadow_floor_amount'],0,.5,'Shadow visibility')
    number(c['layers']['rivers']['river_thickness'],1,24,'Maximum river thickness')
    number(contours['interval'],10,2000,'Contour interval')
    number(contours['level'],-500,9000,'Selected elevation')
    if type(contours['single']) is not bool:raise ValueError('Single contour must be a boolean.')
    for key in ('min_elevation','max_elevation'):number(peaks[key],-500,9000,'Peak elevation')
    if peaks['min_elevation']>peaks['max_elevation']:raise ValueError('Peak minimum must not exceed maximum.')
    if s["backend"] not in ("auto","gpu","cpu","reference","precise","events"):raise ValueError("Choose automatic, GPU, CPU, precise GPU or the DEM ray reference.")
    if s['backend'] in ('gpu','reference','precise','events') and (s['renderer']!='live' or s['terrain']!='dem'):
        raise ValueError('GPU calculation requires Live mode and geographic DEM terrain.')
    if s["date_mode"] not in ("today","fixed"):raise ValueError("Choose today's date or a fixed date.")
    number(s["minutes_per_second"],1/60,60,"Simulated minutes per second")
    if type(s["enabled"]) is not bool:raise ValueError("Shadow overlay must be a boolean.")
    if s["renderer"] not in ("live","mesh") or s["terrain"] not in ("stl","dem"):
        raise ValueError("Choose live/mesh renderer and stl/dem terrain.")
    if s["terrain"]=="dem" and s["renderer"]!="live":
        raise ValueError("Geographic DEM shading uses the live terrain-grid renderer.")
    if s["grid_width"] not in (512,1024,2048,4096):
        raise ValueError("Terrain grid width must be 512, 1024, 2048 or 4096.")
    if s["shadow_size"] not in (512,1024,2048,4096):
        raise ValueError("Shadow map size must be 512, 1024, 2048 or 4096.")
    number(s["latitude"], -89, 89, "Sun latitude")
    number(s["longitude"], -180, 180, "Sun longitude")
    number(s["ambient"], 0, 1, "Ambient")
    number(s["bias_mm"], 0, 10, "Shadow bias")
    number(s["step_minutes"], 1, 120, "Animation step", True)
    if not isinstance(s["time"], str) or len(s["time"]) > 64:
        raise ValueError("Invalid sun time.")
    if not isinstance(s["timezone"],str) or len(s["timezone"])>100:
        raise ValueError("Use an IANA timezone such as Europe/Zurich.")
    try:zone=ZoneInfo(s["timezone"])
    except (ZoneInfoNotFoundError,ValueError) as exc:
        raise ValueError("Unknown timezone; install tzdata and use an IANA name, e.g. Europe/Zurich.") from exc
    moment = datetime.fromisoformat(s["time"].replace("Z", "+00:00"))
    if moment.tzinfo is None:
        naive=moment
        moment=moment.replace(tzinfo=zone)
        if moment.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)!=naive:
            raise ValueError("This local time does not exist because of a daylight-saving clock change.")
    s["time"]=moment.astimezone(zone).isoformat()
    if s['playing']:
        from .playback import PlaybackClock
        PlaybackClock(s,0)  # Validate the selected local calendar day.
    if c["pattern"] not in ("sun", "calibration", "checkerboard", "columns", "rows", "pixel", "white", "black", "chromatic"):
        raise ValueError("Unknown pattern.")
    number(c["pixel"]["x"], 0, o["width"]-1, "Pixel X", True)
    number(c["pixel"]["y"], 0, o["height"]-1, "Pixel Y", True)
    g=c["geo"]
    for key in ("lat_min","lat_max"):
        number(g[key], -90, 90, key)
    for key in ("lon_min","lon_max"):
        number(g[key], -180, 180, key)
    for key in ("width_mm","height_mm"):
        number(g[key], 1, 20000, key)
    if g["lat_min"] >= g["lat_max"] or g["lon_min"] >= g["lon_max"]:
        raise ValueError("Geographic limits must be increasing.")
    return c


def load(path,timezone_override=None):
    stored=json.loads(path.read_text()) if path.exists() else {}
    rivers=stored.get('layers',{}).get('rivers',{})
    if 'width' in rivers:rivers.setdefault('river_thickness',max(5.,rivers.pop('width')))
    # Migrate settings from the earlier bounded-range / target-FPS controls.
    for key in ('frame_rate','range_start','range_end','loop'):
        stored.get('sun',{}).pop(key,None)
    c=merge(DEFAULT,stored)
    if timezone_override:c['sun']['timezone']=timezone_override
    if not path.exists():
        c["sun"]["time"]=datetime.now(ZoneInfo(c["sun"]["timezone"])).replace(hour=12,minute=0,second=0,microsecond=0).isoformat()
    c["sun"]["playing"] = False
    if c["sun"]["date_mode"]=="today":
        moment=datetime.fromisoformat(c["sun"]["time"])
        zone=ZoneInfo(c["sun"]["timezone"])
        today=datetime.now(zone)
        c["sun"]["time"]=moment.replace(tzinfo=None,year=today.year,month=today.month,day=today.day).isoformat()
    return validate(c)


def save(path, config):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    temp.replace(path)
