"""Pinhole projection, planar calibration, and geographic conventions."""
import itertools
import math
import numpy as np


def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def map_box(bounds):
    width, height = bounds[3:5] - bounds[:2]
    return np.array(list(itertools.product((-width/2, width/2), (-height/2, height/2), (bounds[2], bounds[5]))))


def apply(matrix, points, perspective=True):
    points = np.atleast_2d(points)
    q = points @ matrix[:, :3].T + matrix[:, 3]
    return q[:, :2] / q[:, 2:3] if perspective else q


def homography(source, destination):
    source, destination = np.asarray(source, float), np.asarray(destination, float)
    edges = np.roll(destination, -1, axis=0) - destination
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0]*following[:, 1] - edges[:, 1]*following[:, 0]
    # Screen order is NW, NE, SE, SW. Prevent crossed/degenerate calibration.
    if not (np.all(cross > 1) or np.all(cross < -1)):
        raise ValueError("Corners must form a convex NW, NE, SE, SW quadrilateral with nonzero area.")
    rows, rhs = [], []
    for (x, y), (u, v) in zip(source, destination):
        rows.extend(([x,y,1,0,0,0,-u*x,-u*y], [0,0,0,x,y,1,-v*x,-v*y]))
        rhs.extend((u, v))
    result = np.append(np.linalg.solve(rows, rhs), 1).reshape(3, 3)
    if not np.all(np.isfinite(result)) or np.linalg.cond(result) > 1e12:
        raise ValueError("Corner calibration is singular or too extreme.")
    return result


def projector(config, bounds):
    p = config["projector"]
    w, h = config["output"]["width"], config["output"]["height"]
    origin = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
    if p["z"] <= bounds[5] + 10:
        raise ValueError("Lens Z must be at least 10 mm above the tallest mesh point.")
    forward = unit(-origin)  # Always aimed at map-frame origin.
    right = unit(np.cross(forward, [0, 1, 0]))
    down = np.cross(forward, right)
    a = math.radians(p["roll"])
    r, d = math.cos(a)*right + math.sin(a)*down, -math.sin(a)*right + math.cos(a)*down
    rotation = np.vstack([r, d, forward])
    focal = w * p["throw_ratio"]
    intrinsics = np.array([[focal, 0, w/2+p["shift_x"]], [0, focal, h/2+p["shift_y"]], [0,0,1]])
    base = intrinsics @ np.column_stack((rotation, -rotation @ origin))
    if config["output"].get("mirror",False):
        base=np.array([[-1.,0.,w],[0.,1.,0.],[0.,0.,1.]]) @ base
    width, height = bounds[3:5] - bounds[:2]
    corners = np.array([[-width/2,height/2,0], [width/2,height/2,0], [width/2,-height/2,0], [-width/2,-height/2,0]])
    source = apply(base, corners)
    target = source if config["corners"] is None else np.asarray(config["corners"], np.float64)
    matrix = homography(source, target) @ base
    depths = map_box(bounds) @ matrix[2,:3] + matrix[2,3]
    if depths.min() <= 1:
        raise ValueError("The map crosses the projector/homography horizon; change pose or reset corners.")
    return matrix, np.linalg.inv(matrix[:,:3]), origin, target


def unproject(pixel, depth, inverse, origin):
    """Integer (u,v) identifies a pixel; the optical ray goes through its centre."""
    u, v = pixel
    return origin + depth * (inverse @ [u+0.5, v+0.5, 1])


def sun_camera(sun, bounds, size):
    right = unit(np.cross(sun, [0, 0, 1])) if abs(sun[2]) < .999999 else np.array([1.,0.,0.])
    up = np.cross(right, sun)
    rotation = np.vstack((right, -up, -sun))
    projected = map_box(bounds) @ rotation.T
    low, high = projected.min(axis=0), projected.max(axis=0)
    scale = (size-4) / np.maximum(high[:2]-low[:2], 1e-6)
    m = np.zeros((3,4), np.float64)
    m[:2,:3] = rotation[:2] * scale[:,None]
    m[:2,3] = 2-low[:2]*scale
    m[2,:3] = rotation[2]
    return m, float(max(1/scale))


def geo_to_xy(latitude, longitude, bounds, geo):
    # The generator flips Y at export (_flip_mesh_across_x_axis), so STL Y
    # increases north. AEQD chooses aspect ratio only, not vertex XY.
    if not geo["lat_min"] <= latitude <= geo["lat_max"] or not geo["lon_min"] <= longitude <= geo["lon_max"]:
        raise ValueError("Coordinate is outside the configured map extent.")
    raw_x = (longitude-geo["lon_min"])/(geo["lon_max"]-geo["lon_min"])*geo["width_mm"]
    raw_y = (latitude-geo["lat_min"])/(geo["lat_max"]-geo["lat_min"])*geo["height_mm"]
    return raw_x-(bounds[0]+bounds[3])/2, raw_y-(bounds[1]+bounds[4])/2


def xy_to_geo(x, y, bounds, geo):
    raw_x = x+(bounds[0]+bounds[3])/2
    raw_y = y+(bounds[1]+bounds[4])/2
    return {"latitude": geo["lat_min"]+raw_y/geo["height_mm"]*(geo["lat_max"]-geo["lat_min"]),
            "longitude": geo["lon_min"]+raw_x/geo["width_mm"]*(geo["lon_max"]-geo["lon_min"])}
