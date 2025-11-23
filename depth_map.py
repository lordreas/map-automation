from gpx2pdf import build_elevation_map
import numpy as np
from skimage.measure import find_contours

# elev_map, elev_transform = build_elevation_map(43, 48, 5, 17)
elev_map, elev_transform = build_elevation_map(45.5, 46, 11, 11.5)

import matplotlib.pyplot as plt
import matplotlib
from math import ceil
from matplotlib.patches import Rectangle
matplotlib.use("TkAgg")

interval = 200  # elevation interval in meters
min_elev = np.nanmin(elev_map)
max_elev = np.nanmax(elev_map)
levels = np.arange(
    np.floor(min_elev / interval) * interval,
    np.ceil(max_elev / interval) * interval + interval,
    interval,
)

contour_lines = {}
for level in levels:
    contours = find_contours(elev_map, level)
    contour_lines[level] = [
        [(int(round(col)), int(round(row))) for row, col in contour]
        for contour in contours
        if contour.size > 0
    ]

tile_height = 300
tile_width = 300
margin_px = 20

img_height, img_width = elev_map.shape
grid_rows = ceil(img_height / tile_height)
grid_cols = ceil(img_width / tile_width)

def make_box(y0, y1, x0, x1):
    return {"y0": int(y0), "y1": int(y1), "x0": int(x0), "x1": int(x1)}

def clip_ring_segments(line_coords, outer_box, inner_box):
    segments = []
    current = []
    for x, y in line_coords:
        in_outer = (
            outer_box["x0"] <= x < outer_box["x1"]
            and outer_box["y0"] <= y < outer_box["y1"]
        )
        in_inner = (
            inner_box
            and inner_box["x0"] <= x < inner_box["x1"]
            and inner_box["y0"] <= y < inner_box["y1"]
        )
        if in_outer and not in_inner:
            current.append((x, y))
        else:
            if len(current) > 1:
                segments.append(current)
            current = []
    if len(current) > 1:
        segments.append(current)
    return segments

tile_results = []
for row in range(grid_rows):
    for col in range(grid_cols):
        y0 = row * tile_height
        y1 = min((row + 1) * tile_height, img_height)
        x0 = col * tile_width
        x1 = min((col + 1) * tile_width, img_width)

        outer_box = make_box(
            max(0, y0 - margin_px),
            min(img_height, y1 + margin_px),
            max(0, x0 - margin_px),
            min(img_width, x1 + margin_px),
        )

        inner_y0 = y0 + margin_px
        inner_y1 = y1 - margin_px
        inner_x0 = x0 + margin_px
        inner_x1 = x1 - margin_px
        inner_box = (
            make_box(inner_y0, inner_y1, inner_x0, inner_x1)
            if inner_y1 > inner_y0 and inner_x1 > inner_x0
            else None
        )

        cropped = {}
        for level, lines in contour_lines.items():
            ring_segments = []
            for coords in lines:
                ring_segments.extend(
                    clip_ring_segments(coords, outer_box, inner_box)
                )
            if ring_segments:
                cropped[level] = ring_segments

        tile_results.append(
            {
                "row": row,
                "col": col,
                "outer": outer_box,
                "inner": inner_box,
                "segments": cropped,
            }
        )

adjacent_tiles = []
for row in range(grid_rows):
    row_tiles = sorted(
        [tile for tile in tile_results if tile["row"] == row],
        key=lambda item: item["col"],
    )
    if len(row_tiles) >= 3:
        adjacent_tiles = row_tiles[:3]
        break
if not adjacent_tiles:
    adjacent_tiles = tile_results[: min(3, len(tile_results))]

if adjacent_tiles:
    fig_tiles, axes = plt.subplots(
        1, len(adjacent_tiles), figsize=(5 * len(adjacent_tiles), 5)
    )
    if len(adjacent_tiles) == 1:
        axes = [axes]
    for ax, info in zip(axes, adjacent_tiles):
        outer = info["outer"]
        inner = info["inner"]
        tile_img = elev_map[outer["y0"] : outer["y1"], outer["x0"] : outer["x1"]]
        ax.imshow(tile_img, cmap="terrain", origin="upper")
        for segments in info["segments"].values():
            for segment in segments:
                xs = [x - outer["x0"] for x, _ in segment]
                ys = [y - outer["y0"] for _, y in segment]
                ax.plot(xs, ys, color="black", linewidth=0.9)
        ax.add_patch(
            Rectangle(
                (0, 0),
                outer["x1"] - outer["x0"],
                outer["y1"] - outer["y0"],
                edgecolor="white",
                facecolor="none",
                linewidth=1.0,
            )
        )
        if inner:
            ax.add_patch(
                Rectangle(
                    (inner["x0"] - outer["x0"], inner["y0"] - outer["y0"]),
                    inner["x1"] - inner["x0"],
                    inner["y1"] - inner["y0"],
                    edgecolor="red",
                    facecolor="none",
                    linestyle="--",
                    linewidth=1.0,
                )
            )
        ax.set_title(f"Tile ({info['row']}, {info['col']})")
        ax.set_axis_off()

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(elev_map, cmap="terrain")
for level, lines in contour_lines.items():
    for coords in lines:
        if len(coords) < 2:
            continue
        xs, ys = zip(*coords)
        ax.plot(xs, ys, color="black", linewidth=0.8)
ax.set_title("Elevation Contours")
ax.set_axis_off()
plt.show()