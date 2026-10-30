# Geographic projection layers

Open the control page and choose **Layers**. Every checkbox is independent.
Backgrounds blend in this order: White, Satellite, Land cover. Road vectors,
contours, rivers, drainage divides, summits, and GPX sit above the backgrounds.
Sunlight and cast shadows multiply the background colors. With every layer
switched off, physical output is black. No map layer contains rendered text,
place names, elevation labels, or road shields. Attribution appears in the
browser controls, outside the projected image.

- **Elevation lines**: all contours with adjustable spacing, or one adjustable
  elevation. GPU uniforms update while dragging. Line width and color are
  adjustable. Heights are geographic DEM meters, with the same cleaned land/sea
  surface used for terrain generation. Sea-level flats do not become outlines.
- **Rivers**: HydroRIVERS v1 reaches, with mean discharge mapped logarithmically
  to line width. `layers.rivers.river_thickness` is the maximum in projector
  pixels (default 5); the minimum is 1 px. Enabling or replaying reveals the
  network downstream over six seconds. A downstream-distance potential defines
  a single clock over every connected network: shorter tributaries begin later,
  and their ends meet the receiving river's advancing front at the confluence.
  River vertices are interpolated by physical distance, not vertex count.
- **Drainage divides**: shared HydroBASINS level-8 catchment boundaries. The
  outer coastlines are removed. Width and color are adjustable. These are basin
  divides, not every local ridge in the DEM.
- **Mountain summits**: 22,778 catalogued GeoNames mountain/peak locations within
  this installation's extent. Minimum and maximum elevations filter before
  rasterization, so overlapping excluded peaks do not hide selected peaks.
  5,828 missing height tags use the cleaned DEM. Marker size and color are
  adjustable. This catalogue is not every unnamed local DEM maximum.
- **Satellite**: a 4096 × 2396 Sentinel-2 cloudless 2016 composite from EOX.
- **Land cover**: ESA WorldCover 2021 classes, including vegetation, water,
  bare ground, built areas, and permanent snow/ice. This is land classification,
  not an invented altitude-based vegetation estimate or today's snow cover.
- **Roads**: label-free OSM motorway/trunk vectors from OpenFreeMap, with width, color, and opacity. Minor roads, ramps, ferries and tunnels are filtered out.
- **GPX**: the existing Geneva–Salzburg route, animated independently of sunlight.
  Width uses geographic map texels; the other vector widths use output pixels.

## Alignment and performance

Every physical pixel uses its **first intersection with the original STL**.
The hit's XY is converted through the existing geographic registration, including
STL offsets, calibration corners, mirror, and projector pose. There is no flat
image homography shortcut. A compact 16-bit geographic lookup preserves better
than 1/16 of a geographic raster texel at 4096 width. Contours additionally cache
projected DEM heights; a single GPU fetch per pixel is sufficient for the
contours-only shader. Recalibration rebuilds lookups and projected heights,
without downloading source maps again. Chromatic compensation still applies to
the final colored image, including every overlay.

All downloaded data stay local. GPU textures and CPU products for disabled map
layers are released when switching configurations; the small projected-height
cache remains available for interactive contours. Dense peak filtering uses a
native exact distance transform outside the UI thread. GPU draws retain full
3840 × 2160 output; preview captures are downsampled from the composed HDMI image.

## Install or update

From the laptop repository root:

```sh
./deploy.sh --setup              # First installation on an already SSH-ready map account
./deploy.sh                      # Subsequent code/data deploy, preserve calibration/settings
./deploy.sh --skip-layers        # Code update, use the installed geographic data
./deploy.sh --caches             # Also transfer prepared horizon/daily event banks
```

The script prepares missing map products on the laptop, rsyncs the source and
bounded data products, installs the display and console services, and enables
next-day generation at **23:30 in the installation timezone**, with a **10 GB**
budget for daily events. It does not configure Git access on the board. Existing
SSH and DHCP/Wi-Fi settings are preserved. `--host` overrides the SSH alias;
`--no-restart` stages a version for inspection.

Preparation alone (laptop preparation requirements are in
`requirements-prepare.txt`):

```sh
.venv/bin/python -m map_projection.prepare_layers --dem elevation_tiles/elevation_43.5_48.4_4.9_16.4_shape_11697x20000_resamp_average_bboxmerge_v2.npy
```

The rendered geographic products occupy about 180 MB, separate from the much
larger sunlight horizon bank. Downloads are shared and reused across deployments.
The HydroRIVERS definition includes catchments ≥10 km² or mean discharge ≥0.1
m³/s, so very small streams are outside its coverage.

Sources and attribution are stored in the product manifest and displayed when
that layer is enabled. The download script validates ZIP checksums and raster
sizes. It does not use the OpenStreetMap public raster tile service or require
an imagery API key.

For an actual board compositing benchmark (requires the running installation):

```sh
ssh projector 'cd /home/map/map-automation && .venv/bin/python -m map_projection.experiments.benchmark_layers'
```

The benchmark finishes every GPU frame separately, because batching opaque
frames on a tile-based GPU can discard overwritten frames and report a false
rate. The reported rate covers compositing; HDMI scanout, sunlight computation,
chromatic compensation, and preview captures add their own costs.

## Monthly imagery and lighting

The Satellite image-date selector offers the detailed annual Sentinel-2 mosaic,
twelve real NASA Blue Marble monthly composites, or an automatic choice from the
selected calendar month. NASA's cloud-free data are observations from **2004 at
500 m**, not current snow forecasts or synthetic snow painted on the DEM. The
monthly images use the same registration and 4K output. This map covers about
889 × 545 km: a physical 3840 × 2160 pixel averages about **232 × 252 m**; a
4096 × 2396 geographic texel covers **217 × 227 m**. Projector calibration and
latitude cause local variation.

Higher-frequency future sources include Sentinel-2 L2A (10 m RGB, nominal
5-day revisit) and NASA Harmonized Landsat–Sentinel (30 m, approximately 1.6-day
observations). Those observations need cloud/shadow-masked temporal composites;
snow must remain a valid surface class. Monthly/10-day/weekly mosaics should be
generated on the laptop, with a declared acquisition window and gap handling,
then transferred as compact region-sized images. An observation cadence is not
a guarantee of a cloud-free map at every interval.

Lighting uses one deterministic exposure reference for the entire selected local
day, sampled from the terrain normals and the daylight solar arc. It does not
normalize each frame or change exposure during sunset. Lighting is applied with
a display gamma correction. **Keep shadows visible** adds a configurable floor,
default **10%**, even at night; disabling it restores black nights. Image
brightness separately controls the satellite image (default 1.25×). Existing
ambient-light controls remain available in advanced lighting settings.

The elevation view supports a single contour or a white/color fill above or
below the selected altitude. A fill includes sea-level terrain when below the
threshold; invalid projector rays remain black.

Map products include predecoded RGB NPYs. Three decoded image products are kept
in a bounded CPU cache, and the current satellite/land-cover GPU textures remain
resident while toggled off, avoiding repeated PNG decompression and texture
uploads. Actual month changes load the chosen image rather than retaining twelve
full-size GPU textures on the 2 GB board.

The road layer now uses OSM motorway and trunk-highway geometry from OpenFreeMap,
filtered before rasterization. Minor roads, ramps, ferries, and tunnels are
excluded. Motorway/trunk class sets relative widths between 1 px and the chosen
maximum; it is an estimate of size, not a surveyed lane-count classification.
No road labels are downloaded or rendered into the map.

River playback now projects distance/flow/timing fields through the STL hits
once in C. A sparse tiled GLES shader reveals only river footprints; static
backgrounds and their lighting are reused throughout the reveal. Chromatic
correction is combined with this pass when GPX is not active. The render loop
uses display backpressure without an extra 10 ms wait during a reveal. On the
Radxa, completed-frame offscreen tests measured about **38 fps** for rivers on
black and **28 fps** over a cached satellite background, before HDMI/preview
costs. These are rendering measurements, not the 60 Hz HDMI scanout rate.

The final live 3840 × 2160 check, with the 960 × 540 browser preview and 0.5 px
chromatic correction active, measured **21.1 fps** for rivers on black and
**16.5 fps** for rivers over satellite. Warm satellite switches took
**0.12–0.15 s** and land-cover switches **0.14–0.30 s**. Full scanout remains
60 Hz; these measurements count changed frames presented by the application.
The GUI integration check also verified all three elevation styles, different
January/July snow images, and a 10% night floor that becomes exactly black when
disabled.

## Compute river replay

River coverage and reveal times are compiled by native C into two packed pixel
lists: red/blue and chromatically corrected green. Coordinates, coverage and
phase occupy 8 bytes per event. A linear counting sort groups events by phase;
the shader does not ray-trace or resample the DEM during playback.

A GLES 3.1 compute kernel updates only events in the advancing reveal band,
retaining already completed pixels. A packed R32UI image stores the final RGBA8
values losslessly, allowing safe masked channel updates on GLES hardware that
forbids read/write RGBA8 images. Dispatch barriers serialize red/blue and green
updates. Native rendering unpacks this retained pixel buffer into the 4K output. The
interactive preview uses hardware linear downsampling of that normalized image;
full-resolution exports can read the packed buffer directly. No map is rendered
a second time for the preview. Replaying, seeking backwards, recoloring, changing
width, recalibrating, or changing the background rebuilds the appropriate state.

The PowerVR driver requires initialization with image stores: a framebuffer fast
clear can otherwise remain visible to the first presentation after partial
compute updates. Regression tests compare the compute output with the previous
shader, test repeated replays, verify channel synchronization, and compare the
preview with the downsampled full-resolution image on the actual Radxa.

Completed-frame offscreen measurements on the Radxa were approximately **62 fps
on black** and **45 fps over a static satellite image**, up from about 38/28 fps.
These figures exclude scanout and preview costs; use the live check below for
end-to-end measurements. No target-frame-rate option is introduced.


On the Radxa, passing a Python integer to `np.searchsorted` on the 1.6-million
entry uint16 event list took about 2.27 ms per lookup, versus 0.008 ms with an
explicit uint16 key. The renderer uses native-dtype keys, avoiding four expensive
array conversions per steady-state frame. C counting sort also replaces
comparison sorting during one-time event preparation.

The benchmark's `--preview` option includes 15 Hz hardware downsampling, RGBA
readback and JPEG encoding. On this board, manually averaging packed integer
texels in a fragment loop is substantially slower than the hardware blit path;
that integer downsampler is retained for exact full-resolution exports and
regression checks, not the interactive stream.

With the final compute renderer, 0.5 px chromatic correction, and 15 Hz hardware
preview downsampling/readback/JPEG encoding, 120-frame offscreen runs measured
**75.9 fps on black** and **62.5 fps over satellite** at 3840 × 2160. The HDMI
connector was disconnected during these final runs, so these are rendering
capacity measurements, not observed on-screen frame rates. Live verification is
available when a powered display is connected:

```sh
.venv/bin/python -m map_projection.experiments.check_river_playback --url http://projector.local:8080
```

The check requires HDMI before changing the view, counts distinct presentation
events with the browser preview active, and restores its settings unless the
user changes the view independently during the check.
