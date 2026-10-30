# Horizon and daily-event cache

Deployed event playback, with the sampling limitations below. Performance and
exact replay are tested; angular approximation remains uncertified.

## Data and playback

`bake_horizons` computes a full terrain-space skyline bank on the laptop GPU,
using the reference bilinear-ray intersection and 18 binary-search iterations
per azimuth. Geometry identity hashes actual float32 DEM contents, geographic
size, receiver bias and model version. Projector corners and pose are excluded.
The bank is memory mapped, resumable, and published complete only after all
azimuths are written. Disk-space checks run before allocating it.

`bake_events` compiles one UTC day from the skyline bank. It evaluates visibility
at one-second intervals and bisects detected transitions to 50 ms. Fractional
seconds now reach the solar calculation. Each cell retains all detected events,
not an assumed sunrise/sunset pair. CSR lists support diagnostics; chronological
(time, cell) streams support playback. A manifest prevents reading partial data.

`backend=events` uses the installation's `MAP_PROJECTION_EVENTS` directory,
defaulting to `map_projection/.cache/events`, with UTC-date subdirectories.
Geometry, bias and solar-location mismatches are rejected. Visibility uses one
bit per terrain cell (1.23 MB at 4096 × 2396), updated by an OpenMP C routine.
Chronological event cells are pinned in RAM; a compact run-length timeline avoids
paging large time arrays during jumps. Forward/backward crossings XOR bits,
preserving repeated-event parity. Long seeks choose the cheapest starting state:
current, initial, or final visibility. GPU normals retain 10 bits per component
and are projected once when calibration changes. Cached shadow events remain in
terrain coordinates, independent of projector recalibration.
Sun direction and diffuse shading remain live. No per-pixel event searches or
terrain rays run during normal replay. Normal quantization can change final grayscale by roughly one code value;
visibility remains binary and exact relative to the event data. Preview lighting
is materialized only when requested. Direct output and HDMI recovery use the
existing paths. Cache generation is independent of projector recalibration.

## Important remaining accuracy work

The initial bank uses **1440 uniformly spaced azimuths (0.25°)**, float32 skyline
angles, and approximately 0.00034° elevation-search resolution. Its full size is
56.5 GB. Angular interpolation is NOT certified: narrow peaks between samples
can be missed. Likewise, a shadow episode wholly between two one-second samples
can be missed. Event timestamp precision does not certify geometric accuracy.
Manifests explicitly record `interval_certified=false`, angular spacing,
sampling step and timestamp refinement. The present solar model remains the
simplified NOAA approximation, with fractional-second support; SPA is not yet
integrated. No subsecond astronomical-accuracy claim is made.

Adaptive skyline refinement with conservative bounds and reference comparison
is still required before calling this a precise cache. The default one-day compiler is bounded-memory C/OpenMP, including archive
decoding, solar calculations, root refinement and chronological stream writing.
The Python reference remains available with `--compiler python`. It does not yet have checkpoint/resume within a day. Completed daily
caches are immutable: choose a new output directory to regenerate.

## Performance and checks

The new CPU/GPU tests cover azimuth wrap, cache identity, exact event boundary
semantics, multiple events, backward seeks, previews, and fractional-second sun
positions. An analytic ridge was also checked against the horizon baker.

A synthetic 4096 × 2396 two-event-per-cell workload with 3840 × 2160 offscreen
output now takes **24.9 ms median, 25.6 ms p95** on PowerVR at one simulated
second per frame (120 frames). That is roughly **40 fps**, not 60. An earlier
161 ms result was dominated by NumPy promoting the full uint32 event-time
array for a Python-int search key on every frame. Using an explicitly uint32
key avoids that copy. The current benchmark includes event processing and GPU
completion, excludes static preparation, HDMI swap and browser previews, and
is synthetic rather than proof of real-cache accuracy. Further profiling of
the direct shader/presentation path is required. Do not deploy as a claimed
4K60 solution. The 50 local tests and four new PowerVR cache tests pass.

## Jobs started for this implementation

Master: `map_projection/.cache/horizons4096/`, 4096 × 2396, 1440 azimuths,
receiver bias matching the projector's 0.05 mm setting. Its `manifest.json`
records all 1440 slices complete. The source is the
prepared land-aware 4096 terrain copied from the projector, not the raw DEM.

The background `map_projection.cache_pipeline` completed 2026-10-25 into
`map_projection/.cache/events/2026-10-25/`: 19,658,590 transitions, 285 MB total.
Its log is `map_projection/.cache/event-bake.log`. These are experimental caches
for validation, not production-ready data.

The complete lossless 55°–305° master is in
`map_projection/.cache/horizons4096-compressed/`, 9.60 GB including index and
metadata. All 2400 tiles were compared bit-for-bit against the raw source.

## Commands

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m map_projection.bake_horizons --input map_projection/.cache/horizon-input.npz --terrain map_projection/.cache/horizon-source4096.npy --output map_projection/.cache/horizons4096 --angles 1440
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m map_projection.bake_events --horizons map_projection/.cache/horizons4096 --output map_projection/.cache/events --date 2026-10-25 --latitude 45.95 --longitude 10.65
OPENBLAS_NUM_THREADS=1 MAP_PROJECTION_GPU_TESTS=1 .venv/bin/python -m unittest map_projection.tests.test_horizon_cache
```

After validation, copy the selected date directory to the projector's event
root with rsync and select the event backend. Copying the full master to the
2 GB board is unnecessary for playback. A compressed master can also live on
the SD card for local daily generation; see [the compression study](SKYLINE_COMPRESSION.md).
Keep cache versions separate when replacing data;
never rewrite files while a playback process has them mapped.
