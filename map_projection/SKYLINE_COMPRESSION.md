# Skyline compression study

Measurements use the actual prepared 4096 × 2396 terrain and its completed
1440-direction skyline bank. Sizes below use decimal GB. This study changes
staging code only; the production display service is untouched.

## Recommended storage

The original float32 bank contains 14,132,183,040 samples and occupies
56.529 GB. At the installation latitude (45.95° N), the Sun's annual daytime
azimuth range is approximately 55.1°–304.9°. Retaining the enclosing 55°–305°
sector leaves **1001 directions per cell**, or 39.295 GB before compression.
This sector is installation-specific; it must be reconsidered if relocating
the map. It is unrelated to projector calibration.

`skyline_store.py` stores independently compressed 16 × 256 geographic tiles.
It first recovers the exact 18-bit elevation codes used by the skyline baker's
binary search. This introduces no further quantization: reconstruction must
match every original float32 bit, otherwise that tile uses raw float bits.
Reversible differences across azimuth and geographic neighbours, zigzag coding,
byte shuffling and Zstandard then compress the values. Every tile is decoded
and compared bit-for-bit with its source during packing.

The **completed full archive is 9,598,245,825 bytes (9.60 GB / 8.94 GiB)**,
including metadata and index. All 2400 tiles used exact elevation codes, with
no raw-bit fallback, and passed full source comparison. Packing and verification
took 942 seconds on the laptop. This saves 83.0% relative to the full-circle
original, including the removal of unreachable directions.

The compressed reader uses a configurable decoded-tile LRU, defaulting to
256 MiB. A normal 65,536-cell daily-builder chunk spans 16 tiles and fits in
that budget. Query buffers and temporary decoding allocations add memory beyond
the LRU budget. The entire master is never expanded into RAM.

The daily-event compiler now accepts either the raw or the compressed bank.
An integration test checks matching event lists from both. Playback continues
to use daily events, not on-demand skyline decompression. Storage compression
therefore does not itself increase rendering frame rate.

## Alternatives measured on representative samples

The following are **whole-bank size estimates**, extrapolated from 16 geographic
tiles; they are not completed full archives:

| Method, after restricting azimuth | Estimated GB | Additional accuracy cost |
|---|---:|---|
| Ordinary Zstandard level 6 on float32 | 24.05 | None |
| Exact elevation codes and predictive compression | 9.94 | None |
| Adaptive knots, 0.001° tolerance, compressed | 8.41 | Bounded error at stored directions |
| Adaptive knots, 0.01° tolerance, compressed | 6.01 | Larger angular error |
| Adaptive knots, 0.05° tolerance, compressed | 2.89 | Substantial angular error |
| Exact codes plus annual solar elevation-band pruning | 6.29 | Only preserves solar visibility inside the tested band |

At 0.001° tolerance, adaptive simplification retains about 587 of the 1001
points on average. Exact piecewise-linear simplification retains about 868.
The modest saving at small tolerances does not justify changing shadow timing
when the lossless master fits comfortably on a 64 GB card. Angular errors do
not imply a fixed temporal error: nearly tangent Sun/skyline crossings are
particularly sensitive.

Zero-height receivers can still see complicated mountain silhouettes. In the
sample, sea-level receivers retained about 695 knots at 0.001°, compared with
572 for other receivers. Receiver elevation alone cannot identify a flat skyline.

## Solar elevation-band experiment

At fixed latitude, azimuth and elevation of the Sun obey

`sin(declination) = sin(latitude) sin(elevation) + cos(latitude) cos(elevation) cos(azimuth)`.

Bounding declination by ±24° gives a conservative annual solar elevation band
for the present installation. For each stored azimuth interval, the experiment
compares bounds on this band with bounds on the existing linear skyline.
Intervals entirely below all reachable solar elevations are always lit;
intervals entirely above them are always shadowed. Heights are retained wherever
an interval is ambiguous. Constant-visibility flags replace the other intervals.

On 65,601,536 sample values, approximately 39.5% of vertices become unnecessary.
Tests compare original and reconstructed visibility at random points inside
each solar band. The projected total, including compressed flags, is 6.29 GB.
This is an experiment, not a supported archive format or a completed full bake.
It preserves visibility relative to the existing interpolated master inside
the declared solar band, not the general skyline at arbitrary viewing angles.

## Limits and self-contained operation

Lossless storage preserves the existing master exactly; it does not certify
the original 0.25° azimuth sampling. Narrow terrain features between samples
still require conservative refinement/reference validation. Likewise, 50 ms
event refinement does not prove no event was missed between one-second samples.

A 64 GB card provides roughly 59.6 GiB before partitions and the operating
system. The lossless master leaves ample space for the application and a rolling
set of daily caches. The completed 2026-10-25 event cache contains 19,658,590
transitions and occupies 284,974,271 bytes (285 MB). Recalibrating projector corners does not invalidate these
terrain-space caches. Changing terrain, receiver bias, geographic geometry or
solar location requires the corresponding identity checks/rebuilds.

The reader runs on the Radxa using system libzstd, with no additional Python
compression package. Six storage tests pass there; all 56 local tests pass.
The full-width ARM test decoded a 65,536-cell stripe (262.4 MB of skyline data)
in 7.71 seconds, then queried it at 0.350 ms median / 0.402 ms p95. Peak process
RSS was 361 MiB, including the 250 MiB decoded chunk, interpreter and temporary
allocations. An initial single-tile test queried 4096 cached cells at 0.040 ms.
These are **not frame-rate measurements**, and full daily generation on the
Radxa has not yet been timed. Storage capacity and
bounded decoding are established separately from daily compilation speed.

The Radxa also generated a complete one-second-sampled day for one actual
65,536-cell full-width terrain stripe: **308.4 seconds, 131,230 events, 367 MiB
peak RSS**. A simple cell-count extrapolation gives approximately **12.8 hours
for the full terrain** with the current CPU compiler; this is an estimate, not
a full-board measurement. The current Python bisection/query workflow needs
optimization before promising quick autonomous generation. An initial rolling
set can be prepared on the laptop. Compression makes local storage practical;
it does not eliminate daily computation or establish 60 fps playback.

Follow-up: the optional batched compiler reduces this same Radxa strip to
92.55 seconds, with every timestamp unchanged, lowering the extrapolation to
3.85 hours. Actual native 4K event playback measures 23–30 fps depending on
speed and preview load. See [the full pipeline benchmark](PIPELINE_BENCHMARK.md).

Raw results and reproducible experiment scripts are under
`map_projection/experiments/results/2026-10-25/` and
`map_projection/experiments/measure_*skyline*`.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m map_projection.skyline_store --source map_projection/.cache/horizons4096 --output map_projection/.cache/horizons4096-compressed --start 55 --end 305 --level 6
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m map_projection.bake_events --horizons map_projection/.cache/horizons4096-compressed --output map_projection/.cache/events --date YYYY-MM-DD --latitude 45.95 --longitude 10.65
```


Update 2026-10-25: the native C compiler has now completed the **entire**
4096 × 2396 day from this compressed master on the Radxa in **209.36 seconds**
(4 threads). Initial states, offsets and all 19,658,590 timestamps match the
previous full-day reference exactly. This replaces the earlier 12.8-hour and
3.85-hour extrapolations. The complete 9.60 GB master is installed on the Radxa;
daily generation no longer depends on the laptop. See the updated
[full pipeline benchmark](PIPELINE_BENCHMARK.md) for replay pacing and limitations.
