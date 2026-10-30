# Native Radxa pipeline measurements — 2026-10-25

These are measurements on the 2 GB Cubie A7Z, using its PowerVR BXM-4-64.
The production service was temporarily stopped for exclusive KMS access and
restored after every run. Its renderer files and persisted configuration were
not replaced. Experiments run from `/home/map/map-projection-staging`.

## Actual HDMI output, real daily cache

`experiments/benchmark_full_events.py` uses the real 2026-10-25 daily cache,
4096 × 2396 terrain, actual DEM normals, current projector calibration, the
Engine worker and clock, HTTP server, and SDL KMS/DRM at 3840 × 2160 / 60 Hz.
It measures unique-frame presentation through `Display.present`, including
SDL swaps, rather than isolated shader submission. No glFinish is inserted.
The physical projector's optics and browser painting/network delays are not
measured. Browser load is represented by actual loopback HTTP state requests
once per second and 960 × 540 preview requests every 750 ms.

Each case runs for 12 wall-clock seconds after preparation/warmup. Sunset
samples begin at local 16:20 or 15:00; the fastest test spans roughly 12 simulated
hours. These are short samples, not full-day stability or midnight-loop tests.

| Workload | Current event renderer fps | Dense-upload experiment fps |
|---|---:|---:|
| Sunset, 1 simulated min/s, preview closed | 30.04 | 30.04 |
| Sunset, 15 simulated min/s | 25.36 | 28.12 |
| Day sweep, 60 simulated min/s | 23.66 | 25.49 |
| Sunset, 1 min/s, HTTP browser preview active | 23.37 | 23.72 |

The usual frame interval is 33–34 ms despite the HDMI mode being 60 Hz. In the
baseline, the median draw-plus-swap call takes approximately 23–26 ms; including
engine work and synchronization misses the 16.7 ms budget. This is not sustained
60 fps. The prior synthetic offscreen result of 40 fps did not include this
whole pipeline and is not the end-to-end rate.

Dense changes are a separate problem. At 15 min/s, the longest measured baseline
interval is 809 ms, versus 205 ms with dense uploads. At 60 min/s it is 865 ms,
versus 476 ms. The experiment uses parity-preserving CPU XOR followed by a
contiguous packed-texture upload when at least 500,000 events are crossed;
sparse updates retain the existing compute scatter. It improves bursts but
does not solve them. Forward/reverse/repeated-event GPU tests pass on PowerVR
with the dense path forced even for tiny inputs.

Preview requests produce roughly 182 ms p95 intervals in both versions. The
current event preview first renders the entire terrain light grid, then runs
the general multisample preview and readback. A direct preview-sized event
shader should avoid the full-grid pass and needs its own timing/quality check.
Simply calling the readback asynchronous does not remove the GPU work.

### Preview follow-up

Directly shading the 960 × 540 thumbnail with fixed 4 × 4 supersampling removes
the full-grid pass, but measures only 23.86 fps with 165 ms p95 intervals on
this board. The supersampling work remains expensive.

The implemented moving-frame path instead captures the completed HDMI image
**before SDL swaps buffers**. GPU framebuffer blits reduce it in successive
steps of at most 2×, preserving area coverage for the usual 4× reduction, and
only the thumbnail is read back through a PBO. Requests remain queued until a
fresh presentation, rather than reading undefined post-swap back-buffer data.
Paused frames use the direct thumbnail shader from existing event data; no
full terrain-grid render is needed.

In a 20-second real native-output run, previews active at the existing 750 ms
period now achieve **29.34 fps, 41.76 ms p95 intervals**, versus the earlier
23.37 fps / 182.33 ms. The maximum interval is still 108 ms, so this does not
claim perfectly smooth presentation. Native 60 Hz mode and the same 4096
terrain/calibration/cache remain in use. Results are in
`full-events-output-preview.json`. Full-image orientation, thumbnail averaging,
forward/reverse event parity and both capture paths pass GPU tests locally and
on PowerVR. The original production service is restored after benchmarking;
these source changes are not deployed to production.

Reducing CPython's thread-switch interval from its default 5 ms to 1 ms yields
29.29 fps and 42.46 ms p95 over another 20-second preview run. This does not
establish an average-throughput improvement. The maximum interval is lower
(65 ms versus 108 ms), but these short samples cannot attribute that difference
to scheduling. Pure GPU time and GIL-wait time have not been isolated; the
draw-plus-swap duration must not be presented as either of those measurements.

Other isolated experiments are retained for comparison: a resident GPU event
stream with atomic XOR performs much worse on large bursts (up to nine-second
intervals), and mediump shading does not improve the normal 30 fps rate. Neither
is selected for production. No useful precision tradeoff is established.

Peak process RSS in the baseline is about 770 MiB, excluding GPU/kernel/OS
allocations. All raw JSON measurements are in
`experiments/results/2026-10-25/full-events*.json`.

## Daily compiler: measured improvement

The earlier 12.8-hour estimate extrapolated a **308.36-second** Radxa run for
65,536 cells to the full 9,814,016 cells. It was not a complete 13-hour bake.
The implementation sweeps every second of the day and bisects each detected
change using many tiny Python queries.

Profiling the same strip on the laptop reports 45.7 million Python calls;
skyline lookup accounts for 112.7 of 141.8 profiled seconds. Decompression is
under one percent. Bisection queries repeatedly invalidate the cached full
sweep query and sort/index tiny geographic groups. About 355,000 scalar solar
calculations are performed for one strip.

The optional `--vectorized` compiler now:

- evaluates the same solar model directly on arrays, including fractional
  seconds, leap years and UTC year boundaries;
- evaluates different cell/azimuth pairs in one skyline query without discarding
  the main sweep's cached planes;
- writes event lists in bulk instead of making one file write per receiver;
- preserves the scalar compiler's float32 comparison rounding.

On the Radxa, this takes **92.55 seconds** for the same strip: **3.33× faster**.
All 131,230 event timestamps, initial states and offsets match the original
exactly. Peak process RSS is about 367 MiB. Cell-count extrapolation is now
**3.85 hours**, not a measured full-board bake. A laptop run takes 28.54 seconds.
An additional minute-block visibility-bound experiment also matches the strip
exactly, but its 28.51-second laptop time establishes no further speedup.
Both optimizations remain opt-in; the scalar compiler remains the default.

All 57 local tests pass, including GPU tests. Test coverage includes the vector
solar model across leap/year boundaries, mixed-azimuth skyline batches and
scalar/vector/bounded event compilation. The PowerVR tests also exercise the
isolated replay variants. The full master still has uncertified 0.25° azimuth
sampling; these optimizations do not repair that approximation.

## Next algorithmic steps

The largest remaining compiler improvement is to avoid the second-by-second
sweep. Intersect the Sun's continuous path with each piecewise-linear skyline
segment, using bounds/refinement to retain multiple crossings and grazing
events. Evaluate candidate roots in large batches. This changes work from
testing billions of unchanged states toward processing the skyline segments
and actual crossings. Validate against the current sampled reference and the
continuous DEM renderer; do not assume exactly two transitions per cell.

Decoded geographic tiles can be processed on the GPU, with per-cell private
work/output counts and a packing pass, rather than contentious per-event
atomics. The whole compressed master does not fit RAM, so stream/decode one
tile batch at a time. CPU decode and SD throughput then become relevant lower
bounds; a seconds-long full-day bake is not established by these measurements.

For replay, prioritize a direct low-resolution preview path and dense-update
handling. Then profile/reduce the calibrated 4K shading and presentation cost
until the complete loop fits under 16.7 ms. The actual board measurements do
not currently justify promising 60 fps, even though the HDMI link is 4K60.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m map_projection.bake_events --horizons map_projection/.cache/horizons4096-compressed --output map_projection/.cache/events-vectorized --date 2026-10-25 --latitude 45.95 --longitude 10.65 --vectorized
```


## Native compiler and stutter fixes — deployed 2026-10-25

These full-size measurements supersede the compiler extrapolations above.
The standalone C/OpenMP compiler is now the default. Python handles manifests
and process launch; decoding, solar evaluation, visibility bounds, refinement,
CSR output and chronological sorting run in C. Minute-block conservative bounds
skip constant states while preserving the original one-second sampled states
and 50 ms transition refinement. It retains all crossings, not a two-event limit.

| Complete 4096 × 2396 day; 19,658,590 events | Measured time |
|---|---:|
| Laptop, 8 threads | 27.12 s |
| Radxa, 4 threads, low priority, compressed master on SD | 209.36 s |

Both runs match **every initial state, offset and event timestamp** in the
previous full-day reference cache exactly. The full compressed skyline bank
(9.60 GB) and native compiler are installed on the Radxa. It can compile days
locally. This does not certify the bank's 0.25° angular approximation or shadow
episodes missed between the original one-second samples.

Replay previously sorted/scattered millions of updates and uploaded up to
39 MB on a large seek. The new path uses a 1.23 MB visibility bitset, native
parallel XOR updates, resident chronological cells and initial/final snapshots
for long seeks. Calibrated normals are prepared once. The PowerVR shader avoids
integer division by a uniform, which proved particularly expensive on this GPU.
The event path also skips unused terrain/ray-tracing GPU allocations.

The display waits for new frames via a condition variable. Browser thumbnails
are downsampled from completed HDMI frames before swap, with asynchronous CPU
readback; paused frames retain a half-resolution GPU snapshot. Neither path
requires rerendering the terrain for normal thumbnails.

Actual full pipeline, real cache, current calibration, 4K KMS output:

| Case | Average fps | p95 frame interval | Maximum interval |
|---|---:|---:|---:|
| Sunset, 1 simulated min/s | 30.04 | 34.50 ms | 35.29 ms |
| 15 min/s | 40.53 | 34.35 ms | 50.13 ms |
| 60 min/s | 34.44 | 34.44 ms | 43.42 ms |
| Sunset with HTTP thumbnails | 28.83 | 48.29 ms | 51.26 ms |

Faster cases include nighttime black frames, so their higher average is **not**
a claim of daytime 40–60 fps. Daylight is approximately 30 fps. The old 15/60
min/s cases had maxima of 809/865 ms. With HTTP previews active, three Now
transitions produced 50/50/51 frames in 1.05–1.08 s; Stop/return produced
45/44/45 in 1.06–1.07 s. The ease curve and exact event times are unchanged.

A subsequent lightweight-allocation repeat measured 28.50 fps with thumbnails
and one 105 ms outlier; Now had 48–51 frames and Stop/return 42–44.
These changes remove the large observed event-processing stalls, but do not
establish perfect pacing or 60 fps. The final paused-preview snapshot is covered
by local and PowerVR tests; the table predates that final idle-only change.

All 63 local tests pass, including GPU tests. The latest five cache/GPU tests
also pass on PowerVR; native compiler and bit updates were tested there earlier.
The new replay and native compiler are deployed; calibration and preferences
were preserved. Raw results: `experiments/results/2026-10-25/` files
`native-full-laptop.json`, `native-full-radxa-final-result.json`,
`full-events-native-bits-final.json`, and
`full-events-native-bits-lightweight.json`.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m map_projection.bake_events --horizons map_projection/.cache/horizons4096-compressed --output map_projection/.cache/events --date YYYY-MM-DD --latitude 45.95 --longitude 10.65 --threads 4
```

Completed daily caches are immutable; select a fresh output directory when
regenerating an existing date. `MAP_PROJECTION_BITSET=0` selects the previous
replay path for diagnosis. It is not the default.


## Replacement 64 GB card: self-contained verification, 2026-10-27

The full horizon bank and two daily caches are installed. A complete new day
compiled on the Radxa in **362.59 seconds including storage flush**, using four
native C workers and 420 MiB peak compiler RSS. All five data arrays match the
laptop reference exactly. The current uncooled board reached 85–89 °C, with both
CPU clusters capped at 416 MHz by thermal controls.

Current real-cache daylight playback: **30.04 fps** at 1 min/s, **28.51 fps** at
15 min/s, and **29.61 fps** at 60 min/s; with browser thumbnails, **28.66 fps**
(960-wide) or **28.49 fps** (1920-wide). Occasional longer frames remain, up to
136 ms in the fast sunset case. These are actual hot-board measurements, not
60 fps rendering or a prediction after cooling. Full measurements and raw data:
[2026-10-27 verification](experiments/results/2026-10-27/README.md).
