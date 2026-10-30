# Radxa live-rendering experiments — 2026-10-24

These measurements document the development of the new rendering backend.
The GPU texture scan, land-aware surface and playback controls were deployed
to the projector with rsync on 2026-10-24 after user approval.
The final section records tests of the integrated application backend.

The physical 2 GB A7Z runs Debian 11 R6 and PowerVR BXM-4-64 with GLES 3.2
DDK 24.2. All output buffers are **3840 × 2160**. The shadow grid is a separate
quality setting. Exact original-STL projector intersections stay at 4K.

## Fresh-frame CPU measurements

Medians for three fresh sun positions (09:00, 12:00, 16:00 on 2026-10-24),
using the actual geographic DEM. Experimental CPU workers were restricted to
cores 6 and 7, the two Cortex-A76 cores. No pre-rendered sun frames are read.

| Shadow grid width | Current CPU renderer | Experimental CPU, 2 workers | Slowest of three experimental frames |
| --- | ---: | ---: | ---: |
| 512 | 323.5 ms | 9.2 ms | 9.2 ms |
| 1024 | 339.2 ms | 15.7 ms | 15.8 ms |
| 2048 | 465.1 ms | 42.9 ms | 43.5 ms |

These are calculation times, **excluding SDL upload, HDMI presentation, web
preview generation and long-run scheduling/thermal variability**. Three frames
are not a sustained-frame-rate guarantee. 15 fps allows 66.7 ms; 60 fps allows
16.7 ms for the entire frame pipeline.

The experiment prepares fixed pixel-to-terrain indices and DEM normals once,
then recalculates the horizon, lighting and output pixels for every sun time.
It shades each terrain cell once and gathers the results into the 4K pixel
buffer. It sweeps contiguous rows, stores the transposed terrain once, uses a
32 × 32 tiled transpose for the resulting horizon, and runs two CPU workers.
Eight workers were slower in earlier tests on this heterogeneous CPU.

Additional static preparation was about 0.10–0.16 s after the exact STL lookup
was available. The existing STL intersection pass can still take 12–40 s after
a calibration change; these timings do not include that pass.

The tested frames differed from the current renderer by at most 1/255 in grey
value at the same shadow-grid setting. That is a limited comparison, not proof
of equivalence across all dates, directions or configurations. The float32 SIMD
sweep assumes finite terrain inputs and needs broader validation before release.
Reducing the grid changes shadow detail; it does not lower HDMI resolution.

## Actual GPU experiments

The offscreen EGL context reported `PowerVR B-Series BXM-4-64`,
`OpenGL ES 3.2 build 24.2@6603887`. It ran alongside the projection service.
Warm GPU medians exclude the first shader-compilation frame.

- A straightforward OpenCL lighting/gather path with RGB readback took about
  245 ms, including a slower CPU horizon. Moving CPU code to GPU kernels alone
  did not make it fast.
- An initial GLES mapping pass took 46.6 ms. Replacing per-fragment integer
  division with fixed packed UV coordinates reduced that 4K pass to 9.7 ms.
- The combined GLES prototype recomputed the CPU shadow field, uploaded it,
  and rendered lighting and the projected 4K image directly on the GPU. It took
  **55.8 ms**: about 25.9 ms CPU shadows, 7.1 ms upload, and 22.6 ms GPU rendering.
  No full RGB readback is included in that timed pipeline.
- A readback outside the timed GLES loop differed from a CPU reference by at
  most 1/255. This checks numerical output, not projected optics.

The combined prototype is about 18 fps offscreen, not demonstrated 60 fps.
CPU and GPU experiments used different sampling counts and scheduling; their
figures are indicative rather than a controlled backend ranking. GPU-resident
geometry and lighting avoid full-frame CPU readback, but require a new display
backend, buffering and sustained end-to-end measurement.

A live 15 fps implementation at the current 2048 grid looks realistic. Fresh
4K60 at that quality has not been demonstrated. Lower grids, improved shadow
algorithms or temporal reconstruction would need explicit quality decisions.

## Reproduce on the board

From `/home/map/map-automation`:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m map_projection.experiments.benchmark_cpu
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m map_projection.experiments.benchmark_opencl
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m map_projection.experiments.benchmark_gles
```

The CPU experiment builds `fast_dem.so` with GCC/OpenMP and writes
`gpu-input.npz` for the GPU experiments. The GLES experiment also needs
`PyOpenGL` (installed in this board's virtual environment for development).
None of these scripts changes application settings or takes over HDMI.
Raw measurements are in [results/2026-10-24](results/2026-10-24).

## Initial animation proposal (now implemented)

- Date defaults to today in the chosen timezone and remains editable.
- Speed uses **simulated minutes per real second**, independently of frame rate.
- At 15 min/s, 15 fps advances one simulated minute per frame; 60 fps advances
  15 simulated seconds per frame. One minute per frame at 60 fps instead means
  60 min/s, a different playback speed.
- Use a monotonic wall-clock timeline rather than adding time after each render.
  A slow frame should not silently change the requested timelapse speed.
- Keep playback smoothness, terrain/shadow quality and the policy for missed
  deadlines explicit. Decide the quality tradeoff before integrating a backend.

## Follow-up: shadows generated on the GPU

At the time of these experiments, the coastal correction was local. The following benchmarks
ran from `/home/map/gpu-shadow-experiments`, outside the deployed application.
The application service was not restarted; its settings and source were not
modified. Offscreen contexts do not take over HDMI.

The A7Z GPU can execute independent shadow rays. The earlier GPU experiments
above only moved lighting and output to the GPU; their CPU horizon was a
prototype limitation, not a hardware restriction. Radxa lists GLES 3.2 and
OpenCL 3.0 support in its [A7Z specifications](https://docs.radxa.com/en/cubie/a7z).
These follow-up shaders actually executed on the PowerVR device.

### Coastline diagnosis and correction

The original DEM includes bathymetry down to approximately -2624 m near the
Ligurian/Tuscan coast. The old terrain sampler retained negative heights down
to -1000 m, then replaced deeper water with zero. This introduced a roughly
1000 m discontinuity offshore and also shaded underwater slopes. It is visible
in the [before/after comparison](results/2026-10-24/coast-before-after.png).

`terrain.surface_heights` now flattens negative and nonfinite source samples
**before bilinear interpolation**. This avoids negative coastal interpolation
halos as well as the false cliff. Cache version `terrain-v3-flat-sea` prevents
reuse of the old data. The full source DEM remains unchanged. Without a land
mask, this also flattens genuine below-sea-level land; it is an explicit
limitation of this rendering surface, not a change to the original elevation
or mesh files.

The comparison fixture contains 266,002 negative cells before the correction
and none afterward. Its coastal maximum gradient falls from about 1.59 to
0.88 m/m; remaining steep land is retained. Tests cover ocean depths, a void,
coastal interpolation, cached reload and cast shadows on a flat receiver.
All 16 laptop tests pass. The production live renderer already computes
occlusion shadows as well as normal-based diffuse shading.

### Independent GPU rays

`benchmark_gpu_shadows.py` launches a fragment invocation per terrain cell.
It follows each light ray independently through a discrete heightfield.
A static max-height mip hierarchy skips blocks that the ray passes above.
The hierarchy is an acceleration structure, not a collection of cached sun
frames. The straightforward cell-by-cell version was much slower.

At widths 512/1024/2048, the accelerated path took approximately
20.8/60.7/232.5 ms including the projected 4K output. Across four sun positions,
4096 CPU DDA queries at each resolution had **zero shadow disagreements**;
there were no iteration-cap hits. Full 4K gather comparisons matched exactly.
This validates the acceleration against piecewise-constant terrain columns.
It does **not** validate exact triangle intersection with the original STL,
or equivalence to the production interpolated terrain sweep.

Independent per-pixel rays are possible. The expensive part is searching the
terrain for each ray; parallel execution alone does not remove that work. An
exact original-mesh tracer would need a suitable acceleration structure and
memory budgeting for the 16.5 million triangles. A terrain grid is a choice of
surface resolution and shared work, not a requirement for independent rays.
The projector's original-STL visibility mapping stays at 4K in these tests.

### Sharing ray work on the GPU

`benchmark_gpu_scan.py` shears the terrain coordinates so samples on each
sunlight ray lie along a row. A prefix maximum of height plus accumulated ray
rise determines which downstream receivers are occluded. Each terrain sample
is visited once, rather than searching the terrain from every receiver.

Both a workgroup parallel prefix scan and one sequential scan per GPU lane
were tested. The latter was substantially faster on this GPU: many sunlight
rays run in parallel, with no workgroup barriers. An **exclusive** prefix keeps
the receiver's own resampled height from casting a false shadow on itself.
Moving the horizon from a storage buffer to a GPU texture then reduced the
2048 lighting pass from about 22.4 to 8.7 ms.

Best path, 100 fresh calculations per resolution: 25 calculations at each of
09:00, 12:00, 16:00 and 18:00 on 2026-10-24. Shader warmup is excluded. Every
calculation reruns shadows, lighting and the 4K gather; no rendered frames or
sun-angle shadow results are cached. Static geometry, normals and lookup
textures are prepared once.

| Terrain grid | GPU shadows | GPU lighting | 4K gather | Median total | 95th percentile total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 × 300 | 1.49 ms | 1.04 ms | 8.60 ms | **11.10 ms** | 11.56 ms |
| 1024 × 599 | 4.72 ms | 2.60 ms | 8.94 ms | **16.18 ms** | 17.89 ms |
| 2048 × 1198 | 18.47 ms | 8.65 ms | 9.45 ms | **36.47 ms** | 43.09 ms |

Stage medians need not sum to the median total. These are **offscreen** times,
including GPU completion but excluding HDMI scanout/presentation, browser
preview generation, initial geometry preparation and validation readbacks.
They are not a sustained display-rate guarantee. A 2048-grid, 15 fps backend
has substantial calculation headroom in these samples. Fresh 4K60 at that
grid has not been demonstrated; even its shadow stage exceeds 16.7 ms.
Width 1024 is close but misses some 60 fps deadlines before presentation;
512 has calculation headroom, at lower shadow detail.

The scan is a different interpolation approximation from the production
sweep. Shadow classifications differ by approximately 0.003–2.52% of terrain
cells across the tested resolutions/times, with the largest differences near
sunset. It requires visual review, not automatic replacement. Against an
independent double-precision implementation of the same sun-aligned surface,
GPU discrepancies were confined to 0–8 cells per frame; the largest threshold
margin at a discrepancy was 0.014503 m. Normal shading differed by at most one
8-bit grey level; every full 4K gather comparison matched exactly.

Raw results:

- [Independent GPU rays](results/2026-10-24/gpu-shadows-results.json)
- [Workgroup parallel scan](results/2026-10-24/gpu-scan-results.json)
- [Sequential GPU lanes, storage buffer](results/2026-10-24/gpu-scan-serial-results.json)
- [Sequential GPU lanes, texture, 100 calculations per grid](results/2026-10-24/gpu-scan-texture-100-results.json)

### Reproduce without deploying

On the laptop, retain the existing exact projector lookup and regenerate only
the DEM lighting surface using the corrected local sampler:

```sh
scp projector:/home/map/map-automation/map_projection/experiments/gpu-input.npz map_projection/.cache/original-gpu-input.npz
.venv/bin/python -m map_projection.experiments.prepare_gpu_input --input map_projection/.cache/original-gpu-input.npz --output map_projection/.cache/corrected-gpu-input.npz
ssh projector 'mkdir -p /home/map/gpu-shadow-experiments'
rsync -a map_projection/experiments/gpu_context.py map_projection/experiments/benchmark_gpu_shadows.py map_projection/experiments/benchmark_gpu_scan.py projector:/home/map/gpu-shadow-experiments/
rsync -a map_projection/.cache/corrected-gpu-input.npz projector:/home/map/gpu-shadow-experiments/input.npz
```

Run the experiment from the scratch directory on the board:

```sh
cd /home/map/gpu-shadow-experiments
export PYTHONPATH=/home/map/map-automation
export OPENBLAS_NUM_THREADS=1
/home/map/map-automation/.venv/bin/python benchmark_gpu_shadows.py --input input.npz --methods 1
/home/map/map-automation/.venv/bin/python benchmark_gpu_scan.py --input input.npz --serial --texture --trials 25
```

`prepare_gpu_input.py` defaults to the repository Alps geographic extent. Use
`--config path/to/settings.json` for a different source extent. The input
lookup must correspond to the same STL and grid shape. No calibration or
production settings are changed by these commands. The existing deployment
sync script must not be run as part of this isolated experiment workflow.

## Integrated application follow-up: land mask and playback

The application now includes the GPU texture-based scan renderer and monotonic
minutes-per-second playback controls. These changes are now deployed on the A7Z.
The original flat-sea-only correction described above is superseded by the
shared, land-aware preprocessing in `elevation_surface.py` and
`map_projection.prepare_terrain`.

Mesh generation and projection now call the same coastline, void and sea-level
functions. Positive elevations survive mask disagreement; water below zero is
flattened; negative land is limited to -20 m and retained only in four-connected
patches of at least ten source samples. The policy is applied before lighting-grid
interpolation. The original DEM is unchanged. On this laptop, the prepared
11697 × 20000 int16 surface retains 2,257,646 below-sea-level land samples and
occupies 467.9 MB. The sampled 2048 grid retains 25,377 negative land cells.
GIS and labeling work run once on the laptop, never in the animation loop.

The **actual application Renderer** was exercised on the physical A7Z in a
separate offscreen process, alongside the existing projection service. For
100 fresh 3840 × 2160 calculations at the 2048 × 1198 land-aware grid:

- Median calculation: **36.374 ms**; 95th percentile: **42.960 ms**.
- Four sun times: 09:00, 12:00, 16:00, 18:00 on 2026-10-24.
- Full-resolution capture matched the rendered framebuffer byte-for-byte.
- Day → night → day recovered without stale light or shadow data.
- No HDMI takeover, service restart or deployment was involved in this benchmark.

[Raw integrated-renderer results](results/2026-10-24/integrated-gpu-land-results.json).
These timings exclude HDMI swap/presentation and preview capture. The integrated
SDL/KMS window has since been validated at 4K60 with approximately 15 fresh
shadow frames per second at a 15 fps target. The offscreen results
are not proof of sustained 60 fresh frames per second. Direct output avoids the CPU image readback and
texture re-upload used by the original SDL streaming path.

The local browser integration was checked with actual SDL/GLES output and the
full STL, including 4K output/capture, sunset selection, speed, pause/resume,
loop/stop behavior, scene transitions and mobile layout. On the Radxa, 27 native,
playback and reconnect tests plus three GLES pixel/shadow/preview tests passed;
additional CPU-equivalence and watchdog checks were run separately.

HDMI recovery now monitors status/EDID and kernel hotplug events, reopens the
SDL/KMS surface, and retries transient failures without a settings change.
Disconnected startup preserves the web server. A simulated disconnect/reconnect
with actual laptop SDL/GLES contexts restored the same frame; physical cable
reconnection on the board remains a deployment check.

Physical deployment exposed two additional A733 problems: repeated full SDL
device initialization could hang or abort on a reused logind descriptor, and
an unchanged DRM mode could remain logically active while hotplug had disabled
the HDMI hardware. Rebuilding the window while retaining the SDL device was also insufficient.
The final workaround replaces the app process on hotplug and uses
`drm.reset_scanout` through its fresh SDL master descriptor before the first
swap applies the requested mode. The user confirmed the visible calibration
grid before and after physical reconnection. The systemd watchdog
successfully recovered the initial driver hang, with about a 25-second delay.

The authorized deployment preserved corner, pose, bounds and output settings
exactly against the pre-update backup. The production service reports PowerVR
GLES 3.2, direct 3840 × 2160 at 60 Hz, no renderer error and successful browser
preview/full-frame capture during playback. First-time geometry preparation
took about 17 seconds (13.385 seconds in the projection pass). Subsequent frames
reuse this geometry. The service uses the new readiness/watchdog unit.

## Playback and highlight refinement — 2026-10-25

Playback now runs without a target-FPS setting. The time slider renders during
input, a logarithmic speed slider spans 1 s/s–1 h/s, and playback loops the local
calendar day. Stop eases back to the pre-animation selection in one second;
scrubbing interrupts playback directly. Now eases to synchronized wall time.
The clock follows presented frames through server-sent events instead of the
old once-per-second status poll. The latter caused multi-minute clock jumps
at accelerated speeds even when shadow frames changed more frequently.

Isolated bright terrain cells are identified from the 3 × 3 light neighborhood.
Their coordinates are compacted into a GPU list, then an indirect compute
launch traces exact column-heightfield rays against a static max-height pyramid.
Occluded highlights are replaced with ambient light; genuinely illuminated peaks
are retained. Traversal safety-limit hits fail dark rather than emitting bright
markers. Barriers protect horizon read/write reuse between animation frames.
This does not blur the output or cache rendered shadow frames.

On the actual PowerVR device, five sunset positions on 2026-10-25, five warmed
fresh frames per position, all with a 3840 × 2160 output:

| Lighting grid | Range of per-position median times | Corrected false highlights per position |
| --- | ---: | ---: |
| 2048 × 1198 | 62.257–86.700 ms | 8,015–17,396 |
| 1024 × 599 | 25.209–31.519 ms | 2,370–6,856 |

These offscreen timings exclude HDMI swap and preview capture. At each position,
128 randomly selected corrections matched independent CPU cell-by-cell ray
queries: **1,280 checks total, no disagreement**. Repeated static frames were
byte-identical. The synthetic GPU suite additionally verifies that true isolated
sunlit peaks survive and that asynchronous captures retain the requested image
after the sun changes. The added ray checks cost throughput; quality remains
2048 on the installed projector. No claim of 60 fresh 4K shadow frames per second
is made at that quality.

A monolithic fragment ray filter took approximately 102–149 ms across the first
four sunset positions. GPU compaction plus compute rays reduced that cost;
compaction in a separate compute pass was slower on this PowerVR, so production
uses fragment compaction and compute refinement. All geometry, normals and the
height pyramid are prepared once per scene. Preview readback now uses pixel
buffers and GPU fences, avoiding a synchronous readback stall in the display loop.

[2048 results](results/2026-10-25/refinement-2048.json),
[1024 results](results/2026-10-25/refinement-1024.json).
Reproduce without taking over HDMI:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. /home/map/map-automation/.venv/bin/python \
  -m map_projection.experiments.benchmark_refinement --input input.npz \
  --output refinement-2048.json --grid 2048
```

The full laptop suite passes **39 tests**; the ARM staging checks pass **20 tests**,
including five actual PowerVR GLES tests. Desktop/mobile browser checks cover
live dragging, speed edits, per-frame clock updates, Stop/Now easing and the
shadow toggle. A live HDMI connector reset preserved calibration, resumed
playback, reconnected the browser stream without reloading and retained the
original return position for Stop. Network time synchronization was enabled
and confirmed after discovering the board's reset clock.

Three captures of a paused production frame had identical pixel hashes and an
unchanged frame identifier. GLES dithering is disabled, and unchanged frames
are not repeatedly submitted. Any visible moving dark-grey texture while paused
therefore originates downstream of the application frame; projector grayscale
modulation is the likely source, rather than simulated-time updates.

The deployed control preview uses asynchronous pixel buffers and polls fences
without waiting in the display loop. During playback, thumbnail requests are
sampled less often to preserve GPU time for projection; manual scrubbing and
one-second transitions request faster previews. The clock still receives every
presentation event. The displayed FPS averages a two-second presentation window
so preview overhead is included rather than showing a favorable single interval.
The installed service is healthy at 4K60, with zero watchdog restarts during the
final checks and approximately 1 GB of memory available on the 2 GB board.

## Temporal aliasing diagnosis and continuous DEM reference — 2026-10-25

The user confirmed that isolated white speckles were removed, but many shadow
pixels still alternated rapidly in brightness. The effect weakened at larger
terrain grids. This is a reproducible renderer error, separate from the
projector's modulation of a static grey image.

`temporal_shadows.py` compares a 121-frame moving-sun sequence, 16:20–17:00
Europe/Zurich on 2026-10-25, at 20 simulated seconds per frame. Geometry, sun,
receiver bias and sampling resolution are held fixed. The reference computes
an independent ray at **every cell**, intersecting the continuous bilinear
patches between DEM samples. Along a ray each patch reduces to a quadratic;
its endpoints and possible interior maximum are checked analytically. A
static maximum-of-patch-corners mip hierarchy skips provably clear intervals.
Traversal follows integer cell boundaries without march-step gaps. Iteration
exhaustion is an explicit error, read through a four-byte diagnostic buffer,
rather than silently classified as shadow.

The independent float64 CPU oracle (`reference.bilinear_blocked`) visits every
crossed patch, without the GPU hierarchy. Analytic tests include ramps, a
bilinear patch with an interior occluder despite clear endpoints, flat ground,
edge-parallel rays and a monotonic ridge shadow. On each of NVIDIA and PowerVR,
all 3,078 synthetic GPU classifications match the CPU. Each real-terrain sunset
sequence below also has 448 sampled CPU checks with zero disagreements and
no exhausted traversals.

| Grid | Fast: pixels switching more than once | Reference: pixels switching more than once | Maximum switches, fast / reference |
| --- | ---: | ---: | ---: |
| Actual deployed 512 × 299 surface | 28,123 | 0 | 98 / 1 |
| 2048 × 1198 surface | 284,256 | 297 | 105 / 5 |

The raw scan **before** isolated-highlight refinement already has 29,476
reversing pixels at 512 × 299. Sparse refinement reduces that to 28,123 but
cannot fix the moving-lattice approximation. At the same grid, 4,784 fast-mode
reversing pixels are completely constant in the reference throughout the
sequence. The discrepancy therefore cannot be explained by real moving
shadows or the projector. Increasing grid size reduces the spatial scale of
these errors but does not validate the underlying approximation.

A height field need not be geometrically convex. For the three most active
reference cells at 2048 (x/y: 1397/393, 1355/520, 207/798), the CPU independently
reproduces all four/five transitions, with zero per-frame disagreements. Those
rare remaining reversals reflect actual occlusion as solar azimuth changes.

On the PowerVR board, warm median **grid-sized offscreen output** timings were:

| Grid | Fast scan + sparse refinement | All-cell continuous ray reference |
| --- | ---: | ---: |
| 512 × 299 | 7.42 ms | 30.30 ms |
| 2048 × 1198 | 76.97 ms | 627.75 ms |

These measurements deliberately use the grid itself as the output to isolate
shadow calculations. They exclude native 4K gather, HDMI swap, browser preview
capture and sustained playback scheduling. The fixture's receiver bias is
47.418 m (0.08 mm on the printed map); later interactive settings can differ.
The benchmark ran alongside the projection service, so timings are indicative.
In the actual 4K HDMI session the 4096-wide reference was about 2.5 seconds per
draw. The reference is intentionally slow and does not promise smooth 60 fps.

The new **Calculation → Reference · continuous DEM rays (slow)** mode was
rsynced to the projector. Switching between it and fast mode preserves the
selected geographic grid, lighting and exact 4K projector lookup. Playback,
scrubbing, stop/return, live speed changes, captures and browser controls use
the existing paths. Approximate CPU day caching is disabled in this mode.
The user visually checked playback around sunset and confirmed:
**“Repeated flicker is gone.”** This establishes the reference for future
acceleration work. No temporal smoothing or forced monotonicity is used.

This is an accuracy oracle for the **selected bilinear DEM grid and point-sun
model**, within floating point tolerance. It retains land-mask cleanup, fixed
normals, receiver bias and the flat geographic approximation. It is not an
exact tracer of all 20,000 source columns, the printed STL, Earth curvature or
a finite solar disc. Compare methods at the same grid before changing spatial
resolution. A fast replacement must match this reference in both individual
frames and moving-sun sequences; additional speckle filtering is insufficient.

Reproduction, using the prepared land-aware input on either machine:

```sh
OPENBLAS_NUM_THREADS=1 MAP_PROJECTION_GPU_TESTS=1 .venv/bin/python -m unittest map_projection.tests.test_reference
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m map_projection.experiments.temporal_shadows --input land-aware-gpu-input.npz --grid 2048 --output temporal-2048.json
```

For an exact deployed grid, add `--terrain /path/to/prepared-grid-terrain.npy`;
otherwise the experiment resizes the provided input grid. The script stores
packed per-frame masks and accumulates transition counts in bounded memory;
unpacked full-resolution sequences can exhaust the 2 GB board. Source hashes
are included in new result files. JSON measurements are in
[results/2026-10-25/temporal-actual-512.json](results/2026-10-25/temporal-actual-512.json)
and [temporal-2048.json](results/2026-10-25/temporal-2048.json).

Validation: **45 local tests**, **26 board runtime tests** (including both real
GPU backends and the independent ray oracle), and local/live-projector browser
checks passed. Two offline terrain-preparation tests additionally require
SciPy, which is installed on the laptop but deliberately absent from the board;
they pass locally. The earlier 28-test board invocation included those two
preparation tests and reported missing SciPy; the runtime-only 26-test suite
passes. Backup before deployment:
`/home/map/map-projection-backups/before-bilinear-reference-20261025.tgz`.
