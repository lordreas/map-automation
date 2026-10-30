# 4096 precise rendering: first implementation and handoff

## Status and user contract

The user requested a single implementation pass, with further verification and
improvement assigned to a subsequent model. The target is an actual 4096-wide
DEM, continuous bilinear occlusion matching the visually approved reference,
and at least 30 fresh frames/second without pre-rendered animation frames.
The Python host remains; expensive work runs in custom GLES 3.1 GPU kernels.

**The target is NOT met.** The implementation is in the local repo and in
`/home/map/map-projection-staging` on `ssh projector`. Production was not
restarted or replaced. No remote Git access was configured. This is an
experimental path, not an announced 30 fps solution.

Focused actual-board results, 4096 × 2396 DEM and 3840 × 2160 output:

- Reference: approximately 2400 ms/frame.
- New conservative-beam implementation: approximately 1965 ms/frame (0.51 fps).
- About 18% less frame time; still about 59 times the 33.33 ms budget.
- All 9,814,016 terrain cells compared for each of three consecutive sun positions:
  **zero byte differences**, including cached-witness reuse.
- Still approximately 4.67 million exact fallback rays per frame. This is the
  primary unresolved problem. The dispatch reduction alone is insufficient.
- These are synchronized offscreen GPU timings including native-sized output,
  excluding HDMI swap and browser previews. Static preparation/shader warmup
  is excluded. The first timed frame has warmed witnesses; later timed frames
  advance the sun by 20 seconds, so they are not repeated-frame benchmarks.

Raw results: [precise-4096-first-pass.json](results/2026-10-25/precise-4096-first-pass.json).
Do not interpret these as sustained performance, full accuracy certification,
or evidence of 30 fps feasibility. The projector's validated reference remains
available for comparison.

## Implementation

`gpu_precise.py` contains all new GPU stages. `backend="precise"` selects them;
the advanced UI adds **Precise GPU · experimental** without changing defaults.

1. At terrain preparation, build min/max bounds on the complete bilinear patches
   in each 8 × 8 cell block. Inclusive corner extrema bound every point of a
   bilinear patch. This is static geometry preparation, not a time cache.
2. For the current sun, sweep coarse corridors. Each corridor bounds every ray
   in its entire width, rather than interpolating an approximate horizon.
   Lower bounds use guaranteed terrain heights at upstream planes; upper bounds
   enclose the whole intervening surface. The receiver slab's top is kept
   separate to avoid adding fictitious ray rise beyond the actual receiver.
3. A fragment pass resolves guaranteed clear or blocked pixels and shades them.
   Negative Lambert contributions require no occlusion work. Border padding and
   floating point margins intentionally increase uncertainty rather than create
   falsely confident classifications.
4. For unresolved pixels, re-intersect the last successful blocker patch with
   the **current** sun ray. The cache stores geometry witnesses, never old
   visibility. A sun/time/date jump cannot blindly reuse an old shadow.
5. Compact remaining cells into an SSBO and indirectly dispatch only those rays.
   Use the validated reference's continuous bilinear intersection and hierarchy
   traversal. If the coarse upper bound proves all upstream slabs clear, stop
   the exact traversal at the sunward edge of the receiver slab (at most eight
   dominant-axis cells). Otherwise search until an occluder or domain exit.
6. Write the successful blocker for subsequent frames. Shade and gather into the
   native projector buffer using the existing exact mapping. Every ray uses the
   current sun. There is no temporal blending or forced shadow monotonicity.

The GPU-only dispatch avoids synchronous CPU reads of the candidate count.
Traversal-error diagnostics are polled behind a fence; failures surface as
errors. Non-scan modes no longer allocate the legacy full-size sheared horizon
and column-height hierarchy, saving roughly 190 MB of GPU memory at 4096.
The reference algorithm itself has not been edited.

## Files

- `gpu_precise.py`: coarse bounds, classification/witness checks, indirect exact
  rays, persistent geometry-witness texture and diagnostic fence.
- `gpu.py`: lazy method-specific allocations, new renderer lifecycle/dispatch.
- `config.py`, `engine.py`: explicit `precise` backend, current SunFrame method,
  GPU requirement and prevention of approximate CPU cached-day fallback.
- `web/index.html`, `web/app.js`: one additional backend choice; the existing
  slider, playback, timezone, preview and HDMI recovery behavior is retained.
- `tests/test_precise.py`: reference comparison on partial 8-cell blocks,
  negative elevations, all quadrants, axis-aligned and near-axis rays, high and
  low sun, nearby and reversed time/direction steps with persistent witnesses.
- `experiments/benchmark_precise.py`: bounded-memory board benchmark and full
  terrain-image comparison against reference; saves output rather than holding
  several full-resolution sequences in the 2 GB RAM.

## Verification performed

All **46 local tests** passed with real NVIDIA GLES enabled. The new focused
GLES test also passed on PowerVR. Three final 4096 sunset frames match the
reference byte-for-byte on PowerVR. No broad date/latitude/speed sweep or visual
user acceptance of the new path has been performed; that is the next model's
verification task. No production deployment or benchmark-setting changes were
made to the running projector service.

## Reproduce

On the laptop:

```sh
OPENBLAS_NUM_THREADS=1 MAP_PROJECTION_GPU_TESTS=1 .venv/bin/python -m unittest map_projection.tests.test_precise
```

On the board, run the isolated staging checkout (no HDMI ownership):

```sh
cd /home/map/map-projection-staging
OPENBLAS_NUM_THREADS=1 /home/map/map-automation/.venv/bin/python -m map_projection.experiments.benchmark_precise --input input.npz --terrain /home/map/map-automation/map_projection/.cache/c59fc8ce222277e3849c3a3c-terrain.npy --output precise-4096.json --frames 6
```

The benchmark's 4K lookup is rescaled from the existing 2048 fixture for equal
output workload. Accuracy comparison reads the **entire 4096 terrain lighting
texture**, not just that rescaled lookup. The application uses its usual exact
4096 terrain lookup. Fixture bias is 47.418 m; respect the user's actual bias
when doing live comparisons.

## Where further work is needed

The coarse beam bounds are too loose over mountainous terrain: millions of
receivers still need exact rays. Even after local-ray clipping, current timings
remain dominated by fallback. Before claiming a speedup from a change, record
candidate fraction, fraction of locally limited rays, witness hit rate, and
GPU time for each pass. These metrics were deliberately not added as permanent
per-frame CPU synchronization.

Potential directions need measurement, not assumptions: narrower anisotropic
corridors, tighter height bounds near the receiver, spatial grouping of fallback
rays, and longer-lived conservative geometric visibility certificates. The
30 fps target needs an algorithmic reduction of roughly two orders of magnitude
from the current fallback workload. Merely moving the Python host to C would
not eliminate the measured GPU work. Preserve the reference and compare temporal
sequences, large sun jumps and grazing cases before replacing the live backend.
