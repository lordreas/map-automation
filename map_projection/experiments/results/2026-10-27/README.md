# Self-contained A7Z cache and playback verification — 2026-10-27

The replacement 64 GB SD card holds the complete lossless compressed skyline
bank (9,598,187,442-byte archive plus index/manifest, approximately 9.60 GB),
today's event cache, and the next day's cache generated on the board. Each day
occupies approximately 285 MB. About 40 GiB remains free. Runtime files and
native compilers are under `/home/map/map-automation`; no remote Git access.
The full horizon transfer was resumed and verified by rsync `--append-verify`.

## Full day generated on the board

The board compiled **2026-10-28**, with four native C/OpenMP workers, while the
normal projector service showed a paused map. No laptop computation, Internet
service or remote cache lookup was used during generation.

- 9,814,016 receivers: 4096 × 2396 terrain grid.
- 19,650,388 transitions.
- Native numerical/compiler time: **361.43 s**.
- Complete command including Python orchestration: **362.49 s**.
- Complete command plus storage flush: **362.59 s — 6 min 3 s**.
- Peak compiler child RSS: **430,404 KiB (420 MiB)**.
- Result size: **284,876,130 bytes**.
- All five NPY files match the independently generated laptop reference
  byte for byte (SHA-256), including both chronological streams.
- The application successfully loaded and displayed this on-board-generated
  date at native 4K60 before returning to the previous day/time.

The archive retains its existing 0.25-degree angular sampling, one-second state
sampling and 50 ms event refinement. Exact agreement certifies unchanged event
compilation, not unsampled geometric accuracy.

## Native 4K playback

The benchmark includes the real daily cache, Engine worker, current calibration,
solar calculation, GPU updates/shading, native 3840 × 2160 KMS presentation,
HTTP server, and explicit preview requests. It excludes phone/browser painting
and optical verification. The display refresh is 60 Hz; fresh frames are slower.
No cache generation or bank transfer ran concurrently with playback measurements.
Each case sampled 12 seconds following preparation/warmup.

| Case | Daylight fps | p95 interval | Worst interval |
| --- | ---: | ---: | ---: |
| Sunset, 1 simulated minute/real second | 30.04 | 34.21 ms | 35.17 ms |
| Fast playback, 15 simulated min/s | 28.51 | 46.58 ms | 136.14 ms |
| Fast playback, 60 simulated min/s | 29.61 | 34.86 ms | 52.34 ms |
| Sunset + 960 × 540 browser preview | 28.66 | 45.72 ms | 51.59 ms |
| Sunset + 1920 × 1080 browser preview | 28.49 | 47.42 ms | 94.32 ms |

The faster playback cases include night in their overall averages. The fps and
p95 above use consecutive **daylight** frames, avoiding inflated averages from
cheap black nighttime frames; maxima are from the complete respective run.
Preview requests repeat every 750 ms, matching the UI's playback throttle.
Transitions and detailed frame statistics are in the raw JSON files.

## Cooling limitation

There is no heatsink or fan fitted. During the bake, CPU temperatures reached
approximately 85–89 °C. Both CPU cooling controls reached maximum state,
limiting their maximum frequency to **416 MHz** (hardware maxima: 1.794/2.002 GHz).
The fan control requested maximum output, but no fan was present. The first
playback run started at 79.36 °C and ended at 84.84 °C; the large-preview run
started at 81.07 °C and ended at 85.96 °C. Clock caps varied during playback,
and both runs ended with the big cluster capped at 416 MHz.

These are measurements of the current **uncooled** board, not an estimate of
performance after adding cooling. The earlier 209.36-second day bake was a
separate run under different thermal conditions. A fitted heatsink, preferably
with airflow for sustained loads, should precede another comparison. Thermal
limits have not been overridden.

## Artifacts and reproduction

- `day-benchmark-2026-10-28.json`: timing, storage, memory and exact reference hashes.
- `thermal-during-bake.json`: thermal controls and clock evidence.
- `full-pipeline-2026-10-27.json`: full pipeline and transitions, 960-wide preview.
- `full-pipeline-preview1920-2026-10-27.json`: large preview case.

```sh
# Use a fresh date/output: completed day caches are immutable.
ssh projector 'cd /home/map/map-automation && OPENBLAS_NUM_THREADS=1 .venv/bin/python -m map_projection.experiments.benchmark_native_day --horizons map_projection/.cache/horizons4096-compressed --output map_projection/.cache/events --date YYYY-MM-DD --threads 4 --report map_projection/.cache/day-benchmark-YYYY-MM-DD.json'
```

For KMS playback benchmarks stop `map-projection.service` first and restore it
on exit; the benchmark requires exclusive ownership of HDMI. Use
`benchmark_full_events --date YYYY-MM-DD --switch-interval .001`; it keeps
installation settings unchanged. The normal service was restored afterwards.
