# Map projection

A local native-pixel output window and a LAN web control interface. Implementation,
settings, development dependencies and caches live in `map_projection/`. The mesh
generator and lighting preparation share the coastline policy in `elevation_surface.py`.

## Deploy from Linux to the Radxa

Use `./deploy.sh --setup` for the initial installation and `./deploy.sh` for updates.
The SSH alias `projector` must reach the board's `map` account with passwordless sudo;
use `--host` to override it. Deployment uses rsync and configures the projection,
console toggle, and daily cache services without installing Git access on the board.

Source data is kept outside Git. Before deploying, provide these local files:

- `stl_files/full_map_0.4mm.stl`: the original printed terrain mesh.
- `elevation_tiles/elevation_43.5_48.4_4.9_16.4_shape_11697x20000_resamp_average_bboxmerge_v2.npy`:
  the source elevation raster.
- `Genf__Salzburg.gpx`: the route used by the GPX layer.

Create a laptop Python environment with `python3 -m venv .venv`, then install
`map_projection/requirements-prepare.txt`. Missing geographic layers are downloaded
and prepared on the laptop during deployment. Use `--caches` to transfer an existing
horizon bank and daily event caches too; see [HORIZON_CACHE.md](HORIZON_CACHE.md) for
generation commands. Personal calibration, Wi-Fi credentials, generated data, native
binaries, and Python environments are excluded from Git. Deployment preserves the
board's calibration and uses defaults when no local or remote settings file exists.

## Start on this laptop

From `D:\git\map-automation`:

```powershell
.\map_projection\.venv\Scripts\python.exe -m map_projection --open-browser
# Alternatively:
.\map_projection\start.ps1
```

Controls: **http://127.0.0.1:8080**. On another device, use `http://<laptop-LAN-IP>:8080`;
Windows Firewall must permit the server on the private network. The service binds all
interfaces by default. It is intended for a trusted LAN, without Internet exposure or
authentication. Use `--host 127.0.0.1` for laptop-only access.

The initial output is a movable, unscaled **1920 × 1080** window. Its close button exits
the application; Escape leaves fullscreen. Every control is also available over HTTP.
Initial scene: today's date at noon in **Europe/Zurich**, so a daytime demonstration is
visible even when started at night. Set the installation timezone with `--timezone`
(or `MAP_PROJECTION_TIMEZONE`); all installed IANA zones are supported. Daylight saving time
is handled by IANA timezone data. On an ambiguous fall-back time, the first occurrence
is used for an unqualified local time; an explicit offset in the API disambiguates it.

For a fresh Windows environment:

```powershell
py -3 -m venv map_projection\.venv
.\map_projection\.venv\Scripts\python.exe -m pip install -r map_projection\requirements.txt
.\map_projection\.venv\Scripts\python.exe -m pip install ziglang==0.14.1
```

The C renderer compiles automatically. On this laptop, the existing PlatformIO Python
was used to create this isolated environment; neither its installation nor the root
repository's Linux `.venv` was changed. Browser QA dependencies are development-only.

## Banana Pi M2 Zero / Allwinner H3

Use a current 32-bit ARM Linux image with working **sun4i DRM/KMS**, a Mesa/Lima graphics
stack and SDL2's **kmsdrm** driver. The native C99 calculation code supports ARMv7 and
does not require CUDA, Vulkan, OpenGL compute, Numba, a desktop browser, or a desktop
compositor. SDL uses the GPU for texture presentation when its driver permits it.
Shadow calculation uses a linear-time C heightfield sweep on the CPU; this avoids
requiring modern GPU compute facilities on the H3's OpenGL ES 2.0-era Mali-400.

Example Debian/Armbian package setup (package availability depends on the image):

```sh
sudo apt update
sudo apt install gcc libc6-dev python3 python3-numpy python3-pil python3-sdl2 \
  libsdl2-2.0-0 libgbm1 libegl1 libgl1-mesa-dri libdrm-tests tzdata
cd /opt/map-automation
python3 -m map_projection --direct --width 1920 --height 1080 --refresh 60
```

Run direct output from a local VT with DRM device access and no compositor holding the
connector. `--direct` explicitly selects KMS/DRM, requires fullscreen and **refuses to
fall back to X11/Wayland**. Unsupported modes are rejected. The control server remains
available for mode-selection errors after SDL initialization. Failure to initialize
SDL/KMS itself is a startup failure and is reported in the process log.

Inspect `modetest -M sun4i-drm -c` and the app's Output diagnostics to see advertised and
active modes. The board's Linux image and its display driver must be tested before
enabling the supplied [systemd service](deploy/map-projection.service). Adapt its user,
checkout path, device groups and VT to the image; make the checkout writable by that
user for compilation, settings and cache creation. Do not install a desktop Chromium
on the 512 MB board for this application. The browser runs on your phone/laptop.

Copy the source STL onto the Pi. No laptop preprocessing is required: the Pi streams
the mesh to build both its projection lookup and compact exposed-terrain grid. After
preparation, arbitrary sun positions need neither the original mesh nor network data.
A changed projector pose rebuilds intersections from the original mesh.

**This board has not been physically tested in this session.** The C renderer was
cross-compiled successfully for `arm-linux-gnueabihf`; this verifies compilation, not
frame rate or DRM operation on a specific image. HDMI scanout refresh and the rate of
new shadow calculations are separate. A 60 Hz signal may repeat a slower-changing
shadow frame.

## Radxa Cubie A7Z, 2 GB

Validated on the physical A7Z on 2026-10-24 with Radxa's A733 Debian 11 KDE
R6 image, ARM64 kernel `5.15.147-21-a733`, SDL 2.0.14 and the supplied PowerVR
graphics libraries. Direct KMS/DRM output activates **3840 × 2160 at 60 Hz**
using the PowerVR BXM-4-64 GLES 3.2 renderer. A full 4K RGB readback matched a
single-white-pixel test exactly. This verifies the render buffer, not the optics.

The deployed copy is `/home/map/map-automation`, with an isolated `.venv`
(NumPy 1.26.4, Pillow 11.3.0, PySDL2 0.9.17, PyOpenGL 3.1.10). It contains the
original STL, the `bboxmerge_v2.npy` elevation raster and the prepared land-aware
lighting surface. There is no Git checkout or Git
authentication on the board. Laptop settings are backed up on the board as
`map_projection/settings.laptop.json`; the initial 4K corner coordinates are
scaled from the saved 1080p calibration and must be physically rechecked.

The installed `map-projection.service` comes from
[deploy/radxa-a7z.service](deploy/radxa-a7z.service). It runs as `map` on VT1,
selects DRM card 0, and exposes controls at `http://<board-IP>:8080`.
SDDM is disabled and the default boot target is `multi-user.target`.
The app's `--headless` flag would disable HDMI output; the service uses `--direct`.
The service needs the distribution's `kbd` package for `/bin/chvt`.

Update the board from this laptop, preserving board settings, its virtual
environment and caches:

```sh
bash map_projection/deploy/sync-projector.sh
ssh projector 'sudo systemctl status map-projection --no-pager'
ssh projector 'sudo journalctl -u map-projection -n 50 --no-pager'
```

The sync script prepares the lighting surface locally, copies source, the STL,
DEM and prepared surface with its metadata using rsync, installs the service
unit, then restarts the service. It never copies `.git` or the laptop's virtual
environment. It expects the existing SSH alias `projector`, remote user `map`,
and sudo access for restarting the service.

To temporarily restore the desktop:

```sh
ssh projector 'sudo systemctl stop map-projection && sudo systemctl start sddm'
```

To return to projection:

```sh
ssh projector 'sudo systemctl stop sddm && sudo systemctl start map-projection'
```

The application requests GLES before creating a direct-output window. For SDL
older than 2.0.16 it uses the nearest-neighbour texture creation hint rather
than `SDL_SetTextureScaleMode`, which crashes in the R6 SDL GLES driver.
The renderer also compares depths at float32 storage precision so coincident
back faces cannot win because of ARM floating-point rounding. The local suite
passes all 45 tests, including real GLES rendering checks; native,
playback, recovery, watchdog and GLES checks also passed on the board.

The deployed GPU renderer has been checked in the actual 4K60 HDMI session:
HDMI scanout remains 60 Hz; fresh terrain-frame throughput depends on grid
size and algorithm (see the measurements in experiments/README.md). Preview and full-resolution captures work during
playback. Initial geometry preparation took about 17 seconds; later sunlight
changes reuse it. HDMI refresh remains independent of new shadow-frame cadence.
The pre-update app and unit are backed up in
`/home/map/map-projection-backups/before-timelapse-20261024.*`.

## Display hardware and what “one pixel” means

| Device | Evidence about display output |
| --- | --- |
| Radxa Cubie A7Z, **A733**, 2 GB | Physically verified at 3840 × 2160, 60 Hz using direct KMS/DRM and accelerated OpenGL ES output; see deployment notes above. |
| BPI-M2 Zero, H3/H2+ | Manufacturer lists mini HDMI 1.4 and 1080p. Its pages disagree on 30/60 Hz; neither establishes 4K60 output. 4K video decoding is not proof of 4K HDMI output. |
| Original Radxa ZERO, **S905Y2** | Radxa's current specification explicitly lists **Micro HDMI × 1, 4K60p**, and **65 × 30 mm**. This is a manufacturer output claim; a chosen Linux image still needs validation. |
| Radxa ZERO **3W/3E**, RK3566 | Radxa explicitly lists **1080p60 output**, despite the chip's other video capabilities. This is a different board from the original ZERO. |
| Acer Vero HL6810 | Acer lists UHD output, HDMI 2.0b and optical throw ratio **1.15–1.5**. The requested zoom must be calibrated within that range. |

Primary sources:

- [Banana Pi M2 Zero specification](https://wiki.banana-pi.org/Banana_Pi_BPI-M2_ZERO)
- [Radxa ZERO: Display and Size rows](https://docs.radxa.com/en/zero/zero)
- [Radxa ZERO hardware: HDMI maximum resolution](https://wiki.radxa.com/Zero/hardware/zero)
- [Radxa ZERO 3: Display row](https://docs.radxa.com/en/zero/zero3)
- [Acer HL6810 specification](https://www.acer.com/gb-en/projectors/vero/pdp/MR.JX511.007)
- [Acer HL6810 manual](https://global-download.acer.com/GDFiles/Document/User%20Manual/User%20Manual_Acer_1.0_A_A.pdf?BC=ACER&LC=en&OS=ALL&SC=PA_6&Step1=&Step2=&Step3=HL6810&acerid=638473066935062939)

The application calculates exactly one sample per **HDMI input pixel**. It creates an
RGB24 texture, disables blending/filtering, does not request logical scaling, and
requires texture dimensions to equal the actual SDL drawable. On Windows it requests
per-monitor DPI awareness before opening a window. In fullscreen it requires an exact
advertised width/height/refresh tuple and checks the activated display mode. Desktop
windowing is reported as desktop output; it is not claimed to bypass the compositor.

For the Acer's user-reported 1080p DMD plus four-position pixel shifting, 4K input pixels
are effective projected samples across subframes, not 8.3 million separate physical
micromirrors. The application cannot control the projector's XPR phase or guarantee an
optical point-spread function. Calibrate the **actual selected input mode** and verify
with the one-pixel checkerboard, columns, rows and single-pixel patterns. Keep keystone,
digital zoom, overscan and image enhancement off; use suitable aspect/input settings.
For colour-pixel verification, use RGB or YCbCr 4:4:4 at a supported bit depth; chroma
subsampling can blend neighbouring colour samples. Optical zoom changes intrinsics.
Changing 1080p/4K may also change projector processing and requires rechecking alignment.

The software reports `projector_pixels_verified: false` deliberately: an SDL readback
cannot verify the HDMI cable, projector input processing or optics. The control preview
is explicitly scaled and is never the physical output surface.

When the projector is connected, select its display in Output and request fullscreen
3840 × 2160 at an advertised refresh rate. The laptop's built-in display currently does
not advertise that mode; no silent fallback is allowed.

## Geometry and calibration

- Map frame: origin at STL XY bounding-box center on **Z = 0**; X east, Y north/up the
  wall, Z out from the wall. Default lens center: **(0, 750, 2064) mm**.
- Projector frame: X right, Y down, Z forward; it aims at the map-frame origin. Roll,
  throw ratio and optical-center shifts are editable. The default throw ratio 1.2 is
  within Acer's range but is an estimate until zoom is measured/calibrated.
- The generator's `_flip_mesh_across_x_axis` flips raster Y when exporting. **Exported
  STL Y already increases north.** Geographic mapping follows the original lon/lat
  raster, not an assumed UTM transformation. AEQD was used to choose the print's aspect
  ratio, not to position individual raster vertices.
- The saved nominal geographic extent is 43.5–48.4° N, 4.9–16.4° E and 1500 ×
  877.260605 mm. Actual STL edges are slightly inset. STL does not carry georeferencing;
  these are inferred from the generator and tile metadata. The Mont Blanc check reaches
  a roughly 44.84 mm-high point, near the mesh maximum of 45 mm. Confirm landmarks on
  the assembled print before relying on geographic targeting.
- `P = H K R [I | -C]`. Four corners calibrate the map's base plane; they modify the
  ray projection before intersection. Each output pixel is sampled at `(u+0.5,v+0.5)`.
  Streaming triangle rasterization with perspective-correct depth gives the nearest
  triangle hit, including occlusion and boundary walls. It is mathematically the same
  intersection as a pinhole ray against those triangles, up to floating-point error.
- Pixel-to-world: invert the 3×3 left block of P and use the stored depth. World/geo-to-
  pixel: intersect a vertical ray with the STL, project that point, and check lens
  visibility with an independent mesh ray. The interface reports the **actual nearest
  pixel-center hit** and its distance from the target; arbitrary coordinates generally
  do not coincide exactly with a pixel center.
- Corner buttons move one physical output pixel per request; holds repeat until
  release, pointer cancellation, or loss of window focus. Calibration grid and blackout
  updates remain immediate while an expensive geometry job finishes.
- Horizontal mirror defaults off. It is part of the projection matrix, so forward and
  inverse mappings remain consistent. Raw pixel test patterns still address physical
  output coordinates. A future physical mirror also changes the effective lens pose;
  a horizontal image flip alone does not model that optical path.

Four planar correspondences cannot correct a wrong lens position's relief parallax,
radial lens distortion, tile placement errors or unknown XPR offsets. Those need further
3D landmarks/intrinsic calibration. No spatial antialiasing or post-render keystone
resampling is applied to the projector frame.

## Live shadows and DEM feasibility

The default **Live · terrain grid** mode samples only the exposed top of the original
STL, using a top-down depth pass, at 2048 × 1198 samples (~9.8 MB float32). It does not
retain duplicate or hidden geometry. A native directional horizon sweep is O(number
of grid samples); shade lookup is O(number of output pixels). This handles arbitrary
sun azimuth/elevation continuously without a finite precomputed set of sun angles.
The sweep interpolates upstream terrain horizons, so cast-shadow edges are an
approximation at the configured grid resolution. This is separate from the **exact
original-STL projector intersections**, which always remain at output resolution.

**Reference · original STL shadow map** streams all original triangles from the sun
view into a 512–4096-square orthographic depth map. It is slower but gives a useful
independent reference. Shadow maps still have finite sampling and depth bias; they are
not infinite-resolution exact secondary rays. Flat STL normals provide diffuse light.
Both modes include terrain cast shadows, not only hillshading from slope normals.

The supplied STL has 16,467,674 triangles. A measured upward-facing filter retained
16,443,114 (~99.85%), so simply dropping bottom/walls did **not** halve this particular
file. The live terrain grid avoids that storage issue entirely. `prepare_top` remains
available as a lossless triangle filter, but no redundant near-full-size STL is kept
by the runtime. Projection uses the original file to retain silhouette occlusion.

The optional **Actual geographic elevations** mode uses the existing full-resolution
`bboxmerge_v2.npy` DEM as its source, samples it in bounded row blocks, and computes
shadows and normals in geographic metres. Projection still follows the printed STL.
A prepared land-aware surface uses the **same** OSM mask construction, HGT-void
handling and `apply_sea_level_model` function as `depth_map.py`, shared through
`elevation_surface.py`. Positive elevations survive mask disagreements. Negative
sea is zero; negative land is kept down to -20 m only in four-connected patches
of at least 10 source samples. The source DEM remains unchanged. Preprocessing
runs once on the laptop, before interpolation, and runtime reads bounded rows
of the resulting raster. Cached terrain and sun frames use the new surface key.

Prepare the surface once (the synchronization script also checks it):

```sh
.venv/bin/python -m pip install -r map_projection/requirements-prepare.txt
.venv/bin/python -m map_projection.prepare_terrain --if-needed
```

Only the laptop needs the GIS and SciPy dependencies. The prepared int16 file
adds approximately 468 MB on the SD card; the projector never labels the full
raster in RAM. The sea-level floor and minimum patch size can be overridden
with the same CLI options as terrain generation. For a different DEM extent,
pass its settings file with `--config`.
The 11697 × 20000 int16 source is **467.9 MB**: with a 512 MB board, the whole DEM plus
render buffers and OS cannot be resident. Full-source-resolution live shading is not
claimed feasible on this board. A sampled 2048/4096-column grid is the supported path.
DEM mode uses one regional sun direction and a local equirectangular metres approximation;
it does not model curvature or atmospheric refraction across the Alps.

The default DEM path is inferred from this repository. For another source:

```sh
python3 -m map_projection --dem /path/to/north-up-elevation-metres.npy
```

The NPY must be a numeric 2D north-up raster **covering exactly the configured geographic
extent** with elevations in metres. Do not use the older padded, non-`bboxmerge_v2`
caches as if they had those bounds. No pickle metadata is executed.

Time-to-sun uses the [NOAA fractional-year equations](https://gml.noaa.gov/grad/solcalc/solareqns.PDF).
This is an astronomical approximation, not the higher-precision SPA model. Night is
black; ambient light is applied during daytime only. Shadows cannot remove ambient
room light from the physical print, so the room needs appropriate lighting.

## Preview, calibration and pending sunlight

The browser preview uses Lanczos antialiasing and requests a raster sized for
its element and device pixel ratio (up to 1920 × 1080). Browser filtering is
smooth. `/frame.png` and the physical projector remain unfiltered native pixels.

Jogging offers 1, 5 and 10 px steps, including hold-to-repeat. Browser mutations
are sent in order; selecting another scene stops hold-repeat. Repeated identical
settings do not restart a pending render. The API reports `pending` and
`displayed_pattern`, so the interface explains when it is preparing sunlight
while keeping calibration visible. New calibration still requires the original
STL intersection pass, which can take tens of seconds on the board.

Direct output explicitly resets its viewport and scale and uses full-size source
and destination rectangles. Unchanged frames are not repeatedly submitted to the
GPU; HDMI continues scanning at 60 Hz. These guards address possible rendering
state issues, but the reported rare zoom glitch has not been reproduced or
attributed to a hardware fault.

## Sunlight and animation

Drag the time slider to render that time immediately, including while holding
or moving the handle. **Start Animation** records that selection and loops the
selected calendar day. The logarithmic speed slider spans **1 simulated second
per real second** to **1 simulated hour per real second**. Speed changes take
effect during playback without changing the selected return time.

While playing, the button reads **Stop Animation**, the slider follows the
presented frame and its handle shows a play marker. Stop eases back to the
selection made before playback, over one second. Dragging the time slider
instead cancels playback and takes direct control, without returning first.
**Now** eases to the actual current date/time over one second. Both transitions
use smoothstep easing. **Enable/Disable Shadows** toggles the lighting overlay;
disabling it leaves a white map with the exact registered footprint.

The displayed clock uses whole minutes but follows every presented frame over
`/api/events`. Shadow motion remains continuous; it is not rounded to minutes.
Rendering runs as fast as computation and HDMI presentation allow, with GPU
backpressure instead of a configured target frame rate. Simulated time uses a
monotonic wall clock, independently of achieved rendering throughput. The loop
respects 23/25-hour daylight-saving days. `step_minutes` only controls optional
cached-day spacing. Lighting and terrain settings apply immediately under the
collapsed advanced section. The timezone is an installation setting rather
than a browser control, for example `--timezone Asia/Kathmandu`.

On the A7Z, direct output automatically uses the fastest measured GLES 3.1
method: one GPU lane scans each sun-aligned ray, with the horizon, normals,
lighting and exact 4K projector lookup retained in GPU memory. Final pixels
are rendered directly into the display surface, without full-frame readback
and re-upload. Browser previews use an on-demand area-filtered GPU capture
with asynchronous pixel-buffer readback and nonblocking fence polling;
full-resolution captures remain available. `--cpu` selects the optimized CPU
fallback; `--gpu` explicitly requests a GLES display on a desktop. Automatic
mode falls back to CPU output if the compute API cannot be initialized.
Printed-STL lighting and the original-STL reference mode use the CPU renderer.

The fast GPU scan is an interpolation approximation. Isolated bright regions
are now checked with exact heightfield rays against a static max-height pyramid.
Only proven occlusion replaces a highlight; genuine illuminated peaks survive.
The GPU builds its candidate list and dispatches refinement without CPU readback.
Memory barriers protect horizon reuse between frames. This improves sunset
isolated-highlight behavior but does not resolve broader temporal aliasing
from the moving scan lattice. Measurements and quality checks are recorded in
[experiments/README.md](experiments/README.md). Lower lighting-grid widths can
update faster while HDMI dimensions and projector mapping remain unchanged.

### Continuous DEM ray reference

In advanced lighting, choose **Calculation → Reference · continuous DEM rays
(slow)**. This is a separate reference from the original-STL shadow map. It
intersects every receiver's sun ray with the **continuous bilinear surface** of
the selected, land-corrected geographic DEM grid. On each crossed patch it
analytically checks the quadratic height difference, including interior maxima;
a conservative max-height hierarchy skips only provably clear patches. There
is no sheared resampling, isolated-pixel candidate selection, random sampling,
or temporal smoothing. Traversal exhaustion raises an error rather than
silently inventing a shadow. A separate float64 CPU implementation visits
all crossed patches without a hierarchy and serves as a numerical oracle.

This is ground truth for the **selected sampled-grid, point-sun model**, within
floating point tolerance (1 mm intersection threshold in geographic metres).
It retains the same normal field, receiver bias, sun vector, sea-mask cleanup
and flat geographic approximation as live mode. It does not claim exact
intersection with every sample of the 20,000-wide source DEM, the printed STL,
Earth curvature, or the finite solar disc. Grid width controls geometric detail;
HDMI and the registered projector lookup stay at 4K. The comparison deliberately
uses the same grid in both methods to expose algorithmic error independently
of geometry resolution. This reference is live only; cached-day rendering is
disabled to prevent using the approximate CPU sweep under a reference label.

A 121-frame test of 16:20–17:00 on 2026-10-25, at 20 simulated seconds/frame,
reproduces severe scan aliasing on the board. At the actual 512 × 299 grid,
28,123 terrain pixels switch lit/shadow more than once in fast mode, with a
maximum of 98 transitions. The reference has zero reversals in that sequence.
At 2048 × 1198, fast mode has 284,256 reversing pixels, versus 297 in the
reference. The three most active reference cells were checked independently
on the CPU: their four/five transitions are genuine changing-azimuth occlusion.
A DEM is a height field, but need not be geometrically convex.

All-cell GPU reference classifications match 3,078 independent CPU queries on
synthetic terrain, plus sampled real-terrain sunset rays, on both NVIDIA and
PowerVR. Analytic tests cover a flat surface, ramps, an interior bilinear
occluder, edge-parallel rays and a monotonic ridge shadow. See
[experiments/README.md](experiments/README.md) for reproduction and timings.

## HDMI reconnection

Direct output watches DRM connector status, EDID/mode changes and hotplug
uevents, including short unplug/replug cycles. On the A7Z it replaces the app
process when HDMI changes, avoiding stale SDL/PowerVR state. The browser briefly
reconnects; settings, current playback and the pre-animation return position
survive. Startup without a connected HDMI sink keeps the controls available.
The user confirmed that the calibration grid returns after physical replugging.

The A733 kernel can leave the CRTC marked active after hotplug disabled the
physical HDMI output. Direct configuration disables stale scanout through SDL's
existing DRM master descriptor before the fresh window applies the requested
mode. Reusing only the window was insufficient on SDL 2.0.14. The systemd unit
also has readiness notifications and a 20-second watchdog for driver hangs.
Unsupported modes remain errors; output is never silently rescaled.

## Validation and measured limits

Run the geometry, occlusion, pixel-addressing, timezone and HTTP checks:

```powershell
.\map_projection\.venv\Scripts\python.exe -m unittest discover -s map_projection/tests -v
.\map_projection\.venv\Scripts\python.exe -m map_projection.scripts.check_display
.\map_projection\.venv\Scripts\python.exe -m map_projection.benchmark
```

Browser QA uses `scripts/browser_check.py` and `scripts/browser_playback_check.py`
with Playwright and the installed Chrome;
neither is required on the Pi. It exercises desktop/mobile layouts, hold/release
nudging, pixel patterns, DEM rendering and a Mont Blanc geographic query.

Measured on this laptop, not the H3: original-STL projection ~1.25 s; an initial exact
STL shadow-map calculation ~2.66 s; live printed-terrain shading ~0.08 s and sampled
DEM shading ~0.13 s at 1080p/2048-column terrain. The headless steady working set was
~73 MB. Driver/GPU buffers and 4K output increase memory; benchmark the actual board.
Calculation runs without an artificial frame-rate limit; achieved cadence
depends on backend, resolution and hardware. HDMI refresh is independently
selected. Slower devices show the newest completed frame while maintaining
the requested simulated-time speed. Requested state and displayed frame
information are separately reported by `/api/state`.

Settings persist atomically in `settings.json` (ignored by Git). Projection/terrain
caches are keyed by mesh file identity, geometry, rendering code and parameters;
sun frames are cached only for reference rendering or explicit day caching. Cache
files are ignored by Git and can be removed while the app is stopped. First startup
then regenerates them. The original STL and DEM are always preserved.

The Radxa system clock was found reset after reboot with NTP disabled. Network
synchronization is now enabled (`sudo timedatectl set-ntp true`) and synchronized.
This is necessary for reliable initial dates and the **Now** action.

Geographic map layers, river animation, data sources, and the reproducible
installation entry point are described in [LAYERS.md](LAYERS.md).
