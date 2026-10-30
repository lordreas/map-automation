#!/usr/bin/env bash
# Reproducible laptop -> Radxa installation. No Git credentials on the board.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
host=${PROJECTOR_HOST:-projector}
remote=/home/map/map-automation
python=${MAP_PROJECTION_PYTHON:-$repo/.venv/bin/python}
setup=0
layers=1
restart=1
caches=0
while (($#)); do
  case "$1" in
    --host) host=$2; shift 2;;
    --setup) setup=1; shift;;
    --prepare-layers) layers=1; shift;;
    --skip-layers) layers=0; shift;;
    --caches) caches=1; shift;;
    --no-restart) restart=0; shift;;
    -h|--help) cat <<'HELP'
Usage: ./deploy.sh [--host projector] [--setup] [--prepare-layers]
                   [--skip-layers] [--caches] [--no-restart]
Default: prepare/download missing map layers on this laptop, rsync code and
terrain, install systemd units, and restart projection. Preserve calibration,
Wi-Fi, SSH, settings, and existing caches. The remote SSH user must be map with
passwordless sudo. --setup also installs OS/Python dependencies and enables
console-only boot. --caches transfers prepared horizons and daily events.
All map download products are cached locally; repeated deploys reuse them.
HELP
      exit 0;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2;;
  esac
done
cd "$repo"
config=$(mktemp)
trap 'rm -f "$config"' EXIT
if ! ssh "$host" "cat $remote/map_projection/settings.json" > "$config"; then
  "$python" - "$config" <<'PY'
import json
from pathlib import Path
import sys
from map_projection.config import ROOT,load
Path(sys.argv[1]).write_text(json.dumps(load(ROOT/'settings.json'))+'\n')
PY
fi
dem=elevation_tiles/elevation_43.5_48.4_4.9_16.4_shape_11697x20000_resamp_average_bboxmerge_v2.npy
if ((setup || layers)); then
  if ! "$python" -c 'import scipy, rasterio, geopandas, mapbox_vector_tile' >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1; then
      uv pip install --python "$python" -r map_projection/requirements-prepare.txt
    else
      "$python" -m ensurepip
      "$python" -m pip install -r map_projection/requirements-prepare.txt
    fi
  fi
fi
"$python" -m map_projection.prepare_terrain --if-needed --dem "$dem" --config "$config"
if ((layers)); then
  "$python" -m map_projection.prepare_layers --config "$config" --dem "$dem"
fi
ssh "$host" "mkdir -p $remote/map_projection $remote/elevation_tiles $remote/stl_files"
rsync -az --itemize-changes --exclude='.venv/' --exclude='.cache/' --exclude='__pycache__/' --exclude='settings*.json' map_projection/ "$host:$remote/map_projection/"
rsync -az --itemize-changes deploy.sh elevation_surface.py Genf__Salzburg.gpx "$host:$remote/"
rsync -az --relative --itemize-changes stl_files/full_map_0.4mm.stl "$dem" "${dem%.npy}-land-surface.npy" "${dem%.npy}-land-surface.json" "$host:$remote/"
if ((layers)); then
  layer_dir=$("$python" -c 'import sys; from pathlib import Path; from map_projection.config import load; from map_projection.prepare_layers import directory; print(directory(load(Path(sys.argv[1]))["geo"]))' "$config")
  layer_key=$(basename "$layer_dir")
  ssh "$host" "mkdir -p $remote/map_projection/.cache/layers/$layer_key"
  rsync -az --itemize-changes --exclude='peaks-osm.json' "$layer_dir/" "$host:$remote/map_projection/.cache/layers/$layer_key/"
fi
if ((caches)); then
  for bank in horizons4096-compressed events; do
    if [[ -d map_projection/.cache/$bank ]]; then
      ssh "$host" "mkdir -p $remote/map_projection/.cache/$bank"
      rsync -az --itemize-changes "map_projection/.cache/$bank/" "$host:$remote/map_projection/.cache/$bank/"
    fi
  done
fi
ssh "$host" "bash $remote/map_projection/deploy/install-projector.sh $setup $restart"
printf '\nProjector addresses: '
ssh "$host" 'hostname -I'
