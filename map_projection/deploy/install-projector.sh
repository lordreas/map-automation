#!/usr/bin/env bash
# Run by the root deployment entry point, over an already configured SSH link.
set -euo pipefail
cd /home/map/map-automation
setup=${1:-0}
restart=${2:-1}
if ((setup)) || [[ ! -x .venv/bin/python ]]; then
  # Debian 11 repositories moved to the signed Debian archive after EOL.
  if [[ -f /etc/os-release ]] && . /etc/os-release && [[ ${VERSION_CODENAME:-} == bullseye ]]; then
    sudo -n python3 map_projection/deploy/archive-bullseye.py
  fi
  sudo -n apt-get update
  sudo -n apt-get install -y python3-venv python3-dev build-essential rsync libsdl2-2.0-0 libegl1 libgles2 kbd libdrm-tests mesa-utils avahi-daemon
  python3 -m venv .venv
  .venv/bin/python -m pip install -r map_projection/requirements.txt
  sudo -n usermod -aG video,render,input map
fi
# Never replace the installation's calibration, date, colors, or Wi-Fi setup.
if [[ ! -f map_projection/settings.json ]]; then
  .venv/bin/python - <<'PY'
import copy
from map_projection.config import DEFAULT,ROOT,save
c=copy.deepcopy(DEFAULT)
c['output'].update(width=3840,height=2160,fullscreen=True)
c['pixel'].update(x=1920,y=1080)
c['sun'].update(terrain='dem',grid_width=4096,shadow_size=4096,backend='events' if (ROOT/'.cache/horizons4096-compressed/manifest.json').exists() else 'auto')
save(ROOT/'settings.json',c)
PY
fi
sudo -n install -m 644 map_projection/deploy/radxa-a7z.service /etc/systemd/system/map-projection.service
sudo -n install -m 644 map_projection/deploy/map-projection-console.service map_projection/deploy/map-projection-day-cache.service /etc/systemd/system/
sudo -n install -m 755 map_projection/deploy/console_toggle.py /usr/local/lib/map-projection-console.py
# Use the installation timezone, not the laptop or systemd host timezone.
zone=$(.venv/bin/python -c 'from map_projection.config import load,ROOT; print(load(ROOT/"settings.json")["sun"]["timezone"])')
sed "s@Europe/Zurich@$zone@" map_projection/deploy/map-projection-day-cache.timer | sudo -n tee /etc/systemd/system/map-projection-day-cache.timer >/dev/null
sudo -n systemctl daemon-reload
sudo -n systemctl enable --now map-projection-day-cache.timer map-projection-console.service getty@tty2.service
sudo -n systemctl enable map-projection.service
if ((setup)); then
  sudo -n systemctl set-default multi-user.target
  sudo -n systemctl disable --now sddm.service || true
fi
if ((restart)); then
  sudo -n systemctl restart map-projection.service
  sudo -n systemctl --no-pager --full status map-projection.service
fi
