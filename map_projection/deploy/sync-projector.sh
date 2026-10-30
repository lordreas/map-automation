#!/usr/bin/env bash
# Compatibility entry point; all installation logic lives in deploy.sh.
set -euo pipefail
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/deploy.sh" "$@"
