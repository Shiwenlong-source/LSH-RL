#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

export UV_CACHE="${UV_CACHE:-$DIR/.cache_uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DIR/.cache_xdg}"

# Set proxy variables outside this script if your environment needs them.
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1,0.0.0.0}"
export no_proxy="${no_proxy:-$NO_PROXY}"

exec "$@"
