#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MPLCONFIGDIR="$PROJECT_ROOT/.matplotlib"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"

mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/.venv/bin/activate"
