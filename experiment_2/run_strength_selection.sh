#!/usr/bin/env bash
# Preserve the original entry point and its no-argument parallel launch.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if (( $# == 0 )); then
    set -- parallel
fi
exec "$repo/lessmark" "$@"
