#!/usr/bin/env bash
# Called from the repository root by ./lessmark after loading the source server settings.
set -euo pipefail

usage() {
    echo 'Usage: ./lessmark copy-inputs [--dry-run] user@host:/absolute/input-root/'
    echo 'A local destination directory is also accepted.'
}

rsync_options=(--archive --copy-links --partial --protect-args --stats)
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    --dry-run) rsync_options+=(--dry-run); shift ;;
    *) rsync_options+=(--info=progress2) ;;
esac
if (( $# != 1 )) || [[ -z "$1" || "$1" == -* ]]; then
    usage >&2
    exit 2
fi

source_root="${LERRA_DATA_ROOT:-/data}"
destination="$1"
file_list="$(mktemp)"
trap 'rm -f -- "$file_list"' EXIT

"${LERRA_PYTHON:-python3}" scripts/assets.py --root "$source_root" --files-from "$file_list"
rsync "${rsync_options[@]}" --files-from="$file_list" -- "${source_root%/}/" "$destination"
