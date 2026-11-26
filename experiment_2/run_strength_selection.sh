#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "$experiment_dir")"
cd -- "$project_dir"
if [[ -f server.env ]]; then
    set -a
    source ./server.env
    set +a
fi
python_bin="${LERRA_PYTHON:-python3}"
export LERRA_OUTPUT="${LERRA_OUTPUT:-$project_dir/runs/experiment_2_strength_selection}"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

action=${1:-parallel}
if (( $# )); then shift; fi
case "$action" in
    parallel|status)
        script=launch
        if [[ "$action" = status ]]; then script=status; fi
        exec "$python_bin" -m "experiment_2.parallel_strength.$script" "$@"
        ;;
    verify)
        exec "$python_bin" "$experiment_dir/code/verify_design.py" "$@"
        ;;
    run|report)
        exec "$python_bin" "$experiment_dir/code/run_strength_selection.py" "$action" --output "$LERRA_OUTPUT" "$@"
        ;;
    retry-mail)
        if (( $# > 1 )); then
            echo "Usage: $0 retry-mail [campaign-output-directory]" >&2
            exit 2
        fi
        output=${1:-$LERRA_OUTPUT}
        exec "$python_bin" -m runtime.comparison.protocol_v2.notifications --retry-pending "${output%/}.alerts"
        ;;
    *)
        exec "$python_bin" "$experiment_dir/code/run_strength_selection.py" "$action" "$@"
        ;;
esac
