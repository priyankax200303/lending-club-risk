#!/usr/bin/env bash
# Full rebuild. Run scripts/get_data.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f outputs/metrics.json
python3 scripts/build_notebooks.py
python3 scripts/reconcile.py
