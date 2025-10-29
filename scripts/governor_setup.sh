#!/usr/bin/env bash
# Usage: sudo bash scripts/governor_setup.sh performance|ondemand|powersave
set -euo pipefail
GOV=${1:-performance}
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
  echo "$GOV" > "$cpu/cpufreq/scaling_governor" || true
done
echo "Set governor=$GOV"
