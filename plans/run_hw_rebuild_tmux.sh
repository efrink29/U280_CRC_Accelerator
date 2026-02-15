#!/usr/bin/env bash
set -euo pipefail

PLATFORM="xilinx_u280_gen3x16_xdma_1_202211_1"
SESSION_NAME="${1:-hw_rebuild_manual}"
LOG_PATH="${2:-plans/logs/${SESSION_NAME}.log}"

cd /users/arashs/U280_CRC_Accelerator || exit 1
mkdir -p "$(dirname "$LOG_PATH")"

exec > >(tee -a "$LOG_PATH") 2>&1

finish() {
  rc=$?
  echo "[END $(date -Is)] RC=${rc}"
}
trap finish EXIT

echo "[START $(date -Is)] session=${SESSION_NAME}"
if [[ -f /share/Xilinx/Vitis/2023.1/settings64.sh ]]; then
  source /share/Xilinx/Vitis/2023.1/settings64.sh
elif [[ -f /tools/Xilinx/Vitis/2023.1/settings64.sh ]]; then
  source /tools/Xilinx/Vitis/2023.1/settings64.sh
else
  echo "ERROR: Vitis settings64.sh not found in /share or /tools"
  exit 1
fi

if [[ -f /opt/xilinx/xrt/setup.sh ]]; then
  source /opt/xilinx/xrt/setup.sh
else
  echo "ERROR: XRT setup.sh not found at /opt/xilinx/xrt/setup.sh"
  exit 1
fi

echo "[VERSIONS $(date -Is)]"
v++ --version | head -n 1
xbutil --version | sed -n '1,3p'

echo "[CLEAN $(date -Is)] removing previous hw artifacts"
rm -rf "_x.hw.${PLATFORM}" "build_dir.hw.${PLATFORM}" "package.hw"

echo "[BUILD HOST $(date -Is)]"
make host PLATFORM="${PLATFORM}"

echo "[BUILD HW $(date -Is)]"
make build TARGET=hw PLATFORM="${PLATFORM}"
