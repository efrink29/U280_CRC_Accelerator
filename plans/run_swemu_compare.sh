#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PLATFORM_DEFAULT="xilinx_u280_gen3x16_xdma_1_202211_1"
PLATFORM="${PLATFORM:-${PLATFORM_DEFAULT}}"
XCLBIN_PATH="${REPO_ROOT}/build_dir.sw_emu.${PLATFORM}/kernel.xclbin"
HOST_BIN="${REPO_ROOT}/host_xrt"

setup_swemu_env() {
    if [[ ! -f /tools/Xilinx/Vitis/2023.1/settings64.sh ]]; then
        echo "Missing /tools/Xilinx/Vitis/2023.1/settings64.sh"
        exit 1
    fi
    if [[ ! -f /opt/xilinx/xrt/setup.sh ]]; then
        echo "Missing /opt/xilinx/xrt/setup.sh"
        exit 1
    fi

    # shellcheck disable=SC1091
    source /tools/Xilinx/Vitis/2023.1/settings64.sh >/dev/null 2>&1
    # shellcheck disable=SC1091
    source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

    # Workaround for environments where sw_emu searches cpu_em path.
    local fixroot="${HOME}/.vitis_emu_fix"
    mkdir -p "${fixroot}/data/emulation/unified/cpu_em/generic_pcie/model"
    ln -sf \
        /tools/Xilinx/Vitis/2023.1/data/emulation/unified/sw_emu/generic_pcie/model/genericpciemodel \
        "${fixroot}/data/emulation/unified/cpu_em/generic_pcie/model/genericpciemodel"

    export XILINX_VITIS="${fixroot}"
    export LD_LIBRARY_PATH="/tools/Xilinx/Vitis/2023.1/lib/lnx64.o:/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}"
    export XCL_EMULATION_MODE="sw_emu"

    local runtime_ini="${REPO_ROOT}/plans/.xrt_runtime.ini"
    cat > "${runtime_ini}" <<'EOF'
[Runtime]
verbosity=0
EOF
    export XRT_INI_PATH="${runtime_ini}"
}

ensure_build_artifacts() {
    if [[ ! -f "${REPO_ROOT}/xrt.ini" && -f "${REPO_ROOT}/xrt.ini.bak" ]]; then
        cp "${REPO_ROOT}/xrt.ini.bak" "${REPO_ROOT}/xrt.ini"
    fi

    if [[ ! -x "${HOST_BIN}" ]]; then
        echo "Building host binary..."
        make host
    fi

    if [[ ! -f "${XCLBIN_PATH}" ]]; then
        echo "Building sw_emu xclbin for platform ${PLATFORM}..."
        make build TARGET=sw_emu PLATFORM="${PLATFORM}"
    fi
}

prepare_input_file() {
    local default_input="${REPO_ROOT}/BinaryData/hash_10mb.bin"
    read -r -p "Input binary file path [${default_input}]: " INPUT_FILE
    INPUT_FILE="${INPUT_FILE:-${default_input}}"

    if [[ ! -f "${INPUT_FILE}" ]]; then
        echo "Creating 10MB random test file at ${INPUT_FILE}"
        mkdir -p "$(dirname -- "${INPUT_FILE}")"
        dd if=/dev/urandom of="${INPUT_FILE}" bs=1M count=10 status=none
    fi

    local input_size
    input_size="$(stat -c%s "${INPUT_FILE}")"
    echo "Using input: ${INPUT_FILE} (${input_size} bytes)"
}

print_python_sha256_info() {
    python3 - <<'PY'
import hashlib
from pathlib import Path
import os
p = Path(os.environ["INPUT_FILE"])
b = p.read_bytes()
print(f"Whole-file sha256sum: {hashlib.sha256(b).hexdigest()}")
chunk = 1024
print(f"Chunk(0,1024) sha256: {hashlib.sha256(b[:chunk]).hexdigest()}")
PY
}

run_one_mode() {
    local cfg="$1"
    local log_file
    log_file="$(mktemp)"

    echo
    echo "Running mode: ${cfg}"
    if ! "${HOST_BIN}" \
        -x "${XCLBIN_PATH}" \
        -c "${cfg}" \
        -i "${INPUT_FILE}" \
        -n 1 \
        -l 1 > "${log_file}" 2>&1; then
        echo "Run failed for ${cfg}. Full log:"
        cat "${log_file}"
        rm -f "${log_file}"
        return 1
    fi

    if [[ "${cfg}" == "SHA_256" ]]; then
        print_python_sha256_info
    fi

    echo "Summary (${cfg}):"
    grep -E "Loaded input file|${cfg} FPGA Time|${cfg} Host Time|${cfg} Host:|${cfg} FPGA:" "${log_file}" || true

    if grep -q "\[MISMATCH\]" "${log_file}"; then
        echo "Result: FAIL (${cfg})"
        echo "Mismatches found."
        rm -f "${log_file}"
        return 1
    fi

    if ! grep -q "${cfg} Host:" "${log_file}"; then
        echo "Result: FAIL (${cfg})"
        echo "Digest/checksum lines were not found in output."
        echo "Full log:"
        cat "${log_file}"
        rm -f "${log_file}"
        return 1
    fi

    echo "Result: PASS (${cfg})"
    rm -f "${log_file}"
    return 0
}

choose_modes() {
    echo
    echo "Select mode to run in sw_emu:"
    echo "  1) CRC_32"
    echo "  2) TCP_CHECKSUM"
    echo "  3) SHA_256"
    echo "  4) ALL"
    read -r -p "Choice [1-4]: " choice

    case "${choice}" in
        1) MODES=("CRC_32") ;;
        2) MODES=("TCP_CHECKSUM") ;;
        3) MODES=("SHA_256") ;;
        4) MODES=("CRC_32" "TCP_CHECKSUM" "SHA_256") ;;
        *)
            echo "Invalid choice: ${choice}"
            exit 1
            ;;
    esac
}

main() {
    setup_swemu_env
    ensure_build_artifacts
    choose_modes
    prepare_input_file

    export INPUT_FILE
    local failures=0
    for mode in "${MODES[@]}"; do
        if ! run_one_mode "${mode}"; then
            failures=$((failures + 1))
        fi
    done

    echo
    if (( failures == 0 )); then
        echo "All selected mode(s) passed host-vs-FPGA comparison."
    else
        echo "${failures} mode(s) failed."
        exit 1
    fi
}

main "$@"
