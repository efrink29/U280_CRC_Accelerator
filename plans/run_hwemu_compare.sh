#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PLATFORM_DEFAULT="xilinx_u280_gen3x16_xdma_1_202211_1"
PLATFORM="${PLATFORM:-${PLATFORM_DEFAULT}}"
EMU_TARGET="hw_emu"
XCLBIN_PATH="${REPO_ROOT}/build_dir.${EMU_TARGET}.${PLATFORM}/kernel.xclbin"
HOST_BIN="${REPO_ROOT}/host_xrt"
EMCONFIG_SRC="${REPO_ROOT}/_x.${EMU_TARGET}.${PLATFORM}/emconfig.json"
EMCONFIG_DST="${REPO_ROOT}/emconfig.json"
TOOL_VITIS_VERSION=""
TOOL_XRT_BRANCH=""
HWEMU_FORCE_REBUILD="${HWEMU_FORCE_REBUILD:-0}"
HWEMU_DISABLE_DEBUG_BUILD="${HWEMU_DISABLE_DEBUG_BUILD:-1}"

resolve_vitis_settings() {
    local candidate=""
    for candidate in \
        /share/Xilinx/Vitis/2023.1/settings64.sh \
        /tools/Xilinx/Vitis/2023.1/settings64.sh; do
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

detect_tool_versions() {
    TOOL_VITIS_VERSION="$(v++ --version 2>/dev/null | sed -n 's/.*v++ v\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)"
    TOOL_XRT_BRANCH="$(xbutil --version 2>/dev/null | awk -F: '/Branch/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')"
}

check_toolchain_compatibility() {
    detect_tool_versions
    if [[ -z "${TOOL_VITIS_VERSION}" || -z "${TOOL_XRT_BRANCH}" ]]; then
        echo "Warning: unable to detect Vitis/XRT versions. Continuing without compatibility check."
        return 0
    fi

    if [[ "${TOOL_VITIS_VERSION}" != "${TOOL_XRT_BRANCH}" ]]; then
        echo "ERROR: Toolchain version mismatch detected:"
        echo "  Vitis (v++): ${TOOL_VITIS_VERSION}"
        echo "  XRT (xbutil): ${TOOL_XRT_BRANCH}"
        echo "hw_emu requires matching versions (for example, both 2023.1)."
        echo "Set SKIP_TOOLCHAIN_CHECK=1 to bypass this guard."
        if [[ "${SKIP_TOOLCHAIN_CHECK:-0}" != "1" ]]; then
            exit 1
        fi
    fi
}

setup_hwemu_env() {
    local vitis_settings=""
    if ! vitis_settings="$(resolve_vitis_settings)"; then
        echo "Missing Vitis settings64.sh."
        echo "Checked: /share/Xilinx/Vitis/2023.1/settings64.sh and /tools/Xilinx/Vitis/2023.1/settings64.sh"
        exit 1
    fi
    if [[ ! -f /opt/xilinx/xrt/setup.sh ]]; then
        echo "Missing /opt/xilinx/xrt/setup.sh"
        exit 1
    fi

    # shellcheck disable=SC1091
    source "${vitis_settings}" >/dev/null 2>&1
    # shellcheck disable=SC1091
    source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

    export XCL_EMULATION_MODE="${EMU_TARGET}"
    # Ensure hw_emu xsim runs headless and exits without waiting for a GUI session.
    export VITIS_LAUNCH_WAVEFORM_BATCH="${VITIS_LAUNCH_WAVEFORM_BATCH:-1}"
    unset VITIS_LAUNCH_WAVEFORM_GUI

    local runtime_ini="${REPO_ROOT}/plans/.xrt_runtime.ini"
    cat > "${runtime_ini}" <<'EOF'
[Runtime]
verbosity=0
EOF
    export XRT_INI_PATH="${runtime_ini}"
    check_toolchain_compatibility
}

xclbin_has_debug_kernel() {
    local xclbin="$1"
    local meta_xml=""

    if [[ ! -f "${xclbin}" ]]; then
        return 1
    fi
    if ! command -v xclbinutil >/dev/null 2>&1; then
        return 1
    fi

    meta_xml="$(mktemp)"
    if ! xclbinutil --quiet --force \
        --input "${xclbin}" \
        --dump-section "EMBEDDED_METADATA:RAW:${meta_xml}" >/dev/null 2>&1; then
        rm -f "${meta_xml}"
        return 1
    fi

    if grep -q 'debug="true"' "${meta_xml}"; then
        rm -f "${meta_xml}"
        return 0
    fi

    rm -f "${meta_xml}"
    return 1
}

build_hwemu_xclbin() {
    local make_args=(
        build
        TARGET="${EMU_TARGET}"
        PLATFORM="${PLATFORM}"
    )

    if [[ "${HWEMU_DISABLE_DEBUG_BUILD}" == "1" ]]; then
        make_args+=(VPP_FLAGS=)
        echo "Building ${EMU_TARGET} xclbin without debug waveform flags (VPP_FLAGS=)."
    else
        echo "Building ${EMU_TARGET} xclbin with Makefile defaults."
    fi

    make "${make_args[@]}"
}

ensure_build_artifacts() {
    if [[ ! -f "${REPO_ROOT}/xrt.ini" && -f "${REPO_ROOT}/xrt.ini.bak" ]]; then
        cp "${REPO_ROOT}/xrt.ini.bak" "${REPO_ROOT}/xrt.ini"
    fi

    if [[ ! -x "${HOST_BIN}" ]]; then
        echo "Building host binary..."
        make host PLATFORM="${PLATFORM}"
    fi

    if [[ -f "${XCLBIN_PATH}" ]] && [[ "${HWEMU_DISABLE_DEBUG_BUILD}" == "1" ]] && xclbin_has_debug_kernel "${XCLBIN_PATH}"; then
        echo "Existing ${EMU_TARGET} xclbin is debug-enabled; forcing rebuild without debug."
        HWEMU_FORCE_REBUILD=1
    fi

    if [[ "${HWEMU_FORCE_REBUILD}" == "1" ]]; then
        echo "Forcing clean ${EMU_TARGET} rebuild."
        rm -rf \
            "${REPO_ROOT}/_x.${EMU_TARGET}.${PLATFORM}" \
            "${REPO_ROOT}/build_dir.${EMU_TARGET}.${PLATFORM}" \
            "${REPO_ROOT}/package.${EMU_TARGET}"
    fi

    if [[ ! -f "${XCLBIN_PATH}" ]]; then
        echo "Building ${EMU_TARGET} xclbin for platform ${PLATFORM}..."
        build_hwemu_xclbin
    fi

    if [[ ! -f "${EMCONFIG_SRC}" ]]; then
        echo "Generating emconfig.json for ${EMU_TARGET}..."
        make emconfig TARGET="${EMU_TARGET}" PLATFORM="${PLATFORM}"
    fi

    if [[ -f "${EMCONFIG_SRC}" ]]; then
        cp "${EMCONFIG_SRC}" "${EMCONFIG_DST}"
    fi
}

prepare_input_file() {
    local default_input="${REPO_ROOT}/BinaryData/hash_64kb.bin"
    local default_input_bytes="${HWEMU_DEFAULT_INPUT_BYTES:-65536}"
    read -r -p "Input binary file path [${default_input}]: " INPUT_FILE
    INPUT_FILE="${INPUT_FILE:-${default_input}}"

    if [[ ! -f "${INPUT_FILE}" ]]; then
        echo "Creating ${default_input_bytes}-byte random test file at ${INPUT_FILE}"
        mkdir -p "$(dirname -- "${INPUT_FILE}")"
        dd if=/dev/urandom of="${INPUT_FILE}" bs="${default_input_bytes}" count=1 status=none
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

        if grep -q "libprotobuf ERROR" "${log_file}"; then
            echo "Detected hw_emu runtime protocol error (protobuf)."
            echo "This is commonly caused by simulator crashes (for example from debug-waveform hw_emu builds)"
            echo "or by a Vitis/XRT version mismatch."
            echo "Detected versions: Vitis=${TOOL_VITIS_VERSION:-unknown}, XRT=${TOOL_XRT_BRANCH:-unknown}"
            rm -f "${log_file}"
            return 2
        fi

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
    echo "Select mode to run in ${EMU_TARGET}:"
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
    setup_hwemu_env
    ensure_build_artifacts
    choose_modes
    prepare_input_file

    export INPUT_FILE
    local failures=0
    local mode_rc=0
    for mode in "${MODES[@]}"; do
        mode_rc=0
        run_one_mode "${mode}" || mode_rc=$?
        if (( mode_rc != 0 )); then
            failures=$((failures + 1))
            if (( mode_rc == 2 )); then
                echo "Fatal runtime/toolchain issue detected; stopping remaining modes."
                exit 1
            fi
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
