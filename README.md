# U280 CRC/TCP/SHA Accelerator

This project builds one FPGA binary (`kernel.xclbin`) that contains:

- `calculate_crc` (16 CUs: `CRC_0..CRC_15`)
- `calculate_tcp_checksum` (1 CU: `TCP_0`)
- `calculate_sha256` (1 CU: `SHA_0`)

The host app is `host_xrt` and dispatches by config mode:

- `CRC_32` (CRC path)
- `TCP_CHECKSUM` (TCP checksum path)
- `SHA_256` (SHA-256 path)

## Prerequisites

- Xilinx Vitis 2023.1
- XRT installed
- U280 platform: `xilinx_u280_gen3x16_xdma_1_202211_1`

Set environment for every new shell:

```bash
cd ~/U280_CRC_Accelerator
source /share/Xilinx/Vitis/2023.1/settings64.sh
source /opt/xilinx/xrt/setup.sh
```

## Build Host

```bash
cd ~/U280_CRC_Accelerator
make -B host PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
```

## Build Software Emulation (`sw_emu`)

```bash
cd ~/U280_CRC_Accelerator
source /share/Xilinx/Vitis/2023.1/settings64.sh
source /opt/xilinx/xrt/setup.sh

make -B host PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
make -B build TARGET=sw_emu PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
make emconfig TARGET=sw_emu PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
cp _x.sw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/emconfig.json .
```

Run tests:

```bash
export XCL_EMULATION_MODE=sw_emu
./host_xrt -x build_dir.sw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin -c CRC_32 -i BinaryData/hash_10mb.bin -n 1 -l 1
./host_xrt -x build_dir.sw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin -c TCP_CHECKSUM -i BinaryData/hash_10mb.bin -n 1 -l 1
./host_xrt -x build_dir.sw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin -c SHA_256 -i BinaryData/hash_10mb.bin -n 1 -l 1
```

## Build Hardware Emulation (`hw_emu`)

```bash
cd ~/U280_CRC_Accelerator
source /share/Xilinx/Vitis/2023.1/settings64.sh
source /opt/xilinx/xrt/setup.sh

make -B host PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
make -B build TARGET=hw_emu PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
make emconfig TARGET=hw_emu PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
cp _x.hw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/emconfig.json .
```

Run tests:

```bash
export XCL_EMULATION_MODE=hw_emu
export VITIS_LAUNCH_WAVEFORM_BATCH=1
./host_xrt -x build_dir.hw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin -c CRC_32 -i BinaryData/hash_10mb.bin -n 1 -l 1
./host_xrt -x build_dir.hw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin -c TCP_CHECKSUM -i BinaryData/hash_10mb.bin -n 1 -l 1
./host_xrt -x build_dir.hw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin -c SHA_256 -i BinaryData/hash_10mb.bin -n 1 -l 1
```

## Build Real Hardware (`hw`) for U280

Use 220 MHz link frequency (timing closed on this design):

```bash
cd ~/U280_CRC_Accelerator
source /share/Xilinx/Vitis/2023.1/settings64.sh
source /opt/xilinx/xrt/setup.sh

make -B host PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
make -B build TARGET=hw PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1 VPP_LDFLAGS="--kernel_frequency 220"
```

Quick kernel check:

```bash
xclbinutil --info -i build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin | grep -E "Kernels:|calculate_crc|calculate_tcp_checksum|calculate_sha256|Signature:"
```

Run on card:

```bash
unset XCL_EMULATION_MODE
xbutil examine

./host_xrt -x build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin -c CRC_32 -i BinaryData/hash_10mb.bin -n 1 -l 1
./host_xrt -x build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin -c TCP_CHECKSUM -i BinaryData/hash_10mb.bin -n 1 -l 1
./host_xrt -x build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin -c SHA_256 -i BinaryData/hash_10mb.bin -n 1 -l 1
```

## Recommended tmux Command for Long HW Builds

```bash
tmux new -s u280_hw_build
cd ~/U280_CRC_Accelerator
source /share/Xilinx/Vitis/2023.1/settings64.sh
source /opt/xilinx/xrt/setup.sh
make -B host PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
make -B build TARGET=hw PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1 VPP_LDFLAGS="--kernel_frequency 220" 2>&1 | tee hw_build.log
```

Detach with `Ctrl+b d`, re-attach with:

```bash
tmux attach -t u280_hw_build
```

## Important: Host/XCLBIN Must Match

If you see `bad kernel argument index ...`, your `host_xrt` and `kernel.xclbin` are from different revisions.

Fix:

```bash
make -B host PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
make -B build TARGET=hw PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1 VPP_LDFLAGS="--kernel_frequency 220"
```

Then run `host_xrt` with that rebuilt `kernel.xclbin`.

## What to Push to GitHub

Push source/config changes:

- `src/kernel.cpp`
- `src/manager.hpp`
- `src/host.cpp` (if changed)
- `makefile_us_alveo.mk`
- `vpp_connectivity.cfg`
- `xrt.ini`
- `Configs/*` (if changed)
- `README.md`

Do not push generated build outputs by default:

- `build_dir.*`
- `_x*`
- `package.*`
- `host_xrt`
- `*.log`

Optional:

- If another machine needs the exact hardware image immediately, you can copy `build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin` manually or publish it as a release artifact.
