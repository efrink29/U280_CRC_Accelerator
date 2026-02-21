# U280 CRC/TCP/SHA Accelerator

This project builds one FPGA binary (`kernel.xclbin`) that contains:

- `calculate_crc` (16 CUs: `CRC_0..CRC_15`)
- `calculate_tcp_checksum` (2 CUs: `TCP_0..TCP_1`)
- `calculate_sha256` (2 CUs: `SHA_0..SHA_1`)

The host app is `host_xrt` and supports:

- `CRC_32`
- `TCP_CHECKSUM`
- `SHA_256`

## 1) Tool Setup (every new shell)

```bash
cd ~/U280_CRC_Accelerator

if [ -f /share/Xilinx/Vitis/2023.1/settings64.sh ]; then
  source /share/Xilinx/Vitis/2023.1/settings64.sh
elif [ -f /tools/Xilinx/Vitis/2023.1/settings64.sh ]; then
  source /tools/Xilinx/Vitis/2023.1/settings64.sh
else
  echo "Vitis settings64.sh not found" && exit 1
fi

source /opt/xilinx/xrt/setup.sh
```

Platform used in this repo:

- `xilinx_u280_gen3x16_xdma_1_202211_1`

## 2) Build Host

```bash
make -B host PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
```

## 3) Run Software Emulation (`sw_emu`)

### Option A: helper script (recommended)

```bash
bash plans/run_swemu_compare.sh
```

### Option B: manual commands

```bash
make -B host PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
make -B build TARGET=sw_emu PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
make emconfig TARGET=sw_emu PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
cp _x.sw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/emconfig.json .

export XCL_EMULATION_MODE=sw_emu

./host_xrt -x build_dir.sw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin -c CRC_32 -i BinaryData/hash_10mb.bin -n 1 -l 1
./host_xrt -x build_dir.sw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin -c TCP_CHECKSUM -i BinaryData/hash_10mb.bin -n 1 -l 1
./host_xrt -x build_dir.sw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin -c SHA_256 -i BinaryData/hash_10mb.bin -n 1 -l 1
```

## 4) Run Hardware Emulation (`hw_emu`)

```bash
make -B host PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
make -B build TARGET=hw_emu PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
make emconfig TARGET=hw_emu PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
cp _x.hw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/emconfig.json .

export XCL_EMULATION_MODE=hw_emu
export VITIS_LAUNCH_WAVEFORM_BATCH=1

./host_xrt -x build_dir.hw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin -c CRC_32 -i BinaryData/hash_10mb.bin -n 1 -l 1
./host_xrt -x build_dir.hw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin -c TCP_CHECKSUM -i BinaryData/hash_10mb.bin -n 1 -l 1
./host_xrt -x build_dir.hw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin -c SHA_256 -i BinaryData/hash_10mb.bin -n 1 -l 1
```

## 5) Build Real Hardware (`hw`) for U280

Current working frequency for this checked-in design is `210` MHz.

```bash
make -B host PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
make -B build TARGET=hw PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1 VPP_LDFLAGS='--kernel_frequency 210'
```

If timing fails, retry with lower frequency:

```bash
make -B build TARGET=hw PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1 VPP_LDFLAGS='--kernel_frequency 200'
```

## 6) Run on Real U280

```bash
unset XCL_EMULATION_MODE
xbutil examine
```

Set xclbin path once:

```bash
XCLBIN=build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin
```

Run:

```bash
./host_xrt -x "$XCLBIN" -c CRC_32 -i BinaryData/hash_10mb.bin -n 10 -l 1
./host_xrt -x "$XCLBIN" -c TCP_CHECKSUM -i BinaryData/hash_10mb.bin -n 2 -l 1
./host_xrt -x "$XCLBIN" -c SHA_256 -i BinaryData/hash_10mb.bin -n 2 -l 1
```

`-n` suggestions above match available CUs (`CRC=16`, `TCP=2`, `SHA=2`) and host limits.

## 7) Verify Built xclbin

```bash
xclbinutil --info -i build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin | grep -E "Kernels:|calculate_crc|calculate_tcp_checksum|calculate_sha256|connectivity.nk|kernel_frequency"
```

Expected:

- kernels include all 3 functions
- connectivity shows `calculate_tcp_checksum:2` and `calculate_sha256:2`
- command line contains `--kernel_frequency 210`

## 8) Known Runtime Message (harmless)

You may see:

- `No compute units matching 'calculate_tcp_checksum:{TCP_2}'`
- `No compute units matching 'calculate_sha256:{SHA_2}'`

This is a probe message from CU discovery. It is not a run failure when output checks show `[OK]`.

## 9) tmux for Long HW Builds

```bash
tmux new -s u280_hw_build
cd ~/U280_CRC_Accelerator
# run tool setup commands from section 1
make -B host PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
make -B build TARGET=hw PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1 VPP_LDFLAGS='--kernel_frequency 210' 2>&1 | tee hw_build.log
```

Detach: `Ctrl+b d`  
Re-attach:

```bash
tmux attach -t u280_hw_build
```

## 10) What to Push to GitHub

Push source/config/docs:

- `src/kernel.cpp`
- `src/manager.hpp`
- `src/helpers/crc.cpp`
- `src/host.cpp` (if changed)
- `vpp_connectivity.cfg`
- `makefile_us_alveo.mk`
- `xrt.ini`
- `README.md`
- `plans/run_swemu_compare.sh`

Do not push generated artifacts:

- `_x*`
- `build_dir.*`
- `package.*`
- `host_xrt`
- `*.log`
