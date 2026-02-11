Includes:

The includes can be found in the Vitis Examples Repository in the common folder. https://github.com/Xilinx/Vitis_Accel_Examples/tree/main/common

Software Emulation

make all TARGET=sw_emu PLATFORM=/opt/xilinx/platforms/xilinx_u280_gen3x16_xdma_1_202211_1/xilinx_u280_gen3x16_xdma_1_202211_1.xpfm

./host_xrt -x ./build_dir.sw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin

Hardware Emulation

make all TARGET=hw_emu PLATFORM=/opt/xilinx/platforms/xilinx_u280_gen3x16_xdma_1_202211_1/xilinx_u280_gen3x16_xdma_1_202211_1.xpfm

./host_xrt -x ./build_dir.hw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin

Hardware (Run on on CloudLab OCT)

make all TARGET=hw PLATFORM=/opt/xilinx/platforms/xilinx_u280_gen3x16_xdma_1_202211_1/xilinx_u280_gen3x16_xdma_1_202211_1.xpfm

./host_xrt -x ./build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin

TCP Checksum Validation

This kernel now supports dual mode:
- CRC mode (`CHECK_MODE_CRC`)
- TCP/Internet 16-bit checksum mode (`CHECK_MODE_TCP_CHECKSUM`)

Host validation uses:
- CRC: `parallel_compute(...)` from `src/helpers/crc.cpp`
- TCP checksum: same logic as MLED `CheckMethodChecksum::calculate` from `mledcpp/core/check_method.cpp`

Use the TCP checksum config:

./host_xrt -x ./build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin -c TCP_CHECKSUM

Use CRC config (example):

./host_xrt -x ./build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/kernel.xclbin -c CRC_32

Optional C-sim style validation (kernel function vs MLED-equivalent reference):

g++ -std=c++17 -I/tools/Xilinx/Vitis_HLS/2023.1/include -Isrc src/tcp_checksum_validation.cpp src/helpers/crc.cpp src/kernel.cpp -o tcp_checksum_validation
./tcp_checksum_validation

Command to compile just Host Binary:

g++ -o host_xrt /home/efrink/Documents/Examples/Kernel_Tools/parallel_crc/common/includes/cmdparser/cmdlineparser.cpp /home/efrink/Documents/Examples/Kernel_Tools/parallel_crc/common/includes/xcl2/xcl2.cpp /home/efrink/Documents/Examples/Kernel_Tools/parallel_crc/common/includes/logger/logger.cpp src/host.cpp src/manager.hpp -I/opt/xilinx/xrt/include -I/tools/Xilinx/Vivado/2023.1/include -Wall -O0 -g -std=c++17 -I/home/efrink/Documents/Examples/Kernel_Tools/parallel_crc/common/includes/cmdparser -I/home/efrink/Documents/Examples/Kernel_Tools/parallel_crc/common/includes/logger -I/home/efrink/Documents/Examples/Kernel_Tools/parallel_crc/common/includes/xcl2 -fmessage-length=0 -L/opt/xilinx/xrt/lib -pthread -lOpenCL -lrt -lstdc++  -luuid -lxrt_coreutil
