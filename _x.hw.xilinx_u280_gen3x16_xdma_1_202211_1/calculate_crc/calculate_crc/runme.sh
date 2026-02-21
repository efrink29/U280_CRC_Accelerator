#!/bin/sh

# 
# v++(TM)
# runme.sh: a v++-generated Runs Script for UNIX
# Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
# Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
# 

if [ -z "$PATH" ]; then
  PATH=/share/Xilinx/Vitis_HLS/2023.1/bin:/share/Xilinx/Vitis/2023.1/bin:/share/Xilinx/Vitis/2023.1/bin
else
  PATH=/share/Xilinx/Vitis_HLS/2023.1/bin:/share/Xilinx/Vitis/2023.1/bin:/share/Xilinx/Vitis/2023.1/bin:$PATH
fi
export PATH

if [ -z "$LD_LIBRARY_PATH" ]; then
  LD_LIBRARY_PATH=
else
  LD_LIBRARY_PATH=:$LD_LIBRARY_PATH
fi
export LD_LIBRARY_PATH

HD_PWD='/users/arashs/U280_CRC_Accelerator/_x.hw.xilinx_u280_gen3x16_xdma_1_202211_1/calculate_crc/calculate_crc'
cd "$HD_PWD"

HD_LOG=runme.log
/bin/touch $HD_LOG

ISEStep="./ISEWrap.sh"
EAStep()
{
     $ISEStep $HD_LOG "$@" >> $HD_LOG 2>&1
     if [ $? -ne 0 ]
     then
         exit
     fi
}

EAStep vitis_hls -f calculate_crc.tcl -messageDb vitis_hls.pb
