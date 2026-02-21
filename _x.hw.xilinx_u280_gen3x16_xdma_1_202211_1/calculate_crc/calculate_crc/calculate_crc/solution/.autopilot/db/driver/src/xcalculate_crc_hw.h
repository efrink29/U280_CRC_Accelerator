// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.1 (64-bit)
// Tool Version Limit: 2023.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
// control
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read/COR)
//        bit 4  - ap_continue (Read/Write/SC)
//        bit 7  - auto_restart (Read/Write)
//        bit 9  - interrupt (Read)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0 - enable ap_done interrupt (Read/Write)
//        bit 1 - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0 - ap_done (Read/TOW)
//        bit 1 - ap_ready (Read/TOW)
//        others - reserved
// 0x10 : Data signal of data_in
//        bit 31~0 - data_in[31:0] (Read/Write)
// 0x14 : Data signal of data_in
//        bit 31~0 - data_in[63:32] (Read/Write)
// 0x18 : reserved
// 0x1c : Data signal of crc_out
//        bit 31~0 - crc_out[31:0] (Read/Write)
// 0x20 : Data signal of crc_out
//        bit 31~0 - crc_out[63:32] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of tables
//        bit 31~0 - tables[31:0] (Read/Write)
// 0x2c : Data signal of tables
//        bit 31~0 - tables[63:32] (Read/Write)
// 0x30 : reserved
// 0x34 : Data signal of numChunks
//        bit 31~0 - numChunks[31:0] (Read/Write)
// 0x38 : reserved
// 0x3c : Data signal of chunkSize
//        bit 31~0 - chunkSize[31:0] (Read/Write)
// 0x40 : reserved
// 0x44 : Data signal of crc_size
//        bit 31~0 - crc_size[31:0] (Read/Write)
// 0x48 : reserved
// 0x4c : Data signal of init_value
//        bit 31~0 - init_value[31:0] (Read/Write)
// 0x50 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XCALCULATE_CRC_CONTROL_ADDR_AP_CTRL         0x00
#define XCALCULATE_CRC_CONTROL_ADDR_GIE             0x04
#define XCALCULATE_CRC_CONTROL_ADDR_IER             0x08
#define XCALCULATE_CRC_CONTROL_ADDR_ISR             0x0c
#define XCALCULATE_CRC_CONTROL_ADDR_DATA_IN_DATA    0x10
#define XCALCULATE_CRC_CONTROL_BITS_DATA_IN_DATA    64
#define XCALCULATE_CRC_CONTROL_ADDR_CRC_OUT_DATA    0x1c
#define XCALCULATE_CRC_CONTROL_BITS_CRC_OUT_DATA    64
#define XCALCULATE_CRC_CONTROL_ADDR_TABLES_DATA     0x28
#define XCALCULATE_CRC_CONTROL_BITS_TABLES_DATA     64
#define XCALCULATE_CRC_CONTROL_ADDR_NUMCHUNKS_DATA  0x34
#define XCALCULATE_CRC_CONTROL_BITS_NUMCHUNKS_DATA  32
#define XCALCULATE_CRC_CONTROL_ADDR_CHUNKSIZE_DATA  0x3c
#define XCALCULATE_CRC_CONTROL_BITS_CHUNKSIZE_DATA  32
#define XCALCULATE_CRC_CONTROL_ADDR_CRC_SIZE_DATA   0x44
#define XCALCULATE_CRC_CONTROL_BITS_CRC_SIZE_DATA   32
#define XCALCULATE_CRC_CONTROL_ADDR_INIT_VALUE_DATA 0x4c
#define XCALCULATE_CRC_CONTROL_BITS_INIT_VALUE_DATA 32

