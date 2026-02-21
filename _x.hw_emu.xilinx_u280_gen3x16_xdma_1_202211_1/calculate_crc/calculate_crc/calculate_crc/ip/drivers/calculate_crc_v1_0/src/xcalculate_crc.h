// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.1 (64-bit)
// Tool Version Limit: 2023.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XCALCULATE_CRC_H
#define XCALCULATE_CRC_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xcalculate_crc_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
    u16 DeviceId;
    u64 Control_BaseAddress;
} XCalculate_crc_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XCalculate_crc;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XCalculate_crc_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XCalculate_crc_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XCalculate_crc_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XCalculate_crc_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
int XCalculate_crc_Initialize(XCalculate_crc *InstancePtr, u16 DeviceId);
XCalculate_crc_Config* XCalculate_crc_LookupConfig(u16 DeviceId);
int XCalculate_crc_CfgInitialize(XCalculate_crc *InstancePtr, XCalculate_crc_Config *ConfigPtr);
#else
int XCalculate_crc_Initialize(XCalculate_crc *InstancePtr, const char* InstanceName);
int XCalculate_crc_Release(XCalculate_crc *InstancePtr);
#endif

void XCalculate_crc_Start(XCalculate_crc *InstancePtr);
u32 XCalculate_crc_IsDone(XCalculate_crc *InstancePtr);
u32 XCalculate_crc_IsIdle(XCalculate_crc *InstancePtr);
u32 XCalculate_crc_IsReady(XCalculate_crc *InstancePtr);
void XCalculate_crc_Continue(XCalculate_crc *InstancePtr);
void XCalculate_crc_EnableAutoRestart(XCalculate_crc *InstancePtr);
void XCalculate_crc_DisableAutoRestart(XCalculate_crc *InstancePtr);

void XCalculate_crc_Set_data_in(XCalculate_crc *InstancePtr, u64 Data);
u64 XCalculate_crc_Get_data_in(XCalculate_crc *InstancePtr);
void XCalculate_crc_Set_crc_out(XCalculate_crc *InstancePtr, u64 Data);
u64 XCalculate_crc_Get_crc_out(XCalculate_crc *InstancePtr);
void XCalculate_crc_Set_tables(XCalculate_crc *InstancePtr, u64 Data);
u64 XCalculate_crc_Get_tables(XCalculate_crc *InstancePtr);
void XCalculate_crc_Set_numChunks(XCalculate_crc *InstancePtr, u32 Data);
u32 XCalculate_crc_Get_numChunks(XCalculate_crc *InstancePtr);
void XCalculate_crc_Set_chunkSize(XCalculate_crc *InstancePtr, u32 Data);
u32 XCalculate_crc_Get_chunkSize(XCalculate_crc *InstancePtr);
void XCalculate_crc_Set_crc_size(XCalculate_crc *InstancePtr, u32 Data);
u32 XCalculate_crc_Get_crc_size(XCalculate_crc *InstancePtr);
void XCalculate_crc_Set_init_value(XCalculate_crc *InstancePtr, u32 Data);
u32 XCalculate_crc_Get_init_value(XCalculate_crc *InstancePtr);

void XCalculate_crc_InterruptGlobalEnable(XCalculate_crc *InstancePtr);
void XCalculate_crc_InterruptGlobalDisable(XCalculate_crc *InstancePtr);
void XCalculate_crc_InterruptEnable(XCalculate_crc *InstancePtr, u32 Mask);
void XCalculate_crc_InterruptDisable(XCalculate_crc *InstancePtr, u32 Mask);
void XCalculate_crc_InterruptClear(XCalculate_crc *InstancePtr, u32 Mask);
u32 XCalculate_crc_InterruptGetEnabled(XCalculate_crc *InstancePtr);
u32 XCalculate_crc_InterruptGetStatus(XCalculate_crc *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
