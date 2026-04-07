// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef XCALCULATE_SHA256_H
#define XCALCULATE_SHA256_H

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
#include "xcalculate_sha256_hw.h"

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
} XCalculate_sha256_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XCalculate_sha256;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XCalculate_sha256_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XCalculate_sha256_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XCalculate_sha256_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XCalculate_sha256_ReadReg(BaseAddress, RegOffset) \
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
int XCalculate_sha256_Initialize(XCalculate_sha256 *InstancePtr, u16 DeviceId);
XCalculate_sha256_Config* XCalculate_sha256_LookupConfig(u16 DeviceId);
int XCalculate_sha256_CfgInitialize(XCalculate_sha256 *InstancePtr, XCalculate_sha256_Config *ConfigPtr);
#else
int XCalculate_sha256_Initialize(XCalculate_sha256 *InstancePtr, const char* InstanceName);
int XCalculate_sha256_Release(XCalculate_sha256 *InstancePtr);
#endif

void XCalculate_sha256_Start(XCalculate_sha256 *InstancePtr);
u32 XCalculate_sha256_IsDone(XCalculate_sha256 *InstancePtr);
u32 XCalculate_sha256_IsIdle(XCalculate_sha256 *InstancePtr);
u32 XCalculate_sha256_IsReady(XCalculate_sha256 *InstancePtr);
void XCalculate_sha256_Continue(XCalculate_sha256 *InstancePtr);
void XCalculate_sha256_EnableAutoRestart(XCalculate_sha256 *InstancePtr);
void XCalculate_sha256_DisableAutoRestart(XCalculate_sha256 *InstancePtr);

void XCalculate_sha256_Set_data_in(XCalculate_sha256 *InstancePtr, u64 Data);
u64 XCalculate_sha256_Get_data_in(XCalculate_sha256 *InstancePtr);
void XCalculate_sha256_Set_crc_out(XCalculate_sha256 *InstancePtr, u64 Data);
u64 XCalculate_sha256_Get_crc_out(XCalculate_sha256 *InstancePtr);
void XCalculate_sha256_Set_numChunks(XCalculate_sha256 *InstancePtr, u32 Data);
u32 XCalculate_sha256_Get_numChunks(XCalculate_sha256 *InstancePtr);
void XCalculate_sha256_Set_chunkSize(XCalculate_sha256 *InstancePtr, u32 Data);
u32 XCalculate_sha256_Get_chunkSize(XCalculate_sha256 *InstancePtr);

void XCalculate_sha256_InterruptGlobalEnable(XCalculate_sha256 *InstancePtr);
void XCalculate_sha256_InterruptGlobalDisable(XCalculate_sha256 *InstancePtr);
void XCalculate_sha256_InterruptEnable(XCalculate_sha256 *InstancePtr, u32 Mask);
void XCalculate_sha256_InterruptDisable(XCalculate_sha256 *InstancePtr, u32 Mask);
void XCalculate_sha256_InterruptClear(XCalculate_sha256 *InstancePtr, u32 Mask);
u32 XCalculate_sha256_InterruptGetEnabled(XCalculate_sha256 *InstancePtr);
u32 XCalculate_sha256_InterruptGetStatus(XCalculate_sha256 *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
