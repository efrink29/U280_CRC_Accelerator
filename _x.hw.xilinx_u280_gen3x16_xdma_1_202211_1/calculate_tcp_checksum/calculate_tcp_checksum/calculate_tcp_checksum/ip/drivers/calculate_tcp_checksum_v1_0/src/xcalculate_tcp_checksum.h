// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef XCALCULATE_TCP_CHECKSUM_H
#define XCALCULATE_TCP_CHECKSUM_H

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
#include "xcalculate_tcp_checksum_hw.h"

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
} XCalculate_tcp_checksum_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XCalculate_tcp_checksum;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XCalculate_tcp_checksum_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XCalculate_tcp_checksum_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XCalculate_tcp_checksum_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XCalculate_tcp_checksum_ReadReg(BaseAddress, RegOffset) \
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
int XCalculate_tcp_checksum_Initialize(XCalculate_tcp_checksum *InstancePtr, u16 DeviceId);
XCalculate_tcp_checksum_Config* XCalculate_tcp_checksum_LookupConfig(u16 DeviceId);
int XCalculate_tcp_checksum_CfgInitialize(XCalculate_tcp_checksum *InstancePtr, XCalculate_tcp_checksum_Config *ConfigPtr);
#else
int XCalculate_tcp_checksum_Initialize(XCalculate_tcp_checksum *InstancePtr, const char* InstanceName);
int XCalculate_tcp_checksum_Release(XCalculate_tcp_checksum *InstancePtr);
#endif

void XCalculate_tcp_checksum_Start(XCalculate_tcp_checksum *InstancePtr);
u32 XCalculate_tcp_checksum_IsDone(XCalculate_tcp_checksum *InstancePtr);
u32 XCalculate_tcp_checksum_IsIdle(XCalculate_tcp_checksum *InstancePtr);
u32 XCalculate_tcp_checksum_IsReady(XCalculate_tcp_checksum *InstancePtr);
void XCalculate_tcp_checksum_Continue(XCalculate_tcp_checksum *InstancePtr);
void XCalculate_tcp_checksum_EnableAutoRestart(XCalculate_tcp_checksum *InstancePtr);
void XCalculate_tcp_checksum_DisableAutoRestart(XCalculate_tcp_checksum *InstancePtr);

void XCalculate_tcp_checksum_Set_data_in(XCalculate_tcp_checksum *InstancePtr, u64 Data);
u64 XCalculate_tcp_checksum_Get_data_in(XCalculate_tcp_checksum *InstancePtr);
void XCalculate_tcp_checksum_Set_crc_out(XCalculate_tcp_checksum *InstancePtr, u64 Data);
u64 XCalculate_tcp_checksum_Get_crc_out(XCalculate_tcp_checksum *InstancePtr);
void XCalculate_tcp_checksum_Set_numChunks(XCalculate_tcp_checksum *InstancePtr, u32 Data);
u32 XCalculate_tcp_checksum_Get_numChunks(XCalculate_tcp_checksum *InstancePtr);
void XCalculate_tcp_checksum_Set_chunkSize(XCalculate_tcp_checksum *InstancePtr, u32 Data);
u32 XCalculate_tcp_checksum_Get_chunkSize(XCalculate_tcp_checksum *InstancePtr);

void XCalculate_tcp_checksum_InterruptGlobalEnable(XCalculate_tcp_checksum *InstancePtr);
void XCalculate_tcp_checksum_InterruptGlobalDisable(XCalculate_tcp_checksum *InstancePtr);
void XCalculate_tcp_checksum_InterruptEnable(XCalculate_tcp_checksum *InstancePtr, u32 Mask);
void XCalculate_tcp_checksum_InterruptDisable(XCalculate_tcp_checksum *InstancePtr, u32 Mask);
void XCalculate_tcp_checksum_InterruptClear(XCalculate_tcp_checksum *InstancePtr, u32 Mask);
u32 XCalculate_tcp_checksum_InterruptGetEnabled(XCalculate_tcp_checksum *InstancePtr);
u32 XCalculate_tcp_checksum_InterruptGetStatus(XCalculate_tcp_checksum *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
