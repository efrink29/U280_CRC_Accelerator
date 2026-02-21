// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.1 (64-bit)
// Tool Version Limit: 2023.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xcalculate_sha256.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XCalculate_sha256_CfgInitialize(XCalculate_sha256 *InstancePtr, XCalculate_sha256_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XCalculate_sha256_Start(XCalculate_sha256 *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_sha256_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_AP_CTRL) & 0x80;
    XCalculate_sha256_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XCalculate_sha256_IsDone(XCalculate_sha256 *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_sha256_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XCalculate_sha256_IsIdle(XCalculate_sha256 *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_sha256_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XCalculate_sha256_IsReady(XCalculate_sha256 *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_sha256_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XCalculate_sha256_Continue(XCalculate_sha256 *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_sha256_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_AP_CTRL) & 0x80;
    XCalculate_sha256_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_AP_CTRL, Data | 0x10);
}

void XCalculate_sha256_EnableAutoRestart(XCalculate_sha256 *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_sha256_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XCalculate_sha256_DisableAutoRestart(XCalculate_sha256 *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_sha256_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_AP_CTRL, 0);
}

void XCalculate_sha256_Set_data_in(XCalculate_sha256 *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_sha256_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_DATA_IN_DATA, (u32)(Data));
    XCalculate_sha256_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_DATA_IN_DATA + 4, (u32)(Data >> 32));
}

u64 XCalculate_sha256_Get_data_in(XCalculate_sha256 *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_sha256_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_DATA_IN_DATA);
    Data += (u64)XCalculate_sha256_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_DATA_IN_DATA + 4) << 32;
    return Data;
}

void XCalculate_sha256_Set_crc_out(XCalculate_sha256 *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_sha256_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_CRC_OUT_DATA, (u32)(Data));
    XCalculate_sha256_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_CRC_OUT_DATA + 4, (u32)(Data >> 32));
}

u64 XCalculate_sha256_Get_crc_out(XCalculate_sha256 *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_sha256_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_CRC_OUT_DATA);
    Data += (u64)XCalculate_sha256_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_CRC_OUT_DATA + 4) << 32;
    return Data;
}

void XCalculate_sha256_Set_numChunks(XCalculate_sha256 *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_sha256_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_NUMCHUNKS_DATA, Data);
}

u32 XCalculate_sha256_Get_numChunks(XCalculate_sha256 *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_sha256_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_NUMCHUNKS_DATA);
    return Data;
}

void XCalculate_sha256_Set_chunkSize(XCalculate_sha256 *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_sha256_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_CHUNKSIZE_DATA, Data);
}

u32 XCalculate_sha256_Get_chunkSize(XCalculate_sha256 *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_sha256_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_CHUNKSIZE_DATA);
    return Data;
}

void XCalculate_sha256_InterruptGlobalEnable(XCalculate_sha256 *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_sha256_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_GIE, 1);
}

void XCalculate_sha256_InterruptGlobalDisable(XCalculate_sha256 *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_sha256_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_GIE, 0);
}

void XCalculate_sha256_InterruptEnable(XCalculate_sha256 *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XCalculate_sha256_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_IER);
    XCalculate_sha256_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_IER, Register | Mask);
}

void XCalculate_sha256_InterruptDisable(XCalculate_sha256 *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XCalculate_sha256_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_IER);
    XCalculate_sha256_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_IER, Register & (~Mask));
}

void XCalculate_sha256_InterruptClear(XCalculate_sha256 *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_sha256_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_ISR, Mask);
}

u32 XCalculate_sha256_InterruptGetEnabled(XCalculate_sha256 *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XCalculate_sha256_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_IER);
}

u32 XCalculate_sha256_InterruptGetStatus(XCalculate_sha256 *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XCalculate_sha256_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_SHA256_CONTROL_ADDR_ISR);
}

