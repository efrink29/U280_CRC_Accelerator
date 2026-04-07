// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
/***************************** Include Files *********************************/
#include "xcalculate_tcp_checksum.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XCalculate_tcp_checksum_CfgInitialize(XCalculate_tcp_checksum *InstancePtr, XCalculate_tcp_checksum_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XCalculate_tcp_checksum_Start(XCalculate_tcp_checksum *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_tcp_checksum_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_AP_CTRL) & 0x80;
    XCalculate_tcp_checksum_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XCalculate_tcp_checksum_IsDone(XCalculate_tcp_checksum *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_tcp_checksum_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XCalculate_tcp_checksum_IsIdle(XCalculate_tcp_checksum *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_tcp_checksum_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XCalculate_tcp_checksum_IsReady(XCalculate_tcp_checksum *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_tcp_checksum_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XCalculate_tcp_checksum_Continue(XCalculate_tcp_checksum *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_tcp_checksum_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_AP_CTRL) & 0x80;
    XCalculate_tcp_checksum_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_AP_CTRL, Data | 0x10);
}

void XCalculate_tcp_checksum_EnableAutoRestart(XCalculate_tcp_checksum *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_tcp_checksum_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XCalculate_tcp_checksum_DisableAutoRestart(XCalculate_tcp_checksum *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_tcp_checksum_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_AP_CTRL, 0);
}

void XCalculate_tcp_checksum_Set_data_in(XCalculate_tcp_checksum *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_tcp_checksum_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_DATA_IN_DATA, (u32)(Data));
    XCalculate_tcp_checksum_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_DATA_IN_DATA + 4, (u32)(Data >> 32));
}

u64 XCalculate_tcp_checksum_Get_data_in(XCalculate_tcp_checksum *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_tcp_checksum_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_DATA_IN_DATA);
    Data += (u64)XCalculate_tcp_checksum_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_DATA_IN_DATA + 4) << 32;
    return Data;
}

void XCalculate_tcp_checksum_Set_crc_out(XCalculate_tcp_checksum *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_tcp_checksum_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_CRC_OUT_DATA, (u32)(Data));
    XCalculate_tcp_checksum_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_CRC_OUT_DATA + 4, (u32)(Data >> 32));
}

u64 XCalculate_tcp_checksum_Get_crc_out(XCalculate_tcp_checksum *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_tcp_checksum_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_CRC_OUT_DATA);
    Data += (u64)XCalculate_tcp_checksum_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_CRC_OUT_DATA + 4) << 32;
    return Data;
}

void XCalculate_tcp_checksum_Set_numChunks(XCalculate_tcp_checksum *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_tcp_checksum_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_NUMCHUNKS_DATA, Data);
}

u32 XCalculate_tcp_checksum_Get_numChunks(XCalculate_tcp_checksum *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_tcp_checksum_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_NUMCHUNKS_DATA);
    return Data;
}

void XCalculate_tcp_checksum_Set_chunkSize(XCalculate_tcp_checksum *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_tcp_checksum_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_CHUNKSIZE_DATA, Data);
}

u32 XCalculate_tcp_checksum_Get_chunkSize(XCalculate_tcp_checksum *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_tcp_checksum_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_CHUNKSIZE_DATA);
    return Data;
}

void XCalculate_tcp_checksum_InterruptGlobalEnable(XCalculate_tcp_checksum *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_tcp_checksum_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_GIE, 1);
}

void XCalculate_tcp_checksum_InterruptGlobalDisable(XCalculate_tcp_checksum *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_tcp_checksum_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_GIE, 0);
}

void XCalculate_tcp_checksum_InterruptEnable(XCalculate_tcp_checksum *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XCalculate_tcp_checksum_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_IER);
    XCalculate_tcp_checksum_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_IER, Register | Mask);
}

void XCalculate_tcp_checksum_InterruptDisable(XCalculate_tcp_checksum *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XCalculate_tcp_checksum_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_IER);
    XCalculate_tcp_checksum_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_IER, Register & (~Mask));
}

void XCalculate_tcp_checksum_InterruptClear(XCalculate_tcp_checksum *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    //XCalculate_tcp_checksum_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_ISR, Mask);
}

u32 XCalculate_tcp_checksum_InterruptGetEnabled(XCalculate_tcp_checksum *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XCalculate_tcp_checksum_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_IER);
}

u32 XCalculate_tcp_checksum_InterruptGetStatus(XCalculate_tcp_checksum *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    // Current Interrupt Clear Behavior is Clear on Read(COR).
    return XCalculate_tcp_checksum_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_TCP_CHECKSUM_CONTROL_ADDR_ISR);
}

