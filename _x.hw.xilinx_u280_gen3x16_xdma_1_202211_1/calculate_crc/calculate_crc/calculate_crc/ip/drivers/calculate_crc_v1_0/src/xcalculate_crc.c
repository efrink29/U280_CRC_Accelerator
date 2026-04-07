// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
/***************************** Include Files *********************************/
#include "xcalculate_crc.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XCalculate_crc_CfgInitialize(XCalculate_crc *InstancePtr, XCalculate_crc_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XCalculate_crc_Start(XCalculate_crc *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_AP_CTRL) & 0x80;
    XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XCalculate_crc_IsDone(XCalculate_crc *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XCalculate_crc_IsIdle(XCalculate_crc *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XCalculate_crc_IsReady(XCalculate_crc *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XCalculate_crc_Continue(XCalculate_crc *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_AP_CTRL) & 0x80;
    XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_AP_CTRL, Data | 0x10);
}

void XCalculate_crc_EnableAutoRestart(XCalculate_crc *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XCalculate_crc_DisableAutoRestart(XCalculate_crc *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_AP_CTRL, 0);
}

void XCalculate_crc_Set_data_in(XCalculate_crc *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_DATA_IN_DATA, (u32)(Data));
    XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_DATA_IN_DATA + 4, (u32)(Data >> 32));
}

u64 XCalculate_crc_Get_data_in(XCalculate_crc *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_DATA_IN_DATA);
    Data += (u64)XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_DATA_IN_DATA + 4) << 32;
    return Data;
}

void XCalculate_crc_Set_crc_out(XCalculate_crc *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_CRC_OUT_DATA, (u32)(Data));
    XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_CRC_OUT_DATA + 4, (u32)(Data >> 32));
}

u64 XCalculate_crc_Get_crc_out(XCalculate_crc *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_CRC_OUT_DATA);
    Data += (u64)XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_CRC_OUT_DATA + 4) << 32;
    return Data;
}

void XCalculate_crc_Set_tables(XCalculate_crc *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_TABLES_DATA, (u32)(Data));
    XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_TABLES_DATA + 4, (u32)(Data >> 32));
}

u64 XCalculate_crc_Get_tables(XCalculate_crc *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_TABLES_DATA);
    Data += (u64)XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_TABLES_DATA + 4) << 32;
    return Data;
}

void XCalculate_crc_Set_numChunks(XCalculate_crc *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_NUMCHUNKS_DATA, Data);
}

u32 XCalculate_crc_Get_numChunks(XCalculate_crc *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_NUMCHUNKS_DATA);
    return Data;
}

void XCalculate_crc_Set_chunkSize(XCalculate_crc *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_CHUNKSIZE_DATA, Data);
}

u32 XCalculate_crc_Get_chunkSize(XCalculate_crc *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_CHUNKSIZE_DATA);
    return Data;
}

void XCalculate_crc_Set_crc_size(XCalculate_crc *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_CRC_SIZE_DATA, Data);
}

u32 XCalculate_crc_Get_crc_size(XCalculate_crc *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_CRC_SIZE_DATA);
    return Data;
}

void XCalculate_crc_Set_init_value(XCalculate_crc *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_INIT_VALUE_DATA, Data);
}

u32 XCalculate_crc_Get_init_value(XCalculate_crc *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_INIT_VALUE_DATA);
    return Data;
}

void XCalculate_crc_InterruptGlobalEnable(XCalculate_crc *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_GIE, 1);
}

void XCalculate_crc_InterruptGlobalDisable(XCalculate_crc *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_GIE, 0);
}

void XCalculate_crc_InterruptEnable(XCalculate_crc *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_IER);
    XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_IER, Register | Mask);
}

void XCalculate_crc_InterruptDisable(XCalculate_crc *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_IER);
    XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_IER, Register & (~Mask));
}

void XCalculate_crc_InterruptClear(XCalculate_crc *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    //XCalculate_crc_WriteReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_ISR, Mask);
}

u32 XCalculate_crc_InterruptGetEnabled(XCalculate_crc *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_IER);
}

u32 XCalculate_crc_InterruptGetStatus(XCalculate_crc *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    // Current Interrupt Clear Behavior is Clear on Read(COR).
    return XCalculate_crc_ReadReg(InstancePtr->Control_BaseAddress, XCALCULATE_CRC_CONTROL_ADDR_ISR);
}

