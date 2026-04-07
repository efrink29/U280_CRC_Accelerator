// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xcalculate_tcp_checksum.h"

extern XCalculate_tcp_checksum_Config XCalculate_tcp_checksum_ConfigTable[];

XCalculate_tcp_checksum_Config *XCalculate_tcp_checksum_LookupConfig(u16 DeviceId) {
	XCalculate_tcp_checksum_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XCALCULATE_TCP_CHECKSUM_NUM_INSTANCES; Index++) {
		if (XCalculate_tcp_checksum_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XCalculate_tcp_checksum_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XCalculate_tcp_checksum_Initialize(XCalculate_tcp_checksum *InstancePtr, u16 DeviceId) {
	XCalculate_tcp_checksum_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XCalculate_tcp_checksum_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XCalculate_tcp_checksum_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

