// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xcalculate_crc.h"

extern XCalculate_crc_Config XCalculate_crc_ConfigTable[];

XCalculate_crc_Config *XCalculate_crc_LookupConfig(u16 DeviceId) {
	XCalculate_crc_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XCALCULATE_CRC_NUM_INSTANCES; Index++) {
		if (XCalculate_crc_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XCalculate_crc_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XCalculate_crc_Initialize(XCalculate_crc *InstancePtr, u16 DeviceId) {
	XCalculate_crc_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XCalculate_crc_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XCalculate_crc_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

