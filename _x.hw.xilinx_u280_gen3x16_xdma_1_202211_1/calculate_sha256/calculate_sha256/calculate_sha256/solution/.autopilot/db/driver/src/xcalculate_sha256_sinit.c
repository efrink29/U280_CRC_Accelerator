// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xcalculate_sha256.h"

extern XCalculate_sha256_Config XCalculate_sha256_ConfigTable[];

XCalculate_sha256_Config *XCalculate_sha256_LookupConfig(u16 DeviceId) {
	XCalculate_sha256_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XCALCULATE_SHA256_NUM_INSTANCES; Index++) {
		if (XCalculate_sha256_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XCalculate_sha256_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XCalculate_sha256_Initialize(XCalculate_sha256 *InstancePtr, u16 DeviceId) {
	XCalculate_sha256_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XCalculate_sha256_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XCalculate_sha256_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

