set moduleName crc_dataflow_region
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 1
set pipeline_type dataflow
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set C_modelName {crc_dataflow_region}
set C_modelType { void 0 }
set C_modelArgList {
	{ gmem0 int 128 regular {axi_master 2}  }
	{ data_in int 64 regular  }
	{ crc_out int 64 regular  }
	{ crcTables_0 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_1 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_2 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_3 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_4 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_5 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_6 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_7 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_8 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_9 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_10 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_11 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_12 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_13 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_14 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_15 int 32 regular {array 256 { 1 } 1 1 }  }
	{ numChunks int 32 regular  }
	{ chunkSize int 32 regular  }
	{ crc_size int 32 regular  }
	{ init_value int 32 regular  }
}
set C_modelArgMapList {[ 
	{ "Name" : "gmem0", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READWRITE", "bitSlice":[ {"cElement": [{"cName": "data_in","offset": { "type": "dynamic","port_name": "data_in","bundle": "control"},"direction": "READONLY"},{"cName": "crc_out","offset": { "type": "dynamic","port_name": "crc_out","bundle": "control"},"direction": "WRITEONLY"}]}]} , 
 	{ "Name" : "data_in", "interface" : "wire", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "crc_out", "interface" : "wire", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_0", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_1", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_2", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_3", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_4", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_5", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_6", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_7", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_8", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_9", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_10", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_11", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_12", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_13", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_14", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_15", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "numChunks", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "chunkSize", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crc_size", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "init_value", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} ]}
# RTL Port declarations: 
set portNum 145
set portList { 
	{ m_axi_gmem0_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem0_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem0_AWADDR sc_out sc_lv 64 signal 0 } 
	{ m_axi_gmem0_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem0_AWLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_gmem0_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem0_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem0_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem0_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem0_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem0_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem0_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem0_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem0_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem0_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem0_WDATA sc_out sc_lv 128 signal 0 } 
	{ m_axi_gmem0_WSTRB sc_out sc_lv 16 signal 0 } 
	{ m_axi_gmem0_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem0_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem0_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem0_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem0_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem0_ARADDR sc_out sc_lv 64 signal 0 } 
	{ m_axi_gmem0_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem0_ARLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_gmem0_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem0_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem0_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem0_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem0_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem0_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem0_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem0_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem0_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem0_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem0_RDATA sc_in sc_lv 128 signal 0 } 
	{ m_axi_gmem0_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem0_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem0_RFIFONUM sc_in sc_lv 9 signal 0 } 
	{ m_axi_gmem0_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem0_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem0_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem0_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem0_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem0_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem0_BUSER sc_in sc_lv 1 signal 0 } 
	{ data_in sc_in sc_lv 64 signal 1 } 
	{ crc_out sc_in sc_lv 64 signal 2 } 
	{ crcTables_0_address0 sc_out sc_lv 8 signal 3 } 
	{ crcTables_0_ce0 sc_out sc_logic 1 signal 3 } 
	{ crcTables_0_d0 sc_out sc_lv 32 signal 3 } 
	{ crcTables_0_q0 sc_in sc_lv 32 signal 3 } 
	{ crcTables_0_we0 sc_out sc_logic 1 signal 3 } 
	{ crcTables_1_address0 sc_out sc_lv 8 signal 4 } 
	{ crcTables_1_ce0 sc_out sc_logic 1 signal 4 } 
	{ crcTables_1_d0 sc_out sc_lv 32 signal 4 } 
	{ crcTables_1_q0 sc_in sc_lv 32 signal 4 } 
	{ crcTables_1_we0 sc_out sc_logic 1 signal 4 } 
	{ crcTables_2_address0 sc_out sc_lv 8 signal 5 } 
	{ crcTables_2_ce0 sc_out sc_logic 1 signal 5 } 
	{ crcTables_2_d0 sc_out sc_lv 32 signal 5 } 
	{ crcTables_2_q0 sc_in sc_lv 32 signal 5 } 
	{ crcTables_2_we0 sc_out sc_logic 1 signal 5 } 
	{ crcTables_3_address0 sc_out sc_lv 8 signal 6 } 
	{ crcTables_3_ce0 sc_out sc_logic 1 signal 6 } 
	{ crcTables_3_d0 sc_out sc_lv 32 signal 6 } 
	{ crcTables_3_q0 sc_in sc_lv 32 signal 6 } 
	{ crcTables_3_we0 sc_out sc_logic 1 signal 6 } 
	{ crcTables_4_address0 sc_out sc_lv 8 signal 7 } 
	{ crcTables_4_ce0 sc_out sc_logic 1 signal 7 } 
	{ crcTables_4_d0 sc_out sc_lv 32 signal 7 } 
	{ crcTables_4_q0 sc_in sc_lv 32 signal 7 } 
	{ crcTables_4_we0 sc_out sc_logic 1 signal 7 } 
	{ crcTables_5_address0 sc_out sc_lv 8 signal 8 } 
	{ crcTables_5_ce0 sc_out sc_logic 1 signal 8 } 
	{ crcTables_5_d0 sc_out sc_lv 32 signal 8 } 
	{ crcTables_5_q0 sc_in sc_lv 32 signal 8 } 
	{ crcTables_5_we0 sc_out sc_logic 1 signal 8 } 
	{ crcTables_6_address0 sc_out sc_lv 8 signal 9 } 
	{ crcTables_6_ce0 sc_out sc_logic 1 signal 9 } 
	{ crcTables_6_d0 sc_out sc_lv 32 signal 9 } 
	{ crcTables_6_q0 sc_in sc_lv 32 signal 9 } 
	{ crcTables_6_we0 sc_out sc_logic 1 signal 9 } 
	{ crcTables_7_address0 sc_out sc_lv 8 signal 10 } 
	{ crcTables_7_ce0 sc_out sc_logic 1 signal 10 } 
	{ crcTables_7_d0 sc_out sc_lv 32 signal 10 } 
	{ crcTables_7_q0 sc_in sc_lv 32 signal 10 } 
	{ crcTables_7_we0 sc_out sc_logic 1 signal 10 } 
	{ crcTables_8_address0 sc_out sc_lv 8 signal 11 } 
	{ crcTables_8_ce0 sc_out sc_logic 1 signal 11 } 
	{ crcTables_8_d0 sc_out sc_lv 32 signal 11 } 
	{ crcTables_8_q0 sc_in sc_lv 32 signal 11 } 
	{ crcTables_8_we0 sc_out sc_logic 1 signal 11 } 
	{ crcTables_9_address0 sc_out sc_lv 8 signal 12 } 
	{ crcTables_9_ce0 sc_out sc_logic 1 signal 12 } 
	{ crcTables_9_d0 sc_out sc_lv 32 signal 12 } 
	{ crcTables_9_q0 sc_in sc_lv 32 signal 12 } 
	{ crcTables_9_we0 sc_out sc_logic 1 signal 12 } 
	{ crcTables_10_address0 sc_out sc_lv 8 signal 13 } 
	{ crcTables_10_ce0 sc_out sc_logic 1 signal 13 } 
	{ crcTables_10_d0 sc_out sc_lv 32 signal 13 } 
	{ crcTables_10_q0 sc_in sc_lv 32 signal 13 } 
	{ crcTables_10_we0 sc_out sc_logic 1 signal 13 } 
	{ crcTables_11_address0 sc_out sc_lv 8 signal 14 } 
	{ crcTables_11_ce0 sc_out sc_logic 1 signal 14 } 
	{ crcTables_11_d0 sc_out sc_lv 32 signal 14 } 
	{ crcTables_11_q0 sc_in sc_lv 32 signal 14 } 
	{ crcTables_11_we0 sc_out sc_logic 1 signal 14 } 
	{ crcTables_12_address0 sc_out sc_lv 8 signal 15 } 
	{ crcTables_12_ce0 sc_out sc_logic 1 signal 15 } 
	{ crcTables_12_d0 sc_out sc_lv 32 signal 15 } 
	{ crcTables_12_q0 sc_in sc_lv 32 signal 15 } 
	{ crcTables_12_we0 sc_out sc_logic 1 signal 15 } 
	{ crcTables_13_address0 sc_out sc_lv 8 signal 16 } 
	{ crcTables_13_ce0 sc_out sc_logic 1 signal 16 } 
	{ crcTables_13_d0 sc_out sc_lv 32 signal 16 } 
	{ crcTables_13_q0 sc_in sc_lv 32 signal 16 } 
	{ crcTables_13_we0 sc_out sc_logic 1 signal 16 } 
	{ crcTables_14_address0 sc_out sc_lv 8 signal 17 } 
	{ crcTables_14_ce0 sc_out sc_logic 1 signal 17 } 
	{ crcTables_14_d0 sc_out sc_lv 32 signal 17 } 
	{ crcTables_14_q0 sc_in sc_lv 32 signal 17 } 
	{ crcTables_14_we0 sc_out sc_logic 1 signal 17 } 
	{ crcTables_15_address0 sc_out sc_lv 8 signal 18 } 
	{ crcTables_15_ce0 sc_out sc_logic 1 signal 18 } 
	{ crcTables_15_d0 sc_out sc_lv 32 signal 18 } 
	{ crcTables_15_q0 sc_in sc_lv 32 signal 18 } 
	{ crcTables_15_we0 sc_out sc_logic 1 signal 18 } 
	{ numChunks sc_in sc_lv 32 signal 19 } 
	{ chunkSize sc_in sc_lv 32 signal 20 } 
	{ crc_size sc_in sc_lv 32 signal 21 } 
	{ init_value sc_in sc_lv 32 signal 22 } 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ crc_out_ap_vld sc_in sc_logic 1 invld 2 } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ data_in_ap_vld sc_in sc_logic 1 invld 1 } 
	{ numChunks_ap_vld sc_in sc_logic 1 invld 19 } 
	{ chunkSize_ap_vld sc_in sc_logic 1 invld 20 } 
	{ crc_size_ap_vld sc_in sc_logic 1 invld 21 } 
	{ init_value_ap_vld sc_in sc_logic 1 invld 22 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_continue sc_in sc_logic 1 continue -1 } 
}
set NewPortList {[ 
	{ "name": "m_axi_gmem0_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem0_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem0_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem0", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem0_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem0_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem0", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem0_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem0", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem0_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem0", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem0_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem0", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem0_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem0", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem0_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem0", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem0_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem0", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem0_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem0", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem0_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem0_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem0_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem0_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem0", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem0_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "gmem0", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem0_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem0_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "WID" }} , 
 	{ "name": "m_axi_gmem0_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem0_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem0_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem0_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem0", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem0_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem0_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem0", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem0_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem0", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem0_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem0", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem0_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem0", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem0_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem0", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem0_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem0", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem0_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem0", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem0_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem0", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem0_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem0_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem0_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem0_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":128, "type": "signal", "bundle":{"name": "gmem0", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem0_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem0_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "RID" }} , 
 	{ "name": "m_axi_gmem0_RFIFONUM", "direction": "in", "datatype": "sc_lv", "bitwidth":9, "type": "signal", "bundle":{"name": "gmem0", "role": "RFIFONUM" }} , 
 	{ "name": "m_axi_gmem0_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem0_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem0", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem0_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem0_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem0_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem0", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem0_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "BID" }} , 
 	{ "name": "m_axi_gmem0_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "BUSER" }} , 
 	{ "name": "data_in", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "data_in", "role": "default" }} , 
 	{ "name": "crc_out", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "crc_out", "role": "default" }} , 
 	{ "name": "crcTables_0_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_0", "role": "address0" }} , 
 	{ "name": "crcTables_0_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_0", "role": "ce0" }} , 
 	{ "name": "crcTables_0_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_0", "role": "d0" }} , 
 	{ "name": "crcTables_0_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_0", "role": "q0" }} , 
 	{ "name": "crcTables_0_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_0", "role": "we0" }} , 
 	{ "name": "crcTables_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_1", "role": "address0" }} , 
 	{ "name": "crcTables_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_1", "role": "ce0" }} , 
 	{ "name": "crcTables_1_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_1", "role": "d0" }} , 
 	{ "name": "crcTables_1_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_1", "role": "q0" }} , 
 	{ "name": "crcTables_1_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_1", "role": "we0" }} , 
 	{ "name": "crcTables_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_2", "role": "address0" }} , 
 	{ "name": "crcTables_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_2", "role": "ce0" }} , 
 	{ "name": "crcTables_2_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_2", "role": "d0" }} , 
 	{ "name": "crcTables_2_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_2", "role": "q0" }} , 
 	{ "name": "crcTables_2_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_2", "role": "we0" }} , 
 	{ "name": "crcTables_3_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_3", "role": "address0" }} , 
 	{ "name": "crcTables_3_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_3", "role": "ce0" }} , 
 	{ "name": "crcTables_3_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_3", "role": "d0" }} , 
 	{ "name": "crcTables_3_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_3", "role": "q0" }} , 
 	{ "name": "crcTables_3_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_3", "role": "we0" }} , 
 	{ "name": "crcTables_4_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_4", "role": "address0" }} , 
 	{ "name": "crcTables_4_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_4", "role": "ce0" }} , 
 	{ "name": "crcTables_4_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_4", "role": "d0" }} , 
 	{ "name": "crcTables_4_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_4", "role": "q0" }} , 
 	{ "name": "crcTables_4_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_4", "role": "we0" }} , 
 	{ "name": "crcTables_5_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_5", "role": "address0" }} , 
 	{ "name": "crcTables_5_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_5", "role": "ce0" }} , 
 	{ "name": "crcTables_5_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_5", "role": "d0" }} , 
 	{ "name": "crcTables_5_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_5", "role": "q0" }} , 
 	{ "name": "crcTables_5_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_5", "role": "we0" }} , 
 	{ "name": "crcTables_6_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_6", "role": "address0" }} , 
 	{ "name": "crcTables_6_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_6", "role": "ce0" }} , 
 	{ "name": "crcTables_6_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_6", "role": "d0" }} , 
 	{ "name": "crcTables_6_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_6", "role": "q0" }} , 
 	{ "name": "crcTables_6_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_6", "role": "we0" }} , 
 	{ "name": "crcTables_7_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_7", "role": "address0" }} , 
 	{ "name": "crcTables_7_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_7", "role": "ce0" }} , 
 	{ "name": "crcTables_7_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_7", "role": "d0" }} , 
 	{ "name": "crcTables_7_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_7", "role": "q0" }} , 
 	{ "name": "crcTables_7_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_7", "role": "we0" }} , 
 	{ "name": "crcTables_8_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_8", "role": "address0" }} , 
 	{ "name": "crcTables_8_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_8", "role": "ce0" }} , 
 	{ "name": "crcTables_8_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_8", "role": "d0" }} , 
 	{ "name": "crcTables_8_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_8", "role": "q0" }} , 
 	{ "name": "crcTables_8_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_8", "role": "we0" }} , 
 	{ "name": "crcTables_9_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_9", "role": "address0" }} , 
 	{ "name": "crcTables_9_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_9", "role": "ce0" }} , 
 	{ "name": "crcTables_9_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_9", "role": "d0" }} , 
 	{ "name": "crcTables_9_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_9", "role": "q0" }} , 
 	{ "name": "crcTables_9_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_9", "role": "we0" }} , 
 	{ "name": "crcTables_10_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_10", "role": "address0" }} , 
 	{ "name": "crcTables_10_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_10", "role": "ce0" }} , 
 	{ "name": "crcTables_10_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_10", "role": "d0" }} , 
 	{ "name": "crcTables_10_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_10", "role": "q0" }} , 
 	{ "name": "crcTables_10_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_10", "role": "we0" }} , 
 	{ "name": "crcTables_11_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_11", "role": "address0" }} , 
 	{ "name": "crcTables_11_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_11", "role": "ce0" }} , 
 	{ "name": "crcTables_11_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_11", "role": "d0" }} , 
 	{ "name": "crcTables_11_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_11", "role": "q0" }} , 
 	{ "name": "crcTables_11_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_11", "role": "we0" }} , 
 	{ "name": "crcTables_12_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_12", "role": "address0" }} , 
 	{ "name": "crcTables_12_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_12", "role": "ce0" }} , 
 	{ "name": "crcTables_12_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_12", "role": "d0" }} , 
 	{ "name": "crcTables_12_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_12", "role": "q0" }} , 
 	{ "name": "crcTables_12_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_12", "role": "we0" }} , 
 	{ "name": "crcTables_13_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_13", "role": "address0" }} , 
 	{ "name": "crcTables_13_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_13", "role": "ce0" }} , 
 	{ "name": "crcTables_13_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_13", "role": "d0" }} , 
 	{ "name": "crcTables_13_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_13", "role": "q0" }} , 
 	{ "name": "crcTables_13_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_13", "role": "we0" }} , 
 	{ "name": "crcTables_14_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_14", "role": "address0" }} , 
 	{ "name": "crcTables_14_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_14", "role": "ce0" }} , 
 	{ "name": "crcTables_14_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_14", "role": "d0" }} , 
 	{ "name": "crcTables_14_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_14", "role": "q0" }} , 
 	{ "name": "crcTables_14_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_14", "role": "we0" }} , 
 	{ "name": "crcTables_15_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_15", "role": "address0" }} , 
 	{ "name": "crcTables_15_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_15", "role": "ce0" }} , 
 	{ "name": "crcTables_15_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_15", "role": "d0" }} , 
 	{ "name": "crcTables_15_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_15", "role": "q0" }} , 
 	{ "name": "crcTables_15_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_15", "role": "we0" }} , 
 	{ "name": "numChunks", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "numChunks", "role": "default" }} , 
 	{ "name": "chunkSize", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "chunkSize", "role": "default" }} , 
 	{ "name": "crc_size", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crc_size", "role": "default" }} , 
 	{ "name": "init_value", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "init_value", "role": "default" }} , 
 	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "crc_out_ap_vld", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "crc_out", "role": "ap_vld" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "data_in_ap_vld", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "data_in", "role": "ap_vld" }} , 
 	{ "name": "numChunks_ap_vld", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "numChunks", "role": "ap_vld" }} , 
 	{ "name": "chunkSize_ap_vld", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "chunkSize", "role": "ap_vld" }} , 
 	{ "name": "crc_size_ap_vld", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "crc_size", "role": "ap_vld" }} , 
 	{ "name": "init_value_ap_vld", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "invld", "bundle":{"name": "init_value", "role": "ap_vld" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_continue", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "continue", "bundle":{"name": "ap_continue", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "6", "12", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34"],
		"CDFG" : "crc_dataflow_region",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "Dataflow", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "1",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "-1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "1",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"InputProcess" : [
			{"ID" : "1", "Name" : "entry_proc_U0"},
			{"ID" : "2", "Name" : "read_input_U0"},
			{"ID" : "6", "Name" : "process_crc_chunks_U0"}],
		"OutputProcess" : [
			{"ID" : "12", "Name" : "write_output_U0"}],
		"Port" : [
			{"Name" : "gmem0", "Type" : "MAXI", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "read_input_U0", "Port" : "gmem0"},
					{"ID" : "12", "SubInstance" : "write_output_U0", "Port" : "gmem0"}]},
			{"Name" : "data_in", "Type" : "None", "Direction" : "I"},
			{"Name" : "crc_out", "Type" : "None", "Direction" : "I"},
			{"Name" : "crcTables_0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "process_crc_chunks_U0", "Port" : "crcTables_0"}]},
			{"Name" : "crcTables_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "process_crc_chunks_U0", "Port" : "crcTables_1"}]},
			{"Name" : "crcTables_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "process_crc_chunks_U0", "Port" : "crcTables_2"}]},
			{"Name" : "crcTables_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "process_crc_chunks_U0", "Port" : "crcTables_3"}]},
			{"Name" : "crcTables_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "process_crc_chunks_U0", "Port" : "crcTables_4"}]},
			{"Name" : "crcTables_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "process_crc_chunks_U0", "Port" : "crcTables_5"}]},
			{"Name" : "crcTables_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "process_crc_chunks_U0", "Port" : "crcTables_6"}]},
			{"Name" : "crcTables_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "process_crc_chunks_U0", "Port" : "crcTables_7"}]},
			{"Name" : "crcTables_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "process_crc_chunks_U0", "Port" : "crcTables_8"}]},
			{"Name" : "crcTables_9", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "process_crc_chunks_U0", "Port" : "crcTables_9"}]},
			{"Name" : "crcTables_10", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "process_crc_chunks_U0", "Port" : "crcTables_10"}]},
			{"Name" : "crcTables_11", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "process_crc_chunks_U0", "Port" : "crcTables_11"}]},
			{"Name" : "crcTables_12", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "process_crc_chunks_U0", "Port" : "crcTables_12"}]},
			{"Name" : "crcTables_13", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "process_crc_chunks_U0", "Port" : "crcTables_13"}]},
			{"Name" : "crcTables_14", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "process_crc_chunks_U0", "Port" : "crcTables_14"}]},
			{"Name" : "crcTables_15", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "6", "SubInstance" : "process_crc_chunks_U0", "Port" : "crcTables_15"}]},
			{"Name" : "numChunks", "Type" : "None", "Direction" : "I"},
			{"Name" : "chunkSize", "Type" : "None", "Direction" : "I"},
			{"Name" : "crc_size", "Type" : "None", "Direction" : "I"},
			{"Name" : "init_value", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.entry_proc_U0", "Parent" : "0",
		"CDFG" : "entry_proc",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "1",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "0", "EstimateLatencyMin" : "0", "EstimateLatencyMax" : "0",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "crc_out", "Type" : "None", "Direction" : "I"},
			{"Name" : "crc_out_c", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["12"], "DependentChan" : "15", "DependentChanDepth" : "4", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "crc_out_c_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.read_input_U0", "Parent" : "0", "Child" : ["3", "5"],
		"CDFG" : "read_input",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "-1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "gmem0", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "gmem0_blk_n_AR", "Type" : "RtlSignal"}],
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "gmem0", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "in_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "inByte017", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["6"], "DependentChan" : "16", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte017", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte118", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["6"], "DependentChan" : "17", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte118", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte219", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["6"], "DependentChan" : "18", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte219", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte320", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["6"], "DependentChan" : "19", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte320", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte421", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["6"], "DependentChan" : "20", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte421", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte522", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["6"], "DependentChan" : "21", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte522", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte623", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["6"], "DependentChan" : "22", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte623", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte724", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["6"], "DependentChan" : "23", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte724", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte825", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["6"], "DependentChan" : "24", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte825", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte926", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["6"], "DependentChan" : "25", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte926", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte1027", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["6"], "DependentChan" : "26", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte1027", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte1128", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["6"], "DependentChan" : "27", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte1128", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte1229", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["6"], "DependentChan" : "28", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte1229", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte1330", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["6"], "DependentChan" : "29", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte1330", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte1431", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["6"], "DependentChan" : "30", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte1431", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte1532", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["6"], "DependentChan" : "31", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte1532", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "numChunks", "Type" : "None", "Direction" : "I"},
			{"Name" : "chunkSize", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "3", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.read_input_U0.grp_read_input_Pipeline_mem_rd_fu_112", "Parent" : "2", "Child" : ["4"],
		"CDFG" : "read_input_Pipeline_mem_rd",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "-1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "gmem0", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "gmem0_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "sext_ln323", "Type" : "None", "Direction" : "I"},
			{"Name" : "trunc_ln323_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "inByte017", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte017_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte118", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte118_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte219", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte219_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte320", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte320_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte421", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte421_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte522", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte522_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte623", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte623_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte724", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte724_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte825", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte825_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte926", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte926_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1027", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte1027_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1128", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte1128_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1229", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte1229_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1330", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte1330_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1431", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte1431_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1532", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte1532_blk_n", "Type" : "RtlSignal"}]}],
		"Loop" : [
			{"Name" : "mem_rd", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter2", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter1", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "4", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.read_input_U0.grp_read_input_Pipeline_mem_rd_fu_112.flow_control_loop_pipe_sequential_init_U", "Parent" : "3"},
	{"ID" : "5", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.read_input_U0.mul_32s_32s_32_2_1_U58", "Parent" : "2"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.process_crc_chunks_U0", "Parent" : "0", "Child" : ["7", "10"],
		"CDFG" : "process_crc_chunks",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "-1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "crcTables_0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "crcTables_0", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_0", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_1", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_2", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_3", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_4", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_5", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_6", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_7", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_8", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_9", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_9", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_10", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_10", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_11", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_11", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_12", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_12", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_13", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_13", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_14", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_14", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_15", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_15", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte017", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["2"], "DependentChan" : "16", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte017", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte017", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte118", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["2"], "DependentChan" : "17", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte118", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte118", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte219", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["2"], "DependentChan" : "18", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte219", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte219", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte320", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["2"], "DependentChan" : "19", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte320", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte320", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte421", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["2"], "DependentChan" : "20", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte421", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte421", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte522", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["2"], "DependentChan" : "21", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte522", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte522", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte623", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["2"], "DependentChan" : "22", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte623", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte623", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte724", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["2"], "DependentChan" : "23", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte724", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte724", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte825", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["2"], "DependentChan" : "24", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte825", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte825", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte926", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["2"], "DependentChan" : "25", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte926", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte926", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte1027", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["2"], "DependentChan" : "26", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte1027", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte1027", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte1128", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["2"], "DependentChan" : "27", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte1128", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte1128", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte1229", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["2"], "DependentChan" : "28", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte1229", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte1229", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte1330", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["2"], "DependentChan" : "29", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte1330", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte1330", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte1431", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["2"], "DependentChan" : "30", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte1431", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte1431", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte1532", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["2"], "DependentChan" : "31", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "10", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte1532", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "7", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte1532", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crc_size", "Type" : "None", "Direction" : "I"},
			{"Name" : "init_value", "Type" : "None", "Direction" : "I"},
			{"Name" : "outStream33", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["12"], "DependentChan" : "32", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "outStream33_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "numChunks", "Type" : "None", "Direction" : "I"},
			{"Name" : "chunkSize", "Type" : "None", "Direction" : "I"},
			{"Name" : "numChunks_c", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["12"], "DependentChan" : "33", "DependentChanDepth" : "2", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "numChunks_c_blk_n", "Type" : "RtlSignal"}]}],
		"Loop" : [
			{"Name" : "chunk_loop", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "7", "FirstState" : "ap_ST_fsm_state3", "LastState" : ["ap_ST_fsm_state7"], "QuitState" : ["ap_ST_fsm_state3"], "PreState" : ["ap_ST_fsm_state2"], "PostState" : ["ap_ST_fsm_state1"], "OneDepthLoop" : "0", "OneStateBlock": ""}}]},
	{"ID" : "7", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.process_crc_chunks_U0.grp_crc_process_full_blocks_fu_197", "Parent" : "6", "Child" : ["8"],
		"CDFG" : "crc_process_full_blocks",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "crcTables_0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_0", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_1", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_2", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_3", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_4", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_5", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_6", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_7", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_8", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_9", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_9", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_10", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_10", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_11", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_11", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_12", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_12", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_13", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_13", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_14", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_14", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_15", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_15", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte017", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte017", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte118", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte118", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte219", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte219", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte320", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte320", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte421", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte421", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte522", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte522", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte623", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte623", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte724", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte724", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte825", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte825", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte926", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte926", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte1027", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte1027", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte1128", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte1128", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte1229", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte1229", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte1330", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte1330", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte1431", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte1431", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte1532", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte1532", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "blocks_in_chunk", "Type" : "None", "Direction" : "I"},
			{"Name" : "crc", "Type" : "None", "Direction" : "I"},
			{"Name" : "mask", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "8", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.process_crc_chunks_U0.grp_crc_process_full_blocks_fu_197.grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Parent" : "7", "Child" : ["9"],
		"CDFG" : "crc_process_full_blocks_Pipeline_full_block_loop",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "5", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "crc", "Type" : "None", "Direction" : "I"},
			{"Name" : "blocks_in_chunk", "Type" : "None", "Direction" : "I"},
			{"Name" : "inByte017", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte017_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte118", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte118_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte219", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte219_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte320", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte320_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte421", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte421_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte522", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte522_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte623", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte623_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte724", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte724_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte825", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte825_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte926", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte926_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1027", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1027_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1128", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1128_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1229", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1229_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1330", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1330_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1431", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1431_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1532", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1532_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "crcTables_0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "crcTables_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "crcTables_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "crcTables_3", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "crcTables_4", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "crcTables_5", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "crcTables_6", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "crcTables_7", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "crcTables_8", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "crcTables_9", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "crcTables_10", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "crcTables_11", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "crcTables_12", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "crcTables_13", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "crcTables_14", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "crcTables_15", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "mask", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_out", "Type" : "Vld", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "full_block_loop", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter3", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter3", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "9", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.process_crc_chunks_U0.grp_crc_process_full_blocks_fu_197.grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131.flow_control_loop_pipe_sequential_init_U", "Parent" : "8"},
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.process_crc_chunks_U0.grp_crc_process_tail_bytes_fu_268", "Parent" : "6", "Child" : ["11"],
		"CDFG" : "crc_process_tail_bytes",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "5", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "crcTables_0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inByte017", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte017_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte118", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte118_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte219", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte219_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte320", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte320_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte421", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte421_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte522", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte522_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte623", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte623_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte724", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte724_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte825", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte825_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte926", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte926_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1027", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1027_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1128", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1128_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1229", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1229_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1330", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1330_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1431", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1431_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1532", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1532_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "tail_bytes", "Type" : "None", "Direction" : "I"},
			{"Name" : "crc", "Type" : "None", "Direction" : "I"},
			{"Name" : "mask", "Type" : "None", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "tail_byte_loop", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter3", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter3", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "11", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.process_crc_chunks_U0.grp_crc_process_tail_bytes_fu_268.flow_control_loop_pipe_sequential_init_U", "Parent" : "10"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.write_output_U0", "Parent" : "0", "Child" : ["13"],
		"CDFG" : "write_output",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "-1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"StartSource" : "1",
		"StartFifo" : "start_for_write_output_U0_U",
		"Port" : [
			{"Name" : "outStream33", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["6"], "DependentChan" : "32", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "13", "SubInstance" : "grp_write_output_Pipeline_VITIS_LOOP_350_1_fu_58", "Port" : "outStream33", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "numChunks", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["6"], "DependentChan" : "33", "DependentChanDepth" : "2", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "numChunks_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "gmem0", "Type" : "MAXI", "Direction" : "O",
				"SubConnect" : [
					{"ID" : "13", "SubInstance" : "grp_write_output_Pipeline_VITIS_LOOP_350_1_fu_58", "Port" : "gmem0", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "crc_out", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["1"], "DependentChan" : "15", "DependentChanDepth" : "4", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "crc_out_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.write_output_U0.grp_write_output_Pipeline_VITIS_LOOP_350_1_fu_58", "Parent" : "12", "Child" : ["14"],
		"CDFG" : "write_output_Pipeline_VITIS_LOOP_350_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "-1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "numChunks_load", "Type" : "None", "Direction" : "I"},
			{"Name" : "outStream33", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "outStream33_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "crc_out_load", "Type" : "None", "Direction" : "I"},
			{"Name" : "trunc_ln353_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "gmem0", "Type" : "MAXI", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "gmem0_blk_n_AW", "Type" : "RtlSignal"},
					{"Name" : "gmem0_blk_n_W", "Type" : "RtlSignal"},
					{"Name" : "gmem0_blk_n_B", "Type" : "RtlSignal"}]}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_350_1", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter70", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter70", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "14", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.write_output_U0.grp_write_output_Pipeline_VITIS_LOOP_350_1_fu_58.flow_control_loop_pipe_sequential_init_U", "Parent" : "13"},
	{"ID" : "15", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.crc_out_c_U", "Parent" : "0"},
	{"ID" : "16", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.inByte0_U", "Parent" : "0"},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.inByte1_U", "Parent" : "0"},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.inByte2_U", "Parent" : "0"},
	{"ID" : "19", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.inByte3_U", "Parent" : "0"},
	{"ID" : "20", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.inByte4_U", "Parent" : "0"},
	{"ID" : "21", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.inByte5_U", "Parent" : "0"},
	{"ID" : "22", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.inByte6_U", "Parent" : "0"},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.inByte7_U", "Parent" : "0"},
	{"ID" : "24", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.inByte8_U", "Parent" : "0"},
	{"ID" : "25", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.inByte9_U", "Parent" : "0"},
	{"ID" : "26", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.inByte10_U", "Parent" : "0"},
	{"ID" : "27", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.inByte11_U", "Parent" : "0"},
	{"ID" : "28", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.inByte12_U", "Parent" : "0"},
	{"ID" : "29", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.inByte13_U", "Parent" : "0"},
	{"ID" : "30", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.inByte14_U", "Parent" : "0"},
	{"ID" : "31", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.inByte15_U", "Parent" : "0"},
	{"ID" : "32", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.outStream_U", "Parent" : "0"},
	{"ID" : "33", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.numChunks_c_U", "Parent" : "0"},
	{"ID" : "34", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.start_for_write_output_U0_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	crc_dataflow_region {
		gmem0 {Type IO LastRead 3 FirstWrite -1}
		data_in {Type I LastRead 0 FirstWrite -1}
		crc_out {Type I LastRead 3 FirstWrite -1}
		crcTables_0 {Type I LastRead 2 FirstWrite -1}
		crcTables_1 {Type I LastRead 2 FirstWrite -1}
		crcTables_2 {Type I LastRead 2 FirstWrite -1}
		crcTables_3 {Type I LastRead 2 FirstWrite -1}
		crcTables_4 {Type I LastRead 1 FirstWrite -1}
		crcTables_5 {Type I LastRead 1 FirstWrite -1}
		crcTables_6 {Type I LastRead 1 FirstWrite -1}
		crcTables_7 {Type I LastRead 1 FirstWrite -1}
		crcTables_8 {Type I LastRead 1 FirstWrite -1}
		crcTables_9 {Type I LastRead 1 FirstWrite -1}
		crcTables_10 {Type I LastRead 1 FirstWrite -1}
		crcTables_11 {Type I LastRead 1 FirstWrite -1}
		crcTables_12 {Type I LastRead 2 FirstWrite -1}
		crcTables_13 {Type I LastRead 2 FirstWrite -1}
		crcTables_14 {Type I LastRead 2 FirstWrite -1}
		crcTables_15 {Type I LastRead 2 FirstWrite -1}
		numChunks {Type I LastRead 0 FirstWrite -1}
		chunkSize {Type I LastRead 0 FirstWrite -1}
		crc_size {Type I LastRead 2 FirstWrite -1}
		init_value {Type I LastRead 2 FirstWrite -1}}
	entry_proc {
		crc_out {Type I LastRead 0 FirstWrite -1}
		crc_out_c {Type O LastRead -1 FirstWrite 0}}
	read_input {
		gmem0 {Type I LastRead 3 FirstWrite -1}
		in_r {Type I LastRead 3 FirstWrite -1}
		inByte017 {Type O LastRead -1 FirstWrite 2}
		inByte118 {Type O LastRead -1 FirstWrite 2}
		inByte219 {Type O LastRead -1 FirstWrite 2}
		inByte320 {Type O LastRead -1 FirstWrite 2}
		inByte421 {Type O LastRead -1 FirstWrite 2}
		inByte522 {Type O LastRead -1 FirstWrite 2}
		inByte623 {Type O LastRead -1 FirstWrite 2}
		inByte724 {Type O LastRead -1 FirstWrite 2}
		inByte825 {Type O LastRead -1 FirstWrite 2}
		inByte926 {Type O LastRead -1 FirstWrite 2}
		inByte1027 {Type O LastRead -1 FirstWrite 2}
		inByte1128 {Type O LastRead -1 FirstWrite 2}
		inByte1229 {Type O LastRead -1 FirstWrite 2}
		inByte1330 {Type O LastRead -1 FirstWrite 2}
		inByte1431 {Type O LastRead -1 FirstWrite 2}
		inByte1532 {Type O LastRead -1 FirstWrite 2}
		numChunks {Type I LastRead 0 FirstWrite -1}
		chunkSize {Type I LastRead 0 FirstWrite -1}}
	read_input_Pipeline_mem_rd {
		gmem0 {Type I LastRead 1 FirstWrite -1}
		sext_ln323 {Type I LastRead 0 FirstWrite -1}
		trunc_ln323_1 {Type I LastRead 0 FirstWrite -1}
		inByte017 {Type O LastRead -1 FirstWrite 2}
		inByte118 {Type O LastRead -1 FirstWrite 2}
		inByte219 {Type O LastRead -1 FirstWrite 2}
		inByte320 {Type O LastRead -1 FirstWrite 2}
		inByte421 {Type O LastRead -1 FirstWrite 2}
		inByte522 {Type O LastRead -1 FirstWrite 2}
		inByte623 {Type O LastRead -1 FirstWrite 2}
		inByte724 {Type O LastRead -1 FirstWrite 2}
		inByte825 {Type O LastRead -1 FirstWrite 2}
		inByte926 {Type O LastRead -1 FirstWrite 2}
		inByte1027 {Type O LastRead -1 FirstWrite 2}
		inByte1128 {Type O LastRead -1 FirstWrite 2}
		inByte1229 {Type O LastRead -1 FirstWrite 2}
		inByte1330 {Type O LastRead -1 FirstWrite 2}
		inByte1431 {Type O LastRead -1 FirstWrite 2}
		inByte1532 {Type O LastRead -1 FirstWrite 2}}
	process_crc_chunks {
		crcTables_0 {Type I LastRead 2 FirstWrite -1}
		crcTables_1 {Type I LastRead 2 FirstWrite -1}
		crcTables_2 {Type I LastRead 2 FirstWrite -1}
		crcTables_3 {Type I LastRead 2 FirstWrite -1}
		crcTables_4 {Type I LastRead 1 FirstWrite -1}
		crcTables_5 {Type I LastRead 1 FirstWrite -1}
		crcTables_6 {Type I LastRead 1 FirstWrite -1}
		crcTables_7 {Type I LastRead 1 FirstWrite -1}
		crcTables_8 {Type I LastRead 1 FirstWrite -1}
		crcTables_9 {Type I LastRead 1 FirstWrite -1}
		crcTables_10 {Type I LastRead 1 FirstWrite -1}
		crcTables_11 {Type I LastRead 1 FirstWrite -1}
		crcTables_12 {Type I LastRead 2 FirstWrite -1}
		crcTables_13 {Type I LastRead 2 FirstWrite -1}
		crcTables_14 {Type I LastRead 2 FirstWrite -1}
		crcTables_15 {Type I LastRead 2 FirstWrite -1}
		inByte017 {Type I LastRead 1 FirstWrite -1}
		inByte118 {Type I LastRead 1 FirstWrite -1}
		inByte219 {Type I LastRead 1 FirstWrite -1}
		inByte320 {Type I LastRead 1 FirstWrite -1}
		inByte421 {Type I LastRead 1 FirstWrite -1}
		inByte522 {Type I LastRead 1 FirstWrite -1}
		inByte623 {Type I LastRead 1 FirstWrite -1}
		inByte724 {Type I LastRead 1 FirstWrite -1}
		inByte825 {Type I LastRead 1 FirstWrite -1}
		inByte926 {Type I LastRead 1 FirstWrite -1}
		inByte1027 {Type I LastRead 1 FirstWrite -1}
		inByte1128 {Type I LastRead 1 FirstWrite -1}
		inByte1229 {Type I LastRead 1 FirstWrite -1}
		inByte1330 {Type I LastRead 1 FirstWrite -1}
		inByte1431 {Type I LastRead 1 FirstWrite -1}
		inByte1532 {Type I LastRead 1 FirstWrite -1}
		crc_size {Type I LastRead 1 FirstWrite -1}
		init_value {Type I LastRead 1 FirstWrite -1}
		outStream33 {Type O LastRead -1 FirstWrite 6}
		numChunks {Type I LastRead 0 FirstWrite -1}
		chunkSize {Type I LastRead 0 FirstWrite -1}
		numChunks_c {Type O LastRead -1 FirstWrite 0}}
	crc_process_full_blocks {
		crcTables_0 {Type I LastRead 2 FirstWrite -1}
		crcTables_1 {Type I LastRead 2 FirstWrite -1}
		crcTables_2 {Type I LastRead 2 FirstWrite -1}
		crcTables_3 {Type I LastRead 2 FirstWrite -1}
		crcTables_4 {Type I LastRead 1 FirstWrite -1}
		crcTables_5 {Type I LastRead 1 FirstWrite -1}
		crcTables_6 {Type I LastRead 1 FirstWrite -1}
		crcTables_7 {Type I LastRead 1 FirstWrite -1}
		crcTables_8 {Type I LastRead 1 FirstWrite -1}
		crcTables_9 {Type I LastRead 1 FirstWrite -1}
		crcTables_10 {Type I LastRead 1 FirstWrite -1}
		crcTables_11 {Type I LastRead 1 FirstWrite -1}
		crcTables_12 {Type I LastRead 2 FirstWrite -1}
		crcTables_13 {Type I LastRead 2 FirstWrite -1}
		crcTables_14 {Type I LastRead 2 FirstWrite -1}
		crcTables_15 {Type I LastRead 2 FirstWrite -1}
		inByte017 {Type I LastRead 1 FirstWrite -1}
		inByte118 {Type I LastRead 1 FirstWrite -1}
		inByte219 {Type I LastRead 1 FirstWrite -1}
		inByte320 {Type I LastRead 1 FirstWrite -1}
		inByte421 {Type I LastRead 1 FirstWrite -1}
		inByte522 {Type I LastRead 1 FirstWrite -1}
		inByte623 {Type I LastRead 1 FirstWrite -1}
		inByte724 {Type I LastRead 1 FirstWrite -1}
		inByte825 {Type I LastRead 1 FirstWrite -1}
		inByte926 {Type I LastRead 1 FirstWrite -1}
		inByte1027 {Type I LastRead 1 FirstWrite -1}
		inByte1128 {Type I LastRead 1 FirstWrite -1}
		inByte1229 {Type I LastRead 1 FirstWrite -1}
		inByte1330 {Type I LastRead 1 FirstWrite -1}
		inByte1431 {Type I LastRead 1 FirstWrite -1}
		inByte1532 {Type I LastRead 1 FirstWrite -1}
		blocks_in_chunk {Type I LastRead 0 FirstWrite -1}
		crc {Type I LastRead 0 FirstWrite -1}
		mask {Type I LastRead 0 FirstWrite -1}}
	crc_process_full_blocks_Pipeline_full_block_loop {
		crc {Type I LastRead 0 FirstWrite -1}
		blocks_in_chunk {Type I LastRead 0 FirstWrite -1}
		inByte017 {Type I LastRead 1 FirstWrite -1}
		inByte118 {Type I LastRead 1 FirstWrite -1}
		inByte219 {Type I LastRead 1 FirstWrite -1}
		inByte320 {Type I LastRead 1 FirstWrite -1}
		inByte421 {Type I LastRead 1 FirstWrite -1}
		inByte522 {Type I LastRead 1 FirstWrite -1}
		inByte623 {Type I LastRead 1 FirstWrite -1}
		inByte724 {Type I LastRead 1 FirstWrite -1}
		inByte825 {Type I LastRead 1 FirstWrite -1}
		inByte926 {Type I LastRead 1 FirstWrite -1}
		inByte1027 {Type I LastRead 1 FirstWrite -1}
		inByte1128 {Type I LastRead 1 FirstWrite -1}
		inByte1229 {Type I LastRead 1 FirstWrite -1}
		inByte1330 {Type I LastRead 1 FirstWrite -1}
		inByte1431 {Type I LastRead 1 FirstWrite -1}
		inByte1532 {Type I LastRead 1 FirstWrite -1}
		crcTables_0 {Type I LastRead 2 FirstWrite -1}
		crcTables_1 {Type I LastRead 2 FirstWrite -1}
		crcTables_2 {Type I LastRead 2 FirstWrite -1}
		crcTables_3 {Type I LastRead 2 FirstWrite -1}
		crcTables_4 {Type I LastRead 1 FirstWrite -1}
		crcTables_5 {Type I LastRead 1 FirstWrite -1}
		crcTables_6 {Type I LastRead 1 FirstWrite -1}
		crcTables_7 {Type I LastRead 1 FirstWrite -1}
		crcTables_8 {Type I LastRead 1 FirstWrite -1}
		crcTables_9 {Type I LastRead 1 FirstWrite -1}
		crcTables_10 {Type I LastRead 1 FirstWrite -1}
		crcTables_11 {Type I LastRead 1 FirstWrite -1}
		crcTables_12 {Type I LastRead 2 FirstWrite -1}
		crcTables_13 {Type I LastRead 2 FirstWrite -1}
		crcTables_14 {Type I LastRead 2 FirstWrite -1}
		crcTables_15 {Type I LastRead 2 FirstWrite -1}
		mask {Type I LastRead 0 FirstWrite -1}
		p_out {Type O LastRead -1 FirstWrite 2}}
	crc_process_tail_bytes {
		crcTables_0 {Type I LastRead 2 FirstWrite -1}
		inByte017 {Type I LastRead 1 FirstWrite -1}
		inByte118 {Type I LastRead 1 FirstWrite -1}
		inByte219 {Type I LastRead 1 FirstWrite -1}
		inByte320 {Type I LastRead 1 FirstWrite -1}
		inByte421 {Type I LastRead 1 FirstWrite -1}
		inByte522 {Type I LastRead 1 FirstWrite -1}
		inByte623 {Type I LastRead 1 FirstWrite -1}
		inByte724 {Type I LastRead 1 FirstWrite -1}
		inByte825 {Type I LastRead 1 FirstWrite -1}
		inByte926 {Type I LastRead 1 FirstWrite -1}
		inByte1027 {Type I LastRead 1 FirstWrite -1}
		inByte1128 {Type I LastRead 1 FirstWrite -1}
		inByte1229 {Type I LastRead 1 FirstWrite -1}
		inByte1330 {Type I LastRead 1 FirstWrite -1}
		inByte1431 {Type I LastRead 1 FirstWrite -1}
		inByte1532 {Type I LastRead 1 FirstWrite -1}
		tail_bytes {Type I LastRead 0 FirstWrite -1}
		crc {Type I LastRead 0 FirstWrite -1}
		mask {Type I LastRead 0 FirstWrite -1}}
	write_output {
		outStream33 {Type I LastRead 1 FirstWrite -1}
		numChunks {Type I LastRead 0 FirstWrite -1}
		gmem0 {Type O LastRead 3 FirstWrite 2}
		crc_out {Type I LastRead 0 FirstWrite -1}}
	write_output_Pipeline_VITIS_LOOP_350_1 {
		numChunks_load {Type I LastRead 0 FirstWrite -1}
		outStream33 {Type I LastRead 1 FirstWrite -1}
		crc_out_load {Type I LastRead 0 FirstWrite -1}
		trunc_ln353_1 {Type I LastRead 0 FirstWrite -1}
		gmem0 {Type O LastRead 3 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "-1", "Max" : "-1"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	 { m_axi {  { m_axi_gmem0_AWVALID VALID 1 1 }  { m_axi_gmem0_AWREADY READY 0 1 }  { m_axi_gmem0_AWADDR ADDR 1 64 }  { m_axi_gmem0_AWID ID 1 1 }  { m_axi_gmem0_AWLEN SIZE 1 32 }  { m_axi_gmem0_AWSIZE BURST 1 3 }  { m_axi_gmem0_AWBURST LOCK 1 2 }  { m_axi_gmem0_AWLOCK CACHE 1 2 }  { m_axi_gmem0_AWCACHE PROT 1 4 }  { m_axi_gmem0_AWPROT QOS 1 3 }  { m_axi_gmem0_AWQOS REGION 1 4 }  { m_axi_gmem0_AWREGION USER 1 4 }  { m_axi_gmem0_AWUSER DATA 1 1 }  { m_axi_gmem0_WVALID VALID 1 1 }  { m_axi_gmem0_WREADY READY 0 1 }  { m_axi_gmem0_WDATA FIFONUM 1 128 }  { m_axi_gmem0_WSTRB STRB 1 16 }  { m_axi_gmem0_WLAST LAST 1 1 }  { m_axi_gmem0_WID ID 1 1 }  { m_axi_gmem0_WUSER DATA 1 1 }  { m_axi_gmem0_ARVALID VALID 1 1 }  { m_axi_gmem0_ARREADY READY 0 1 }  { m_axi_gmem0_ARADDR ADDR 1 64 }  { m_axi_gmem0_ARID ID 1 1 }  { m_axi_gmem0_ARLEN SIZE 1 32 }  { m_axi_gmem0_ARSIZE BURST 1 3 }  { m_axi_gmem0_ARBURST LOCK 1 2 }  { m_axi_gmem0_ARLOCK CACHE 1 2 }  { m_axi_gmem0_ARCACHE PROT 1 4 }  { m_axi_gmem0_ARPROT QOS 1 3 }  { m_axi_gmem0_ARQOS REGION 1 4 }  { m_axi_gmem0_ARREGION USER 1 4 }  { m_axi_gmem0_ARUSER DATA 1 1 }  { m_axi_gmem0_RVALID VALID 0 1 }  { m_axi_gmem0_RREADY READY 1 1 }  { m_axi_gmem0_RDATA FIFONUM 0 128 }  { m_axi_gmem0_RLAST LAST 0 1 }  { m_axi_gmem0_RID ID 0 1 }  { m_axi_gmem0_RFIFONUM LEN 0 9 }  { m_axi_gmem0_RUSER DATA 0 1 }  { m_axi_gmem0_RRESP RESP 0 2 }  { m_axi_gmem0_BVALID VALID 0 1 }  { m_axi_gmem0_BREADY READY 1 1 }  { m_axi_gmem0_BRESP RESP 0 2 }  { m_axi_gmem0_BID ID 0 1 }  { m_axi_gmem0_BUSER DATA 0 1 } } }
	data_in { ap_none {  { data_in in_data 0 64 }  { data_in_ap_vld in_vld 0 1 } } }
	crc_out { ap_none {  { crc_out in_data 0 64 }  { crc_out_ap_vld in_vld 0 1 } } }
	crcTables_0 { ap_memory {  { crcTables_0_address0 mem_address 1 8 }  { crcTables_0_ce0 mem_ce 1 1 }  { crcTables_0_d0 mem_din 1 32 }  { crcTables_0_q0 mem_dout 0 32 }  { crcTables_0_we0 mem_we 1 1 } } }
	crcTables_1 { ap_memory {  { crcTables_1_address0 mem_address 1 8 }  { crcTables_1_ce0 mem_ce 1 1 }  { crcTables_1_d0 mem_din 1 32 }  { crcTables_1_q0 mem_dout 0 32 }  { crcTables_1_we0 mem_we 1 1 } } }
	crcTables_2 { ap_memory {  { crcTables_2_address0 mem_address 1 8 }  { crcTables_2_ce0 mem_ce 1 1 }  { crcTables_2_d0 mem_din 1 32 }  { crcTables_2_q0 mem_dout 0 32 }  { crcTables_2_we0 mem_we 1 1 } } }
	crcTables_3 { ap_memory {  { crcTables_3_address0 mem_address 1 8 }  { crcTables_3_ce0 mem_ce 1 1 }  { crcTables_3_d0 mem_din 1 32 }  { crcTables_3_q0 mem_dout 0 32 }  { crcTables_3_we0 mem_we 1 1 } } }
	crcTables_4 { ap_memory {  { crcTables_4_address0 mem_address 1 8 }  { crcTables_4_ce0 mem_ce 1 1 }  { crcTables_4_d0 mem_din 1 32 }  { crcTables_4_q0 mem_dout 0 32 }  { crcTables_4_we0 mem_we 1 1 } } }
	crcTables_5 { ap_memory {  { crcTables_5_address0 mem_address 1 8 }  { crcTables_5_ce0 mem_ce 1 1 }  { crcTables_5_d0 mem_din 1 32 }  { crcTables_5_q0 mem_dout 0 32 }  { crcTables_5_we0 mem_we 1 1 } } }
	crcTables_6 { ap_memory {  { crcTables_6_address0 mem_address 1 8 }  { crcTables_6_ce0 mem_ce 1 1 }  { crcTables_6_d0 mem_din 1 32 }  { crcTables_6_q0 mem_dout 0 32 }  { crcTables_6_we0 mem_we 1 1 } } }
	crcTables_7 { ap_memory {  { crcTables_7_address0 mem_address 1 8 }  { crcTables_7_ce0 mem_ce 1 1 }  { crcTables_7_d0 mem_din 1 32 }  { crcTables_7_q0 mem_dout 0 32 }  { crcTables_7_we0 mem_we 1 1 } } }
	crcTables_8 { ap_memory {  { crcTables_8_address0 mem_address 1 8 }  { crcTables_8_ce0 mem_ce 1 1 }  { crcTables_8_d0 mem_din 1 32 }  { crcTables_8_q0 mem_dout 0 32 }  { crcTables_8_we0 mem_we 1 1 } } }
	crcTables_9 { ap_memory {  { crcTables_9_address0 mem_address 1 8 }  { crcTables_9_ce0 mem_ce 1 1 }  { crcTables_9_d0 mem_din 1 32 }  { crcTables_9_q0 mem_dout 0 32 }  { crcTables_9_we0 mem_we 1 1 } } }
	crcTables_10 { ap_memory {  { crcTables_10_address0 mem_address 1 8 }  { crcTables_10_ce0 mem_ce 1 1 }  { crcTables_10_d0 mem_din 1 32 }  { crcTables_10_q0 mem_dout 0 32 }  { crcTables_10_we0 mem_we 1 1 } } }
	crcTables_11 { ap_memory {  { crcTables_11_address0 mem_address 1 8 }  { crcTables_11_ce0 mem_ce 1 1 }  { crcTables_11_d0 mem_din 1 32 }  { crcTables_11_q0 mem_dout 0 32 }  { crcTables_11_we0 mem_we 1 1 } } }
	crcTables_12 { ap_memory {  { crcTables_12_address0 mem_address 1 8 }  { crcTables_12_ce0 mem_ce 1 1 }  { crcTables_12_d0 mem_din 1 32 }  { crcTables_12_q0 mem_dout 0 32 }  { crcTables_12_we0 mem_we 1 1 } } }
	crcTables_13 { ap_memory {  { crcTables_13_address0 mem_address 1 8 }  { crcTables_13_ce0 mem_ce 1 1 }  { crcTables_13_d0 mem_din 1 32 }  { crcTables_13_q0 mem_dout 0 32 }  { crcTables_13_we0 mem_we 1 1 } } }
	crcTables_14 { ap_memory {  { crcTables_14_address0 mem_address 1 8 }  { crcTables_14_ce0 mem_ce 1 1 }  { crcTables_14_d0 mem_din 1 32 }  { crcTables_14_q0 mem_dout 0 32 }  { crcTables_14_we0 mem_we 1 1 } } }
	crcTables_15 { ap_memory {  { crcTables_15_address0 mem_address 1 8 }  { crcTables_15_ce0 mem_ce 1 1 }  { crcTables_15_d0 mem_din 1 32 }  { crcTables_15_q0 mem_dout 0 32 }  { crcTables_15_we0 mem_we 1 1 } } }
	numChunks { ap_none {  { numChunks in_data 0 32 }  { numChunks_ap_vld in_vld 0 1 } } }
	chunkSize { ap_none {  { chunkSize in_data 0 32 }  { chunkSize_ap_vld in_vld 0 1 } } }
	crc_size { ap_none {  { crc_size in_data 0 32 }  { crc_size_ap_vld in_vld 0 1 } } }
	init_value { ap_none {  { init_value in_data 0 32 }  { init_value_ap_vld in_vld 0 1 } } }
}
