set moduleName process_crc_Loop_init_lut_proc
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {process_crc_Loop_init_lut_proc}
set C_modelType { void 0 }
set C_modelArgList {
	{ tables int 64 regular  }
	{ gmem1 int 512 regular {axi_master 0}  }
	{ crcTables_15 int 32 regular {array 256 { 0 } 0 1 }  }
	{ crcTables_14 int 32 regular {array 256 { 0 } 0 1 }  }
	{ crcTables_13 int 32 regular {array 256 { 0 } 0 1 }  }
	{ crcTables_12 int 32 regular {array 256 { 0 } 0 1 }  }
	{ crcTables_11 int 32 regular {array 256 { 0 } 0 1 }  }
	{ crcTables_10 int 32 regular {array 256 { 0 } 0 1 }  }
	{ crcTables_9 int 32 regular {array 256 { 0 } 0 1 }  }
	{ crcTables_8 int 32 regular {array 256 { 0 } 0 1 }  }
	{ crcTables_7 int 32 regular {array 256 { 0 } 0 1 }  }
	{ crcTables_6 int 32 regular {array 256 { 0 } 0 1 }  }
	{ crcTables_5 int 32 regular {array 256 { 0 } 0 1 }  }
	{ crcTables_4 int 32 regular {array 256 { 0 } 0 1 }  }
	{ crcTables_3 int 32 regular {array 256 { 0 } 0 1 }  }
	{ crcTables_2 int 32 regular {array 256 { 0 } 0 1 }  }
	{ crcTables_1 int 32 regular {array 256 { 0 } 0 1 }  }
	{ crcTables int 32 regular {array 256 { 0 } 0 1 }  }
}
set hasAXIMCache 0
set C_modelArgMapList {[ 
	{ "Name" : "tables", "interface" : "wire", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "gmem1", "interface" : "axi_master", "bitwidth" : 512, "direction" : "READONLY", "bitSlice":[ {"cElement": [{"cName": "tables","offset": { "type": "dynamic","port_name": "tables","bundle": "control"},"direction": "READONLY"}]}]} , 
 	{ "Name" : "crcTables_15", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "crcTables_14", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "crcTables_13", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "crcTables_12", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "crcTables_11", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "crcTables_10", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "crcTables_9", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "crcTables_8", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "crcTables_7", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "crcTables_6", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "crcTables_5", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "crcTables_4", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "crcTables_3", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "crcTables_2", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "crcTables_1", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "crcTables", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 118
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_continue sc_in sc_logic 1 continue -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ tables sc_in sc_lv 64 signal 0 } 
	{ m_axi_gmem1_AWVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem1_AWREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem1_AWADDR sc_out sc_lv 64 signal 1 } 
	{ m_axi_gmem1_AWID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem1_AWLEN sc_out sc_lv 32 signal 1 } 
	{ m_axi_gmem1_AWSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem1_AWBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem1_AWLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem1_AWCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem1_AWPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem1_AWQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem1_AWREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem1_AWUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem1_WVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem1_WREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem1_WDATA sc_out sc_lv 512 signal 1 } 
	{ m_axi_gmem1_WSTRB sc_out sc_lv 64 signal 1 } 
	{ m_axi_gmem1_WLAST sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem1_WID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem1_WUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem1_ARVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem1_ARREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem1_ARADDR sc_out sc_lv 64 signal 1 } 
	{ m_axi_gmem1_ARID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem1_ARLEN sc_out sc_lv 32 signal 1 } 
	{ m_axi_gmem1_ARSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem1_ARBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem1_ARLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem1_ARCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem1_ARPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem1_ARQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem1_ARREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem1_ARUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem1_RVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem1_RREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem1_RDATA sc_in sc_lv 512 signal 1 } 
	{ m_axi_gmem1_RLAST sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem1_RID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem1_RFIFONUM sc_in sc_lv 9 signal 1 } 
	{ m_axi_gmem1_RUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem1_RRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem1_BVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem1_BREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem1_BRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem1_BID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem1_BUSER sc_in sc_lv 1 signal 1 } 
	{ crcTables_15_address0 sc_out sc_lv 8 signal 2 } 
	{ crcTables_15_ce0 sc_out sc_logic 1 signal 2 } 
	{ crcTables_15_we0 sc_out sc_logic 1 signal 2 } 
	{ crcTables_15_d0 sc_out sc_lv 32 signal 2 } 
	{ crcTables_14_address0 sc_out sc_lv 8 signal 3 } 
	{ crcTables_14_ce0 sc_out sc_logic 1 signal 3 } 
	{ crcTables_14_we0 sc_out sc_logic 1 signal 3 } 
	{ crcTables_14_d0 sc_out sc_lv 32 signal 3 } 
	{ crcTables_13_address0 sc_out sc_lv 8 signal 4 } 
	{ crcTables_13_ce0 sc_out sc_logic 1 signal 4 } 
	{ crcTables_13_we0 sc_out sc_logic 1 signal 4 } 
	{ crcTables_13_d0 sc_out sc_lv 32 signal 4 } 
	{ crcTables_12_address0 sc_out sc_lv 8 signal 5 } 
	{ crcTables_12_ce0 sc_out sc_logic 1 signal 5 } 
	{ crcTables_12_we0 sc_out sc_logic 1 signal 5 } 
	{ crcTables_12_d0 sc_out sc_lv 32 signal 5 } 
	{ crcTables_11_address0 sc_out sc_lv 8 signal 6 } 
	{ crcTables_11_ce0 sc_out sc_logic 1 signal 6 } 
	{ crcTables_11_we0 sc_out sc_logic 1 signal 6 } 
	{ crcTables_11_d0 sc_out sc_lv 32 signal 6 } 
	{ crcTables_10_address0 sc_out sc_lv 8 signal 7 } 
	{ crcTables_10_ce0 sc_out sc_logic 1 signal 7 } 
	{ crcTables_10_we0 sc_out sc_logic 1 signal 7 } 
	{ crcTables_10_d0 sc_out sc_lv 32 signal 7 } 
	{ crcTables_9_address0 sc_out sc_lv 8 signal 8 } 
	{ crcTables_9_ce0 sc_out sc_logic 1 signal 8 } 
	{ crcTables_9_we0 sc_out sc_logic 1 signal 8 } 
	{ crcTables_9_d0 sc_out sc_lv 32 signal 8 } 
	{ crcTables_8_address0 sc_out sc_lv 8 signal 9 } 
	{ crcTables_8_ce0 sc_out sc_logic 1 signal 9 } 
	{ crcTables_8_we0 sc_out sc_logic 1 signal 9 } 
	{ crcTables_8_d0 sc_out sc_lv 32 signal 9 } 
	{ crcTables_7_address0 sc_out sc_lv 8 signal 10 } 
	{ crcTables_7_ce0 sc_out sc_logic 1 signal 10 } 
	{ crcTables_7_we0 sc_out sc_logic 1 signal 10 } 
	{ crcTables_7_d0 sc_out sc_lv 32 signal 10 } 
	{ crcTables_6_address0 sc_out sc_lv 8 signal 11 } 
	{ crcTables_6_ce0 sc_out sc_logic 1 signal 11 } 
	{ crcTables_6_we0 sc_out sc_logic 1 signal 11 } 
	{ crcTables_6_d0 sc_out sc_lv 32 signal 11 } 
	{ crcTables_5_address0 sc_out sc_lv 8 signal 12 } 
	{ crcTables_5_ce0 sc_out sc_logic 1 signal 12 } 
	{ crcTables_5_we0 sc_out sc_logic 1 signal 12 } 
	{ crcTables_5_d0 sc_out sc_lv 32 signal 12 } 
	{ crcTables_4_address0 sc_out sc_lv 8 signal 13 } 
	{ crcTables_4_ce0 sc_out sc_logic 1 signal 13 } 
	{ crcTables_4_we0 sc_out sc_logic 1 signal 13 } 
	{ crcTables_4_d0 sc_out sc_lv 32 signal 13 } 
	{ crcTables_3_address0 sc_out sc_lv 8 signal 14 } 
	{ crcTables_3_ce0 sc_out sc_logic 1 signal 14 } 
	{ crcTables_3_we0 sc_out sc_logic 1 signal 14 } 
	{ crcTables_3_d0 sc_out sc_lv 32 signal 14 } 
	{ crcTables_2_address0 sc_out sc_lv 8 signal 15 } 
	{ crcTables_2_ce0 sc_out sc_logic 1 signal 15 } 
	{ crcTables_2_we0 sc_out sc_logic 1 signal 15 } 
	{ crcTables_2_d0 sc_out sc_lv 32 signal 15 } 
	{ crcTables_1_address0 sc_out sc_lv 8 signal 16 } 
	{ crcTables_1_ce0 sc_out sc_logic 1 signal 16 } 
	{ crcTables_1_we0 sc_out sc_logic 1 signal 16 } 
	{ crcTables_1_d0 sc_out sc_lv 32 signal 16 } 
	{ crcTables_address0 sc_out sc_lv 8 signal 17 } 
	{ crcTables_ce0 sc_out sc_logic 1 signal 17 } 
	{ crcTables_we0 sc_out sc_logic 1 signal 17 } 
	{ crcTables_d0 sc_out sc_lv 32 signal 17 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_continue", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "continue", "bundle":{"name": "ap_continue", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "tables", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "tables", "role": "default" }} , 
 	{ "name": "m_axi_gmem1_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem1_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem1_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem1", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem1_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem1_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem1", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem1_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem1", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem1_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem1", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem1_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem1", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem1_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem1", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem1_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem1", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem1_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem1", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem1_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem1", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem1_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem1_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem1_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem1_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":512, "type": "signal", "bundle":{"name": "gmem1", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem1_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem1", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem1_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem1_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "WID" }} , 
 	{ "name": "m_axi_gmem1_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem1_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem1_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem1_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem1", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem1_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem1_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem1", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem1_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem1", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem1_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem1", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem1_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem1", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem1_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem1", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem1_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem1", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem1_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem1", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem1_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem1", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem1_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem1_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem1_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem1_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":512, "type": "signal", "bundle":{"name": "gmem1", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem1_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem1_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "RID" }} , 
 	{ "name": "m_axi_gmem1_RFIFONUM", "direction": "in", "datatype": "sc_lv", "bitwidth":9, "type": "signal", "bundle":{"name": "gmem1", "role": "RFIFONUM" }} , 
 	{ "name": "m_axi_gmem1_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem1_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem1", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem1_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem1_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem1_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem1", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem1_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "BID" }} , 
 	{ "name": "m_axi_gmem1_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "BUSER" }} , 
 	{ "name": "crcTables_15_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_15", "role": "address0" }} , 
 	{ "name": "crcTables_15_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_15", "role": "ce0" }} , 
 	{ "name": "crcTables_15_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_15", "role": "we0" }} , 
 	{ "name": "crcTables_15_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_15", "role": "d0" }} , 
 	{ "name": "crcTables_14_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_14", "role": "address0" }} , 
 	{ "name": "crcTables_14_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_14", "role": "ce0" }} , 
 	{ "name": "crcTables_14_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_14", "role": "we0" }} , 
 	{ "name": "crcTables_14_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_14", "role": "d0" }} , 
 	{ "name": "crcTables_13_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_13", "role": "address0" }} , 
 	{ "name": "crcTables_13_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_13", "role": "ce0" }} , 
 	{ "name": "crcTables_13_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_13", "role": "we0" }} , 
 	{ "name": "crcTables_13_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_13", "role": "d0" }} , 
 	{ "name": "crcTables_12_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_12", "role": "address0" }} , 
 	{ "name": "crcTables_12_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_12", "role": "ce0" }} , 
 	{ "name": "crcTables_12_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_12", "role": "we0" }} , 
 	{ "name": "crcTables_12_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_12", "role": "d0" }} , 
 	{ "name": "crcTables_11_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_11", "role": "address0" }} , 
 	{ "name": "crcTables_11_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_11", "role": "ce0" }} , 
 	{ "name": "crcTables_11_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_11", "role": "we0" }} , 
 	{ "name": "crcTables_11_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_11", "role": "d0" }} , 
 	{ "name": "crcTables_10_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_10", "role": "address0" }} , 
 	{ "name": "crcTables_10_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_10", "role": "ce0" }} , 
 	{ "name": "crcTables_10_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_10", "role": "we0" }} , 
 	{ "name": "crcTables_10_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_10", "role": "d0" }} , 
 	{ "name": "crcTables_9_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_9", "role": "address0" }} , 
 	{ "name": "crcTables_9_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_9", "role": "ce0" }} , 
 	{ "name": "crcTables_9_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_9", "role": "we0" }} , 
 	{ "name": "crcTables_9_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_9", "role": "d0" }} , 
 	{ "name": "crcTables_8_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_8", "role": "address0" }} , 
 	{ "name": "crcTables_8_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_8", "role": "ce0" }} , 
 	{ "name": "crcTables_8_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_8", "role": "we0" }} , 
 	{ "name": "crcTables_8_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_8", "role": "d0" }} , 
 	{ "name": "crcTables_7_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_7", "role": "address0" }} , 
 	{ "name": "crcTables_7_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_7", "role": "ce0" }} , 
 	{ "name": "crcTables_7_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_7", "role": "we0" }} , 
 	{ "name": "crcTables_7_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_7", "role": "d0" }} , 
 	{ "name": "crcTables_6_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_6", "role": "address0" }} , 
 	{ "name": "crcTables_6_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_6", "role": "ce0" }} , 
 	{ "name": "crcTables_6_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_6", "role": "we0" }} , 
 	{ "name": "crcTables_6_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_6", "role": "d0" }} , 
 	{ "name": "crcTables_5_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_5", "role": "address0" }} , 
 	{ "name": "crcTables_5_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_5", "role": "ce0" }} , 
 	{ "name": "crcTables_5_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_5", "role": "we0" }} , 
 	{ "name": "crcTables_5_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_5", "role": "d0" }} , 
 	{ "name": "crcTables_4_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_4", "role": "address0" }} , 
 	{ "name": "crcTables_4_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_4", "role": "ce0" }} , 
 	{ "name": "crcTables_4_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_4", "role": "we0" }} , 
 	{ "name": "crcTables_4_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_4", "role": "d0" }} , 
 	{ "name": "crcTables_3_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_3", "role": "address0" }} , 
 	{ "name": "crcTables_3_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_3", "role": "ce0" }} , 
 	{ "name": "crcTables_3_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_3", "role": "we0" }} , 
 	{ "name": "crcTables_3_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_3", "role": "d0" }} , 
 	{ "name": "crcTables_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_2", "role": "address0" }} , 
 	{ "name": "crcTables_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_2", "role": "ce0" }} , 
 	{ "name": "crcTables_2_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_2", "role": "we0" }} , 
 	{ "name": "crcTables_2_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_2", "role": "d0" }} , 
 	{ "name": "crcTables_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_1", "role": "address0" }} , 
 	{ "name": "crcTables_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_1", "role": "ce0" }} , 
 	{ "name": "crcTables_1_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_1", "role": "we0" }} , 
 	{ "name": "crcTables_1_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_1", "role": "d0" }} , 
 	{ "name": "crcTables_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables", "role": "address0" }} , 
 	{ "name": "crcTables_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables", "role": "ce0" }} , 
 	{ "name": "crcTables_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables", "role": "we0" }} , 
 	{ "name": "crcTables_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables", "role": "d0" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1"],
		"CDFG" : "process_crc_Loop_init_lut_proc",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4171", "EstimateLatencyMax" : "4171",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "1",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "tables", "Type" : "None", "Direction" : "I"},
			{"Name" : "gmem1", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "gmem1_blk_n_AR", "Type" : "RtlSignal"}],
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "gmem1", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_15", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_15", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_14", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_14", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_13", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_13", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_12", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_12", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_11", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_11", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_10", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_10", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_9", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_9", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_8", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_8", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_7", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_7", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_6", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_6", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_5", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_5", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_4", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_4", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_3", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_3", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_2", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_2", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_1", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_1", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables", "Inst_start_state" : "72", "Inst_end_state" : "73"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Parent" : "0", "Child" : ["2"],
		"CDFG" : "process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "4099", "EstimateLatencyMax" : "4099",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "gmem1", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "gmem1_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "sext_ln676", "Type" : "None", "Direction" : "I"},
			{"Name" : "crcTables", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "crcTables_1", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "crcTables_2", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "crcTables_3", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "crcTables_4", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "crcTables_5", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "crcTables_6", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "crcTables_7", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "crcTables_8", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "crcTables_9", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "crcTables_10", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "crcTables_11", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "crcTables_12", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "crcTables_13", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "crcTables_14", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "crcTables_15", "Type" : "Memory", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "init_lut_VITIS_LOOP_678_1", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter2", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter2", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "2", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91.flow_control_loop_pipe_sequential_init_U", "Parent" : "1"}]}


set ArgLastReadFirstWriteLatency {
	process_crc_Loop_init_lut_proc {
		tables {Type I LastRead 0 FirstWrite -1}
		gmem1 {Type I LastRead 1 FirstWrite -1}
		crcTables_15 {Type O LastRead -1 FirstWrite 2}
		crcTables_14 {Type O LastRead -1 FirstWrite 2}
		crcTables_13 {Type O LastRead -1 FirstWrite 2}
		crcTables_12 {Type O LastRead -1 FirstWrite 2}
		crcTables_11 {Type O LastRead -1 FirstWrite 2}
		crcTables_10 {Type O LastRead -1 FirstWrite 2}
		crcTables_9 {Type O LastRead -1 FirstWrite 2}
		crcTables_8 {Type O LastRead -1 FirstWrite 2}
		crcTables_7 {Type O LastRead -1 FirstWrite 2}
		crcTables_6 {Type O LastRead -1 FirstWrite 2}
		crcTables_5 {Type O LastRead -1 FirstWrite 2}
		crcTables_4 {Type O LastRead -1 FirstWrite 2}
		crcTables_3 {Type O LastRead -1 FirstWrite 2}
		crcTables_2 {Type O LastRead -1 FirstWrite 2}
		crcTables_1 {Type O LastRead -1 FirstWrite 2}
		crcTables {Type O LastRead -1 FirstWrite 2}}
	process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1 {
		gmem1 {Type I LastRead 1 FirstWrite -1}
		sext_ln676 {Type I LastRead 0 FirstWrite -1}
		crcTables {Type O LastRead -1 FirstWrite 2}
		crcTables_1 {Type O LastRead -1 FirstWrite 2}
		crcTables_2 {Type O LastRead -1 FirstWrite 2}
		crcTables_3 {Type O LastRead -1 FirstWrite 2}
		crcTables_4 {Type O LastRead -1 FirstWrite 2}
		crcTables_5 {Type O LastRead -1 FirstWrite 2}
		crcTables_6 {Type O LastRead -1 FirstWrite 2}
		crcTables_7 {Type O LastRead -1 FirstWrite 2}
		crcTables_8 {Type O LastRead -1 FirstWrite 2}
		crcTables_9 {Type O LastRead -1 FirstWrite 2}
		crcTables_10 {Type O LastRead -1 FirstWrite 2}
		crcTables_11 {Type O LastRead -1 FirstWrite 2}
		crcTables_12 {Type O LastRead -1 FirstWrite 2}
		crcTables_13 {Type O LastRead -1 FirstWrite 2}
		crcTables_14 {Type O LastRead -1 FirstWrite 2}
		crcTables_15 {Type O LastRead -1 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "4171", "Max" : "4171"}
	, {"Name" : "Interval", "Min" : "4171", "Max" : "4171"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	tables { ap_none {  { tables in_data 0 64 } } }
	 { m_axi {  { m_axi_gmem1_AWVALID VALID 1 1 }  { m_axi_gmem1_AWREADY READY 0 1 }  { m_axi_gmem1_AWADDR ADDR 1 64 }  { m_axi_gmem1_AWID ID 1 1 }  { m_axi_gmem1_AWLEN SIZE 1 32 }  { m_axi_gmem1_AWSIZE BURST 1 3 }  { m_axi_gmem1_AWBURST LOCK 1 2 }  { m_axi_gmem1_AWLOCK CACHE 1 2 }  { m_axi_gmem1_AWCACHE PROT 1 4 }  { m_axi_gmem1_AWPROT QOS 1 3 }  { m_axi_gmem1_AWQOS REGION 1 4 }  { m_axi_gmem1_AWREGION USER 1 4 }  { m_axi_gmem1_AWUSER DATA 1 1 }  { m_axi_gmem1_WVALID VALID 1 1 }  { m_axi_gmem1_WREADY READY 0 1 }  { m_axi_gmem1_WDATA FIFONUM 1 512 }  { m_axi_gmem1_WSTRB STRB 1 64 }  { m_axi_gmem1_WLAST LAST 1 1 }  { m_axi_gmem1_WID ID 1 1 }  { m_axi_gmem1_WUSER DATA 1 1 }  { m_axi_gmem1_ARVALID VALID 1 1 }  { m_axi_gmem1_ARREADY READY 0 1 }  { m_axi_gmem1_ARADDR ADDR 1 64 }  { m_axi_gmem1_ARID ID 1 1 }  { m_axi_gmem1_ARLEN SIZE 1 32 }  { m_axi_gmem1_ARSIZE BURST 1 3 }  { m_axi_gmem1_ARBURST LOCK 1 2 }  { m_axi_gmem1_ARLOCK CACHE 1 2 }  { m_axi_gmem1_ARCACHE PROT 1 4 }  { m_axi_gmem1_ARPROT QOS 1 3 }  { m_axi_gmem1_ARQOS REGION 1 4 }  { m_axi_gmem1_ARREGION USER 1 4 }  { m_axi_gmem1_ARUSER DATA 1 1 }  { m_axi_gmem1_RVALID VALID 0 1 }  { m_axi_gmem1_RREADY READY 1 1 }  { m_axi_gmem1_RDATA FIFONUM 0 512 }  { m_axi_gmem1_RLAST LAST 0 1 }  { m_axi_gmem1_RID ID 0 1 }  { m_axi_gmem1_RFIFONUM LEN 0 9 }  { m_axi_gmem1_RUSER DATA 0 1 }  { m_axi_gmem1_RRESP RESP 0 2 }  { m_axi_gmem1_BVALID VALID 0 1 }  { m_axi_gmem1_BREADY READY 1 1 }  { m_axi_gmem1_BRESP RESP 0 2 }  { m_axi_gmem1_BID ID 0 1 }  { m_axi_gmem1_BUSER DATA 0 1 } } }
	crcTables_15 { ap_memory {  { crcTables_15_address0 mem_address 1 8 }  { crcTables_15_ce0 mem_ce 1 1 }  { crcTables_15_we0 mem_we 1 1 }  { crcTables_15_d0 mem_din 1 32 } } }
	crcTables_14 { ap_memory {  { crcTables_14_address0 mem_address 1 8 }  { crcTables_14_ce0 mem_ce 1 1 }  { crcTables_14_we0 mem_we 1 1 }  { crcTables_14_d0 mem_din 1 32 } } }
	crcTables_13 { ap_memory {  { crcTables_13_address0 mem_address 1 8 }  { crcTables_13_ce0 mem_ce 1 1 }  { crcTables_13_we0 mem_we 1 1 }  { crcTables_13_d0 mem_din 1 32 } } }
	crcTables_12 { ap_memory {  { crcTables_12_address0 mem_address 1 8 }  { crcTables_12_ce0 mem_ce 1 1 }  { crcTables_12_we0 mem_we 1 1 }  { crcTables_12_d0 mem_din 1 32 } } }
	crcTables_11 { ap_memory {  { crcTables_11_address0 mem_address 1 8 }  { crcTables_11_ce0 mem_ce 1 1 }  { crcTables_11_we0 mem_we 1 1 }  { crcTables_11_d0 mem_din 1 32 } } }
	crcTables_10 { ap_memory {  { crcTables_10_address0 mem_address 1 8 }  { crcTables_10_ce0 mem_ce 1 1 }  { crcTables_10_we0 mem_we 1 1 }  { crcTables_10_d0 mem_din 1 32 } } }
	crcTables_9 { ap_memory {  { crcTables_9_address0 mem_address 1 8 }  { crcTables_9_ce0 mem_ce 1 1 }  { crcTables_9_we0 mem_we 1 1 }  { crcTables_9_d0 mem_din 1 32 } } }
	crcTables_8 { ap_memory {  { crcTables_8_address0 mem_address 1 8 }  { crcTables_8_ce0 mem_ce 1 1 }  { crcTables_8_we0 mem_we 1 1 }  { crcTables_8_d0 mem_din 1 32 } } }
	crcTables_7 { ap_memory {  { crcTables_7_address0 mem_address 1 8 }  { crcTables_7_ce0 mem_ce 1 1 }  { crcTables_7_we0 mem_we 1 1 }  { crcTables_7_d0 mem_din 1 32 } } }
	crcTables_6 { ap_memory {  { crcTables_6_address0 mem_address 1 8 }  { crcTables_6_ce0 mem_ce 1 1 }  { crcTables_6_we0 mem_we 1 1 }  { crcTables_6_d0 mem_din 1 32 } } }
	crcTables_5 { ap_memory {  { crcTables_5_address0 mem_address 1 8 }  { crcTables_5_ce0 mem_ce 1 1 }  { crcTables_5_we0 mem_we 1 1 }  { crcTables_5_d0 mem_din 1 32 } } }
	crcTables_4 { ap_memory {  { crcTables_4_address0 mem_address 1 8 }  { crcTables_4_ce0 mem_ce 1 1 }  { crcTables_4_we0 mem_we 1 1 }  { crcTables_4_d0 mem_din 1 32 } } }
	crcTables_3 { ap_memory {  { crcTables_3_address0 mem_address 1 8 }  { crcTables_3_ce0 mem_ce 1 1 }  { crcTables_3_we0 mem_we 1 1 }  { crcTables_3_d0 mem_din 1 32 } } }
	crcTables_2 { ap_memory {  { crcTables_2_address0 mem_address 1 8 }  { crcTables_2_ce0 mem_ce 1 1 }  { crcTables_2_we0 mem_we 1 1 }  { crcTables_2_d0 mem_din 1 32 } } }
	crcTables_1 { ap_memory {  { crcTables_1_address0 mem_address 1 8 }  { crcTables_1_ce0 mem_ce 1 1 }  { crcTables_1_we0 mem_we 1 1 }  { crcTables_1_d0 mem_din 1 32 } } }
	crcTables { ap_memory {  { crcTables_address0 mem_address 1 8 }  { crcTables_ce0 mem_ce 1 1 }  { crcTables_we0 mem_we 1 1 }  { crcTables_d0 mem_din 1 32 } } }
}
