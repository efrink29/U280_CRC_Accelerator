set moduleName read_input
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 1
set StallSigGenFlag 1
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {read_input}
set C_modelType { void 0 }
set C_modelArgList {
	{ gmem0 int 128 regular {axi_master 0}  }
	{ in_r int 64 regular  }
	{ inByte0 int 8 regular {fifo 1 volatile }  }
	{ inByte1 int 8 regular {fifo 1 volatile }  }
	{ inByte2 int 8 regular {fifo 1 volatile }  }
	{ inByte3 int 8 regular {fifo 1 volatile }  }
	{ inByte4 int 8 regular {fifo 1 volatile }  }
	{ inByte5 int 8 regular {fifo 1 volatile }  }
	{ inByte6 int 8 regular {fifo 1 volatile }  }
	{ inByte7 int 8 regular {fifo 1 volatile }  }
	{ inByte8 int 8 regular {fifo 1 volatile }  }
	{ inByte9 int 8 regular {fifo 1 volatile }  }
	{ inByte10 int 8 regular {fifo 1 volatile }  }
	{ inByte11 int 8 regular {fifo 1 volatile }  }
	{ inByte12 int 8 regular {fifo 1 volatile }  }
	{ inByte13 int 8 regular {fifo 1 volatile }  }
	{ inByte14 int 8 regular {fifo 1 volatile }  }
	{ inByte15 int 8 regular {fifo 1 volatile }  }
	{ numChunks int 32 regular  }
	{ chunkSize int 32 regular  }
	{ numChunks_c12 int 32 regular {fifo 1}  }
	{ chunkSize_c int 32 regular {fifo 1}  }
}
set hasAXIMCache 0
set C_modelArgMapList {[ 
	{ "Name" : "gmem0", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY", "bitSlice":[ {"cElement": [{"cName": "data_in","offset": { "type": "dynamic","port_name": "data_in","bundle": "control"},"direction": "READONLY"},{"cName": "crc_out","offset": { "type": "dynamic","port_name": "crc_out","bundle": "control"},"direction": "WRITEONLY"}]}]} , 
 	{ "Name" : "in_r", "interface" : "wire", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "inByte0", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte1", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte2", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte3", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte4", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte5", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte6", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte7", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte8", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte9", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte10", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte11", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte12", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte13", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte14", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte15", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "numChunks", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "chunkSize", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "numChunks_c12", "interface" : "fifo", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "chunkSize_c", "interface" : "fifo", "bitwidth" : 32, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 149
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_continue sc_in sc_logic 1 continue -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
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
	{ in_r sc_in sc_lv 64 signal 1 } 
	{ inByte0_din sc_out sc_lv 8 signal 2 } 
	{ inByte0_num_data_valid sc_in sc_lv 7 signal 2 } 
	{ inByte0_fifo_cap sc_in sc_lv 7 signal 2 } 
	{ inByte0_full_n sc_in sc_logic 1 signal 2 } 
	{ inByte0_write sc_out sc_logic 1 signal 2 } 
	{ inByte1_din sc_out sc_lv 8 signal 3 } 
	{ inByte1_num_data_valid sc_in sc_lv 7 signal 3 } 
	{ inByte1_fifo_cap sc_in sc_lv 7 signal 3 } 
	{ inByte1_full_n sc_in sc_logic 1 signal 3 } 
	{ inByte1_write sc_out sc_logic 1 signal 3 } 
	{ inByte2_din sc_out sc_lv 8 signal 4 } 
	{ inByte2_num_data_valid sc_in sc_lv 7 signal 4 } 
	{ inByte2_fifo_cap sc_in sc_lv 7 signal 4 } 
	{ inByte2_full_n sc_in sc_logic 1 signal 4 } 
	{ inByte2_write sc_out sc_logic 1 signal 4 } 
	{ inByte3_din sc_out sc_lv 8 signal 5 } 
	{ inByte3_num_data_valid sc_in sc_lv 7 signal 5 } 
	{ inByte3_fifo_cap sc_in sc_lv 7 signal 5 } 
	{ inByte3_full_n sc_in sc_logic 1 signal 5 } 
	{ inByte3_write sc_out sc_logic 1 signal 5 } 
	{ inByte4_din sc_out sc_lv 8 signal 6 } 
	{ inByte4_num_data_valid sc_in sc_lv 7 signal 6 } 
	{ inByte4_fifo_cap sc_in sc_lv 7 signal 6 } 
	{ inByte4_full_n sc_in sc_logic 1 signal 6 } 
	{ inByte4_write sc_out sc_logic 1 signal 6 } 
	{ inByte5_din sc_out sc_lv 8 signal 7 } 
	{ inByte5_num_data_valid sc_in sc_lv 7 signal 7 } 
	{ inByte5_fifo_cap sc_in sc_lv 7 signal 7 } 
	{ inByte5_full_n sc_in sc_logic 1 signal 7 } 
	{ inByte5_write sc_out sc_logic 1 signal 7 } 
	{ inByte6_din sc_out sc_lv 8 signal 8 } 
	{ inByte6_num_data_valid sc_in sc_lv 7 signal 8 } 
	{ inByte6_fifo_cap sc_in sc_lv 7 signal 8 } 
	{ inByte6_full_n sc_in sc_logic 1 signal 8 } 
	{ inByte6_write sc_out sc_logic 1 signal 8 } 
	{ inByte7_din sc_out sc_lv 8 signal 9 } 
	{ inByte7_num_data_valid sc_in sc_lv 7 signal 9 } 
	{ inByte7_fifo_cap sc_in sc_lv 7 signal 9 } 
	{ inByte7_full_n sc_in sc_logic 1 signal 9 } 
	{ inByte7_write sc_out sc_logic 1 signal 9 } 
	{ inByte8_din sc_out sc_lv 8 signal 10 } 
	{ inByte8_num_data_valid sc_in sc_lv 7 signal 10 } 
	{ inByte8_fifo_cap sc_in sc_lv 7 signal 10 } 
	{ inByte8_full_n sc_in sc_logic 1 signal 10 } 
	{ inByte8_write sc_out sc_logic 1 signal 10 } 
	{ inByte9_din sc_out sc_lv 8 signal 11 } 
	{ inByte9_num_data_valid sc_in sc_lv 7 signal 11 } 
	{ inByte9_fifo_cap sc_in sc_lv 7 signal 11 } 
	{ inByte9_full_n sc_in sc_logic 1 signal 11 } 
	{ inByte9_write sc_out sc_logic 1 signal 11 } 
	{ inByte10_din sc_out sc_lv 8 signal 12 } 
	{ inByte10_num_data_valid sc_in sc_lv 7 signal 12 } 
	{ inByte10_fifo_cap sc_in sc_lv 7 signal 12 } 
	{ inByte10_full_n sc_in sc_logic 1 signal 12 } 
	{ inByte10_write sc_out sc_logic 1 signal 12 } 
	{ inByte11_din sc_out sc_lv 8 signal 13 } 
	{ inByte11_num_data_valid sc_in sc_lv 7 signal 13 } 
	{ inByte11_fifo_cap sc_in sc_lv 7 signal 13 } 
	{ inByte11_full_n sc_in sc_logic 1 signal 13 } 
	{ inByte11_write sc_out sc_logic 1 signal 13 } 
	{ inByte12_din sc_out sc_lv 8 signal 14 } 
	{ inByte12_num_data_valid sc_in sc_lv 7 signal 14 } 
	{ inByte12_fifo_cap sc_in sc_lv 7 signal 14 } 
	{ inByte12_full_n sc_in sc_logic 1 signal 14 } 
	{ inByte12_write sc_out sc_logic 1 signal 14 } 
	{ inByte13_din sc_out sc_lv 8 signal 15 } 
	{ inByte13_num_data_valid sc_in sc_lv 7 signal 15 } 
	{ inByte13_fifo_cap sc_in sc_lv 7 signal 15 } 
	{ inByte13_full_n sc_in sc_logic 1 signal 15 } 
	{ inByte13_write sc_out sc_logic 1 signal 15 } 
	{ inByte14_din sc_out sc_lv 8 signal 16 } 
	{ inByte14_num_data_valid sc_in sc_lv 7 signal 16 } 
	{ inByte14_fifo_cap sc_in sc_lv 7 signal 16 } 
	{ inByte14_full_n sc_in sc_logic 1 signal 16 } 
	{ inByte14_write sc_out sc_logic 1 signal 16 } 
	{ inByte15_din sc_out sc_lv 8 signal 17 } 
	{ inByte15_num_data_valid sc_in sc_lv 7 signal 17 } 
	{ inByte15_fifo_cap sc_in sc_lv 7 signal 17 } 
	{ inByte15_full_n sc_in sc_logic 1 signal 17 } 
	{ inByte15_write sc_out sc_logic 1 signal 17 } 
	{ numChunks sc_in sc_lv 32 signal 18 } 
	{ chunkSize sc_in sc_lv 32 signal 19 } 
	{ numChunks_c12_din sc_out sc_lv 32 signal 20 } 
	{ numChunks_c12_num_data_valid sc_in sc_lv 2 signal 20 } 
	{ numChunks_c12_fifo_cap sc_in sc_lv 2 signal 20 } 
	{ numChunks_c12_full_n sc_in sc_logic 1 signal 20 } 
	{ numChunks_c12_write sc_out sc_logic 1 signal 20 } 
	{ chunkSize_c_din sc_out sc_lv 32 signal 21 } 
	{ chunkSize_c_num_data_valid sc_in sc_lv 2 signal 21 } 
	{ chunkSize_c_fifo_cap sc_in sc_lv 2 signal 21 } 
	{ chunkSize_c_full_n sc_in sc_logic 1 signal 21 } 
	{ chunkSize_c_write sc_out sc_logic 1 signal 21 } 
	{ ap_ext_blocking_n sc_out sc_logic 1 signal -1 } 
	{ ap_str_blocking_n sc_out sc_logic 1 signal -1 } 
	{ ap_int_blocking_n sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_continue", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "continue", "bundle":{"name": "ap_continue", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
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
 	{ "name": "in_r", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "in_r", "role": "default" }} , 
 	{ "name": "inByte0_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte0", "role": "din" }} , 
 	{ "name": "inByte0_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte0", "role": "num_data_valid" }} , 
 	{ "name": "inByte0_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte0", "role": "fifo_cap" }} , 
 	{ "name": "inByte0_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte0", "role": "full_n" }} , 
 	{ "name": "inByte0_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte0", "role": "write" }} , 
 	{ "name": "inByte1_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1", "role": "din" }} , 
 	{ "name": "inByte1_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1", "role": "num_data_valid" }} , 
 	{ "name": "inByte1_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1", "role": "fifo_cap" }} , 
 	{ "name": "inByte1_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1", "role": "full_n" }} , 
 	{ "name": "inByte1_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1", "role": "write" }} , 
 	{ "name": "inByte2_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte2", "role": "din" }} , 
 	{ "name": "inByte2_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte2", "role": "num_data_valid" }} , 
 	{ "name": "inByte2_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte2", "role": "fifo_cap" }} , 
 	{ "name": "inByte2_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte2", "role": "full_n" }} , 
 	{ "name": "inByte2_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte2", "role": "write" }} , 
 	{ "name": "inByte3_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte3", "role": "din" }} , 
 	{ "name": "inByte3_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte3", "role": "num_data_valid" }} , 
 	{ "name": "inByte3_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte3", "role": "fifo_cap" }} , 
 	{ "name": "inByte3_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte3", "role": "full_n" }} , 
 	{ "name": "inByte3_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte3", "role": "write" }} , 
 	{ "name": "inByte4_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte4", "role": "din" }} , 
 	{ "name": "inByte4_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte4", "role": "num_data_valid" }} , 
 	{ "name": "inByte4_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte4", "role": "fifo_cap" }} , 
 	{ "name": "inByte4_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte4", "role": "full_n" }} , 
 	{ "name": "inByte4_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte4", "role": "write" }} , 
 	{ "name": "inByte5_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte5", "role": "din" }} , 
 	{ "name": "inByte5_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte5", "role": "num_data_valid" }} , 
 	{ "name": "inByte5_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte5", "role": "fifo_cap" }} , 
 	{ "name": "inByte5_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte5", "role": "full_n" }} , 
 	{ "name": "inByte5_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte5", "role": "write" }} , 
 	{ "name": "inByte6_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte6", "role": "din" }} , 
 	{ "name": "inByte6_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte6", "role": "num_data_valid" }} , 
 	{ "name": "inByte6_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte6", "role": "fifo_cap" }} , 
 	{ "name": "inByte6_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte6", "role": "full_n" }} , 
 	{ "name": "inByte6_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte6", "role": "write" }} , 
 	{ "name": "inByte7_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte7", "role": "din" }} , 
 	{ "name": "inByte7_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte7", "role": "num_data_valid" }} , 
 	{ "name": "inByte7_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte7", "role": "fifo_cap" }} , 
 	{ "name": "inByte7_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte7", "role": "full_n" }} , 
 	{ "name": "inByte7_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte7", "role": "write" }} , 
 	{ "name": "inByte8_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte8", "role": "din" }} , 
 	{ "name": "inByte8_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte8", "role": "num_data_valid" }} , 
 	{ "name": "inByte8_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte8", "role": "fifo_cap" }} , 
 	{ "name": "inByte8_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte8", "role": "full_n" }} , 
 	{ "name": "inByte8_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte8", "role": "write" }} , 
 	{ "name": "inByte9_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte9", "role": "din" }} , 
 	{ "name": "inByte9_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte9", "role": "num_data_valid" }} , 
 	{ "name": "inByte9_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte9", "role": "fifo_cap" }} , 
 	{ "name": "inByte9_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte9", "role": "full_n" }} , 
 	{ "name": "inByte9_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte9", "role": "write" }} , 
 	{ "name": "inByte10_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte10", "role": "din" }} , 
 	{ "name": "inByte10_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte10", "role": "num_data_valid" }} , 
 	{ "name": "inByte10_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte10", "role": "fifo_cap" }} , 
 	{ "name": "inByte10_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte10", "role": "full_n" }} , 
 	{ "name": "inByte10_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte10", "role": "write" }} , 
 	{ "name": "inByte11_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte11", "role": "din" }} , 
 	{ "name": "inByte11_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte11", "role": "num_data_valid" }} , 
 	{ "name": "inByte11_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte11", "role": "fifo_cap" }} , 
 	{ "name": "inByte11_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte11", "role": "full_n" }} , 
 	{ "name": "inByte11_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte11", "role": "write" }} , 
 	{ "name": "inByte12_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte12", "role": "din" }} , 
 	{ "name": "inByte12_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte12", "role": "num_data_valid" }} , 
 	{ "name": "inByte12_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte12", "role": "fifo_cap" }} , 
 	{ "name": "inByte12_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte12", "role": "full_n" }} , 
 	{ "name": "inByte12_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte12", "role": "write" }} , 
 	{ "name": "inByte13_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte13", "role": "din" }} , 
 	{ "name": "inByte13_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte13", "role": "num_data_valid" }} , 
 	{ "name": "inByte13_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte13", "role": "fifo_cap" }} , 
 	{ "name": "inByte13_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte13", "role": "full_n" }} , 
 	{ "name": "inByte13_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte13", "role": "write" }} , 
 	{ "name": "inByte14_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte14", "role": "din" }} , 
 	{ "name": "inByte14_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte14", "role": "num_data_valid" }} , 
 	{ "name": "inByte14_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte14", "role": "fifo_cap" }} , 
 	{ "name": "inByte14_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte14", "role": "full_n" }} , 
 	{ "name": "inByte14_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte14", "role": "write" }} , 
 	{ "name": "inByte15_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte15", "role": "din" }} , 
 	{ "name": "inByte15_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte15", "role": "num_data_valid" }} , 
 	{ "name": "inByte15_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte15", "role": "fifo_cap" }} , 
 	{ "name": "inByte15_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte15", "role": "full_n" }} , 
 	{ "name": "inByte15_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte15", "role": "write" }} , 
 	{ "name": "numChunks", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "numChunks", "role": "default" }} , 
 	{ "name": "chunkSize", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "chunkSize", "role": "default" }} , 
 	{ "name": "numChunks_c12_din", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "numChunks_c12", "role": "din" }} , 
 	{ "name": "numChunks_c12_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "numChunks_c12", "role": "num_data_valid" }} , 
 	{ "name": "numChunks_c12_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "numChunks_c12", "role": "fifo_cap" }} , 
 	{ "name": "numChunks_c12_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "numChunks_c12", "role": "full_n" }} , 
 	{ "name": "numChunks_c12_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "numChunks_c12", "role": "write" }} , 
 	{ "name": "chunkSize_c_din", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "chunkSize_c", "role": "din" }} , 
 	{ "name": "chunkSize_c_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "chunkSize_c", "role": "num_data_valid" }} , 
 	{ "name": "chunkSize_c_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "chunkSize_c", "role": "fifo_cap" }} , 
 	{ "name": "chunkSize_c_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "chunkSize_c", "role": "full_n" }} , 
 	{ "name": "chunkSize_c_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "chunkSize_c", "role": "write" }} , 
 	{ "name": "ap_ext_blocking_n", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ap_ext_blocking_n", "role": "default" }} , 
 	{ "name": "ap_str_blocking_n", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ap_str_blocking_n", "role": "default" }} , 
 	{ "name": "ap_int_blocking_n", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ap_int_blocking_n", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "3"],
		"CDFG" : "read_input",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "79", "EstimateLatencyMax" : "134217806",
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
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "gmem0", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "in_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "inByte0", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte0", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte1", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte1", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte2", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte2", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte3", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte3", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte4", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte4", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte5", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte5", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte6", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte6", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte7", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte7", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte8", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte8", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte9", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte9", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte10", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte10", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte11", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte11", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte12", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte12", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte13", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte13", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte14", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte14", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte15", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte15", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "numChunks", "Type" : "None", "Direction" : "I"},
			{"Name" : "chunkSize", "Type" : "None", "Direction" : "I"},
			{"Name" : "numChunks_c12", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "2", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "numChunks_c12_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "chunkSize_c", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "2", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "chunkSize_c_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_read_input_Pipeline_mem_rd_fu_142", "Parent" : "0", "Child" : ["2"],
		"CDFG" : "read_input_Pipeline_mem_rd",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "3", "EstimateLatencyMax" : "134217730",
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
			{"Name" : "sext_ln287", "Type" : "None", "Direction" : "I"},
			{"Name" : "loop_count", "Type" : "None", "Direction" : "I"},
			{"Name" : "inByte0", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte0_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte1_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte2", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte2_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte3", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte3_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte4", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte4_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte5", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte5_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte6", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte6_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte7", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte7_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte8", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte8_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte9", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte9_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte10", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte10_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte11", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte11_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte12", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte12_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte13", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte13_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte14", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte14_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte15", "Type" : "Fifo", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "inByte15_blk_n", "Type" : "RtlSignal"}]}],
		"Loop" : [
			{"Name" : "mem_rd", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter2", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter2", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "2", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_read_input_Pipeline_mem_rd_fu_142.flow_control_loop_pipe_sequential_init_U", "Parent" : "1"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_32s_32s_32_2_1_U62", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	read_input {
		gmem0 {Type I LastRead 4 FirstWrite -1}
		in_r {Type I LastRead 4 FirstWrite -1}
		inByte0 {Type O LastRead -1 FirstWrite 2}
		inByte1 {Type O LastRead -1 FirstWrite 2}
		inByte2 {Type O LastRead -1 FirstWrite 2}
		inByte3 {Type O LastRead -1 FirstWrite 2}
		inByte4 {Type O LastRead -1 FirstWrite 2}
		inByte5 {Type O LastRead -1 FirstWrite 2}
		inByte6 {Type O LastRead -1 FirstWrite 2}
		inByte7 {Type O LastRead -1 FirstWrite 2}
		inByte8 {Type O LastRead -1 FirstWrite 2}
		inByte9 {Type O LastRead -1 FirstWrite 2}
		inByte10 {Type O LastRead -1 FirstWrite 2}
		inByte11 {Type O LastRead -1 FirstWrite 2}
		inByte12 {Type O LastRead -1 FirstWrite 2}
		inByte13 {Type O LastRead -1 FirstWrite 2}
		inByte14 {Type O LastRead -1 FirstWrite 2}
		inByte15 {Type O LastRead -1 FirstWrite 2}
		numChunks {Type I LastRead 0 FirstWrite -1}
		chunkSize {Type I LastRead 0 FirstWrite -1}
		numChunks_c12 {Type O LastRead -1 FirstWrite 0}
		chunkSize_c {Type O LastRead -1 FirstWrite 0}}
	read_input_Pipeline_mem_rd {
		gmem0 {Type I LastRead 1 FirstWrite -1}
		sext_ln287 {Type I LastRead 0 FirstWrite -1}
		loop_count {Type I LastRead 0 FirstWrite -1}
		inByte0 {Type O LastRead -1 FirstWrite 2}
		inByte1 {Type O LastRead -1 FirstWrite 2}
		inByte2 {Type O LastRead -1 FirstWrite 2}
		inByte3 {Type O LastRead -1 FirstWrite 2}
		inByte4 {Type O LastRead -1 FirstWrite 2}
		inByte5 {Type O LastRead -1 FirstWrite 2}
		inByte6 {Type O LastRead -1 FirstWrite 2}
		inByte7 {Type O LastRead -1 FirstWrite 2}
		inByte8 {Type O LastRead -1 FirstWrite 2}
		inByte9 {Type O LastRead -1 FirstWrite 2}
		inByte10 {Type O LastRead -1 FirstWrite 2}
		inByte11 {Type O LastRead -1 FirstWrite 2}
		inByte12 {Type O LastRead -1 FirstWrite 2}
		inByte13 {Type O LastRead -1 FirstWrite 2}
		inByte14 {Type O LastRead -1 FirstWrite 2}
		inByte15 {Type O LastRead -1 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "79", "Max" : "134217806"}
	, {"Name" : "Interval", "Min" : "79", "Max" : "134217806"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	 { m_axi {  { m_axi_gmem0_AWVALID VALID 1 1 }  { m_axi_gmem0_AWREADY READY 0 1 }  { m_axi_gmem0_AWADDR ADDR 1 64 }  { m_axi_gmem0_AWID ID 1 1 }  { m_axi_gmem0_AWLEN SIZE 1 32 }  { m_axi_gmem0_AWSIZE BURST 1 3 }  { m_axi_gmem0_AWBURST LOCK 1 2 }  { m_axi_gmem0_AWLOCK CACHE 1 2 }  { m_axi_gmem0_AWCACHE PROT 1 4 }  { m_axi_gmem0_AWPROT QOS 1 3 }  { m_axi_gmem0_AWQOS REGION 1 4 }  { m_axi_gmem0_AWREGION USER 1 4 }  { m_axi_gmem0_AWUSER DATA 1 1 }  { m_axi_gmem0_WVALID VALID 1 1 }  { m_axi_gmem0_WREADY READY 0 1 }  { m_axi_gmem0_WDATA FIFONUM 1 128 }  { m_axi_gmem0_WSTRB STRB 1 16 }  { m_axi_gmem0_WLAST LAST 1 1 }  { m_axi_gmem0_WID ID 1 1 }  { m_axi_gmem0_WUSER DATA 1 1 }  { m_axi_gmem0_ARVALID VALID 1 1 }  { m_axi_gmem0_ARREADY READY 0 1 }  { m_axi_gmem0_ARADDR ADDR 1 64 }  { m_axi_gmem0_ARID ID 1 1 }  { m_axi_gmem0_ARLEN SIZE 1 32 }  { m_axi_gmem0_ARSIZE BURST 1 3 }  { m_axi_gmem0_ARBURST LOCK 1 2 }  { m_axi_gmem0_ARLOCK CACHE 1 2 }  { m_axi_gmem0_ARCACHE PROT 1 4 }  { m_axi_gmem0_ARPROT QOS 1 3 }  { m_axi_gmem0_ARQOS REGION 1 4 }  { m_axi_gmem0_ARREGION USER 1 4 }  { m_axi_gmem0_ARUSER DATA 1 1 }  { m_axi_gmem0_RVALID VALID 0 1 }  { m_axi_gmem0_RREADY READY 1 1 }  { m_axi_gmem0_RDATA FIFONUM 0 128 }  { m_axi_gmem0_RLAST LAST 0 1 }  { m_axi_gmem0_RID ID 0 1 }  { m_axi_gmem0_RFIFONUM LEN 0 9 }  { m_axi_gmem0_RUSER DATA 0 1 }  { m_axi_gmem0_RRESP RESP 0 2 }  { m_axi_gmem0_BVALID VALID 0 1 }  { m_axi_gmem0_BREADY READY 1 1 }  { m_axi_gmem0_BRESP RESP 0 2 }  { m_axi_gmem0_BID ID 0 1 }  { m_axi_gmem0_BUSER DATA 0 1 } } }
	in_r { ap_none {  { in_r in_data 0 64 } } }
	inByte0 { ap_fifo {  { inByte0_din fifo_port_we 1 8 }  { inByte0_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte0_fifo_cap fifo_update 0 7 }  { inByte0_full_n fifo_status 0 1 }  { inByte0_write fifo_data 1 1 } } }
	inByte1 { ap_fifo {  { inByte1_din fifo_port_we 1 8 }  { inByte1_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte1_fifo_cap fifo_update 0 7 }  { inByte1_full_n fifo_status 0 1 }  { inByte1_write fifo_data 1 1 } } }
	inByte2 { ap_fifo {  { inByte2_din fifo_port_we 1 8 }  { inByte2_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte2_fifo_cap fifo_update 0 7 }  { inByte2_full_n fifo_status 0 1 }  { inByte2_write fifo_data 1 1 } } }
	inByte3 { ap_fifo {  { inByte3_din fifo_port_we 1 8 }  { inByte3_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte3_fifo_cap fifo_update 0 7 }  { inByte3_full_n fifo_status 0 1 }  { inByte3_write fifo_data 1 1 } } }
	inByte4 { ap_fifo {  { inByte4_din fifo_port_we 1 8 }  { inByte4_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte4_fifo_cap fifo_update 0 7 }  { inByte4_full_n fifo_status 0 1 }  { inByte4_write fifo_data 1 1 } } }
	inByte5 { ap_fifo {  { inByte5_din fifo_port_we 1 8 }  { inByte5_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte5_fifo_cap fifo_update 0 7 }  { inByte5_full_n fifo_status 0 1 }  { inByte5_write fifo_data 1 1 } } }
	inByte6 { ap_fifo {  { inByte6_din fifo_port_we 1 8 }  { inByte6_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte6_fifo_cap fifo_update 0 7 }  { inByte6_full_n fifo_status 0 1 }  { inByte6_write fifo_data 1 1 } } }
	inByte7 { ap_fifo {  { inByte7_din fifo_port_we 1 8 }  { inByte7_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte7_fifo_cap fifo_update 0 7 }  { inByte7_full_n fifo_status 0 1 }  { inByte7_write fifo_data 1 1 } } }
	inByte8 { ap_fifo {  { inByte8_din fifo_port_we 1 8 }  { inByte8_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte8_fifo_cap fifo_update 0 7 }  { inByte8_full_n fifo_status 0 1 }  { inByte8_write fifo_data 1 1 } } }
	inByte9 { ap_fifo {  { inByte9_din fifo_port_we 1 8 }  { inByte9_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte9_fifo_cap fifo_update 0 7 }  { inByte9_full_n fifo_status 0 1 }  { inByte9_write fifo_data 1 1 } } }
	inByte10 { ap_fifo {  { inByte10_din fifo_port_we 1 8 }  { inByte10_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte10_fifo_cap fifo_update 0 7 }  { inByte10_full_n fifo_status 0 1 }  { inByte10_write fifo_data 1 1 } } }
	inByte11 { ap_fifo {  { inByte11_din fifo_port_we 1 8 }  { inByte11_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte11_fifo_cap fifo_update 0 7 }  { inByte11_full_n fifo_status 0 1 }  { inByte11_write fifo_data 1 1 } } }
	inByte12 { ap_fifo {  { inByte12_din fifo_port_we 1 8 }  { inByte12_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte12_fifo_cap fifo_update 0 7 }  { inByte12_full_n fifo_status 0 1 }  { inByte12_write fifo_data 1 1 } } }
	inByte13 { ap_fifo {  { inByte13_din fifo_port_we 1 8 }  { inByte13_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte13_fifo_cap fifo_update 0 7 }  { inByte13_full_n fifo_status 0 1 }  { inByte13_write fifo_data 1 1 } } }
	inByte14 { ap_fifo {  { inByte14_din fifo_port_we 1 8 }  { inByte14_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte14_fifo_cap fifo_update 0 7 }  { inByte14_full_n fifo_status 0 1 }  { inByte14_write fifo_data 1 1 } } }
	inByte15 { ap_fifo {  { inByte15_din fifo_port_we 1 8 }  { inByte15_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte15_fifo_cap fifo_update 0 7 }  { inByte15_full_n fifo_status 0 1 }  { inByte15_write fifo_data 1 1 } } }
	numChunks { ap_none {  { numChunks in_data 0 32 } } }
	chunkSize { ap_none {  { chunkSize in_data 0 32 } } }
	numChunks_c12 { ap_fifo {  { numChunks_c12_din fifo_port_we 1 32 }  { numChunks_c12_num_data_valid fifo_status_num_data_valid 0 2 }  { numChunks_c12_fifo_cap fifo_update 0 2 }  { numChunks_c12_full_n fifo_status 0 1 }  { numChunks_c12_write fifo_data 1 1 } } }
	chunkSize_c { ap_fifo {  { chunkSize_c_din fifo_port_we 1 32 }  { chunkSize_c_num_data_valid fifo_status_num_data_valid 0 2 }  { chunkSize_c_fifo_cap fifo_update 0 2 }  { chunkSize_c_full_n fifo_status 0 1 }  { chunkSize_c_write fifo_data 1 1 } } }
}
