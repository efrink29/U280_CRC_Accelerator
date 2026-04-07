set moduleName read_input
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
set C_modelName {read_input}
set C_modelType { void 0 }
set C_modelArgList {
	{ gmem0 int 128 regular {axi_master 0}  }
	{ in_r int 64 regular  }
	{ inByte017 int 8 regular {fifo 1 volatile }  }
	{ inByte118 int 8 regular {fifo 1 volatile }  }
	{ inByte219 int 8 regular {fifo 1 volatile }  }
	{ inByte320 int 8 regular {fifo 1 volatile }  }
	{ inByte421 int 8 regular {fifo 1 volatile }  }
	{ inByte522 int 8 regular {fifo 1 volatile }  }
	{ inByte623 int 8 regular {fifo 1 volatile }  }
	{ inByte724 int 8 regular {fifo 1 volatile }  }
	{ inByte825 int 8 regular {fifo 1 volatile }  }
	{ inByte926 int 8 regular {fifo 1 volatile }  }
	{ inByte1027 int 8 regular {fifo 1 volatile }  }
	{ inByte1128 int 8 regular {fifo 1 volatile }  }
	{ inByte1229 int 8 regular {fifo 1 volatile }  }
	{ inByte1330 int 8 regular {fifo 1 volatile }  }
	{ inByte1431 int 8 regular {fifo 1 volatile }  }
	{ inByte1532 int 8 regular {fifo 1 volatile }  }
	{ numChunks int 32 regular  }
	{ chunkSize int 32 regular  }
}
set C_modelArgMapList {[ 
	{ "Name" : "gmem0", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READONLY", "bitSlice":[ {"cElement": [{"cName": "data_in","offset": { "type": "dynamic","port_name": "data_in","bundle": "control"},"direction": "READONLY"},{"cName": "crc_out","offset": { "type": "dynamic","port_name": "crc_out","bundle": "control"},"direction": "WRITEONLY"}]}]} , 
 	{ "Name" : "in_r", "interface" : "wire", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "inByte017", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte118", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte219", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte320", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte421", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte522", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte623", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte724", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte825", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte926", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte1027", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte1128", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte1229", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte1330", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte1431", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "inByte1532", "interface" : "fifo", "bitwidth" : 8, "direction" : "WRITEONLY"} , 
 	{ "Name" : "numChunks", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "chunkSize", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} ]}
# RTL Port declarations: 
set portNum 136
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
	{ inByte017_din sc_out sc_lv 8 signal 2 } 
	{ inByte017_num_data_valid sc_in sc_lv 7 signal 2 } 
	{ inByte017_fifo_cap sc_in sc_lv 7 signal 2 } 
	{ inByte017_full_n sc_in sc_logic 1 signal 2 } 
	{ inByte017_write sc_out sc_logic 1 signal 2 } 
	{ inByte118_din sc_out sc_lv 8 signal 3 } 
	{ inByte118_num_data_valid sc_in sc_lv 7 signal 3 } 
	{ inByte118_fifo_cap sc_in sc_lv 7 signal 3 } 
	{ inByte118_full_n sc_in sc_logic 1 signal 3 } 
	{ inByte118_write sc_out sc_logic 1 signal 3 } 
	{ inByte219_din sc_out sc_lv 8 signal 4 } 
	{ inByte219_num_data_valid sc_in sc_lv 7 signal 4 } 
	{ inByte219_fifo_cap sc_in sc_lv 7 signal 4 } 
	{ inByte219_full_n sc_in sc_logic 1 signal 4 } 
	{ inByte219_write sc_out sc_logic 1 signal 4 } 
	{ inByte320_din sc_out sc_lv 8 signal 5 } 
	{ inByte320_num_data_valid sc_in sc_lv 7 signal 5 } 
	{ inByte320_fifo_cap sc_in sc_lv 7 signal 5 } 
	{ inByte320_full_n sc_in sc_logic 1 signal 5 } 
	{ inByte320_write sc_out sc_logic 1 signal 5 } 
	{ inByte421_din sc_out sc_lv 8 signal 6 } 
	{ inByte421_num_data_valid sc_in sc_lv 7 signal 6 } 
	{ inByte421_fifo_cap sc_in sc_lv 7 signal 6 } 
	{ inByte421_full_n sc_in sc_logic 1 signal 6 } 
	{ inByte421_write sc_out sc_logic 1 signal 6 } 
	{ inByte522_din sc_out sc_lv 8 signal 7 } 
	{ inByte522_num_data_valid sc_in sc_lv 7 signal 7 } 
	{ inByte522_fifo_cap sc_in sc_lv 7 signal 7 } 
	{ inByte522_full_n sc_in sc_logic 1 signal 7 } 
	{ inByte522_write sc_out sc_logic 1 signal 7 } 
	{ inByte623_din sc_out sc_lv 8 signal 8 } 
	{ inByte623_num_data_valid sc_in sc_lv 7 signal 8 } 
	{ inByte623_fifo_cap sc_in sc_lv 7 signal 8 } 
	{ inByte623_full_n sc_in sc_logic 1 signal 8 } 
	{ inByte623_write sc_out sc_logic 1 signal 8 } 
	{ inByte724_din sc_out sc_lv 8 signal 9 } 
	{ inByte724_num_data_valid sc_in sc_lv 7 signal 9 } 
	{ inByte724_fifo_cap sc_in sc_lv 7 signal 9 } 
	{ inByte724_full_n sc_in sc_logic 1 signal 9 } 
	{ inByte724_write sc_out sc_logic 1 signal 9 } 
	{ inByte825_din sc_out sc_lv 8 signal 10 } 
	{ inByte825_num_data_valid sc_in sc_lv 7 signal 10 } 
	{ inByte825_fifo_cap sc_in sc_lv 7 signal 10 } 
	{ inByte825_full_n sc_in sc_logic 1 signal 10 } 
	{ inByte825_write sc_out sc_logic 1 signal 10 } 
	{ inByte926_din sc_out sc_lv 8 signal 11 } 
	{ inByte926_num_data_valid sc_in sc_lv 7 signal 11 } 
	{ inByte926_fifo_cap sc_in sc_lv 7 signal 11 } 
	{ inByte926_full_n sc_in sc_logic 1 signal 11 } 
	{ inByte926_write sc_out sc_logic 1 signal 11 } 
	{ inByte1027_din sc_out sc_lv 8 signal 12 } 
	{ inByte1027_num_data_valid sc_in sc_lv 7 signal 12 } 
	{ inByte1027_fifo_cap sc_in sc_lv 7 signal 12 } 
	{ inByte1027_full_n sc_in sc_logic 1 signal 12 } 
	{ inByte1027_write sc_out sc_logic 1 signal 12 } 
	{ inByte1128_din sc_out sc_lv 8 signal 13 } 
	{ inByte1128_num_data_valid sc_in sc_lv 7 signal 13 } 
	{ inByte1128_fifo_cap sc_in sc_lv 7 signal 13 } 
	{ inByte1128_full_n sc_in sc_logic 1 signal 13 } 
	{ inByte1128_write sc_out sc_logic 1 signal 13 } 
	{ inByte1229_din sc_out sc_lv 8 signal 14 } 
	{ inByte1229_num_data_valid sc_in sc_lv 7 signal 14 } 
	{ inByte1229_fifo_cap sc_in sc_lv 7 signal 14 } 
	{ inByte1229_full_n sc_in sc_logic 1 signal 14 } 
	{ inByte1229_write sc_out sc_logic 1 signal 14 } 
	{ inByte1330_din sc_out sc_lv 8 signal 15 } 
	{ inByte1330_num_data_valid sc_in sc_lv 7 signal 15 } 
	{ inByte1330_fifo_cap sc_in sc_lv 7 signal 15 } 
	{ inByte1330_full_n sc_in sc_logic 1 signal 15 } 
	{ inByte1330_write sc_out sc_logic 1 signal 15 } 
	{ inByte1431_din sc_out sc_lv 8 signal 16 } 
	{ inByte1431_num_data_valid sc_in sc_lv 7 signal 16 } 
	{ inByte1431_fifo_cap sc_in sc_lv 7 signal 16 } 
	{ inByte1431_full_n sc_in sc_logic 1 signal 16 } 
	{ inByte1431_write sc_out sc_logic 1 signal 16 } 
	{ inByte1532_din sc_out sc_lv 8 signal 17 } 
	{ inByte1532_num_data_valid sc_in sc_lv 7 signal 17 } 
	{ inByte1532_fifo_cap sc_in sc_lv 7 signal 17 } 
	{ inByte1532_full_n sc_in sc_logic 1 signal 17 } 
	{ inByte1532_write sc_out sc_logic 1 signal 17 } 
	{ numChunks sc_in sc_lv 32 signal 18 } 
	{ chunkSize sc_in sc_lv 32 signal 19 } 
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
 	{ "name": "inByte017_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte017", "role": "din" }} , 
 	{ "name": "inByte017_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte017", "role": "num_data_valid" }} , 
 	{ "name": "inByte017_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte017", "role": "fifo_cap" }} , 
 	{ "name": "inByte017_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte017", "role": "full_n" }} , 
 	{ "name": "inByte017_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte017", "role": "write" }} , 
 	{ "name": "inByte118_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte118", "role": "din" }} , 
 	{ "name": "inByte118_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte118", "role": "num_data_valid" }} , 
 	{ "name": "inByte118_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte118", "role": "fifo_cap" }} , 
 	{ "name": "inByte118_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte118", "role": "full_n" }} , 
 	{ "name": "inByte118_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte118", "role": "write" }} , 
 	{ "name": "inByte219_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte219", "role": "din" }} , 
 	{ "name": "inByte219_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte219", "role": "num_data_valid" }} , 
 	{ "name": "inByte219_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte219", "role": "fifo_cap" }} , 
 	{ "name": "inByte219_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte219", "role": "full_n" }} , 
 	{ "name": "inByte219_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte219", "role": "write" }} , 
 	{ "name": "inByte320_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte320", "role": "din" }} , 
 	{ "name": "inByte320_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte320", "role": "num_data_valid" }} , 
 	{ "name": "inByte320_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte320", "role": "fifo_cap" }} , 
 	{ "name": "inByte320_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte320", "role": "full_n" }} , 
 	{ "name": "inByte320_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte320", "role": "write" }} , 
 	{ "name": "inByte421_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte421", "role": "din" }} , 
 	{ "name": "inByte421_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte421", "role": "num_data_valid" }} , 
 	{ "name": "inByte421_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte421", "role": "fifo_cap" }} , 
 	{ "name": "inByte421_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte421", "role": "full_n" }} , 
 	{ "name": "inByte421_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte421", "role": "write" }} , 
 	{ "name": "inByte522_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte522", "role": "din" }} , 
 	{ "name": "inByte522_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte522", "role": "num_data_valid" }} , 
 	{ "name": "inByte522_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte522", "role": "fifo_cap" }} , 
 	{ "name": "inByte522_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte522", "role": "full_n" }} , 
 	{ "name": "inByte522_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte522", "role": "write" }} , 
 	{ "name": "inByte623_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte623", "role": "din" }} , 
 	{ "name": "inByte623_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte623", "role": "num_data_valid" }} , 
 	{ "name": "inByte623_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte623", "role": "fifo_cap" }} , 
 	{ "name": "inByte623_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte623", "role": "full_n" }} , 
 	{ "name": "inByte623_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte623", "role": "write" }} , 
 	{ "name": "inByte724_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte724", "role": "din" }} , 
 	{ "name": "inByte724_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte724", "role": "num_data_valid" }} , 
 	{ "name": "inByte724_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte724", "role": "fifo_cap" }} , 
 	{ "name": "inByte724_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte724", "role": "full_n" }} , 
 	{ "name": "inByte724_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte724", "role": "write" }} , 
 	{ "name": "inByte825_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte825", "role": "din" }} , 
 	{ "name": "inByte825_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte825", "role": "num_data_valid" }} , 
 	{ "name": "inByte825_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte825", "role": "fifo_cap" }} , 
 	{ "name": "inByte825_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte825", "role": "full_n" }} , 
 	{ "name": "inByte825_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte825", "role": "write" }} , 
 	{ "name": "inByte926_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte926", "role": "din" }} , 
 	{ "name": "inByte926_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte926", "role": "num_data_valid" }} , 
 	{ "name": "inByte926_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte926", "role": "fifo_cap" }} , 
 	{ "name": "inByte926_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte926", "role": "full_n" }} , 
 	{ "name": "inByte926_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte926", "role": "write" }} , 
 	{ "name": "inByte1027_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1027", "role": "din" }} , 
 	{ "name": "inByte1027_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1027", "role": "num_data_valid" }} , 
 	{ "name": "inByte1027_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1027", "role": "fifo_cap" }} , 
 	{ "name": "inByte1027_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1027", "role": "full_n" }} , 
 	{ "name": "inByte1027_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1027", "role": "write" }} , 
 	{ "name": "inByte1128_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1128", "role": "din" }} , 
 	{ "name": "inByte1128_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1128", "role": "num_data_valid" }} , 
 	{ "name": "inByte1128_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1128", "role": "fifo_cap" }} , 
 	{ "name": "inByte1128_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1128", "role": "full_n" }} , 
 	{ "name": "inByte1128_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1128", "role": "write" }} , 
 	{ "name": "inByte1229_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1229", "role": "din" }} , 
 	{ "name": "inByte1229_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1229", "role": "num_data_valid" }} , 
 	{ "name": "inByte1229_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1229", "role": "fifo_cap" }} , 
 	{ "name": "inByte1229_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1229", "role": "full_n" }} , 
 	{ "name": "inByte1229_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1229", "role": "write" }} , 
 	{ "name": "inByte1330_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1330", "role": "din" }} , 
 	{ "name": "inByte1330_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1330", "role": "num_data_valid" }} , 
 	{ "name": "inByte1330_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1330", "role": "fifo_cap" }} , 
 	{ "name": "inByte1330_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1330", "role": "full_n" }} , 
 	{ "name": "inByte1330_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1330", "role": "write" }} , 
 	{ "name": "inByte1431_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1431", "role": "din" }} , 
 	{ "name": "inByte1431_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1431", "role": "num_data_valid" }} , 
 	{ "name": "inByte1431_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1431", "role": "fifo_cap" }} , 
 	{ "name": "inByte1431_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1431", "role": "full_n" }} , 
 	{ "name": "inByte1431_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1431", "role": "write" }} , 
 	{ "name": "inByte1532_din", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1532", "role": "din" }} , 
 	{ "name": "inByte1532_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1532", "role": "num_data_valid" }} , 
 	{ "name": "inByte1532_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1532", "role": "fifo_cap" }} , 
 	{ "name": "inByte1532_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1532", "role": "full_n" }} , 
 	{ "name": "inByte1532_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1532", "role": "write" }} , 
 	{ "name": "numChunks", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "numChunks", "role": "default" }} , 
 	{ "name": "chunkSize", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "chunkSize", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "3"],
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
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "gmem0", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "in_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "inByte017", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte017", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte118", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte118", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte219", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte219", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte320", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte320", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte421", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte421", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte522", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte522", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte623", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte623", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte724", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte724", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte825", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte825", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte926", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte926", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte1027", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte1027", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte1128", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte1128", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte1229", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte1229", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte1330", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte1330", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte1431", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte1431", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "inByte1532", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_112", "Port" : "inByte1532", "Inst_start_state" : "74", "Inst_end_state" : "75"}]},
			{"Name" : "numChunks", "Type" : "None", "Direction" : "I"},
			{"Name" : "chunkSize", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_read_input_Pipeline_mem_rd_fu_112", "Parent" : "0", "Child" : ["2"],
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
	{"ID" : "2", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_read_input_Pipeline_mem_rd_fu_112.flow_control_loop_pipe_sequential_init_U", "Parent" : "1"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_32s_32s_32_2_1_U58", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
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
		inByte1532 {Type O LastRead -1 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "-1", "Max" : "-1"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	 { m_axi {  { m_axi_gmem0_AWVALID VALID 1 1 }  { m_axi_gmem0_AWREADY READY 0 1 }  { m_axi_gmem0_AWADDR ADDR 1 64 }  { m_axi_gmem0_AWID ID 1 1 }  { m_axi_gmem0_AWLEN SIZE 1 32 }  { m_axi_gmem0_AWSIZE BURST 1 3 }  { m_axi_gmem0_AWBURST LOCK 1 2 }  { m_axi_gmem0_AWLOCK CACHE 1 2 }  { m_axi_gmem0_AWCACHE PROT 1 4 }  { m_axi_gmem0_AWPROT QOS 1 3 }  { m_axi_gmem0_AWQOS REGION 1 4 }  { m_axi_gmem0_AWREGION USER 1 4 }  { m_axi_gmem0_AWUSER DATA 1 1 }  { m_axi_gmem0_WVALID VALID 1 1 }  { m_axi_gmem0_WREADY READY 0 1 }  { m_axi_gmem0_WDATA FIFONUM 1 128 }  { m_axi_gmem0_WSTRB STRB 1 16 }  { m_axi_gmem0_WLAST LAST 1 1 }  { m_axi_gmem0_WID ID 1 1 }  { m_axi_gmem0_WUSER DATA 1 1 }  { m_axi_gmem0_ARVALID VALID 1 1 }  { m_axi_gmem0_ARREADY READY 0 1 }  { m_axi_gmem0_ARADDR ADDR 1 64 }  { m_axi_gmem0_ARID ID 1 1 }  { m_axi_gmem0_ARLEN SIZE 1 32 }  { m_axi_gmem0_ARSIZE BURST 1 3 }  { m_axi_gmem0_ARBURST LOCK 1 2 }  { m_axi_gmem0_ARLOCK CACHE 1 2 }  { m_axi_gmem0_ARCACHE PROT 1 4 }  { m_axi_gmem0_ARPROT QOS 1 3 }  { m_axi_gmem0_ARQOS REGION 1 4 }  { m_axi_gmem0_ARREGION USER 1 4 }  { m_axi_gmem0_ARUSER DATA 1 1 }  { m_axi_gmem0_RVALID VALID 0 1 }  { m_axi_gmem0_RREADY READY 1 1 }  { m_axi_gmem0_RDATA FIFONUM 0 128 }  { m_axi_gmem0_RLAST LAST 0 1 }  { m_axi_gmem0_RID ID 0 1 }  { m_axi_gmem0_RFIFONUM LEN 0 9 }  { m_axi_gmem0_RUSER DATA 0 1 }  { m_axi_gmem0_RRESP RESP 0 2 }  { m_axi_gmem0_BVALID VALID 0 1 }  { m_axi_gmem0_BREADY READY 1 1 }  { m_axi_gmem0_BRESP RESP 0 2 }  { m_axi_gmem0_BID ID 0 1 }  { m_axi_gmem0_BUSER DATA 0 1 } } }
	in_r { ap_none {  { in_r in_data 0 64 } } }
	inByte017 { ap_fifo {  { inByte017_din fifo_port_we 1 8 }  { inByte017_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte017_fifo_cap fifo_update 0 7 }  { inByte017_full_n fifo_status 0 1 }  { inByte017_write fifo_data 1 1 } } }
	inByte118 { ap_fifo {  { inByte118_din fifo_port_we 1 8 }  { inByte118_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte118_fifo_cap fifo_update 0 7 }  { inByte118_full_n fifo_status 0 1 }  { inByte118_write fifo_data 1 1 } } }
	inByte219 { ap_fifo {  { inByte219_din fifo_port_we 1 8 }  { inByte219_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte219_fifo_cap fifo_update 0 7 }  { inByte219_full_n fifo_status 0 1 }  { inByte219_write fifo_data 1 1 } } }
	inByte320 { ap_fifo {  { inByte320_din fifo_port_we 1 8 }  { inByte320_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte320_fifo_cap fifo_update 0 7 }  { inByte320_full_n fifo_status 0 1 }  { inByte320_write fifo_data 1 1 } } }
	inByte421 { ap_fifo {  { inByte421_din fifo_port_we 1 8 }  { inByte421_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte421_fifo_cap fifo_update 0 7 }  { inByte421_full_n fifo_status 0 1 }  { inByte421_write fifo_data 1 1 } } }
	inByte522 { ap_fifo {  { inByte522_din fifo_port_we 1 8 }  { inByte522_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte522_fifo_cap fifo_update 0 7 }  { inByte522_full_n fifo_status 0 1 }  { inByte522_write fifo_data 1 1 } } }
	inByte623 { ap_fifo {  { inByte623_din fifo_port_we 1 8 }  { inByte623_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte623_fifo_cap fifo_update 0 7 }  { inByte623_full_n fifo_status 0 1 }  { inByte623_write fifo_data 1 1 } } }
	inByte724 { ap_fifo {  { inByte724_din fifo_port_we 1 8 }  { inByte724_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte724_fifo_cap fifo_update 0 7 }  { inByte724_full_n fifo_status 0 1 }  { inByte724_write fifo_data 1 1 } } }
	inByte825 { ap_fifo {  { inByte825_din fifo_port_we 1 8 }  { inByte825_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte825_fifo_cap fifo_update 0 7 }  { inByte825_full_n fifo_status 0 1 }  { inByte825_write fifo_data 1 1 } } }
	inByte926 { ap_fifo {  { inByte926_din fifo_port_we 1 8 }  { inByte926_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte926_fifo_cap fifo_update 0 7 }  { inByte926_full_n fifo_status 0 1 }  { inByte926_write fifo_data 1 1 } } }
	inByte1027 { ap_fifo {  { inByte1027_din fifo_port_we 1 8 }  { inByte1027_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte1027_fifo_cap fifo_update 0 7 }  { inByte1027_full_n fifo_status 0 1 }  { inByte1027_write fifo_data 1 1 } } }
	inByte1128 { ap_fifo {  { inByte1128_din fifo_port_we 1 8 }  { inByte1128_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte1128_fifo_cap fifo_update 0 7 }  { inByte1128_full_n fifo_status 0 1 }  { inByte1128_write fifo_data 1 1 } } }
	inByte1229 { ap_fifo {  { inByte1229_din fifo_port_we 1 8 }  { inByte1229_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte1229_fifo_cap fifo_update 0 7 }  { inByte1229_full_n fifo_status 0 1 }  { inByte1229_write fifo_data 1 1 } } }
	inByte1330 { ap_fifo {  { inByte1330_din fifo_port_we 1 8 }  { inByte1330_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte1330_fifo_cap fifo_update 0 7 }  { inByte1330_full_n fifo_status 0 1 }  { inByte1330_write fifo_data 1 1 } } }
	inByte1431 { ap_fifo {  { inByte1431_din fifo_port_we 1 8 }  { inByte1431_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte1431_fifo_cap fifo_update 0 7 }  { inByte1431_full_n fifo_status 0 1 }  { inByte1431_write fifo_data 1 1 } } }
	inByte1532 { ap_fifo {  { inByte1532_din fifo_port_we 1 8 }  { inByte1532_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte1532_fifo_cap fifo_update 0 7 }  { inByte1532_full_n fifo_status 0 1 }  { inByte1532_write fifo_data 1 1 } } }
	numChunks { ap_none {  { numChunks in_data 0 32 } } }
	chunkSize { ap_none {  { chunkSize in_data 0 32 } } }
}
