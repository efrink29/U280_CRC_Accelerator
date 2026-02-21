set moduleName calculate_crc
set isTopModule 1
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_chain
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {calculate_crc}
set C_modelType { void 0 }
set C_modelArgList {
	{ gmem0 int 128 regular {axi_master 2}  }
	{ gmem1 int 512 regular {axi_master 0}  }
	{ data_in int 64 regular {axi_slave 0}  }
	{ crc_out int 64 regular {axi_slave 0}  }
	{ tables int 64 regular {axi_slave 0}  }
	{ numChunks int 32 regular {axi_slave 0}  }
	{ chunkSize int 32 regular {axi_slave 0}  }
	{ crc_size int 32 regular {axi_slave 0}  }
	{ init_value int 32 regular {axi_slave 0}  }
}
set hasAXIMCache 0
set C_modelArgMapList {[ 
	{ "Name" : "gmem0", "interface" : "axi_master", "bitwidth" : 128, "direction" : "READWRITE", "bitSlice":[ {"cElement": [{"cName": "data_in","offset": { "type": "dynamic","port_name": "data_in","bundle": "control"},"direction": "READONLY"},{"cName": "crc_out","offset": { "type": "dynamic","port_name": "crc_out","bundle": "control"},"direction": "WRITEONLY"}]}]} , 
 	{ "Name" : "gmem1", "interface" : "axi_master", "bitwidth" : 512, "direction" : "READONLY", "bitSlice":[ {"cElement": [{"cName": "tables","offset": { "type": "dynamic","port_name": "tables","bundle": "control"},"direction": "READONLY"}]}]} , 
 	{ "Name" : "data_in", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 64, "direction" : "READONLY", "offset" : {"in":16}, "offset_end" : {"in":27}} , 
 	{ "Name" : "crc_out", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 64, "direction" : "READONLY", "offset" : {"in":28}, "offset_end" : {"in":39}} , 
 	{ "Name" : "tables", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 64, "direction" : "READONLY", "offset" : {"in":40}, "offset_end" : {"in":51}} , 
 	{ "Name" : "numChunks", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":52}, "offset_end" : {"in":59}} , 
 	{ "Name" : "chunkSize", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":60}, "offset_end" : {"in":67}} , 
 	{ "Name" : "crc_size", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":68}, "offset_end" : {"in":75}} , 
 	{ "Name" : "init_value", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":76}, "offset_end" : {"in":83}} ]}
# RTL Port declarations: 
set portNum 110
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst_n sc_in sc_logic 1 reset -1 active_low_sync } 
	{ m_axi_gmem0_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem0_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem0_AWADDR sc_out sc_lv 64 signal 0 } 
	{ m_axi_gmem0_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem0_AWLEN sc_out sc_lv 8 signal 0 } 
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
	{ m_axi_gmem0_ARLEN sc_out sc_lv 8 signal 0 } 
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
	{ m_axi_gmem0_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem0_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem0_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem0_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem0_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem0_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem0_BUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem1_AWVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem1_AWREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem1_AWADDR sc_out sc_lv 64 signal 1 } 
	{ m_axi_gmem1_AWID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem1_AWLEN sc_out sc_lv 8 signal 1 } 
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
	{ m_axi_gmem1_ARLEN sc_out sc_lv 8 signal 1 } 
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
	{ m_axi_gmem1_RUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem1_RRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem1_BVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem1_BREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem1_BRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem1_BID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem1_BUSER sc_in sc_lv 1 signal 1 } 
	{ s_axi_control_AWVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_AWREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_AWADDR sc_in sc_lv 7 signal -1 } 
	{ s_axi_control_WVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_WREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_WDATA sc_in sc_lv 32 signal -1 } 
	{ s_axi_control_WSTRB sc_in sc_lv 4 signal -1 } 
	{ s_axi_control_ARVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_ARREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_ARADDR sc_in sc_lv 7 signal -1 } 
	{ s_axi_control_RVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_RREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_RDATA sc_out sc_lv 32 signal -1 } 
	{ s_axi_control_RRESP sc_out sc_lv 2 signal -1 } 
	{ s_axi_control_BVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_BREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_BRESP sc_out sc_lv 2 signal -1 } 
	{ interrupt sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "s_axi_control_AWADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "control", "role": "AWADDR" },"address":[{"name":"calculate_crc","role":"start","value":"0","valid_bit":"0"},{"name":"calculate_crc","role":"continue","value":"0","valid_bit":"4"},{"name":"calculate_crc","role":"auto_start","value":"0","valid_bit":"7"},{"name":"data_in","role":"data","value":"16"},{"name":"crc_out","role":"data","value":"28"},{"name":"tables","role":"data","value":"40"},{"name":"numChunks","role":"data","value":"52"},{"name":"chunkSize","role":"data","value":"60"},{"name":"crc_size","role":"data","value":"68"},{"name":"init_value","role":"data","value":"76"}] },
	{ "name": "s_axi_control_AWVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWVALID" } },
	{ "name": "s_axi_control_AWREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWREADY" } },
	{ "name": "s_axi_control_WVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WVALID" } },
	{ "name": "s_axi_control_WREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WREADY" } },
	{ "name": "s_axi_control_WDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "WDATA" } },
	{ "name": "s_axi_control_WSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "control", "role": "WSTRB" } },
	{ "name": "s_axi_control_ARADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "control", "role": "ARADDR" },"address":[{"name":"calculate_crc","role":"start","value":"0","valid_bit":"0"},{"name":"calculate_crc","role":"done","value":"0","valid_bit":"1"},{"name":"calculate_crc","role":"idle","value":"0","valid_bit":"2"},{"name":"calculate_crc","role":"ready","value":"0","valid_bit":"3"},{"name":"calculate_crc","role":"auto_start","value":"0","valid_bit":"7"}] },
	{ "name": "s_axi_control_ARVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARVALID" } },
	{ "name": "s_axi_control_ARREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARREADY" } },
	{ "name": "s_axi_control_RVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RVALID" } },
	{ "name": "s_axi_control_RREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RREADY" } },
	{ "name": "s_axi_control_RDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "RDATA" } },
	{ "name": "s_axi_control_RRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "RRESP" } },
	{ "name": "s_axi_control_BVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BVALID" } },
	{ "name": "s_axi_control_BREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BREADY" } },
	{ "name": "s_axi_control_BRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "BRESP" } },
	{ "name": "interrupt", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "interrupt" } }, 
 	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst_n", "role": "default" }} , 
 	{ "name": "m_axi_gmem0_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem0_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem0_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem0", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem0_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem0_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem0", "role": "AWLEN" }} , 
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
 	{ "name": "m_axi_gmem0_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem0", "role": "ARLEN" }} , 
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
 	{ "name": "m_axi_gmem0_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem0_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem0", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem0_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem0_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem0_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem0", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem0_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "BID" }} , 
 	{ "name": "m_axi_gmem0_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "BUSER" }} , 
 	{ "name": "m_axi_gmem1_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem1_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem1_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem1", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem1_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem1_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem1", "role": "AWLEN" }} , 
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
 	{ "name": "m_axi_gmem1_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem1", "role": "ARLEN" }} , 
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
 	{ "name": "m_axi_gmem1_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem1_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem1", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem1_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem1_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem1_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem1", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem1_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "BID" }} , 
 	{ "name": "m_axi_gmem1_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "BUSER" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "58", "59", "60"],
		"CDFG" : "calculate_crc",
		"Protocol" : "ap_ctrl_chain",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "-1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "1",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "gmem0", "Type" : "MAXI", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_fu_122", "Port" : "gmem0", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "gmem1", "Type" : "MAXI", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_process_crc_fu_122", "Port" : "gmem1", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "data_in", "Type" : "None", "Direction" : "I"},
			{"Name" : "crc_out", "Type" : "None", "Direction" : "I"},
			{"Name" : "tables", "Type" : "None", "Direction" : "I"},
			{"Name" : "numChunks", "Type" : "None", "Direction" : "I"},
			{"Name" : "chunkSize", "Type" : "None", "Direction" : "I"},
			{"Name" : "crc_size", "Type" : "None", "Direction" : "I"},
			{"Name" : "init_value", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122", "Parent" : "0", "Child" : ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "22", "26", "31", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57"],
		"CDFG" : "process_crc",
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
			{"ID" : "18", "Name" : "entry_proc_U0"},
			{"ID" : "19", "Name" : "process_crc_Loop_init_lut_proc_U0"},
			{"ID" : "22", "Name" : "read_input_U0"}],
		"OutputProcess" : [
			{"ID" : "31", "Name" : "write_output_U0"}],
		"Port" : [
			{"Name" : "gmem0", "Type" : "MAXI", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "22", "SubInstance" : "read_input_U0", "Port" : "gmem0"},
					{"ID" : "31", "SubInstance" : "write_output_U0", "Port" : "gmem0"}]},
			{"Name" : "data_in", "Type" : "None", "Direction" : "I"},
			{"Name" : "crc_out", "Type" : "None", "Direction" : "I"},
			{"Name" : "gmem1", "Type" : "MAXI", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "19", "SubInstance" : "process_crc_Loop_init_lut_proc_U0", "Port" : "gmem1"}]},
			{"Name" : "tables", "Type" : "None", "Direction" : "I"},
			{"Name" : "numChunks", "Type" : "None", "Direction" : "I"},
			{"Name" : "chunkSize", "Type" : "None", "Direction" : "I"},
			{"Name" : "crc_size", "Type" : "None", "Direction" : "I"},
			{"Name" : "init_value", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "2", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.crcTables_U", "Parent" : "1"},
	{"ID" : "3", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.crcTables_1_U", "Parent" : "1"},
	{"ID" : "4", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.crcTables_2_U", "Parent" : "1"},
	{"ID" : "5", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.crcTables_3_U", "Parent" : "1"},
	{"ID" : "6", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.crcTables_4_U", "Parent" : "1"},
	{"ID" : "7", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.crcTables_5_U", "Parent" : "1"},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.crcTables_6_U", "Parent" : "1"},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.crcTables_7_U", "Parent" : "1"},
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.crcTables_8_U", "Parent" : "1"},
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.crcTables_9_U", "Parent" : "1"},
	{"ID" : "12", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.crcTables_10_U", "Parent" : "1"},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.crcTables_11_U", "Parent" : "1"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.crcTables_12_U", "Parent" : "1"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.crcTables_13_U", "Parent" : "1"},
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.crcTables_14_U", "Parent" : "1"},
	{"ID" : "17", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.crcTables_15_U", "Parent" : "1"},
	{"ID" : "18", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.entry_proc_U0", "Parent" : "1",
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
			{"Name" : "crc_out_c", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["31"], "DependentChan" : "34", "DependentChanDepth" : "4", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "crc_out_c_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "crc_size", "Type" : "None", "Direction" : "I"},
			{"Name" : "crc_size_c", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "35", "DependentChanDepth" : "3", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "crc_size_c_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "init_value", "Type" : "None", "Direction" : "I"},
			{"Name" : "init_value_c", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "36", "DependentChanDepth" : "3", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "init_value_c_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.process_crc_Loop_init_lut_proc_U0", "Parent" : "1", "Child" : ["20"],
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
					{"ID" : "20", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "gmem1", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_15", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "17",
				"SubConnect" : [
					{"ID" : "20", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_15", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_14", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "16",
				"SubConnect" : [
					{"ID" : "20", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_14", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_13", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "15",
				"SubConnect" : [
					{"ID" : "20", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_13", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_12", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "14",
				"SubConnect" : [
					{"ID" : "20", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_12", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_11", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "13",
				"SubConnect" : [
					{"ID" : "20", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_11", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_10", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "12",
				"SubConnect" : [
					{"ID" : "20", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_10", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_9", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "11",
				"SubConnect" : [
					{"ID" : "20", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_9", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_8", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "10",
				"SubConnect" : [
					{"ID" : "20", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_8", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_7", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "9",
				"SubConnect" : [
					{"ID" : "20", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_7", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_6", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "8",
				"SubConnect" : [
					{"ID" : "20", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_6", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_5", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "7",
				"SubConnect" : [
					{"ID" : "20", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_5", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_4", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "6",
				"SubConnect" : [
					{"ID" : "20", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_4", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_3", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "5",
				"SubConnect" : [
					{"ID" : "20", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_3", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_2", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "4",
				"SubConnect" : [
					{"ID" : "20", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_2", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables_1", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "3",
				"SubConnect" : [
					{"ID" : "20", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables_1", "Inst_start_state" : "72", "Inst_end_state" : "73"}]},
			{"Name" : "crcTables", "Type" : "Memory", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "2",
				"SubConnect" : [
					{"ID" : "20", "SubInstance" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Port" : "crcTables", "Inst_start_state" : "72", "Inst_end_state" : "73"}]}]},
	{"ID" : "20", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.process_crc_Loop_init_lut_proc_U0.grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91", "Parent" : "19", "Child" : ["21"],
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
	{"ID" : "21", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.process_crc_Loop_init_lut_proc_U0.grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91.flow_control_loop_pipe_sequential_init_U", "Parent" : "20"},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.read_input_U0", "Parent" : "1", "Child" : ["23", "25"],
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
					{"ID" : "23", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "gmem0", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "in_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "inByte0", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "37", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "23", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte0", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte1", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "38", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "23", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte1", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte2", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "39", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "23", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte2", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte3", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "40", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "23", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte3", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte4", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "41", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "23", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte4", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte5", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "42", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "23", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte5", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte6", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "43", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "23", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte6", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte7", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "44", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "23", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte7", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte8", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "45", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "23", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte8", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte9", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "46", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "23", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte9", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte10", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "47", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "23", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte10", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte11", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "48", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "23", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte11", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte12", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "49", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "23", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte12", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte13", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "50", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "23", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte13", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte14", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "51", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "23", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte14", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "inByte15", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "52", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "23", "SubInstance" : "grp_read_input_Pipeline_mem_rd_fu_142", "Port" : "inByte15", "Inst_start_state" : "76", "Inst_end_state" : "77"}]},
			{"Name" : "numChunks", "Type" : "None", "Direction" : "I"},
			{"Name" : "chunkSize", "Type" : "None", "Direction" : "I"},
			{"Name" : "numChunks_c12", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "53", "DependentChanDepth" : "2", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "numChunks_c12_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "chunkSize_c", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["26"], "DependentChan" : "54", "DependentChanDepth" : "2", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "chunkSize_c_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "23", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.read_input_U0.grp_read_input_Pipeline_mem_rd_fu_142", "Parent" : "22", "Child" : ["24"],
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
	{"ID" : "24", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.read_input_U0.grp_read_input_Pipeline_mem_rd_fu_142.flow_control_loop_pipe_sequential_init_U", "Parent" : "23"},
	{"ID" : "25", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.read_input_U0.mul_32s_32s_32_2_1_U62", "Parent" : "22"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.process_blocks_U0", "Parent" : "1", "Child" : ["27", "29"],
		"CDFG" : "process_blocks",
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
			{"Name" : "crcTables_0", "Type" : "Memory", "Direction" : "I", "DependentProc" : ["19"], "DependentChan" : "2",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "crcTables_0", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "29", "SubInstance" : "grp_process_blocks_Pipeline_tail_loop_fu_271", "Port" : "crcTables_0", "Inst_start_state" : "6", "Inst_end_state" : "7"}]},
			{"Name" : "crcTables_1", "Type" : "Memory", "Direction" : "I", "DependentProc" : ["19"], "DependentChan" : "3",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "crcTables_1", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "crcTables_2", "Type" : "Memory", "Direction" : "I", "DependentProc" : ["19"], "DependentChan" : "4",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "crcTables_2", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "crcTables_3", "Type" : "Memory", "Direction" : "I", "DependentProc" : ["19"], "DependentChan" : "5",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "crcTables_3", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "crcTables_4", "Type" : "Memory", "Direction" : "I", "DependentProc" : ["19"], "DependentChan" : "6",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "crcTables_4", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "crcTables_5", "Type" : "Memory", "Direction" : "I", "DependentProc" : ["19"], "DependentChan" : "7",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "crcTables_5", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "crcTables_6", "Type" : "Memory", "Direction" : "I", "DependentProc" : ["19"], "DependentChan" : "8",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "crcTables_6", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "crcTables_7", "Type" : "Memory", "Direction" : "I", "DependentProc" : ["19"], "DependentChan" : "9",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "crcTables_7", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "crcTables_8", "Type" : "Memory", "Direction" : "I", "DependentProc" : ["19"], "DependentChan" : "10",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "crcTables_8", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "crcTables_9", "Type" : "Memory", "Direction" : "I", "DependentProc" : ["19"], "DependentChan" : "11",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "crcTables_9", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "crcTables_10", "Type" : "Memory", "Direction" : "I", "DependentProc" : ["19"], "DependentChan" : "12",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "crcTables_10", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "crcTables_11", "Type" : "Memory", "Direction" : "I", "DependentProc" : ["19"], "DependentChan" : "13",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "crcTables_11", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "crcTables_12", "Type" : "Memory", "Direction" : "I", "DependentProc" : ["19"], "DependentChan" : "14",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "crcTables_12", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "crcTables_13", "Type" : "Memory", "Direction" : "I", "DependentProc" : ["19"], "DependentChan" : "15",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "crcTables_13", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "crcTables_14", "Type" : "Memory", "Direction" : "I", "DependentProc" : ["19"], "DependentChan" : "16",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "crcTables_14", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "crcTables_15", "Type" : "Memory", "Direction" : "I", "DependentProc" : ["19"], "DependentChan" : "17",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "crcTables_15", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "inByte0", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["22"], "DependentChan" : "37", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "inByte0", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "29", "SubInstance" : "grp_process_blocks_Pipeline_tail_loop_fu_271", "Port" : "inByte0", "Inst_start_state" : "6", "Inst_end_state" : "7"}]},
			{"Name" : "inByte1", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["22"], "DependentChan" : "38", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "inByte1", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "29", "SubInstance" : "grp_process_blocks_Pipeline_tail_loop_fu_271", "Port" : "inByte1", "Inst_start_state" : "6", "Inst_end_state" : "7"}]},
			{"Name" : "inByte2", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["22"], "DependentChan" : "39", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "inByte2", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "29", "SubInstance" : "grp_process_blocks_Pipeline_tail_loop_fu_271", "Port" : "inByte2", "Inst_start_state" : "6", "Inst_end_state" : "7"}]},
			{"Name" : "inByte3", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["22"], "DependentChan" : "40", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "inByte3", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "29", "SubInstance" : "grp_process_blocks_Pipeline_tail_loop_fu_271", "Port" : "inByte3", "Inst_start_state" : "6", "Inst_end_state" : "7"}]},
			{"Name" : "inByte4", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["22"], "DependentChan" : "41", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "inByte4", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "29", "SubInstance" : "grp_process_blocks_Pipeline_tail_loop_fu_271", "Port" : "inByte4", "Inst_start_state" : "6", "Inst_end_state" : "7"}]},
			{"Name" : "inByte5", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["22"], "DependentChan" : "42", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "inByte5", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "29", "SubInstance" : "grp_process_blocks_Pipeline_tail_loop_fu_271", "Port" : "inByte5", "Inst_start_state" : "6", "Inst_end_state" : "7"}]},
			{"Name" : "inByte6", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["22"], "DependentChan" : "43", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "inByte6", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "29", "SubInstance" : "grp_process_blocks_Pipeline_tail_loop_fu_271", "Port" : "inByte6", "Inst_start_state" : "6", "Inst_end_state" : "7"}]},
			{"Name" : "inByte7", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["22"], "DependentChan" : "44", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "inByte7", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "29", "SubInstance" : "grp_process_blocks_Pipeline_tail_loop_fu_271", "Port" : "inByte7", "Inst_start_state" : "6", "Inst_end_state" : "7"}]},
			{"Name" : "inByte8", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["22"], "DependentChan" : "45", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "inByte8", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "29", "SubInstance" : "grp_process_blocks_Pipeline_tail_loop_fu_271", "Port" : "inByte8", "Inst_start_state" : "6", "Inst_end_state" : "7"}]},
			{"Name" : "inByte9", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["22"], "DependentChan" : "46", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "inByte9", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "29", "SubInstance" : "grp_process_blocks_Pipeline_tail_loop_fu_271", "Port" : "inByte9", "Inst_start_state" : "6", "Inst_end_state" : "7"}]},
			{"Name" : "inByte10", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["22"], "DependentChan" : "47", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "inByte10", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "29", "SubInstance" : "grp_process_blocks_Pipeline_tail_loop_fu_271", "Port" : "inByte10", "Inst_start_state" : "6", "Inst_end_state" : "7"}]},
			{"Name" : "inByte11", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["22"], "DependentChan" : "48", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "inByte11", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "29", "SubInstance" : "grp_process_blocks_Pipeline_tail_loop_fu_271", "Port" : "inByte11", "Inst_start_state" : "6", "Inst_end_state" : "7"}]},
			{"Name" : "inByte12", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["22"], "DependentChan" : "49", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "inByte12", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "29", "SubInstance" : "grp_process_blocks_Pipeline_tail_loop_fu_271", "Port" : "inByte12", "Inst_start_state" : "6", "Inst_end_state" : "7"}]},
			{"Name" : "inByte13", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["22"], "DependentChan" : "50", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "inByte13", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "29", "SubInstance" : "grp_process_blocks_Pipeline_tail_loop_fu_271", "Port" : "inByte13", "Inst_start_state" : "6", "Inst_end_state" : "7"}]},
			{"Name" : "inByte14", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["22"], "DependentChan" : "51", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "inByte14", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "29", "SubInstance" : "grp_process_blocks_Pipeline_tail_loop_fu_271", "Port" : "inByte14", "Inst_start_state" : "6", "Inst_end_state" : "7"}]},
			{"Name" : "inByte15", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["22"], "DependentChan" : "52", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "27", "SubInstance" : "grp_process_blocks_Pipeline_block_loop_fu_199", "Port" : "inByte15", "Inst_start_state" : "3", "Inst_end_state" : "4"},
					{"ID" : "29", "SubInstance" : "grp_process_blocks_Pipeline_tail_loop_fu_271", "Port" : "inByte15", "Inst_start_state" : "6", "Inst_end_state" : "7"}]},
			{"Name" : "crc_size", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["18"], "DependentChan" : "35", "DependentChanDepth" : "3", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "crc_size_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "init_value", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["18"], "DependentChan" : "36", "DependentChanDepth" : "3", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "init_value_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "outStream", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["31"], "DependentChan" : "55", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "outStream_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "numChunks", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["22"], "DependentChan" : "53", "DependentChanDepth" : "2", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "numChunks_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "chunkSize", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["22"], "DependentChan" : "54", "DependentChanDepth" : "2", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "chunkSize_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "numChunks_c", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["31"], "DependentChan" : "56", "DependentChanDepth" : "2", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "numChunks_c_blk_n", "Type" : "RtlSignal"}]}],
		"Loop" : [
			{"Name" : "chunk_loop", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "8", "FirstState" : "ap_ST_fsm_state3", "LastState" : ["ap_ST_fsm_state8"], "QuitState" : ["ap_ST_fsm_state3"], "PreState" : ["ap_ST_fsm_state2"], "PostState" : ["ap_ST_fsm_state1"], "OneDepthLoop" : "0", "OneStateBlock": ""}}]},
	{"ID" : "27", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.process_blocks_U0.grp_process_blocks_Pipeline_block_loop_fu_199", "Parent" : "26", "Child" : ["28"],
		"CDFG" : "process_blocks_Pipeline_block_loop",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "2", "EstimateLatencyMax" : "134217730",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "init_value_2", "Type" : "None", "Direction" : "I"},
			{"Name" : "blocks_in_chunk", "Type" : "None", "Direction" : "I"},
			{"Name" : "inByte0", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte0_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte2", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte2_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte3", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte3_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte4", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte4_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte5", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte5_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte6", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte6_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte7", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte7_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte8", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte8_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte9", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte9_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte10", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte10_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte11", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte11_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte12", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte12_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte13", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte13_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte14", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte14_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte15", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte15_blk_n", "Type" : "RtlSignal"}]},
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
			{"Name" : "mask_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "crc_out", "Type" : "Vld", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "block_loop", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter2", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter2", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "28", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.process_blocks_U0.grp_process_blocks_Pipeline_block_loop_fu_199.flow_control_loop_pipe_sequential_init_U", "Parent" : "27"},
	{"ID" : "29", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.process_blocks_U0.grp_process_blocks_Pipeline_tail_loop_fu_271", "Parent" : "26", "Child" : ["30"],
		"CDFG" : "process_blocks_Pipeline_tail_loop",
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
			{"Name" : "crc_reload", "Type" : "None", "Direction" : "I"},
			{"Name" : "sub_ln329", "Type" : "None", "Direction" : "I"},
			{"Name" : "crcTables_0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "mask_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "inByte0", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte0_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte2", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte2_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte3", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte3_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte4", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte4_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte5", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte5_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte6", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte6_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte7", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte7_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte8", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte8_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte9", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte9_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte10", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte10_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte11", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte11_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte12", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte12_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte13", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte13_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte14", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte14_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte15", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte15_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "crc_2_out", "Type" : "Vld", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "tail_loop", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter3", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter3", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "30", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.process_blocks_U0.grp_process_blocks_Pipeline_tail_loop_fu_271.flow_control_loop_pipe_sequential_init_U", "Parent" : "29"},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.write_output_U0", "Parent" : "1", "Child" : ["32"],
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
		"StartSource" : "18",
		"StartFifo" : "start_for_write_output_U0_U",
		"Port" : [
			{"Name" : "outStream", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["26"], "DependentChan" : "55", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_write_output_Pipeline_VITIS_LOOP_442_1_fu_58", "Port" : "outStream", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "numChunks", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["26"], "DependentChan" : "56", "DependentChanDepth" : "2", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "numChunks_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "gmem0", "Type" : "MAXI", "Direction" : "O",
				"SubConnect" : [
					{"ID" : "32", "SubInstance" : "grp_write_output_Pipeline_VITIS_LOOP_442_1_fu_58", "Port" : "gmem0", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "crc_out", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["18"], "DependentChan" : "34", "DependentChanDepth" : "4", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "crc_out_blk_n", "Type" : "RtlSignal"}]}]},
	{"ID" : "32", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.write_output_U0.grp_write_output_Pipeline_VITIS_LOOP_442_1_fu_58", "Parent" : "31", "Child" : ["33"],
		"CDFG" : "write_output_Pipeline_VITIS_LOOP_442_1",
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
			{"Name" : "numChunks_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "outStream", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "outStream_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "crc_out_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "trunc_ln445_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "gmem0", "Type" : "MAXI", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "gmem0_blk_n_AW", "Type" : "RtlSignal"},
					{"Name" : "gmem0_blk_n_W", "Type" : "RtlSignal"},
					{"Name" : "gmem0_blk_n_B", "Type" : "RtlSignal"}]}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_442_1", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter70", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter70", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "33", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.write_output_U0.grp_write_output_Pipeline_VITIS_LOOP_442_1_fu_58.flow_control_loop_pipe_sequential_init_U", "Parent" : "32"},
	{"ID" : "34", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.crc_out_c_U", "Parent" : "1"},
	{"ID" : "35", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.crc_size_c_U", "Parent" : "1"},
	{"ID" : "36", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.init_value_c_U", "Parent" : "1"},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.inByte0_U", "Parent" : "1"},
	{"ID" : "38", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.inByte1_U", "Parent" : "1"},
	{"ID" : "39", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.inByte2_U", "Parent" : "1"},
	{"ID" : "40", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.inByte3_U", "Parent" : "1"},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.inByte4_U", "Parent" : "1"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.inByte5_U", "Parent" : "1"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.inByte6_U", "Parent" : "1"},
	{"ID" : "44", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.inByte7_U", "Parent" : "1"},
	{"ID" : "45", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.inByte8_U", "Parent" : "1"},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.inByte9_U", "Parent" : "1"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.inByte10_U", "Parent" : "1"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.inByte11_U", "Parent" : "1"},
	{"ID" : "49", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.inByte12_U", "Parent" : "1"},
	{"ID" : "50", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.inByte13_U", "Parent" : "1"},
	{"ID" : "51", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.inByte14_U", "Parent" : "1"},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.inByte15_U", "Parent" : "1"},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.numChunks_c12_U", "Parent" : "1"},
	{"ID" : "54", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.chunkSize_c_U", "Parent" : "1"},
	{"ID" : "55", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.outStream_U", "Parent" : "1"},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.numChunks_c_U", "Parent" : "1"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_process_crc_fu_122.start_for_write_output_U0_U", "Parent" : "1"},
	{"ID" : "58", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.control_s_axi_U", "Parent" : "0"},
	{"ID" : "59", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.gmem0_m_axi_U", "Parent" : "0"},
	{"ID" : "60", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.gmem1_m_axi_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	calculate_crc {
		gmem0 {Type IO LastRead 4 FirstWrite -1}
		gmem1 {Type I LastRead 1 FirstWrite -1}
		data_in {Type I LastRead 0 FirstWrite -1}
		crc_out {Type I LastRead 0 FirstWrite -1}
		tables {Type I LastRead 0 FirstWrite -1}
		numChunks {Type I LastRead 0 FirstWrite -1}
		chunkSize {Type I LastRead 0 FirstWrite -1}
		crc_size {Type I LastRead 0 FirstWrite -1}
		init_value {Type I LastRead 0 FirstWrite -1}}
	process_crc {
		gmem0 {Type IO LastRead 4 FirstWrite -1}
		data_in {Type I LastRead 0 FirstWrite -1}
		crc_out {Type I LastRead 1 FirstWrite -1}
		gmem1 {Type I LastRead 1 FirstWrite -1}
		tables {Type I LastRead 0 FirstWrite -1}
		numChunks {Type I LastRead 0 FirstWrite -1}
		chunkSize {Type I LastRead 0 FirstWrite -1}
		crc_size {Type I LastRead 1 FirstWrite -1}
		init_value {Type I LastRead 1 FirstWrite -1}}
	entry_proc {
		crc_out {Type I LastRead 0 FirstWrite -1}
		crc_out_c {Type O LastRead -1 FirstWrite 0}
		crc_size {Type I LastRead 0 FirstWrite -1}
		crc_size_c {Type O LastRead -1 FirstWrite 0}
		init_value {Type I LastRead 0 FirstWrite -1}
		init_value_c {Type O LastRead -1 FirstWrite 0}}
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
		crcTables_15 {Type O LastRead -1 FirstWrite 2}}
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
		inByte15 {Type O LastRead -1 FirstWrite 2}}
	process_blocks {
		crcTables_0 {Type I LastRead 2 FirstWrite -1}
		crcTables_1 {Type I LastRead 1 FirstWrite -1}
		crcTables_2 {Type I LastRead 1 FirstWrite -1}
		crcTables_3 {Type I LastRead 1 FirstWrite -1}
		crcTables_4 {Type I LastRead 1 FirstWrite -1}
		crcTables_5 {Type I LastRead 1 FirstWrite -1}
		crcTables_6 {Type I LastRead 1 FirstWrite -1}
		crcTables_7 {Type I LastRead 1 FirstWrite -1}
		crcTables_8 {Type I LastRead 1 FirstWrite -1}
		crcTables_9 {Type I LastRead 1 FirstWrite -1}
		crcTables_10 {Type I LastRead 1 FirstWrite -1}
		crcTables_11 {Type I LastRead 1 FirstWrite -1}
		crcTables_12 {Type I LastRead 1 FirstWrite -1}
		crcTables_13 {Type I LastRead 1 FirstWrite -1}
		crcTables_14 {Type I LastRead 1 FirstWrite -1}
		crcTables_15 {Type I LastRead 1 FirstWrite -1}
		inByte0 {Type I LastRead 1 FirstWrite -1}
		inByte1 {Type I LastRead 1 FirstWrite -1}
		inByte2 {Type I LastRead 1 FirstWrite -1}
		inByte3 {Type I LastRead 1 FirstWrite -1}
		inByte4 {Type I LastRead 1 FirstWrite -1}
		inByte5 {Type I LastRead 1 FirstWrite -1}
		inByte6 {Type I LastRead 1 FirstWrite -1}
		inByte7 {Type I LastRead 1 FirstWrite -1}
		inByte8 {Type I LastRead 1 FirstWrite -1}
		inByte9 {Type I LastRead 1 FirstWrite -1}
		inByte10 {Type I LastRead 1 FirstWrite -1}
		inByte11 {Type I LastRead 1 FirstWrite -1}
		inByte12 {Type I LastRead 1 FirstWrite -1}
		inByte13 {Type I LastRead 1 FirstWrite -1}
		inByte14 {Type I LastRead 1 FirstWrite -1}
		inByte15 {Type I LastRead 1 FirstWrite -1}
		crc_size {Type I LastRead 0 FirstWrite -1}
		init_value {Type I LastRead 0 FirstWrite -1}
		outStream {Type O LastRead -1 FirstWrite 7}
		numChunks {Type I LastRead 0 FirstWrite -1}
		chunkSize {Type I LastRead 0 FirstWrite -1}
		numChunks_c {Type O LastRead -1 FirstWrite 0}}
	process_blocks_Pipeline_block_loop {
		init_value_2 {Type I LastRead 0 FirstWrite -1}
		blocks_in_chunk {Type I LastRead 0 FirstWrite -1}
		inByte0 {Type I LastRead 1 FirstWrite -1}
		inByte1 {Type I LastRead 1 FirstWrite -1}
		inByte2 {Type I LastRead 1 FirstWrite -1}
		inByte3 {Type I LastRead 1 FirstWrite -1}
		inByte4 {Type I LastRead 1 FirstWrite -1}
		inByte5 {Type I LastRead 1 FirstWrite -1}
		inByte6 {Type I LastRead 1 FirstWrite -1}
		inByte7 {Type I LastRead 1 FirstWrite -1}
		inByte8 {Type I LastRead 1 FirstWrite -1}
		inByte9 {Type I LastRead 1 FirstWrite -1}
		inByte10 {Type I LastRead 1 FirstWrite -1}
		inByte11 {Type I LastRead 1 FirstWrite -1}
		inByte12 {Type I LastRead 1 FirstWrite -1}
		inByte13 {Type I LastRead 1 FirstWrite -1}
		inByte14 {Type I LastRead 1 FirstWrite -1}
		inByte15 {Type I LastRead 1 FirstWrite -1}
		crcTables_0 {Type I LastRead 1 FirstWrite -1}
		crcTables_1 {Type I LastRead 1 FirstWrite -1}
		crcTables_2 {Type I LastRead 1 FirstWrite -1}
		crcTables_3 {Type I LastRead 1 FirstWrite -1}
		crcTables_4 {Type I LastRead 1 FirstWrite -1}
		crcTables_5 {Type I LastRead 1 FirstWrite -1}
		crcTables_6 {Type I LastRead 1 FirstWrite -1}
		crcTables_7 {Type I LastRead 1 FirstWrite -1}
		crcTables_8 {Type I LastRead 1 FirstWrite -1}
		crcTables_9 {Type I LastRead 1 FirstWrite -1}
		crcTables_10 {Type I LastRead 1 FirstWrite -1}
		crcTables_11 {Type I LastRead 1 FirstWrite -1}
		crcTables_12 {Type I LastRead 1 FirstWrite -1}
		crcTables_13 {Type I LastRead 1 FirstWrite -1}
		crcTables_14 {Type I LastRead 1 FirstWrite -1}
		crcTables_15 {Type I LastRead 1 FirstWrite -1}
		mask_1 {Type I LastRead 0 FirstWrite -1}
		crc_out {Type O LastRead -1 FirstWrite 1}}
	process_blocks_Pipeline_tail_loop {
		crc_reload {Type I LastRead 0 FirstWrite -1}
		sub_ln329 {Type I LastRead 0 FirstWrite -1}
		crcTables_0 {Type I LastRead 2 FirstWrite -1}
		mask_1 {Type I LastRead 0 FirstWrite -1}
		inByte0 {Type I LastRead 1 FirstWrite -1}
		inByte1 {Type I LastRead 1 FirstWrite -1}
		inByte2 {Type I LastRead 1 FirstWrite -1}
		inByte3 {Type I LastRead 1 FirstWrite -1}
		inByte4 {Type I LastRead 1 FirstWrite -1}
		inByte5 {Type I LastRead 1 FirstWrite -1}
		inByte6 {Type I LastRead 1 FirstWrite -1}
		inByte7 {Type I LastRead 1 FirstWrite -1}
		inByte8 {Type I LastRead 1 FirstWrite -1}
		inByte9 {Type I LastRead 1 FirstWrite -1}
		inByte10 {Type I LastRead 1 FirstWrite -1}
		inByte11 {Type I LastRead 1 FirstWrite -1}
		inByte12 {Type I LastRead 1 FirstWrite -1}
		inByte13 {Type I LastRead 1 FirstWrite -1}
		inByte14 {Type I LastRead 1 FirstWrite -1}
		inByte15 {Type I LastRead 1 FirstWrite -1}
		crc_2_out {Type O LastRead -1 FirstWrite 2}}
	write_output {
		outStream {Type I LastRead 1 FirstWrite -1}
		numChunks {Type I LastRead 0 FirstWrite -1}
		gmem0 {Type O LastRead 3 FirstWrite 2}
		crc_out {Type I LastRead 0 FirstWrite -1}}
	write_output_Pipeline_VITIS_LOOP_442_1 {
		numChunks_1 {Type I LastRead 0 FirstWrite -1}
		outStream {Type I LastRead 1 FirstWrite -1}
		crc_out_1 {Type I LastRead 0 FirstWrite -1}
		trunc_ln445_1 {Type I LastRead 0 FirstWrite -1}
		gmem0 {Type O LastRead 3 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "0", "Max" : "0"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	gmem0 { m_axi {  { m_axi_gmem0_AWVALID VALID 1 1 }  { m_axi_gmem0_AWREADY READY 0 1 }  { m_axi_gmem0_AWADDR ADDR 1 64 }  { m_axi_gmem0_AWID ID 1 1 }  { m_axi_gmem0_AWLEN SIZE 1 8 }  { m_axi_gmem0_AWSIZE BURST 1 3 }  { m_axi_gmem0_AWBURST LOCK 1 2 }  { m_axi_gmem0_AWLOCK CACHE 1 2 }  { m_axi_gmem0_AWCACHE PROT 1 4 }  { m_axi_gmem0_AWPROT QOS 1 3 }  { m_axi_gmem0_AWQOS REGION 1 4 }  { m_axi_gmem0_AWREGION USER 1 4 }  { m_axi_gmem0_AWUSER DATA 1 1 }  { m_axi_gmem0_WVALID VALID 1 1 }  { m_axi_gmem0_WREADY READY 0 1 }  { m_axi_gmem0_WDATA FIFONUM 1 128 }  { m_axi_gmem0_WSTRB STRB 1 16 }  { m_axi_gmem0_WLAST LAST 1 1 }  { m_axi_gmem0_WID ID 1 1 }  { m_axi_gmem0_WUSER DATA 1 1 }  { m_axi_gmem0_ARVALID VALID 1 1 }  { m_axi_gmem0_ARREADY READY 0 1 }  { m_axi_gmem0_ARADDR ADDR 1 64 }  { m_axi_gmem0_ARID ID 1 1 }  { m_axi_gmem0_ARLEN SIZE 1 8 }  { m_axi_gmem0_ARSIZE BURST 1 3 }  { m_axi_gmem0_ARBURST LOCK 1 2 }  { m_axi_gmem0_ARLOCK CACHE 1 2 }  { m_axi_gmem0_ARCACHE PROT 1 4 }  { m_axi_gmem0_ARPROT QOS 1 3 }  { m_axi_gmem0_ARQOS REGION 1 4 }  { m_axi_gmem0_ARREGION USER 1 4 }  { m_axi_gmem0_ARUSER DATA 1 1 }  { m_axi_gmem0_RVALID VALID 0 1 }  { m_axi_gmem0_RREADY READY 1 1 }  { m_axi_gmem0_RDATA FIFONUM 0 128 }  { m_axi_gmem0_RLAST LAST 0 1 }  { m_axi_gmem0_RID ID 0 1 }  { m_axi_gmem0_RUSER DATA 0 1 }  { m_axi_gmem0_RRESP RESP 0 2 }  { m_axi_gmem0_BVALID VALID 0 1 }  { m_axi_gmem0_BREADY READY 1 1 }  { m_axi_gmem0_BRESP RESP 0 2 }  { m_axi_gmem0_BID ID 0 1 }  { m_axi_gmem0_BUSER DATA 0 1 } } }
	gmem1 { m_axi {  { m_axi_gmem1_AWVALID VALID 1 1 }  { m_axi_gmem1_AWREADY READY 0 1 }  { m_axi_gmem1_AWADDR ADDR 1 64 }  { m_axi_gmem1_AWID ID 1 1 }  { m_axi_gmem1_AWLEN SIZE 1 8 }  { m_axi_gmem1_AWSIZE BURST 1 3 }  { m_axi_gmem1_AWBURST LOCK 1 2 }  { m_axi_gmem1_AWLOCK CACHE 1 2 }  { m_axi_gmem1_AWCACHE PROT 1 4 }  { m_axi_gmem1_AWPROT QOS 1 3 }  { m_axi_gmem1_AWQOS REGION 1 4 }  { m_axi_gmem1_AWREGION USER 1 4 }  { m_axi_gmem1_AWUSER DATA 1 1 }  { m_axi_gmem1_WVALID VALID 1 1 }  { m_axi_gmem1_WREADY READY 0 1 }  { m_axi_gmem1_WDATA FIFONUM 1 512 }  { m_axi_gmem1_WSTRB STRB 1 64 }  { m_axi_gmem1_WLAST LAST 1 1 }  { m_axi_gmem1_WID ID 1 1 }  { m_axi_gmem1_WUSER DATA 1 1 }  { m_axi_gmem1_ARVALID VALID 1 1 }  { m_axi_gmem1_ARREADY READY 0 1 }  { m_axi_gmem1_ARADDR ADDR 1 64 }  { m_axi_gmem1_ARID ID 1 1 }  { m_axi_gmem1_ARLEN SIZE 1 8 }  { m_axi_gmem1_ARSIZE BURST 1 3 }  { m_axi_gmem1_ARBURST LOCK 1 2 }  { m_axi_gmem1_ARLOCK CACHE 1 2 }  { m_axi_gmem1_ARCACHE PROT 1 4 }  { m_axi_gmem1_ARPROT QOS 1 3 }  { m_axi_gmem1_ARQOS REGION 1 4 }  { m_axi_gmem1_ARREGION USER 1 4 }  { m_axi_gmem1_ARUSER DATA 1 1 }  { m_axi_gmem1_RVALID VALID 0 1 }  { m_axi_gmem1_RREADY READY 1 1 }  { m_axi_gmem1_RDATA FIFONUM 0 512 }  { m_axi_gmem1_RLAST LAST 0 1 }  { m_axi_gmem1_RID ID 0 1 }  { m_axi_gmem1_RUSER DATA 0 1 }  { m_axi_gmem1_RRESP RESP 0 2 }  { m_axi_gmem1_BVALID VALID 0 1 }  { m_axi_gmem1_BREADY READY 1 1 }  { m_axi_gmem1_BRESP RESP 0 2 }  { m_axi_gmem1_BID ID 0 1 }  { m_axi_gmem1_BUSER DATA 0 1 } } }
}

set maxi_interface_dict [dict create]
dict set maxi_interface_dict gmem0 {NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 READ_WRITE_MODE READ_WRITE}
dict set maxi_interface_dict gmem1 {NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 READ_WRITE_MODE READ_ONLY}

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
	{ gmem0 64 }
	{ gmem1 64 }
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
	{ gmem0 64 }
	{ gmem1 64 }
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
