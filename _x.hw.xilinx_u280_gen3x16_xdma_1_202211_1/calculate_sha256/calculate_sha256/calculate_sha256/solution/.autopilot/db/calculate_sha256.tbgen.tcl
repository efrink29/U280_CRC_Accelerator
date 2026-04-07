set moduleName calculate_sha256
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
set C_modelName {calculate_sha256}
set C_modelType { void 0 }
set C_modelArgList {
	{ gmem0 int 512 regular {axi_master 2}  }
	{ data_in int 64 regular {axi_slave 0}  }
	{ crc_out int 64 regular {axi_slave 0}  }
	{ numChunks int 32 regular {axi_slave 0}  }
	{ chunkSize int 32 regular {axi_slave 0}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "gmem0", "interface" : "axi_master", "bitwidth" : 512, "direction" : "READWRITE", "bitSlice":[ {"cElement": [{"cName": "data_in","offset": { "type": "dynamic","port_name": "data_in","bundle": "control"},"direction": "READONLY"},{"cName": "crc_out","offset": { "type": "dynamic","port_name": "crc_out","bundle": "control"},"direction": "WRITEONLY"}]}]} , 
 	{ "Name" : "data_in", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 64, "direction" : "READONLY", "offset" : {"in":16}, "offset_end" : {"in":27}} , 
 	{ "Name" : "crc_out", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 64, "direction" : "READONLY", "offset" : {"in":28}, "offset_end" : {"in":39}} , 
 	{ "Name" : "numChunks", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":40}, "offset_end" : {"in":47}} , 
 	{ "Name" : "chunkSize", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":48}, "offset_end" : {"in":55}} ]}
# RTL Port declarations: 
set portNum 65
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
	{ m_axi_gmem0_WDATA sc_out sc_lv 512 signal 0 } 
	{ m_axi_gmem0_WSTRB sc_out sc_lv 64 signal 0 } 
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
	{ m_axi_gmem0_RDATA sc_in sc_lv 512 signal 0 } 
	{ m_axi_gmem0_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem0_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem0_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem0_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem0_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem0_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem0_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem0_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem0_BUSER sc_in sc_lv 1 signal 0 } 
	{ s_axi_control_AWVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_AWREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_AWADDR sc_in sc_lv 6 signal -1 } 
	{ s_axi_control_WVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_WREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_WDATA sc_in sc_lv 32 signal -1 } 
	{ s_axi_control_WSTRB sc_in sc_lv 4 signal -1 } 
	{ s_axi_control_ARVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_ARREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_ARADDR sc_in sc_lv 6 signal -1 } 
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
	{ "name": "s_axi_control_AWADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "control", "role": "AWADDR" },"address":[{"name":"calculate_sha256","role":"start","value":"0","valid_bit":"0"},{"name":"calculate_sha256","role":"continue","value":"0","valid_bit":"4"},{"name":"calculate_sha256","role":"auto_start","value":"0","valid_bit":"7"},{"name":"data_in","role":"data","value":"16"},{"name":"crc_out","role":"data","value":"28"},{"name":"numChunks","role":"data","value":"40"},{"name":"chunkSize","role":"data","value":"48"}] },
	{ "name": "s_axi_control_AWVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWVALID" } },
	{ "name": "s_axi_control_AWREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWREADY" } },
	{ "name": "s_axi_control_WVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WVALID" } },
	{ "name": "s_axi_control_WREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WREADY" } },
	{ "name": "s_axi_control_WDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "WDATA" } },
	{ "name": "s_axi_control_WSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "control", "role": "WSTRB" } },
	{ "name": "s_axi_control_ARADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "control", "role": "ARADDR" },"address":[{"name":"calculate_sha256","role":"start","value":"0","valid_bit":"0"},{"name":"calculate_sha256","role":"done","value":"0","valid_bit":"1"},{"name":"calculate_sha256","role":"idle","value":"0","valid_bit":"2"},{"name":"calculate_sha256","role":"ready","value":"0","valid_bit":"3"},{"name":"calculate_sha256","role":"auto_start","value":"0","valid_bit":"7"}] },
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
 	{ "name": "m_axi_gmem0_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":512, "type": "signal", "bundle":{"name": "gmem0", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem0_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem0", "role": "WSTRB" }} , 
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
 	{ "name": "m_axi_gmem0_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":512, "type": "signal", "bundle":{"name": "gmem0", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem0_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem0_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "RID" }} , 
 	{ "name": "m_axi_gmem0_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem0_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem0", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem0_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem0_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem0_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem0", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem0_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "BID" }} , 
 	{ "name": "m_axi_gmem0_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "BUSER" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "26", "27"],
		"CDFG" : "calculate_sha256",
		"Protocol" : "ap_ctrl_chain",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "0",
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
			{"Name" : "gmem0", "Type" : "MAXI", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_sha256_process_and_write_chunk_fu_108", "Port" : "gmem0", "Inst_start_state" : "3", "Inst_end_state" : "4"}]},
			{"Name" : "data_in", "Type" : "None", "Direction" : "I"},
			{"Name" : "crc_out", "Type" : "None", "Direction" : "I"},
			{"Name" : "numChunks", "Type" : "None", "Direction" : "I"},
			{"Name" : "chunkSize", "Type" : "None", "Direction" : "I"},
			{"Name" : "SHA256_K", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_sha256_process_and_write_chunk_fu_108", "Port" : "SHA256_K", "Inst_start_state" : "3", "Inst_end_state" : "4"}]}],
		"Loop" : [
			{"Name" : "hash_chunk_loop", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "4", "FirstState" : "ap_ST_fsm_state2", "LastState" : ["ap_ST_fsm_state4"], "QuitState" : ["ap_ST_fsm_state2"], "PreState" : ["ap_ST_fsm_state1"], "PostState" : ["ap_ST_fsm_state1"], "OneDepthLoop" : "0", "OneStateBlock": ""}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108", "Parent" : "0", "Child" : ["2", "23"],
		"CDFG" : "sha256_process_and_write_chunk",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "273", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "gmem0", "Type" : "MAXI", "Direction" : "IO",
				"BlockSignal" : [
					{"Name" : "gmem0_blk_n_AW", "Type" : "RtlSignal"},
					{"Name" : "gmem0_blk_n_W", "Type" : "RtlSignal"},
					{"Name" : "gmem0_blk_n_B", "Type" : "RtlSignal"}],
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_sha256_process_chunk_fu_97", "Port" : "gmem0", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "chunk", "Type" : "None", "Direction" : "I"},
			{"Name" : "chunk_size", "Type" : "None", "Direction" : "I"},
			{"Name" : "out_words", "Type" : "None", "Direction" : "I"},
			{"Name" : "SHA256_K", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_sha256_process_chunk_fu_97", "Port" : "SHA256_K", "Inst_start_state" : "1", "Inst_end_state" : "2"}]}]},
	{"ID" : "2", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97", "Parent" : "1", "Child" : ["3", "4", "13", "15", "17", "19", "21"],
		"CDFG" : "sha256_process_chunk",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "191", "EstimateLatencyMax" : "-1",
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
					{"Name" : "gmem0_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "gmem0_blk_n_R", "Type" : "RtlSignal"}],
				"SubConnect" : [
					{"ID" : "15", "SubInstance" : "grp_sha256_process_chunk_Pipeline_copy_remainder_fu_2600", "Port" : "gmem0", "Inst_start_state" : "214", "Inst_end_state" : "215"}]},
			{"Name" : "chunk", "Type" : "None", "Direction" : "I"},
			{"Name" : "chunk_size", "Type" : "None", "Direction" : "I"},
			{"Name" : "SHA256_K", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_sha256_compress_block_fu_2572", "Port" : "SHA256_K", "Inst_start_state" : "322", "Inst_end_state" : "323"}]}],
		"Loop" : [
			{"Name" : "full_block_loop_vec", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "323", "FirstState" : "ap_ST_fsm_state209", "LastState" : ["ap_ST_fsm_state212"], "QuitState" : ["ap_ST_fsm_state209"], "PreState" : ["ap_ST_fsm_state3", "ap_ST_fsm_state2", "ap_ST_fsm_state208"], "PostState" : ["ap_ST_fsm_state213"], "OneDepthLoop" : "0", "OneStateBlock": ""}},
			{"Name" : "full_block_loop_scalar", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "323", "FirstState" : "ap_ST_fsm_state3", "LastState" : ["ap_ST_fsm_state139"], "QuitState" : ["ap_ST_fsm_state3"], "PreState" : ["ap_ST_fsm_state2"], "PostState" : ["ap_ST_fsm_state209"], "OneDepthLoop" : "0", "OneStateBlock": ""}}]},
	{"ID" : "3", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.pad_block_U", "Parent" : "2"},
	{"ID" : "4", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_compress_block_fu_2572", "Parent" : "2", "Child" : ["5"],
		"CDFG" : "sha256_compress_block",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "70", "EstimateLatencyMax" : "70",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "block_V_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "state_0_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "state_1_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "state_2_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "state_3_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "state_4_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "state_5_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "state_6_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "state_7_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "SHA256_K", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "5", "SubInstance" : "grp_sha256_compress_block_Pipeline_round_loop_fu_372", "Port" : "SHA256_K", "Inst_start_state" : "1", "Inst_end_state" : "2"}]}]},
	{"ID" : "5", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_compress_block_fu_2572.grp_sha256_compress_block_Pipeline_round_loop_fu_372", "Parent" : "4", "Child" : ["6", "7", "8", "9", "10", "11", "12"],
		"CDFG" : "sha256_compress_block_Pipeline_round_loop",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "68", "EstimateLatencyMax" : "68",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "wbuf_15", "Type" : "None", "Direction" : "I"},
			{"Name" : "wbuf_14", "Type" : "None", "Direction" : "I"},
			{"Name" : "wbuf_13", "Type" : "None", "Direction" : "I"},
			{"Name" : "wbuf_12", "Type" : "None", "Direction" : "I"},
			{"Name" : "wbuf_11", "Type" : "None", "Direction" : "I"},
			{"Name" : "wbuf_10", "Type" : "None", "Direction" : "I"},
			{"Name" : "wbuf_9", "Type" : "None", "Direction" : "I"},
			{"Name" : "wbuf_8", "Type" : "None", "Direction" : "I"},
			{"Name" : "wbuf_7", "Type" : "None", "Direction" : "I"},
			{"Name" : "wbuf_6", "Type" : "None", "Direction" : "I"},
			{"Name" : "wbuf_5", "Type" : "None", "Direction" : "I"},
			{"Name" : "wbuf_4", "Type" : "None", "Direction" : "I"},
			{"Name" : "wbuf_3", "Type" : "None", "Direction" : "I"},
			{"Name" : "wbuf_2", "Type" : "None", "Direction" : "I"},
			{"Name" : "wbuf_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "wbuf", "Type" : "None", "Direction" : "I"},
			{"Name" : "state_0_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "state_1_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "state_2_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "state_3_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "state_4_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "state_5_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "state_6_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "state_7_read", "Type" : "None", "Direction" : "I"},
			{"Name" : "a_2_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "b_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "c_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "d_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "e_2_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "f_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "g_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "h_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "SHA256_K", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "round_loop", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter2", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter2", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "6", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_compress_block_fu_2572.grp_sha256_compress_block_Pipeline_round_loop_fu_372.SHA256_K_U", "Parent" : "5"},
	{"ID" : "7", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_compress_block_fu_2572.grp_sha256_compress_block_Pipeline_round_loop_fu_372.mux_164_32_1_1_U1", "Parent" : "5"},
	{"ID" : "8", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_compress_block_fu_2572.grp_sha256_compress_block_Pipeline_round_loop_fu_372.mux_164_32_1_1_U2", "Parent" : "5"},
	{"ID" : "9", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_compress_block_fu_2572.grp_sha256_compress_block_Pipeline_round_loop_fu_372.mux_164_32_1_1_U3", "Parent" : "5"},
	{"ID" : "10", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_compress_block_fu_2572.grp_sha256_compress_block_Pipeline_round_loop_fu_372.mux_164_32_1_1_U4", "Parent" : "5"},
	{"ID" : "11", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_compress_block_fu_2572.grp_sha256_compress_block_Pipeline_round_loop_fu_372.mux_167_32_1_1_U5", "Parent" : "5"},
	{"ID" : "12", "Level" : "5", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_compress_block_fu_2572.grp_sha256_compress_block_Pipeline_round_loop_fu_372.flow_control_loop_pipe_sequential_init_U", "Parent" : "5"},
	{"ID" : "13", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_clear_pad_block_fu_2595", "Parent" : "2", "Child" : ["14"],
		"CDFG" : "sha256_process_chunk_Pipeline_clear_pad_block",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "pad_block", "Type" : "Memory", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "clear_pad_block", "PipelineType" : "NotSupport"}]},
	{"ID" : "14", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_clear_pad_block_fu_2595.flow_control_loop_pipe_sequential_init_U", "Parent" : "13"},
	{"ID" : "15", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_copy_remainder_fu_2600", "Parent" : "2", "Child" : ["16"],
		"CDFG" : "sha256_process_chunk_Pipeline_copy_remainder",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "2", "EstimateLatencyMax" : "136",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "rem_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "chunk", "Type" : "None", "Direction" : "I"},
			{"Name" : "zext_ln264_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "gmem0", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "gmem0_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "gmem0_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "trunc_ln4", "Type" : "None", "Direction" : "I"},
			{"Name" : "pad_block", "Type" : "Memory", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "copy_remainder", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter72", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter72", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "16", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_copy_remainder_fu_2600.flow_control_loop_pipe_sequential_init_U", "Parent" : "15"},
	{"ID" : "17", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_clear_second_block_fu_2611", "Parent" : "2", "Child" : ["18"],
		"CDFG" : "sha256_process_chunk_Pipeline_clear_second_block",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "66", "EstimateLatencyMax" : "66",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "pad_block", "Type" : "Memory", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "clear_second_block", "PipelineType" : "NotSupport"}]},
	{"ID" : "18", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_clear_second_block_fu_2611.flow_control_loop_pipe_sequential_init_U", "Parent" : "17"},
	{"ID" : "19", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_len_store_double_fu_2616", "Parent" : "2", "Child" : ["20"],
		"CDFG" : "sha256_process_chunk_Pipeline_len_store_double",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "10", "EstimateLatencyMax" : "10",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "zext_ln266", "Type" : "None", "Direction" : "I"},
			{"Name" : "pad_block", "Type" : "Memory", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "len_store_double", "PipelineType" : "NotSupport"}]},
	{"ID" : "20", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_len_store_double_fu_2616.flow_control_loop_pipe_sequential_init_U", "Parent" : "19"},
	{"ID" : "21", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_len_store_single_fu_2622", "Parent" : "2", "Child" : ["22"],
		"CDFG" : "sha256_process_chunk_Pipeline_len_store_single",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "10", "EstimateLatencyMax" : "10",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "zext_ln266", "Type" : "None", "Direction" : "I"},
			{"Name" : "pad_block", "Type" : "Memory", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "len_store_single", "PipelineType" : "NotSupport"}]},
	{"ID" : "22", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_len_store_single_fu_2622.flow_control_loop_pipe_sequential_init_U", "Parent" : "21"},
	{"ID" : "23", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_and_write_chunk_Pipeline_write_digest_fu_109", "Parent" : "1", "Child" : ["24", "25"],
		"CDFG" : "sha256_process_and_write_chunk_Pipeline_write_digest",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "10", "EstimateLatencyMax" : "10",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "digest", "Type" : "None", "Direction" : "I"},
			{"Name" : "digest_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "digest_2", "Type" : "None", "Direction" : "I"},
			{"Name" : "digest_3", "Type" : "None", "Direction" : "I"},
			{"Name" : "digest_4", "Type" : "None", "Direction" : "I"},
			{"Name" : "digest_5", "Type" : "None", "Direction" : "I"},
			{"Name" : "digest_6", "Type" : "None", "Direction" : "I"},
			{"Name" : "digest_7", "Type" : "None", "Direction" : "I"},
			{"Name" : "shiftreg_out", "Type" : "Vld", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "write_digest", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter1", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter1", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter1", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "24", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_and_write_chunk_Pipeline_write_digest_fu_109.mux_84_32_1_1_U66", "Parent" : "23"},
	{"ID" : "25", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_fu_108.grp_sha256_process_and_write_chunk_Pipeline_write_digest_fu_109.flow_control_loop_pipe_sequential_init_U", "Parent" : "23"},
	{"ID" : "26", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.control_s_axi_U", "Parent" : "0"},
	{"ID" : "27", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.gmem0_m_axi_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	calculate_sha256 {
		gmem0 {Type IO LastRead 136 FirstWrite -1}
		data_in {Type I LastRead 0 FirstWrite -1}
		crc_out {Type I LastRead 0 FirstWrite -1}
		numChunks {Type I LastRead 0 FirstWrite -1}
		chunkSize {Type I LastRead 0 FirstWrite -1}
		SHA256_K {Type I LastRead -1 FirstWrite -1}}
	sha256_process_and_write_chunk {
		gmem0 {Type IO LastRead 136 FirstWrite -1}
		chunk {Type I LastRead 0 FirstWrite -1}
		chunk_size {Type I LastRead 0 FirstWrite -1}
		out_words {Type I LastRead 3 FirstWrite -1}
		SHA256_K {Type I LastRead -1 FirstWrite -1}}
	sha256_process_chunk {
		gmem0 {Type I LastRead 136 FirstWrite -1}
		chunk {Type I LastRead 1 FirstWrite -1}
		chunk_size {Type I LastRead 0 FirstWrite -1}
		SHA256_K {Type I LastRead -1 FirstWrite -1}}
	sha256_compress_block {
		block_V_read {Type I LastRead 0 FirstWrite -1}
		state_0_read {Type I LastRead 0 FirstWrite -1}
		state_1_read {Type I LastRead 0 FirstWrite -1}
		state_2_read {Type I LastRead 0 FirstWrite -1}
		state_3_read {Type I LastRead 0 FirstWrite -1}
		state_4_read {Type I LastRead 0 FirstWrite -1}
		state_5_read {Type I LastRead 0 FirstWrite -1}
		state_6_read {Type I LastRead 0 FirstWrite -1}
		state_7_read {Type I LastRead 0 FirstWrite -1}
		SHA256_K {Type I LastRead -1 FirstWrite -1}}
	sha256_compress_block_Pipeline_round_loop {
		wbuf_15 {Type I LastRead 0 FirstWrite -1}
		wbuf_14 {Type I LastRead 0 FirstWrite -1}
		wbuf_13 {Type I LastRead 0 FirstWrite -1}
		wbuf_12 {Type I LastRead 0 FirstWrite -1}
		wbuf_11 {Type I LastRead 0 FirstWrite -1}
		wbuf_10 {Type I LastRead 0 FirstWrite -1}
		wbuf_9 {Type I LastRead 0 FirstWrite -1}
		wbuf_8 {Type I LastRead 0 FirstWrite -1}
		wbuf_7 {Type I LastRead 0 FirstWrite -1}
		wbuf_6 {Type I LastRead 0 FirstWrite -1}
		wbuf_5 {Type I LastRead 0 FirstWrite -1}
		wbuf_4 {Type I LastRead 0 FirstWrite -1}
		wbuf_3 {Type I LastRead 0 FirstWrite -1}
		wbuf_2 {Type I LastRead 0 FirstWrite -1}
		wbuf_1 {Type I LastRead 0 FirstWrite -1}
		wbuf {Type I LastRead 0 FirstWrite -1}
		state_0_read {Type I LastRead 0 FirstWrite -1}
		state_1_read {Type I LastRead 0 FirstWrite -1}
		state_2_read {Type I LastRead 0 FirstWrite -1}
		state_3_read {Type I LastRead 0 FirstWrite -1}
		state_4_read {Type I LastRead 0 FirstWrite -1}
		state_5_read {Type I LastRead 0 FirstWrite -1}
		state_6_read {Type I LastRead 0 FirstWrite -1}
		state_7_read {Type I LastRead 0 FirstWrite -1}
		a_2_out {Type O LastRead -1 FirstWrite 2}
		b_out {Type O LastRead -1 FirstWrite 2}
		c_out {Type O LastRead -1 FirstWrite 2}
		d_out {Type O LastRead -1 FirstWrite 2}
		e_2_out {Type O LastRead -1 FirstWrite 2}
		f_out {Type O LastRead -1 FirstWrite 2}
		g_out {Type O LastRead -1 FirstWrite 2}
		h_out {Type O LastRead -1 FirstWrite 2}
		SHA256_K {Type I LastRead -1 FirstWrite -1}}
	sha256_process_chunk_Pipeline_clear_pad_block {
		pad_block {Type O LastRead -1 FirstWrite 0}}
	sha256_process_chunk_Pipeline_copy_remainder {
		rem_r {Type I LastRead 0 FirstWrite -1}
		chunk {Type I LastRead 0 FirstWrite -1}
		zext_ln264_1 {Type I LastRead 0 FirstWrite -1}
		gmem0 {Type I LastRead 71 FirstWrite -1}
		trunc_ln4 {Type I LastRead 0 FirstWrite -1}
		pad_block {Type O LastRead -1 FirstWrite 72}}
	sha256_process_chunk_Pipeline_clear_second_block {
		pad_block {Type O LastRead -1 FirstWrite 0}}
	sha256_process_chunk_Pipeline_len_store_double {
		zext_ln266 {Type I LastRead 0 FirstWrite -1}
		pad_block {Type O LastRead -1 FirstWrite 0}}
	sha256_process_chunk_Pipeline_len_store_single {
		zext_ln266 {Type I LastRead 0 FirstWrite -1}
		pad_block {Type O LastRead -1 FirstWrite 0}}
	sha256_process_and_write_chunk_Pipeline_write_digest {
		digest {Type I LastRead 0 FirstWrite -1}
		digest_1 {Type I LastRead 0 FirstWrite -1}
		digest_2 {Type I LastRead 0 FirstWrite -1}
		digest_3 {Type I LastRead 0 FirstWrite -1}
		digest_4 {Type I LastRead 0 FirstWrite -1}
		digest_5 {Type I LastRead 0 FirstWrite -1}
		digest_6 {Type I LastRead 0 FirstWrite -1}
		digest_7 {Type I LastRead 0 FirstWrite -1}
		shiftreg_out {Type O LastRead -1 FirstWrite 1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "0", "Max" : "0"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	gmem0 { m_axi {  { m_axi_gmem0_AWVALID VALID 1 1 }  { m_axi_gmem0_AWREADY READY 0 1 }  { m_axi_gmem0_AWADDR ADDR 1 64 }  { m_axi_gmem0_AWID ID 1 1 }  { m_axi_gmem0_AWLEN SIZE 1 8 }  { m_axi_gmem0_AWSIZE BURST 1 3 }  { m_axi_gmem0_AWBURST LOCK 1 2 }  { m_axi_gmem0_AWLOCK CACHE 1 2 }  { m_axi_gmem0_AWCACHE PROT 1 4 }  { m_axi_gmem0_AWPROT QOS 1 3 }  { m_axi_gmem0_AWQOS REGION 1 4 }  { m_axi_gmem0_AWREGION USER 1 4 }  { m_axi_gmem0_AWUSER DATA 1 1 }  { m_axi_gmem0_WVALID VALID 1 1 }  { m_axi_gmem0_WREADY READY 0 1 }  { m_axi_gmem0_WDATA FIFONUM 1 512 }  { m_axi_gmem0_WSTRB STRB 1 64 }  { m_axi_gmem0_WLAST LAST 1 1 }  { m_axi_gmem0_WID ID 1 1 }  { m_axi_gmem0_WUSER DATA 1 1 }  { m_axi_gmem0_ARVALID VALID 1 1 }  { m_axi_gmem0_ARREADY READY 0 1 }  { m_axi_gmem0_ARADDR ADDR 1 64 }  { m_axi_gmem0_ARID ID 1 1 }  { m_axi_gmem0_ARLEN SIZE 1 8 }  { m_axi_gmem0_ARSIZE BURST 1 3 }  { m_axi_gmem0_ARBURST LOCK 1 2 }  { m_axi_gmem0_ARLOCK CACHE 1 2 }  { m_axi_gmem0_ARCACHE PROT 1 4 }  { m_axi_gmem0_ARPROT QOS 1 3 }  { m_axi_gmem0_ARQOS REGION 1 4 }  { m_axi_gmem0_ARREGION USER 1 4 }  { m_axi_gmem0_ARUSER DATA 1 1 }  { m_axi_gmem0_RVALID VALID 0 1 }  { m_axi_gmem0_RREADY READY 1 1 }  { m_axi_gmem0_RDATA FIFONUM 0 512 }  { m_axi_gmem0_RLAST LAST 0 1 }  { m_axi_gmem0_RID ID 0 1 }  { m_axi_gmem0_RUSER DATA 0 1 }  { m_axi_gmem0_RRESP RESP 0 2 }  { m_axi_gmem0_BVALID VALID 0 1 }  { m_axi_gmem0_BREADY READY 1 1 }  { m_axi_gmem0_BRESP RESP 0 2 }  { m_axi_gmem0_BID ID 0 1 }  { m_axi_gmem0_BUSER DATA 0 1 } } }
}

set maxi_interface_dict [dict create]
dict set maxi_interface_dict gmem0 {NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 READ_WRITE_MODE READ_WRITE}

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
	{ gmem0 64 }
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
	{ gmem0 64 }
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
