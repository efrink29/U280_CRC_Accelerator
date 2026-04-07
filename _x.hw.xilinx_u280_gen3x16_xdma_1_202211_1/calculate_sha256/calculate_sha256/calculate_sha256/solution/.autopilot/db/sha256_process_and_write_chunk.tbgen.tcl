set moduleName sha256_process_and_write_chunk
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
set C_modelName {sha256_process_and_write_chunk}
set C_modelType { void 0 }
set C_modelArgList {
	{ gmem0 int 512 regular {axi_master 2}  }
	{ chunk int 64 regular  }
	{ chunk_size int 32 regular  }
	{ out_words int 64 regular  }
}
set C_modelArgMapList {[ 
	{ "Name" : "gmem0", "interface" : "axi_master", "bitwidth" : 512, "direction" : "READWRITE", "bitSlice":[ {"cElement": [{"cName": "data_in","offset": { "type": "dynamic","port_name": "data_in","bundle": "control"},"direction": "READONLY"},{"cName": "crc_out","offset": { "type": "dynamic","port_name": "crc_out","bundle": "control"},"direction": "WRITEONLY"}]}]} , 
 	{ "Name" : "chunk", "interface" : "wire", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "chunk_size", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "out_words", "interface" : "wire", "bitwidth" : 64, "direction" : "READONLY"} ]}
# RTL Port declarations: 
set portNum 55
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
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
	{ m_axi_gmem0_WDATA sc_out sc_lv 512 signal 0 } 
	{ m_axi_gmem0_WSTRB sc_out sc_lv 64 signal 0 } 
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
	{ m_axi_gmem0_RDATA sc_in sc_lv 512 signal 0 } 
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
	{ chunk sc_in sc_lv 64 signal 1 } 
	{ chunk_size sc_in sc_lv 32 signal 2 } 
	{ out_words sc_in sc_lv 64 signal 3 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
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
 	{ "name": "m_axi_gmem0_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":512, "type": "signal", "bundle":{"name": "gmem0", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem0_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem0", "role": "WSTRB" }} , 
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
 	{ "name": "m_axi_gmem0_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":512, "type": "signal", "bundle":{"name": "gmem0", "role": "RDATA" }} , 
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
 	{ "name": "chunk", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "chunk", "role": "default" }} , 
 	{ "name": "chunk_size", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "chunk_size", "role": "default" }} , 
 	{ "name": "out_words", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "out_words", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "22"],
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
					{"ID" : "1", "SubInstance" : "grp_sha256_process_chunk_fu_97", "Port" : "gmem0", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "chunk", "Type" : "None", "Direction" : "I"},
			{"Name" : "chunk_size", "Type" : "None", "Direction" : "I"},
			{"Name" : "out_words", "Type" : "None", "Direction" : "I"},
			{"Name" : "SHA256_K", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_sha256_process_chunk_fu_97", "Port" : "SHA256_K", "Inst_start_state" : "1", "Inst_end_state" : "2"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97", "Parent" : "0", "Child" : ["2", "3", "12", "14", "16", "18", "20"],
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
					{"ID" : "14", "SubInstance" : "grp_sha256_process_chunk_Pipeline_copy_remainder_fu_2600", "Port" : "gmem0", "Inst_start_state" : "214", "Inst_end_state" : "215"}]},
			{"Name" : "chunk", "Type" : "None", "Direction" : "I"},
			{"Name" : "chunk_size", "Type" : "None", "Direction" : "I"},
			{"Name" : "SHA256_K", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "3", "SubInstance" : "grp_sha256_compress_block_fu_2572", "Port" : "SHA256_K", "Inst_start_state" : "322", "Inst_end_state" : "323"}]}],
		"Loop" : [
			{"Name" : "full_block_loop_vec", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "323", "FirstState" : "ap_ST_fsm_state209", "LastState" : ["ap_ST_fsm_state212"], "QuitState" : ["ap_ST_fsm_state209"], "PreState" : ["ap_ST_fsm_state3", "ap_ST_fsm_state2", "ap_ST_fsm_state208"], "PostState" : ["ap_ST_fsm_state213"], "OneDepthLoop" : "0", "OneStateBlock": ""}},
			{"Name" : "full_block_loop_scalar", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "323", "FirstState" : "ap_ST_fsm_state3", "LastState" : ["ap_ST_fsm_state139"], "QuitState" : ["ap_ST_fsm_state3"], "PreState" : ["ap_ST_fsm_state2"], "PostState" : ["ap_ST_fsm_state209"], "OneDepthLoop" : "0", "OneStateBlock": ""}}]},
	{"ID" : "2", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.pad_block_U", "Parent" : "1"},
	{"ID" : "3", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_compress_block_fu_2572", "Parent" : "1", "Child" : ["4"],
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
					{"ID" : "4", "SubInstance" : "grp_sha256_compress_block_Pipeline_round_loop_fu_372", "Port" : "SHA256_K", "Inst_start_state" : "1", "Inst_end_state" : "2"}]}]},
	{"ID" : "4", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_compress_block_fu_2572.grp_sha256_compress_block_Pipeline_round_loop_fu_372", "Parent" : "3", "Child" : ["5", "6", "7", "8", "9", "10", "11"],
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
	{"ID" : "5", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_compress_block_fu_2572.grp_sha256_compress_block_Pipeline_round_loop_fu_372.SHA256_K_U", "Parent" : "4"},
	{"ID" : "6", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_compress_block_fu_2572.grp_sha256_compress_block_Pipeline_round_loop_fu_372.mux_164_32_1_1_U1", "Parent" : "4"},
	{"ID" : "7", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_compress_block_fu_2572.grp_sha256_compress_block_Pipeline_round_loop_fu_372.mux_164_32_1_1_U2", "Parent" : "4"},
	{"ID" : "8", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_compress_block_fu_2572.grp_sha256_compress_block_Pipeline_round_loop_fu_372.mux_164_32_1_1_U3", "Parent" : "4"},
	{"ID" : "9", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_compress_block_fu_2572.grp_sha256_compress_block_Pipeline_round_loop_fu_372.mux_164_32_1_1_U4", "Parent" : "4"},
	{"ID" : "10", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_compress_block_fu_2572.grp_sha256_compress_block_Pipeline_round_loop_fu_372.mux_167_32_1_1_U5", "Parent" : "4"},
	{"ID" : "11", "Level" : "4", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_compress_block_fu_2572.grp_sha256_compress_block_Pipeline_round_loop_fu_372.flow_control_loop_pipe_sequential_init_U", "Parent" : "4"},
	{"ID" : "12", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_clear_pad_block_fu_2595", "Parent" : "1", "Child" : ["13"],
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
	{"ID" : "13", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_clear_pad_block_fu_2595.flow_control_loop_pipe_sequential_init_U", "Parent" : "12"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_copy_remainder_fu_2600", "Parent" : "1", "Child" : ["15"],
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
	{"ID" : "15", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_copy_remainder_fu_2600.flow_control_loop_pipe_sequential_init_U", "Parent" : "14"},
	{"ID" : "16", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_clear_second_block_fu_2611", "Parent" : "1", "Child" : ["17"],
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
	{"ID" : "17", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_clear_second_block_fu_2611.flow_control_loop_pipe_sequential_init_U", "Parent" : "16"},
	{"ID" : "18", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_len_store_double_fu_2616", "Parent" : "1", "Child" : ["19"],
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
	{"ID" : "19", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_len_store_double_fu_2616.flow_control_loop_pipe_sequential_init_U", "Parent" : "18"},
	{"ID" : "20", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_len_store_single_fu_2622", "Parent" : "1", "Child" : ["21"],
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
	{"ID" : "21", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_chunk_fu_97.grp_sha256_process_chunk_Pipeline_len_store_single_fu_2622.flow_control_loop_pipe_sequential_init_U", "Parent" : "20"},
	{"ID" : "22", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_Pipeline_write_digest_fu_109", "Parent" : "0", "Child" : ["23", "24"],
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
	{"ID" : "23", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_Pipeline_write_digest_fu_109.mux_84_32_1_1_U66", "Parent" : "22"},
	{"ID" : "24", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sha256_process_and_write_chunk_Pipeline_write_digest_fu_109.flow_control_loop_pipe_sequential_init_U", "Parent" : "22"}]}


set ArgLastReadFirstWriteLatency {
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
	{"Name" : "Latency", "Min" : "273", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "273", "Max" : "-1"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	 { m_axi {  { m_axi_gmem0_AWVALID VALID 1 1 }  { m_axi_gmem0_AWREADY READY 0 1 }  { m_axi_gmem0_AWADDR ADDR 1 64 }  { m_axi_gmem0_AWID ID 1 1 }  { m_axi_gmem0_AWLEN SIZE 1 32 }  { m_axi_gmem0_AWSIZE BURST 1 3 }  { m_axi_gmem0_AWBURST LOCK 1 2 }  { m_axi_gmem0_AWLOCK CACHE 1 2 }  { m_axi_gmem0_AWCACHE PROT 1 4 }  { m_axi_gmem0_AWPROT QOS 1 3 }  { m_axi_gmem0_AWQOS REGION 1 4 }  { m_axi_gmem0_AWREGION USER 1 4 }  { m_axi_gmem0_AWUSER DATA 1 1 }  { m_axi_gmem0_WVALID VALID 1 1 }  { m_axi_gmem0_WREADY READY 0 1 }  { m_axi_gmem0_WDATA FIFONUM 1 512 }  { m_axi_gmem0_WSTRB STRB 1 64 }  { m_axi_gmem0_WLAST LAST 1 1 }  { m_axi_gmem0_WID ID 1 1 }  { m_axi_gmem0_WUSER DATA 1 1 }  { m_axi_gmem0_ARVALID VALID 1 1 }  { m_axi_gmem0_ARREADY READY 0 1 }  { m_axi_gmem0_ARADDR ADDR 1 64 }  { m_axi_gmem0_ARID ID 1 1 }  { m_axi_gmem0_ARLEN SIZE 1 32 }  { m_axi_gmem0_ARSIZE BURST 1 3 }  { m_axi_gmem0_ARBURST LOCK 1 2 }  { m_axi_gmem0_ARLOCK CACHE 1 2 }  { m_axi_gmem0_ARCACHE PROT 1 4 }  { m_axi_gmem0_ARPROT QOS 1 3 }  { m_axi_gmem0_ARQOS REGION 1 4 }  { m_axi_gmem0_ARREGION USER 1 4 }  { m_axi_gmem0_ARUSER DATA 1 1 }  { m_axi_gmem0_RVALID VALID 0 1 }  { m_axi_gmem0_RREADY READY 1 1 }  { m_axi_gmem0_RDATA FIFONUM 0 512 }  { m_axi_gmem0_RLAST LAST 0 1 }  { m_axi_gmem0_RID ID 0 1 }  { m_axi_gmem0_RFIFONUM LEN 0 9 }  { m_axi_gmem0_RUSER DATA 0 1 }  { m_axi_gmem0_RRESP RESP 0 2 }  { m_axi_gmem0_BVALID VALID 0 1 }  { m_axi_gmem0_BREADY READY 1 1 }  { m_axi_gmem0_BRESP RESP 0 2 }  { m_axi_gmem0_BID ID 0 1 }  { m_axi_gmem0_BUSER DATA 0 1 } } }
	chunk { ap_none {  { chunk in_data 0 64 } } }
	chunk_size { ap_none {  { chunk_size in_data 0 32 } } }
	out_words { ap_none {  { out_words in_data 0 64 } } }
}
