set moduleName sha256_compress_block
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
set C_modelName {sha256_compress_block}
set C_modelType { int 256 }
set C_modelArgList {
	{ block_V_read int 512 regular  }
	{ state_0_read int 32 regular  }
	{ state_1_read int 32 regular  }
	{ state_2_read int 32 regular  }
	{ state_3_read int 32 regular  }
	{ state_4_read int 32 regular  }
	{ state_5_read int 32 regular  }
	{ state_6_read int 32 regular  }
	{ state_7_read int 32 regular  }
}
set C_modelArgMapList {[ 
	{ "Name" : "block_V_read", "interface" : "wire", "bitwidth" : 512, "direction" : "READONLY"} , 
 	{ "Name" : "state_0_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "state_1_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "state_2_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "state_3_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "state_4_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "state_5_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "state_6_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "state_7_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "ap_return", "interface" : "wire", "bitwidth" : 256} ]}
# RTL Port declarations: 
set portNum 23
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ block_V_read sc_in sc_lv 512 signal 0 } 
	{ state_0_read sc_in sc_lv 32 signal 1 } 
	{ state_1_read sc_in sc_lv 32 signal 2 } 
	{ state_2_read sc_in sc_lv 32 signal 3 } 
	{ state_3_read sc_in sc_lv 32 signal 4 } 
	{ state_4_read sc_in sc_lv 32 signal 5 } 
	{ state_5_read sc_in sc_lv 32 signal 6 } 
	{ state_6_read sc_in sc_lv 32 signal 7 } 
	{ state_7_read sc_in sc_lv 32 signal 8 } 
	{ ap_return_0 sc_out sc_lv 32 signal -1 } 
	{ ap_return_1 sc_out sc_lv 32 signal -1 } 
	{ ap_return_2 sc_out sc_lv 32 signal -1 } 
	{ ap_return_3 sc_out sc_lv 32 signal -1 } 
	{ ap_return_4 sc_out sc_lv 32 signal -1 } 
	{ ap_return_5 sc_out sc_lv 32 signal -1 } 
	{ ap_return_6 sc_out sc_lv 32 signal -1 } 
	{ ap_return_7 sc_out sc_lv 32 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "block_V_read", "direction": "in", "datatype": "sc_lv", "bitwidth":512, "type": "signal", "bundle":{"name": "block_V_read", "role": "default" }} , 
 	{ "name": "state_0_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "state_0_read", "role": "default" }} , 
 	{ "name": "state_1_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "state_1_read", "role": "default" }} , 
 	{ "name": "state_2_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "state_2_read", "role": "default" }} , 
 	{ "name": "state_3_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "state_3_read", "role": "default" }} , 
 	{ "name": "state_4_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "state_4_read", "role": "default" }} , 
 	{ "name": "state_5_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "state_5_read", "role": "default" }} , 
 	{ "name": "state_6_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "state_6_read", "role": "default" }} , 
 	{ "name": "state_7_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "state_7_read", "role": "default" }} , 
 	{ "name": "ap_return_0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "ap_return_0", "role": "default" }} , 
 	{ "name": "ap_return_1", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "ap_return_1", "role": "default" }} , 
 	{ "name": "ap_return_2", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "ap_return_2", "role": "default" }} , 
 	{ "name": "ap_return_3", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "ap_return_3", "role": "default" }} , 
 	{ "name": "ap_return_4", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "ap_return_4", "role": "default" }} , 
 	{ "name": "ap_return_5", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "ap_return_5", "role": "default" }} , 
 	{ "name": "ap_return_6", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "ap_return_6", "role": "default" }} , 
 	{ "name": "ap_return_7", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "ap_return_7", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1"],
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
					{"ID" : "1", "SubInstance" : "grp_sha256_compress_block_Pipeline_round_loop_fu_372", "Port" : "SHA256_K", "Inst_start_state" : "1", "Inst_end_state" : "2"}]}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_sha256_compress_block_Pipeline_round_loop_fu_372", "Parent" : "0", "Child" : ["2", "3", "4", "5", "6", "7", "8"],
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
	{"ID" : "2", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sha256_compress_block_Pipeline_round_loop_fu_372.SHA256_K_U", "Parent" : "1"},
	{"ID" : "3", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sha256_compress_block_Pipeline_round_loop_fu_372.mux_164_32_1_1_U1", "Parent" : "1"},
	{"ID" : "4", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sha256_compress_block_Pipeline_round_loop_fu_372.mux_164_32_1_1_U2", "Parent" : "1"},
	{"ID" : "5", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sha256_compress_block_Pipeline_round_loop_fu_372.mux_164_32_1_1_U3", "Parent" : "1"},
	{"ID" : "6", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sha256_compress_block_Pipeline_round_loop_fu_372.mux_164_32_1_1_U4", "Parent" : "1"},
	{"ID" : "7", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sha256_compress_block_Pipeline_round_loop_fu_372.mux_167_32_1_1_U5", "Parent" : "1"},
	{"ID" : "8", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_sha256_compress_block_Pipeline_round_loop_fu_372.flow_control_loop_pipe_sequential_init_U", "Parent" : "1"}]}


set ArgLastReadFirstWriteLatency {
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
		SHA256_K {Type I LastRead -1 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "70", "Max" : "70"}
	, {"Name" : "Interval", "Min" : "70", "Max" : "70"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	block_V_read { ap_none {  { block_V_read in_data 0 512 } } }
	state_0_read { ap_none {  { state_0_read in_data 0 32 } } }
	state_1_read { ap_none {  { state_1_read in_data 0 32 } } }
	state_2_read { ap_none {  { state_2_read in_data 0 32 } } }
	state_3_read { ap_none {  { state_3_read in_data 0 32 } } }
	state_4_read { ap_none {  { state_4_read in_data 0 32 } } }
	state_5_read { ap_none {  { state_5_read in_data 0 32 } } }
	state_6_read { ap_none {  { state_6_read in_data 0 32 } } }
	state_7_read { ap_none {  { state_7_read in_data 0 32 } } }
}
