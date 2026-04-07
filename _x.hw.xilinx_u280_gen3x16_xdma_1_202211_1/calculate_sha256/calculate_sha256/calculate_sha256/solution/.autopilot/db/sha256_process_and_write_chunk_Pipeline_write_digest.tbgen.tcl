set moduleName sha256_process_and_write_chunk_Pipeline_write_digest
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 1
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set C_modelName {sha256_process_and_write_chunk_Pipeline_write_digest}
set C_modelType { void 0 }
set C_modelArgList {
	{ digest int 32 regular  }
	{ digest_1 int 32 regular  }
	{ digest_2 int 32 regular  }
	{ digest_3 int 32 regular  }
	{ digest_4 int 32 regular  }
	{ digest_5 int 32 regular  }
	{ digest_6 int 32 regular  }
	{ digest_7 int 32 regular  }
	{ shiftreg_out int 256 regular {pointer 1}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "digest", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "digest_1", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "digest_2", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "digest_3", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "digest_4", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "digest_5", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "digest_6", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "digest_7", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "shiftreg_out", "interface" : "wire", "bitwidth" : 256, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 16
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ digest sc_in sc_lv 32 signal 0 } 
	{ digest_1 sc_in sc_lv 32 signal 1 } 
	{ digest_2 sc_in sc_lv 32 signal 2 } 
	{ digest_3 sc_in sc_lv 32 signal 3 } 
	{ digest_4 sc_in sc_lv 32 signal 4 } 
	{ digest_5 sc_in sc_lv 32 signal 5 } 
	{ digest_6 sc_in sc_lv 32 signal 6 } 
	{ digest_7 sc_in sc_lv 32 signal 7 } 
	{ shiftreg_out sc_out sc_lv 256 signal 8 } 
	{ shiftreg_out_ap_vld sc_out sc_logic 1 outvld 8 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "digest", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "digest", "role": "default" }} , 
 	{ "name": "digest_1", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "digest_1", "role": "default" }} , 
 	{ "name": "digest_2", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "digest_2", "role": "default" }} , 
 	{ "name": "digest_3", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "digest_3", "role": "default" }} , 
 	{ "name": "digest_4", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "digest_4", "role": "default" }} , 
 	{ "name": "digest_5", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "digest_5", "role": "default" }} , 
 	{ "name": "digest_6", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "digest_6", "role": "default" }} , 
 	{ "name": "digest_7", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "digest_7", "role": "default" }} , 
 	{ "name": "shiftreg_out", "direction": "out", "datatype": "sc_lv", "bitwidth":256, "type": "signal", "bundle":{"name": "shiftreg_out", "role": "default" }} , 
 	{ "name": "shiftreg_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "shiftreg_out", "role": "ap_vld" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2"],
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
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mux_84_32_1_1_U66", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
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
	{"Name" : "Latency", "Min" : "10", "Max" : "10"}
	, {"Name" : "Interval", "Min" : "10", "Max" : "10"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	digest { ap_none {  { digest in_data 0 32 } } }
	digest_1 { ap_none {  { digest_1 in_data 0 32 } } }
	digest_2 { ap_none {  { digest_2 in_data 0 32 } } }
	digest_3 { ap_none {  { digest_3 in_data 0 32 } } }
	digest_4 { ap_none {  { digest_4 in_data 0 32 } } }
	digest_5 { ap_none {  { digest_5 in_data 0 32 } } }
	digest_6 { ap_none {  { digest_6 in_data 0 32 } } }
	digest_7 { ap_none {  { digest_7 in_data 0 32 } } }
	shiftreg_out { ap_vld {  { shiftreg_out out_data 1 256 }  { shiftreg_out_ap_vld out_vld 1 1 } } }
}
