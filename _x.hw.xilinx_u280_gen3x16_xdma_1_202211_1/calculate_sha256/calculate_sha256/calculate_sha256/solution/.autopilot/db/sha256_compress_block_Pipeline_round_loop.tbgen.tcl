set moduleName sha256_compress_block_Pipeline_round_loop
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
set C_modelName {sha256_compress_block_Pipeline_round_loop}
set C_modelType { void 0 }
set C_modelArgList {
	{ wbuf_15 int 32 regular  }
	{ wbuf_14 int 32 regular  }
	{ wbuf_13 int 32 regular  }
	{ wbuf_12 int 32 regular  }
	{ wbuf_11 int 32 regular  }
	{ wbuf_10 int 32 regular  }
	{ wbuf_9 int 32 regular  }
	{ wbuf_8 int 32 regular  }
	{ wbuf_7 int 32 regular  }
	{ wbuf_6 int 32 regular  }
	{ wbuf_5 int 32 regular  }
	{ wbuf_4 int 32 regular  }
	{ wbuf_3 int 32 regular  }
	{ wbuf_2 int 32 regular  }
	{ wbuf_1 int 32 regular  }
	{ wbuf int 32 regular  }
	{ state_0_read int 32 regular  }
	{ state_1_read int 32 regular  }
	{ state_2_read int 32 regular  }
	{ state_3_read int 32 regular  }
	{ state_4_read int 32 regular  }
	{ state_5_read int 32 regular  }
	{ state_6_read int 32 regular  }
	{ state_7_read int 32 regular  }
	{ a_2_out int 32 regular {pointer 1}  }
	{ b_out int 32 regular {pointer 1}  }
	{ c_out int 32 regular {pointer 1}  }
	{ d_out int 32 regular {pointer 1}  }
	{ e_2_out int 32 regular {pointer 1}  }
	{ f_out int 32 regular {pointer 1}  }
	{ g_out int 32 regular {pointer 1}  }
	{ h_out int 32 regular {pointer 1}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "wbuf_15", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "wbuf_14", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "wbuf_13", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "wbuf_12", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "wbuf_11", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "wbuf_10", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "wbuf_9", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "wbuf_8", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "wbuf_7", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "wbuf_6", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "wbuf_5", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "wbuf_4", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "wbuf_3", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "wbuf_2", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "wbuf_1", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "wbuf", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "state_0_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "state_1_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "state_2_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "state_3_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "state_4_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "state_5_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "state_6_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "state_7_read", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "a_2_out", "interface" : "wire", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "b_out", "interface" : "wire", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "c_out", "interface" : "wire", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "d_out", "interface" : "wire", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "e_2_out", "interface" : "wire", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "f_out", "interface" : "wire", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "g_out", "interface" : "wire", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "h_out", "interface" : "wire", "bitwidth" : 32, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 46
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ wbuf_15 sc_in sc_lv 32 signal 0 } 
	{ wbuf_14 sc_in sc_lv 32 signal 1 } 
	{ wbuf_13 sc_in sc_lv 32 signal 2 } 
	{ wbuf_12 sc_in sc_lv 32 signal 3 } 
	{ wbuf_11 sc_in sc_lv 32 signal 4 } 
	{ wbuf_10 sc_in sc_lv 32 signal 5 } 
	{ wbuf_9 sc_in sc_lv 32 signal 6 } 
	{ wbuf_8 sc_in sc_lv 32 signal 7 } 
	{ wbuf_7 sc_in sc_lv 32 signal 8 } 
	{ wbuf_6 sc_in sc_lv 32 signal 9 } 
	{ wbuf_5 sc_in sc_lv 32 signal 10 } 
	{ wbuf_4 sc_in sc_lv 32 signal 11 } 
	{ wbuf_3 sc_in sc_lv 32 signal 12 } 
	{ wbuf_2 sc_in sc_lv 32 signal 13 } 
	{ wbuf_1 sc_in sc_lv 32 signal 14 } 
	{ wbuf sc_in sc_lv 32 signal 15 } 
	{ state_0_read sc_in sc_lv 32 signal 16 } 
	{ state_1_read sc_in sc_lv 32 signal 17 } 
	{ state_2_read sc_in sc_lv 32 signal 18 } 
	{ state_3_read sc_in sc_lv 32 signal 19 } 
	{ state_4_read sc_in sc_lv 32 signal 20 } 
	{ state_5_read sc_in sc_lv 32 signal 21 } 
	{ state_6_read sc_in sc_lv 32 signal 22 } 
	{ state_7_read sc_in sc_lv 32 signal 23 } 
	{ a_2_out sc_out sc_lv 32 signal 24 } 
	{ a_2_out_ap_vld sc_out sc_logic 1 outvld 24 } 
	{ b_out sc_out sc_lv 32 signal 25 } 
	{ b_out_ap_vld sc_out sc_logic 1 outvld 25 } 
	{ c_out sc_out sc_lv 32 signal 26 } 
	{ c_out_ap_vld sc_out sc_logic 1 outvld 26 } 
	{ d_out sc_out sc_lv 32 signal 27 } 
	{ d_out_ap_vld sc_out sc_logic 1 outvld 27 } 
	{ e_2_out sc_out sc_lv 32 signal 28 } 
	{ e_2_out_ap_vld sc_out sc_logic 1 outvld 28 } 
	{ f_out sc_out sc_lv 32 signal 29 } 
	{ f_out_ap_vld sc_out sc_logic 1 outvld 29 } 
	{ g_out sc_out sc_lv 32 signal 30 } 
	{ g_out_ap_vld sc_out sc_logic 1 outvld 30 } 
	{ h_out sc_out sc_lv 32 signal 31 } 
	{ h_out_ap_vld sc_out sc_logic 1 outvld 31 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "wbuf_15", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "wbuf_15", "role": "default" }} , 
 	{ "name": "wbuf_14", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "wbuf_14", "role": "default" }} , 
 	{ "name": "wbuf_13", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "wbuf_13", "role": "default" }} , 
 	{ "name": "wbuf_12", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "wbuf_12", "role": "default" }} , 
 	{ "name": "wbuf_11", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "wbuf_11", "role": "default" }} , 
 	{ "name": "wbuf_10", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "wbuf_10", "role": "default" }} , 
 	{ "name": "wbuf_9", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "wbuf_9", "role": "default" }} , 
 	{ "name": "wbuf_8", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "wbuf_8", "role": "default" }} , 
 	{ "name": "wbuf_7", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "wbuf_7", "role": "default" }} , 
 	{ "name": "wbuf_6", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "wbuf_6", "role": "default" }} , 
 	{ "name": "wbuf_5", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "wbuf_5", "role": "default" }} , 
 	{ "name": "wbuf_4", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "wbuf_4", "role": "default" }} , 
 	{ "name": "wbuf_3", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "wbuf_3", "role": "default" }} , 
 	{ "name": "wbuf_2", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "wbuf_2", "role": "default" }} , 
 	{ "name": "wbuf_1", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "wbuf_1", "role": "default" }} , 
 	{ "name": "wbuf", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "wbuf", "role": "default" }} , 
 	{ "name": "state_0_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "state_0_read", "role": "default" }} , 
 	{ "name": "state_1_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "state_1_read", "role": "default" }} , 
 	{ "name": "state_2_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "state_2_read", "role": "default" }} , 
 	{ "name": "state_3_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "state_3_read", "role": "default" }} , 
 	{ "name": "state_4_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "state_4_read", "role": "default" }} , 
 	{ "name": "state_5_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "state_5_read", "role": "default" }} , 
 	{ "name": "state_6_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "state_6_read", "role": "default" }} , 
 	{ "name": "state_7_read", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "state_7_read", "role": "default" }} , 
 	{ "name": "a_2_out", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "a_2_out", "role": "default" }} , 
 	{ "name": "a_2_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "a_2_out", "role": "ap_vld" }} , 
 	{ "name": "b_out", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "b_out", "role": "default" }} , 
 	{ "name": "b_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "b_out", "role": "ap_vld" }} , 
 	{ "name": "c_out", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "c_out", "role": "default" }} , 
 	{ "name": "c_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "c_out", "role": "ap_vld" }} , 
 	{ "name": "d_out", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "d_out", "role": "default" }} , 
 	{ "name": "d_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "d_out", "role": "ap_vld" }} , 
 	{ "name": "e_2_out", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "e_2_out", "role": "default" }} , 
 	{ "name": "e_2_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "e_2_out", "role": "ap_vld" }} , 
 	{ "name": "f_out", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "f_out", "role": "default" }} , 
 	{ "name": "f_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "f_out", "role": "ap_vld" }} , 
 	{ "name": "g_out", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "g_out", "role": "default" }} , 
 	{ "name": "g_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "g_out", "role": "ap_vld" }} , 
 	{ "name": "h_out", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "h_out", "role": "default" }} , 
 	{ "name": "h_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "h_out", "role": "ap_vld" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7"],
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
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.SHA256_K_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mux_164_32_1_1_U1", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mux_164_32_1_1_U2", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mux_164_32_1_1_U3", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mux_164_32_1_1_U4", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mux_167_32_1_1_U5", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
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
	{"Name" : "Latency", "Min" : "68", "Max" : "68"}
	, {"Name" : "Interval", "Min" : "68", "Max" : "68"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	wbuf_15 { ap_none {  { wbuf_15 in_data 0 32 } } }
	wbuf_14 { ap_none {  { wbuf_14 in_data 0 32 } } }
	wbuf_13 { ap_none {  { wbuf_13 in_data 0 32 } } }
	wbuf_12 { ap_none {  { wbuf_12 in_data 0 32 } } }
	wbuf_11 { ap_none {  { wbuf_11 in_data 0 32 } } }
	wbuf_10 { ap_none {  { wbuf_10 in_data 0 32 } } }
	wbuf_9 { ap_none {  { wbuf_9 in_data 0 32 } } }
	wbuf_8 { ap_none {  { wbuf_8 in_data 0 32 } } }
	wbuf_7 { ap_none {  { wbuf_7 in_data 0 32 } } }
	wbuf_6 { ap_none {  { wbuf_6 in_data 0 32 } } }
	wbuf_5 { ap_none {  { wbuf_5 in_data 0 32 } } }
	wbuf_4 { ap_none {  { wbuf_4 in_data 0 32 } } }
	wbuf_3 { ap_none {  { wbuf_3 in_data 0 32 } } }
	wbuf_2 { ap_none {  { wbuf_2 in_data 0 32 } } }
	wbuf_1 { ap_none {  { wbuf_1 in_data 0 32 } } }
	wbuf { ap_none {  { wbuf in_data 0 32 } } }
	state_0_read { ap_none {  { state_0_read in_data 0 32 } } }
	state_1_read { ap_none {  { state_1_read in_data 0 32 } } }
	state_2_read { ap_none {  { state_2_read in_data 0 32 } } }
	state_3_read { ap_none {  { state_3_read in_data 0 32 } } }
	state_4_read { ap_none {  { state_4_read in_data 0 32 } } }
	state_5_read { ap_none {  { state_5_read in_data 0 32 } } }
	state_6_read { ap_none {  { state_6_read in_data 0 32 } } }
	state_7_read { ap_none {  { state_7_read in_data 0 32 } } }
	a_2_out { ap_vld {  { a_2_out out_data 1 32 }  { a_2_out_ap_vld out_vld 1 1 } } }
	b_out { ap_vld {  { b_out out_data 1 32 }  { b_out_ap_vld out_vld 1 1 } } }
	c_out { ap_vld {  { c_out out_data 1 32 }  { c_out_ap_vld out_vld 1 1 } } }
	d_out { ap_vld {  { d_out out_data 1 32 }  { d_out_ap_vld out_vld 1 1 } } }
	e_2_out { ap_vld {  { e_2_out out_data 1 32 }  { e_2_out_ap_vld out_vld 1 1 } } }
	f_out { ap_vld {  { f_out out_data 1 32 }  { f_out_ap_vld out_vld 1 1 } } }
	g_out { ap_vld {  { g_out out_data 1 32 }  { g_out_ap_vld out_vld 1 1 } } }
	h_out { ap_vld {  { h_out out_data 1 32 }  { h_out_ap_vld out_vld 1 1 } } }
}
