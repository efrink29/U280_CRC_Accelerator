set moduleName entry_proc
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 1
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {entry_proc}
set C_modelType { void 0 }
set C_modelArgList {
	{ crc_out int 64 regular  }
	{ crc_out_c int 64 regular {fifo 1}  }
	{ crc_size int 32 regular  }
	{ crc_size_c int 32 regular {fifo 1}  }
	{ init_value int 32 regular  }
	{ init_value_c int 32 regular {fifo 1}  }
}
set hasAXIMCache 0
set C_modelArgMapList {[ 
	{ "Name" : "crc_out", "interface" : "wire", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "crc_out_c", "interface" : "fifo", "bitwidth" : 64, "direction" : "WRITEONLY"} , 
 	{ "Name" : "crc_size", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crc_size_c", "interface" : "fifo", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "init_value", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "init_value_c", "interface" : "fifo", "bitwidth" : 32, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 28
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ start_full_n sc_in sc_logic 1 signal -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_continue sc_in sc_logic 1 continue -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ start_out sc_out sc_logic 1 signal -1 } 
	{ start_write sc_out sc_logic 1 signal -1 } 
	{ crc_out sc_in sc_lv 64 signal 0 } 
	{ crc_out_c_din sc_out sc_lv 64 signal 1 } 
	{ crc_out_c_num_data_valid sc_in sc_lv 3 signal 1 } 
	{ crc_out_c_fifo_cap sc_in sc_lv 3 signal 1 } 
	{ crc_out_c_full_n sc_in sc_logic 1 signal 1 } 
	{ crc_out_c_write sc_out sc_logic 1 signal 1 } 
	{ crc_size sc_in sc_lv 32 signal 2 } 
	{ crc_size_c_din sc_out sc_lv 32 signal 3 } 
	{ crc_size_c_num_data_valid sc_in sc_lv 3 signal 3 } 
	{ crc_size_c_fifo_cap sc_in sc_lv 3 signal 3 } 
	{ crc_size_c_full_n sc_in sc_logic 1 signal 3 } 
	{ crc_size_c_write sc_out sc_logic 1 signal 3 } 
	{ init_value sc_in sc_lv 32 signal 4 } 
	{ init_value_c_din sc_out sc_lv 32 signal 5 } 
	{ init_value_c_num_data_valid sc_in sc_lv 3 signal 5 } 
	{ init_value_c_fifo_cap sc_in sc_lv 3 signal 5 } 
	{ init_value_c_full_n sc_in sc_logic 1 signal 5 } 
	{ init_value_c_write sc_out sc_logic 1 signal 5 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "start_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_full_n", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_continue", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "continue", "bundle":{"name": "ap_continue", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "start_out", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_out", "role": "default" }} , 
 	{ "name": "start_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "start_write", "role": "default" }} , 
 	{ "name": "crc_out", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "crc_out", "role": "default" }} , 
 	{ "name": "crc_out_c_din", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "crc_out_c", "role": "din" }} , 
 	{ "name": "crc_out_c_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "crc_out_c", "role": "num_data_valid" }} , 
 	{ "name": "crc_out_c_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "crc_out_c", "role": "fifo_cap" }} , 
 	{ "name": "crc_out_c_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crc_out_c", "role": "full_n" }} , 
 	{ "name": "crc_out_c_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crc_out_c", "role": "write" }} , 
 	{ "name": "crc_size", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crc_size", "role": "default" }} , 
 	{ "name": "crc_size_c_din", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crc_size_c", "role": "din" }} , 
 	{ "name": "crc_size_c_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "crc_size_c", "role": "num_data_valid" }} , 
 	{ "name": "crc_size_c_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "crc_size_c", "role": "fifo_cap" }} , 
 	{ "name": "crc_size_c_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crc_size_c", "role": "full_n" }} , 
 	{ "name": "crc_size_c_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crc_size_c", "role": "write" }} , 
 	{ "name": "init_value", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "init_value", "role": "default" }} , 
 	{ "name": "init_value_c_din", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "init_value_c", "role": "din" }} , 
 	{ "name": "init_value_c_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "init_value_c", "role": "num_data_valid" }} , 
 	{ "name": "init_value_c_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "init_value_c", "role": "fifo_cap" }} , 
 	{ "name": "init_value_c_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "init_value_c", "role": "full_n" }} , 
 	{ "name": "init_value_c_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "init_value_c", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "",
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
			{"Name" : "crc_out_c", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "4", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "crc_out_c_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "crc_size", "Type" : "None", "Direction" : "I"},
			{"Name" : "crc_size_c", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "3", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "crc_size_c_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "init_value", "Type" : "None", "Direction" : "I"},
			{"Name" : "init_value_c", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "3", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "init_value_c_blk_n", "Type" : "RtlSignal"}]}]}]}


set ArgLastReadFirstWriteLatency {
	entry_proc {
		crc_out {Type I LastRead 0 FirstWrite -1}
		crc_out_c {Type O LastRead -1 FirstWrite 0}
		crc_size {Type I LastRead 0 FirstWrite -1}
		crc_size_c {Type O LastRead -1 FirstWrite 0}
		init_value {Type I LastRead 0 FirstWrite -1}
		init_value_c {Type O LastRead -1 FirstWrite 0}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "0", "Max" : "0"}
	, {"Name" : "Interval", "Min" : "0", "Max" : "0"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	crc_out { ap_none {  { crc_out in_data 0 64 } } }
	crc_out_c { ap_fifo {  { crc_out_c_din fifo_port_we 1 64 }  { crc_out_c_num_data_valid fifo_status_num_data_valid 0 3 }  { crc_out_c_fifo_cap fifo_update 0 3 }  { crc_out_c_full_n fifo_status 0 1 }  { crc_out_c_write fifo_data 1 1 } } }
	crc_size { ap_none {  { crc_size in_data 0 32 } } }
	crc_size_c { ap_fifo {  { crc_size_c_din fifo_port_we 1 32 }  { crc_size_c_num_data_valid fifo_status_num_data_valid 0 3 }  { crc_size_c_fifo_cap fifo_update 0 3 }  { crc_size_c_full_n fifo_status 0 1 }  { crc_size_c_write fifo_data 1 1 } } }
	init_value { ap_none {  { init_value in_data 0 32 } } }
	init_value_c { ap_fifo {  { init_value_c_din fifo_port_we 1 32 }  { init_value_c_num_data_valid fifo_status_num_data_valid 0 3 }  { init_value_c_fifo_cap fifo_update 0 3 }  { init_value_c_full_n fifo_status 0 1 }  { init_value_c_write fifo_data 1 1 } } }
}
