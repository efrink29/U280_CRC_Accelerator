set moduleName crc_process_full_blocks_Pipeline_full_block_loop
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
set C_modelName {crc_process_full_blocks_Pipeline_full_block_loop}
set C_modelType { void 0 }
set C_modelArgList {
	{ crc int 32 regular  }
	{ blocks_in_chunk int 28 regular  }
	{ inByte017 int 8 regular {fifo 0 volatile }  }
	{ inByte118 int 8 regular {fifo 0 volatile }  }
	{ inByte219 int 8 regular {fifo 0 volatile }  }
	{ inByte320 int 8 regular {fifo 0 volatile }  }
	{ inByte421 int 8 regular {fifo 0 volatile }  }
	{ inByte522 int 8 regular {fifo 0 volatile }  }
	{ inByte623 int 8 regular {fifo 0 volatile }  }
	{ inByte724 int 8 regular {fifo 0 volatile }  }
	{ inByte825 int 8 regular {fifo 0 volatile }  }
	{ inByte926 int 8 regular {fifo 0 volatile }  }
	{ inByte1027 int 8 regular {fifo 0 volatile }  }
	{ inByte1128 int 8 regular {fifo 0 volatile }  }
	{ inByte1229 int 8 regular {fifo 0 volatile }  }
	{ inByte1330 int 8 regular {fifo 0 volatile }  }
	{ inByte1431 int 8 regular {fifo 0 volatile }  }
	{ inByte1532 int 8 regular {fifo 0 volatile }  }
	{ crcTables_0 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_1 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_2 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_3 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_4 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_5 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_6 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_7 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_8 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_9 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_10 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_11 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_12 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_13 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_14 int 32 regular {array 256 { 1 } 1 1 }  }
	{ crcTables_15 int 32 regular {array 256 { 1 } 1 1 }  }
	{ mask int 32 regular  }
	{ p_out int 32 regular {pointer 1}  }
}
set C_modelArgMapList {[ 
	{ "Name" : "crc", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "blocks_in_chunk", "interface" : "wire", "bitwidth" : 28, "direction" : "READONLY"} , 
 	{ "Name" : "inByte017", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte118", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte219", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte320", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte421", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte522", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte623", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte724", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte825", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte926", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte1027", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte1128", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte1229", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte1330", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte1431", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte1532", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_0", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_1", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_2", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_3", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_4", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_5", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_6", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_7", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_8", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_9", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_10", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_11", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_12", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_13", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_14", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crcTables_15", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "mask", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "p_out", "interface" : "wire", "bitwidth" : 32, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 139
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ inByte017_dout sc_in sc_lv 8 signal 2 } 
	{ inByte017_num_data_valid sc_in sc_lv 7 signal 2 } 
	{ inByte017_fifo_cap sc_in sc_lv 7 signal 2 } 
	{ inByte017_empty_n sc_in sc_logic 1 signal 2 } 
	{ inByte017_read sc_out sc_logic 1 signal 2 } 
	{ inByte118_dout sc_in sc_lv 8 signal 3 } 
	{ inByte118_num_data_valid sc_in sc_lv 7 signal 3 } 
	{ inByte118_fifo_cap sc_in sc_lv 7 signal 3 } 
	{ inByte118_empty_n sc_in sc_logic 1 signal 3 } 
	{ inByte118_read sc_out sc_logic 1 signal 3 } 
	{ inByte219_dout sc_in sc_lv 8 signal 4 } 
	{ inByte219_num_data_valid sc_in sc_lv 7 signal 4 } 
	{ inByte219_fifo_cap sc_in sc_lv 7 signal 4 } 
	{ inByte219_empty_n sc_in sc_logic 1 signal 4 } 
	{ inByte219_read sc_out sc_logic 1 signal 4 } 
	{ inByte320_dout sc_in sc_lv 8 signal 5 } 
	{ inByte320_num_data_valid sc_in sc_lv 7 signal 5 } 
	{ inByte320_fifo_cap sc_in sc_lv 7 signal 5 } 
	{ inByte320_empty_n sc_in sc_logic 1 signal 5 } 
	{ inByte320_read sc_out sc_logic 1 signal 5 } 
	{ inByte421_dout sc_in sc_lv 8 signal 6 } 
	{ inByte421_num_data_valid sc_in sc_lv 7 signal 6 } 
	{ inByte421_fifo_cap sc_in sc_lv 7 signal 6 } 
	{ inByte421_empty_n sc_in sc_logic 1 signal 6 } 
	{ inByte421_read sc_out sc_logic 1 signal 6 } 
	{ inByte522_dout sc_in sc_lv 8 signal 7 } 
	{ inByte522_num_data_valid sc_in sc_lv 7 signal 7 } 
	{ inByte522_fifo_cap sc_in sc_lv 7 signal 7 } 
	{ inByte522_empty_n sc_in sc_logic 1 signal 7 } 
	{ inByte522_read sc_out sc_logic 1 signal 7 } 
	{ inByte623_dout sc_in sc_lv 8 signal 8 } 
	{ inByte623_num_data_valid sc_in sc_lv 7 signal 8 } 
	{ inByte623_fifo_cap sc_in sc_lv 7 signal 8 } 
	{ inByte623_empty_n sc_in sc_logic 1 signal 8 } 
	{ inByte623_read sc_out sc_logic 1 signal 8 } 
	{ inByte724_dout sc_in sc_lv 8 signal 9 } 
	{ inByte724_num_data_valid sc_in sc_lv 7 signal 9 } 
	{ inByte724_fifo_cap sc_in sc_lv 7 signal 9 } 
	{ inByte724_empty_n sc_in sc_logic 1 signal 9 } 
	{ inByte724_read sc_out sc_logic 1 signal 9 } 
	{ inByte825_dout sc_in sc_lv 8 signal 10 } 
	{ inByte825_num_data_valid sc_in sc_lv 7 signal 10 } 
	{ inByte825_fifo_cap sc_in sc_lv 7 signal 10 } 
	{ inByte825_empty_n sc_in sc_logic 1 signal 10 } 
	{ inByte825_read sc_out sc_logic 1 signal 10 } 
	{ inByte926_dout sc_in sc_lv 8 signal 11 } 
	{ inByte926_num_data_valid sc_in sc_lv 7 signal 11 } 
	{ inByte926_fifo_cap sc_in sc_lv 7 signal 11 } 
	{ inByte926_empty_n sc_in sc_logic 1 signal 11 } 
	{ inByte926_read sc_out sc_logic 1 signal 11 } 
	{ inByte1027_dout sc_in sc_lv 8 signal 12 } 
	{ inByte1027_num_data_valid sc_in sc_lv 7 signal 12 } 
	{ inByte1027_fifo_cap sc_in sc_lv 7 signal 12 } 
	{ inByte1027_empty_n sc_in sc_logic 1 signal 12 } 
	{ inByte1027_read sc_out sc_logic 1 signal 12 } 
	{ inByte1128_dout sc_in sc_lv 8 signal 13 } 
	{ inByte1128_num_data_valid sc_in sc_lv 7 signal 13 } 
	{ inByte1128_fifo_cap sc_in sc_lv 7 signal 13 } 
	{ inByte1128_empty_n sc_in sc_logic 1 signal 13 } 
	{ inByte1128_read sc_out sc_logic 1 signal 13 } 
	{ inByte1229_dout sc_in sc_lv 8 signal 14 } 
	{ inByte1229_num_data_valid sc_in sc_lv 7 signal 14 } 
	{ inByte1229_fifo_cap sc_in sc_lv 7 signal 14 } 
	{ inByte1229_empty_n sc_in sc_logic 1 signal 14 } 
	{ inByte1229_read sc_out sc_logic 1 signal 14 } 
	{ inByte1330_dout sc_in sc_lv 8 signal 15 } 
	{ inByte1330_num_data_valid sc_in sc_lv 7 signal 15 } 
	{ inByte1330_fifo_cap sc_in sc_lv 7 signal 15 } 
	{ inByte1330_empty_n sc_in sc_logic 1 signal 15 } 
	{ inByte1330_read sc_out sc_logic 1 signal 15 } 
	{ inByte1431_dout sc_in sc_lv 8 signal 16 } 
	{ inByte1431_num_data_valid sc_in sc_lv 7 signal 16 } 
	{ inByte1431_fifo_cap sc_in sc_lv 7 signal 16 } 
	{ inByte1431_empty_n sc_in sc_logic 1 signal 16 } 
	{ inByte1431_read sc_out sc_logic 1 signal 16 } 
	{ inByte1532_dout sc_in sc_lv 8 signal 17 } 
	{ inByte1532_num_data_valid sc_in sc_lv 7 signal 17 } 
	{ inByte1532_fifo_cap sc_in sc_lv 7 signal 17 } 
	{ inByte1532_empty_n sc_in sc_logic 1 signal 17 } 
	{ inByte1532_read sc_out sc_logic 1 signal 17 } 
	{ crc sc_in sc_lv 32 signal 0 } 
	{ blocks_in_chunk sc_in sc_lv 28 signal 1 } 
	{ crcTables_0_address0 sc_out sc_lv 8 signal 18 } 
	{ crcTables_0_ce0 sc_out sc_logic 1 signal 18 } 
	{ crcTables_0_q0 sc_in sc_lv 32 signal 18 } 
	{ crcTables_1_address0 sc_out sc_lv 8 signal 19 } 
	{ crcTables_1_ce0 sc_out sc_logic 1 signal 19 } 
	{ crcTables_1_q0 sc_in sc_lv 32 signal 19 } 
	{ crcTables_2_address0 sc_out sc_lv 8 signal 20 } 
	{ crcTables_2_ce0 sc_out sc_logic 1 signal 20 } 
	{ crcTables_2_q0 sc_in sc_lv 32 signal 20 } 
	{ crcTables_3_address0 sc_out sc_lv 8 signal 21 } 
	{ crcTables_3_ce0 sc_out sc_logic 1 signal 21 } 
	{ crcTables_3_q0 sc_in sc_lv 32 signal 21 } 
	{ crcTables_4_address0 sc_out sc_lv 8 signal 22 } 
	{ crcTables_4_ce0 sc_out sc_logic 1 signal 22 } 
	{ crcTables_4_q0 sc_in sc_lv 32 signal 22 } 
	{ crcTables_5_address0 sc_out sc_lv 8 signal 23 } 
	{ crcTables_5_ce0 sc_out sc_logic 1 signal 23 } 
	{ crcTables_5_q0 sc_in sc_lv 32 signal 23 } 
	{ crcTables_6_address0 sc_out sc_lv 8 signal 24 } 
	{ crcTables_6_ce0 sc_out sc_logic 1 signal 24 } 
	{ crcTables_6_q0 sc_in sc_lv 32 signal 24 } 
	{ crcTables_7_address0 sc_out sc_lv 8 signal 25 } 
	{ crcTables_7_ce0 sc_out sc_logic 1 signal 25 } 
	{ crcTables_7_q0 sc_in sc_lv 32 signal 25 } 
	{ crcTables_8_address0 sc_out sc_lv 8 signal 26 } 
	{ crcTables_8_ce0 sc_out sc_logic 1 signal 26 } 
	{ crcTables_8_q0 sc_in sc_lv 32 signal 26 } 
	{ crcTables_9_address0 sc_out sc_lv 8 signal 27 } 
	{ crcTables_9_ce0 sc_out sc_logic 1 signal 27 } 
	{ crcTables_9_q0 sc_in sc_lv 32 signal 27 } 
	{ crcTables_10_address0 sc_out sc_lv 8 signal 28 } 
	{ crcTables_10_ce0 sc_out sc_logic 1 signal 28 } 
	{ crcTables_10_q0 sc_in sc_lv 32 signal 28 } 
	{ crcTables_11_address0 sc_out sc_lv 8 signal 29 } 
	{ crcTables_11_ce0 sc_out sc_logic 1 signal 29 } 
	{ crcTables_11_q0 sc_in sc_lv 32 signal 29 } 
	{ crcTables_12_address0 sc_out sc_lv 8 signal 30 } 
	{ crcTables_12_ce0 sc_out sc_logic 1 signal 30 } 
	{ crcTables_12_q0 sc_in sc_lv 32 signal 30 } 
	{ crcTables_13_address0 sc_out sc_lv 8 signal 31 } 
	{ crcTables_13_ce0 sc_out sc_logic 1 signal 31 } 
	{ crcTables_13_q0 sc_in sc_lv 32 signal 31 } 
	{ crcTables_14_address0 sc_out sc_lv 8 signal 32 } 
	{ crcTables_14_ce0 sc_out sc_logic 1 signal 32 } 
	{ crcTables_14_q0 sc_in sc_lv 32 signal 32 } 
	{ crcTables_15_address0 sc_out sc_lv 8 signal 33 } 
	{ crcTables_15_ce0 sc_out sc_logic 1 signal 33 } 
	{ crcTables_15_q0 sc_in sc_lv 32 signal 33 } 
	{ mask sc_in sc_lv 32 signal 34 } 
	{ p_out sc_out sc_lv 32 signal 35 } 
	{ p_out_ap_vld sc_out sc_logic 1 outvld 35 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "inByte017_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte017", "role": "dout" }} , 
 	{ "name": "inByte017_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte017", "role": "num_data_valid" }} , 
 	{ "name": "inByte017_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte017", "role": "fifo_cap" }} , 
 	{ "name": "inByte017_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte017", "role": "empty_n" }} , 
 	{ "name": "inByte017_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte017", "role": "read" }} , 
 	{ "name": "inByte118_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte118", "role": "dout" }} , 
 	{ "name": "inByte118_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte118", "role": "num_data_valid" }} , 
 	{ "name": "inByte118_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte118", "role": "fifo_cap" }} , 
 	{ "name": "inByte118_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte118", "role": "empty_n" }} , 
 	{ "name": "inByte118_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte118", "role": "read" }} , 
 	{ "name": "inByte219_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte219", "role": "dout" }} , 
 	{ "name": "inByte219_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte219", "role": "num_data_valid" }} , 
 	{ "name": "inByte219_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte219", "role": "fifo_cap" }} , 
 	{ "name": "inByte219_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte219", "role": "empty_n" }} , 
 	{ "name": "inByte219_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte219", "role": "read" }} , 
 	{ "name": "inByte320_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte320", "role": "dout" }} , 
 	{ "name": "inByte320_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte320", "role": "num_data_valid" }} , 
 	{ "name": "inByte320_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte320", "role": "fifo_cap" }} , 
 	{ "name": "inByte320_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte320", "role": "empty_n" }} , 
 	{ "name": "inByte320_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte320", "role": "read" }} , 
 	{ "name": "inByte421_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte421", "role": "dout" }} , 
 	{ "name": "inByte421_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte421", "role": "num_data_valid" }} , 
 	{ "name": "inByte421_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte421", "role": "fifo_cap" }} , 
 	{ "name": "inByte421_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte421", "role": "empty_n" }} , 
 	{ "name": "inByte421_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte421", "role": "read" }} , 
 	{ "name": "inByte522_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte522", "role": "dout" }} , 
 	{ "name": "inByte522_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte522", "role": "num_data_valid" }} , 
 	{ "name": "inByte522_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte522", "role": "fifo_cap" }} , 
 	{ "name": "inByte522_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte522", "role": "empty_n" }} , 
 	{ "name": "inByte522_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte522", "role": "read" }} , 
 	{ "name": "inByte623_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte623", "role": "dout" }} , 
 	{ "name": "inByte623_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte623", "role": "num_data_valid" }} , 
 	{ "name": "inByte623_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte623", "role": "fifo_cap" }} , 
 	{ "name": "inByte623_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte623", "role": "empty_n" }} , 
 	{ "name": "inByte623_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte623", "role": "read" }} , 
 	{ "name": "inByte724_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte724", "role": "dout" }} , 
 	{ "name": "inByte724_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte724", "role": "num_data_valid" }} , 
 	{ "name": "inByte724_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte724", "role": "fifo_cap" }} , 
 	{ "name": "inByte724_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte724", "role": "empty_n" }} , 
 	{ "name": "inByte724_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte724", "role": "read" }} , 
 	{ "name": "inByte825_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte825", "role": "dout" }} , 
 	{ "name": "inByte825_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte825", "role": "num_data_valid" }} , 
 	{ "name": "inByte825_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte825", "role": "fifo_cap" }} , 
 	{ "name": "inByte825_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte825", "role": "empty_n" }} , 
 	{ "name": "inByte825_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte825", "role": "read" }} , 
 	{ "name": "inByte926_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte926", "role": "dout" }} , 
 	{ "name": "inByte926_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte926", "role": "num_data_valid" }} , 
 	{ "name": "inByte926_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte926", "role": "fifo_cap" }} , 
 	{ "name": "inByte926_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte926", "role": "empty_n" }} , 
 	{ "name": "inByte926_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte926", "role": "read" }} , 
 	{ "name": "inByte1027_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1027", "role": "dout" }} , 
 	{ "name": "inByte1027_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1027", "role": "num_data_valid" }} , 
 	{ "name": "inByte1027_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1027", "role": "fifo_cap" }} , 
 	{ "name": "inByte1027_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1027", "role": "empty_n" }} , 
 	{ "name": "inByte1027_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1027", "role": "read" }} , 
 	{ "name": "inByte1128_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1128", "role": "dout" }} , 
 	{ "name": "inByte1128_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1128", "role": "num_data_valid" }} , 
 	{ "name": "inByte1128_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1128", "role": "fifo_cap" }} , 
 	{ "name": "inByte1128_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1128", "role": "empty_n" }} , 
 	{ "name": "inByte1128_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1128", "role": "read" }} , 
 	{ "name": "inByte1229_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1229", "role": "dout" }} , 
 	{ "name": "inByte1229_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1229", "role": "num_data_valid" }} , 
 	{ "name": "inByte1229_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1229", "role": "fifo_cap" }} , 
 	{ "name": "inByte1229_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1229", "role": "empty_n" }} , 
 	{ "name": "inByte1229_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1229", "role": "read" }} , 
 	{ "name": "inByte1330_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1330", "role": "dout" }} , 
 	{ "name": "inByte1330_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1330", "role": "num_data_valid" }} , 
 	{ "name": "inByte1330_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1330", "role": "fifo_cap" }} , 
 	{ "name": "inByte1330_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1330", "role": "empty_n" }} , 
 	{ "name": "inByte1330_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1330", "role": "read" }} , 
 	{ "name": "inByte1431_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1431", "role": "dout" }} , 
 	{ "name": "inByte1431_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1431", "role": "num_data_valid" }} , 
 	{ "name": "inByte1431_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1431", "role": "fifo_cap" }} , 
 	{ "name": "inByte1431_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1431", "role": "empty_n" }} , 
 	{ "name": "inByte1431_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1431", "role": "read" }} , 
 	{ "name": "inByte1532_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1532", "role": "dout" }} , 
 	{ "name": "inByte1532_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1532", "role": "num_data_valid" }} , 
 	{ "name": "inByte1532_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1532", "role": "fifo_cap" }} , 
 	{ "name": "inByte1532_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1532", "role": "empty_n" }} , 
 	{ "name": "inByte1532_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1532", "role": "read" }} , 
 	{ "name": "crc", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crc", "role": "default" }} , 
 	{ "name": "blocks_in_chunk", "direction": "in", "datatype": "sc_lv", "bitwidth":28, "type": "signal", "bundle":{"name": "blocks_in_chunk", "role": "default" }} , 
 	{ "name": "crcTables_0_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_0", "role": "address0" }} , 
 	{ "name": "crcTables_0_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_0", "role": "ce0" }} , 
 	{ "name": "crcTables_0_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_0", "role": "q0" }} , 
 	{ "name": "crcTables_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_1", "role": "address0" }} , 
 	{ "name": "crcTables_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_1", "role": "ce0" }} , 
 	{ "name": "crcTables_1_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_1", "role": "q0" }} , 
 	{ "name": "crcTables_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_2", "role": "address0" }} , 
 	{ "name": "crcTables_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_2", "role": "ce0" }} , 
 	{ "name": "crcTables_2_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_2", "role": "q0" }} , 
 	{ "name": "crcTables_3_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_3", "role": "address0" }} , 
 	{ "name": "crcTables_3_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_3", "role": "ce0" }} , 
 	{ "name": "crcTables_3_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_3", "role": "q0" }} , 
 	{ "name": "crcTables_4_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_4", "role": "address0" }} , 
 	{ "name": "crcTables_4_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_4", "role": "ce0" }} , 
 	{ "name": "crcTables_4_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_4", "role": "q0" }} , 
 	{ "name": "crcTables_5_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_5", "role": "address0" }} , 
 	{ "name": "crcTables_5_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_5", "role": "ce0" }} , 
 	{ "name": "crcTables_5_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_5", "role": "q0" }} , 
 	{ "name": "crcTables_6_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_6", "role": "address0" }} , 
 	{ "name": "crcTables_6_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_6", "role": "ce0" }} , 
 	{ "name": "crcTables_6_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_6", "role": "q0" }} , 
 	{ "name": "crcTables_7_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_7", "role": "address0" }} , 
 	{ "name": "crcTables_7_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_7", "role": "ce0" }} , 
 	{ "name": "crcTables_7_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_7", "role": "q0" }} , 
 	{ "name": "crcTables_8_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_8", "role": "address0" }} , 
 	{ "name": "crcTables_8_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_8", "role": "ce0" }} , 
 	{ "name": "crcTables_8_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_8", "role": "q0" }} , 
 	{ "name": "crcTables_9_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_9", "role": "address0" }} , 
 	{ "name": "crcTables_9_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_9", "role": "ce0" }} , 
 	{ "name": "crcTables_9_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_9", "role": "q0" }} , 
 	{ "name": "crcTables_10_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_10", "role": "address0" }} , 
 	{ "name": "crcTables_10_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_10", "role": "ce0" }} , 
 	{ "name": "crcTables_10_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_10", "role": "q0" }} , 
 	{ "name": "crcTables_11_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_11", "role": "address0" }} , 
 	{ "name": "crcTables_11_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_11", "role": "ce0" }} , 
 	{ "name": "crcTables_11_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_11", "role": "q0" }} , 
 	{ "name": "crcTables_12_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_12", "role": "address0" }} , 
 	{ "name": "crcTables_12_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_12", "role": "ce0" }} , 
 	{ "name": "crcTables_12_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_12", "role": "q0" }} , 
 	{ "name": "crcTables_13_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_13", "role": "address0" }} , 
 	{ "name": "crcTables_13_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_13", "role": "ce0" }} , 
 	{ "name": "crcTables_13_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_13", "role": "q0" }} , 
 	{ "name": "crcTables_14_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_14", "role": "address0" }} , 
 	{ "name": "crcTables_14_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_14", "role": "ce0" }} , 
 	{ "name": "crcTables_14_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_14", "role": "q0" }} , 
 	{ "name": "crcTables_15_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_15", "role": "address0" }} , 
 	{ "name": "crcTables_15_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_15", "role": "ce0" }} , 
 	{ "name": "crcTables_15_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_15", "role": "q0" }} , 
 	{ "name": "mask", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "mask", "role": "default" }} , 
 	{ "name": "p_out", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "p_out", "role": "default" }} , 
 	{ "name": "p_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "p_out", "role": "ap_vld" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1"],
		"CDFG" : "crc_process_full_blocks_Pipeline_full_block_loop",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "5", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "crc", "Type" : "None", "Direction" : "I"},
			{"Name" : "blocks_in_chunk", "Type" : "None", "Direction" : "I"},
			{"Name" : "inByte017", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte017_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte118", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte118_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte219", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte219_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte320", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte320_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte421", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte421_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte522", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte522_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte623", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte623_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte724", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte724_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte825", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte825_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte926", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte926_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1027", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1027_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1128", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1128_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1229", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1229_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1330", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1330_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1431", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1431_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "inByte1532", "Type" : "Fifo", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "inByte1532_blk_n", "Type" : "RtlSignal"}]},
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
			{"Name" : "mask", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_out", "Type" : "Vld", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "full_block_loop", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter3", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter3", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	crc_process_full_blocks_Pipeline_full_block_loop {
		crc {Type I LastRead 0 FirstWrite -1}
		blocks_in_chunk {Type I LastRead 0 FirstWrite -1}
		inByte017 {Type I LastRead 1 FirstWrite -1}
		inByte118 {Type I LastRead 1 FirstWrite -1}
		inByte219 {Type I LastRead 1 FirstWrite -1}
		inByte320 {Type I LastRead 1 FirstWrite -1}
		inByte421 {Type I LastRead 1 FirstWrite -1}
		inByte522 {Type I LastRead 1 FirstWrite -1}
		inByte623 {Type I LastRead 1 FirstWrite -1}
		inByte724 {Type I LastRead 1 FirstWrite -1}
		inByte825 {Type I LastRead 1 FirstWrite -1}
		inByte926 {Type I LastRead 1 FirstWrite -1}
		inByte1027 {Type I LastRead 1 FirstWrite -1}
		inByte1128 {Type I LastRead 1 FirstWrite -1}
		inByte1229 {Type I LastRead 1 FirstWrite -1}
		inByte1330 {Type I LastRead 1 FirstWrite -1}
		inByte1431 {Type I LastRead 1 FirstWrite -1}
		inByte1532 {Type I LastRead 1 FirstWrite -1}
		crcTables_0 {Type I LastRead 2 FirstWrite -1}
		crcTables_1 {Type I LastRead 2 FirstWrite -1}
		crcTables_2 {Type I LastRead 2 FirstWrite -1}
		crcTables_3 {Type I LastRead 2 FirstWrite -1}
		crcTables_4 {Type I LastRead 1 FirstWrite -1}
		crcTables_5 {Type I LastRead 1 FirstWrite -1}
		crcTables_6 {Type I LastRead 1 FirstWrite -1}
		crcTables_7 {Type I LastRead 1 FirstWrite -1}
		crcTables_8 {Type I LastRead 1 FirstWrite -1}
		crcTables_9 {Type I LastRead 1 FirstWrite -1}
		crcTables_10 {Type I LastRead 1 FirstWrite -1}
		crcTables_11 {Type I LastRead 1 FirstWrite -1}
		crcTables_12 {Type I LastRead 2 FirstWrite -1}
		crcTables_13 {Type I LastRead 2 FirstWrite -1}
		crcTables_14 {Type I LastRead 2 FirstWrite -1}
		crcTables_15 {Type I LastRead 2 FirstWrite -1}
		mask {Type I LastRead 0 FirstWrite -1}
		p_out {Type O LastRead -1 FirstWrite 2}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "5", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "5", "Max" : "-1"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	crc { ap_none {  { crc in_data 0 32 } } }
	blocks_in_chunk { ap_none {  { blocks_in_chunk in_data 0 28 } } }
	inByte017 { ap_fifo {  { inByte017_dout fifo_port_we 0 8 }  { inByte017_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte017_fifo_cap fifo_update 0 7 }  { inByte017_empty_n fifo_status 0 1 }  { inByte017_read fifo_data 1 1 } } }
	inByte118 { ap_fifo {  { inByte118_dout fifo_port_we 0 8 }  { inByte118_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte118_fifo_cap fifo_update 0 7 }  { inByte118_empty_n fifo_status 0 1 }  { inByte118_read fifo_data 1 1 } } }
	inByte219 { ap_fifo {  { inByte219_dout fifo_port_we 0 8 }  { inByte219_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte219_fifo_cap fifo_update 0 7 }  { inByte219_empty_n fifo_status 0 1 }  { inByte219_read fifo_data 1 1 } } }
	inByte320 { ap_fifo {  { inByte320_dout fifo_port_we 0 8 }  { inByte320_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte320_fifo_cap fifo_update 0 7 }  { inByte320_empty_n fifo_status 0 1 }  { inByte320_read fifo_data 1 1 } } }
	inByte421 { ap_fifo {  { inByte421_dout fifo_port_we 0 8 }  { inByte421_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte421_fifo_cap fifo_update 0 7 }  { inByte421_empty_n fifo_status 0 1 }  { inByte421_read fifo_data 1 1 } } }
	inByte522 { ap_fifo {  { inByte522_dout fifo_port_we 0 8 }  { inByte522_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte522_fifo_cap fifo_update 0 7 }  { inByte522_empty_n fifo_status 0 1 }  { inByte522_read fifo_data 1 1 } } }
	inByte623 { ap_fifo {  { inByte623_dout fifo_port_we 0 8 }  { inByte623_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte623_fifo_cap fifo_update 0 7 }  { inByte623_empty_n fifo_status 0 1 }  { inByte623_read fifo_data 1 1 } } }
	inByte724 { ap_fifo {  { inByte724_dout fifo_port_we 0 8 }  { inByte724_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte724_fifo_cap fifo_update 0 7 }  { inByte724_empty_n fifo_status 0 1 }  { inByte724_read fifo_data 1 1 } } }
	inByte825 { ap_fifo {  { inByte825_dout fifo_port_we 0 8 }  { inByte825_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte825_fifo_cap fifo_update 0 7 }  { inByte825_empty_n fifo_status 0 1 }  { inByte825_read fifo_data 1 1 } } }
	inByte926 { ap_fifo {  { inByte926_dout fifo_port_we 0 8 }  { inByte926_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte926_fifo_cap fifo_update 0 7 }  { inByte926_empty_n fifo_status 0 1 }  { inByte926_read fifo_data 1 1 } } }
	inByte1027 { ap_fifo {  { inByte1027_dout fifo_port_we 0 8 }  { inByte1027_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte1027_fifo_cap fifo_update 0 7 }  { inByte1027_empty_n fifo_status 0 1 }  { inByte1027_read fifo_data 1 1 } } }
	inByte1128 { ap_fifo {  { inByte1128_dout fifo_port_we 0 8 }  { inByte1128_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte1128_fifo_cap fifo_update 0 7 }  { inByte1128_empty_n fifo_status 0 1 }  { inByte1128_read fifo_data 1 1 } } }
	inByte1229 { ap_fifo {  { inByte1229_dout fifo_port_we 0 8 }  { inByte1229_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte1229_fifo_cap fifo_update 0 7 }  { inByte1229_empty_n fifo_status 0 1 }  { inByte1229_read fifo_data 1 1 } } }
	inByte1330 { ap_fifo {  { inByte1330_dout fifo_port_we 0 8 }  { inByte1330_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte1330_fifo_cap fifo_update 0 7 }  { inByte1330_empty_n fifo_status 0 1 }  { inByte1330_read fifo_data 1 1 } } }
	inByte1431 { ap_fifo {  { inByte1431_dout fifo_port_we 0 8 }  { inByte1431_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte1431_fifo_cap fifo_update 0 7 }  { inByte1431_empty_n fifo_status 0 1 }  { inByte1431_read fifo_data 1 1 } } }
	inByte1532 { ap_fifo {  { inByte1532_dout fifo_port_we 0 8 }  { inByte1532_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte1532_fifo_cap fifo_update 0 7 }  { inByte1532_empty_n fifo_status 0 1 }  { inByte1532_read fifo_data 1 1 } } }
	crcTables_0 { ap_memory {  { crcTables_0_address0 mem_address 1 8 }  { crcTables_0_ce0 mem_ce 1 1 }  { crcTables_0_q0 in_data 0 32 } } }
	crcTables_1 { ap_memory {  { crcTables_1_address0 mem_address 1 8 }  { crcTables_1_ce0 mem_ce 1 1 }  { crcTables_1_q0 in_data 0 32 } } }
	crcTables_2 { ap_memory {  { crcTables_2_address0 mem_address 1 8 }  { crcTables_2_ce0 mem_ce 1 1 }  { crcTables_2_q0 in_data 0 32 } } }
	crcTables_3 { ap_memory {  { crcTables_3_address0 mem_address 1 8 }  { crcTables_3_ce0 mem_ce 1 1 }  { crcTables_3_q0 in_data 0 32 } } }
	crcTables_4 { ap_memory {  { crcTables_4_address0 mem_address 1 8 }  { crcTables_4_ce0 mem_ce 1 1 }  { crcTables_4_q0 mem_dout 0 32 } } }
	crcTables_5 { ap_memory {  { crcTables_5_address0 mem_address 1 8 }  { crcTables_5_ce0 mem_ce 1 1 }  { crcTables_5_q0 mem_dout 0 32 } } }
	crcTables_6 { ap_memory {  { crcTables_6_address0 mem_address 1 8 }  { crcTables_6_ce0 mem_ce 1 1 }  { crcTables_6_q0 mem_dout 0 32 } } }
	crcTables_7 { ap_memory {  { crcTables_7_address0 mem_address 1 8 }  { crcTables_7_ce0 mem_ce 1 1 }  { crcTables_7_q0 mem_dout 0 32 } } }
	crcTables_8 { ap_memory {  { crcTables_8_address0 mem_address 1 8 }  { crcTables_8_ce0 mem_ce 1 1 }  { crcTables_8_q0 in_data 0 32 } } }
	crcTables_9 { ap_memory {  { crcTables_9_address0 mem_address 1 8 }  { crcTables_9_ce0 mem_ce 1 1 }  { crcTables_9_q0 in_data 0 32 } } }
	crcTables_10 { ap_memory {  { crcTables_10_address0 mem_address 1 8 }  { crcTables_10_ce0 mem_ce 1 1 }  { crcTables_10_q0 in_data 0 32 } } }
	crcTables_11 { ap_memory {  { crcTables_11_address0 mem_address 1 8 }  { crcTables_11_ce0 mem_ce 1 1 }  { crcTables_11_q0 in_data 0 32 } } }
	crcTables_12 { ap_memory {  { crcTables_12_address0 mem_address 1 8 }  { crcTables_12_ce0 mem_ce 1 1 }  { crcTables_12_q0 in_data 0 32 } } }
	crcTables_13 { ap_memory {  { crcTables_13_address0 mem_address 1 8 }  { crcTables_13_ce0 mem_ce 1 1 }  { crcTables_13_q0 in_data 0 32 } } }
	crcTables_14 { ap_memory {  { crcTables_14_address0 mem_address 1 8 }  { crcTables_14_ce0 mem_ce 1 1 }  { crcTables_14_q0 in_data 0 32 } } }
	crcTables_15 { ap_memory {  { crcTables_15_address0 mem_address 1 8 }  { crcTables_15_ce0 mem_ce 1 1 }  { crcTables_15_q0 in_data 0 32 } } }
	mask { ap_none {  { mask in_data 0 32 } } }
	p_out { ap_vld {  { p_out out_data 1 32 }  { p_out_ap_vld out_vld 1 1 } } }
}
