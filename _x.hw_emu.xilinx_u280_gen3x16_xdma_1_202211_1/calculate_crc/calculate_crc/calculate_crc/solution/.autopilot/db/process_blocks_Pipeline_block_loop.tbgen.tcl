set moduleName process_blocks_Pipeline_block_loop
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 1
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 1
set StallSigGenFlag 1
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {process_blocks_Pipeline_block_loop}
set C_modelType { void 0 }
set C_modelArgList {
	{ init_value_2 int 32 regular  }
	{ blocks_in_chunk int 29 regular  }
	{ inByte0 int 8 regular {fifo 0 volatile }  }
	{ inByte1 int 8 regular {fifo 0 volatile }  }
	{ inByte2 int 8 regular {fifo 0 volatile }  }
	{ inByte3 int 8 regular {fifo 0 volatile }  }
	{ inByte4 int 8 regular {fifo 0 volatile }  }
	{ inByte5 int 8 regular {fifo 0 volatile }  }
	{ inByte6 int 8 regular {fifo 0 volatile }  }
	{ inByte7 int 8 regular {fifo 0 volatile }  }
	{ inByte8 int 8 regular {fifo 0 volatile }  }
	{ inByte9 int 8 regular {fifo 0 volatile }  }
	{ inByte10 int 8 regular {fifo 0 volatile }  }
	{ inByte11 int 8 regular {fifo 0 volatile }  }
	{ inByte12 int 8 regular {fifo 0 volatile }  }
	{ inByte13 int 8 regular {fifo 0 volatile }  }
	{ inByte14 int 8 regular {fifo 0 volatile }  }
	{ inByte15 int 8 regular {fifo 0 volatile }  }
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
	{ mask_1 int 32 regular  }
	{ crc_out int 32 regular {pointer 1}  }
}
set hasAXIMCache 0
set C_modelArgMapList {[ 
	{ "Name" : "init_value_2", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "blocks_in_chunk", "interface" : "wire", "bitwidth" : 29, "direction" : "READONLY"} , 
 	{ "Name" : "inByte0", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte1", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte2", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte3", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte4", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte5", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte6", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte7", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte8", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte9", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte10", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte11", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte12", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte13", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte14", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "inByte15", "interface" : "fifo", "bitwidth" : 8, "direction" : "READONLY"} , 
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
 	{ "Name" : "mask_1", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "crc_out", "interface" : "wire", "bitwidth" : 32, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 142
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ inByte0_dout sc_in sc_lv 8 signal 2 } 
	{ inByte0_num_data_valid sc_in sc_lv 7 signal 2 } 
	{ inByte0_fifo_cap sc_in sc_lv 7 signal 2 } 
	{ inByte0_empty_n sc_in sc_logic 1 signal 2 } 
	{ inByte0_read sc_out sc_logic 1 signal 2 } 
	{ inByte1_dout sc_in sc_lv 8 signal 3 } 
	{ inByte1_num_data_valid sc_in sc_lv 7 signal 3 } 
	{ inByte1_fifo_cap sc_in sc_lv 7 signal 3 } 
	{ inByte1_empty_n sc_in sc_logic 1 signal 3 } 
	{ inByte1_read sc_out sc_logic 1 signal 3 } 
	{ inByte2_dout sc_in sc_lv 8 signal 4 } 
	{ inByte2_num_data_valid sc_in sc_lv 7 signal 4 } 
	{ inByte2_fifo_cap sc_in sc_lv 7 signal 4 } 
	{ inByte2_empty_n sc_in sc_logic 1 signal 4 } 
	{ inByte2_read sc_out sc_logic 1 signal 4 } 
	{ inByte3_dout sc_in sc_lv 8 signal 5 } 
	{ inByte3_num_data_valid sc_in sc_lv 7 signal 5 } 
	{ inByte3_fifo_cap sc_in sc_lv 7 signal 5 } 
	{ inByte3_empty_n sc_in sc_logic 1 signal 5 } 
	{ inByte3_read sc_out sc_logic 1 signal 5 } 
	{ inByte4_dout sc_in sc_lv 8 signal 6 } 
	{ inByte4_num_data_valid sc_in sc_lv 7 signal 6 } 
	{ inByte4_fifo_cap sc_in sc_lv 7 signal 6 } 
	{ inByte4_empty_n sc_in sc_logic 1 signal 6 } 
	{ inByte4_read sc_out sc_logic 1 signal 6 } 
	{ inByte5_dout sc_in sc_lv 8 signal 7 } 
	{ inByte5_num_data_valid sc_in sc_lv 7 signal 7 } 
	{ inByte5_fifo_cap sc_in sc_lv 7 signal 7 } 
	{ inByte5_empty_n sc_in sc_logic 1 signal 7 } 
	{ inByte5_read sc_out sc_logic 1 signal 7 } 
	{ inByte6_dout sc_in sc_lv 8 signal 8 } 
	{ inByte6_num_data_valid sc_in sc_lv 7 signal 8 } 
	{ inByte6_fifo_cap sc_in sc_lv 7 signal 8 } 
	{ inByte6_empty_n sc_in sc_logic 1 signal 8 } 
	{ inByte6_read sc_out sc_logic 1 signal 8 } 
	{ inByte7_dout sc_in sc_lv 8 signal 9 } 
	{ inByte7_num_data_valid sc_in sc_lv 7 signal 9 } 
	{ inByte7_fifo_cap sc_in sc_lv 7 signal 9 } 
	{ inByte7_empty_n sc_in sc_logic 1 signal 9 } 
	{ inByte7_read sc_out sc_logic 1 signal 9 } 
	{ inByte8_dout sc_in sc_lv 8 signal 10 } 
	{ inByte8_num_data_valid sc_in sc_lv 7 signal 10 } 
	{ inByte8_fifo_cap sc_in sc_lv 7 signal 10 } 
	{ inByte8_empty_n sc_in sc_logic 1 signal 10 } 
	{ inByte8_read sc_out sc_logic 1 signal 10 } 
	{ inByte9_dout sc_in sc_lv 8 signal 11 } 
	{ inByte9_num_data_valid sc_in sc_lv 7 signal 11 } 
	{ inByte9_fifo_cap sc_in sc_lv 7 signal 11 } 
	{ inByte9_empty_n sc_in sc_logic 1 signal 11 } 
	{ inByte9_read sc_out sc_logic 1 signal 11 } 
	{ inByte10_dout sc_in sc_lv 8 signal 12 } 
	{ inByte10_num_data_valid sc_in sc_lv 7 signal 12 } 
	{ inByte10_fifo_cap sc_in sc_lv 7 signal 12 } 
	{ inByte10_empty_n sc_in sc_logic 1 signal 12 } 
	{ inByte10_read sc_out sc_logic 1 signal 12 } 
	{ inByte11_dout sc_in sc_lv 8 signal 13 } 
	{ inByte11_num_data_valid sc_in sc_lv 7 signal 13 } 
	{ inByte11_fifo_cap sc_in sc_lv 7 signal 13 } 
	{ inByte11_empty_n sc_in sc_logic 1 signal 13 } 
	{ inByte11_read sc_out sc_logic 1 signal 13 } 
	{ inByte12_dout sc_in sc_lv 8 signal 14 } 
	{ inByte12_num_data_valid sc_in sc_lv 7 signal 14 } 
	{ inByte12_fifo_cap sc_in sc_lv 7 signal 14 } 
	{ inByte12_empty_n sc_in sc_logic 1 signal 14 } 
	{ inByte12_read sc_out sc_logic 1 signal 14 } 
	{ inByte13_dout sc_in sc_lv 8 signal 15 } 
	{ inByte13_num_data_valid sc_in sc_lv 7 signal 15 } 
	{ inByte13_fifo_cap sc_in sc_lv 7 signal 15 } 
	{ inByte13_empty_n sc_in sc_logic 1 signal 15 } 
	{ inByte13_read sc_out sc_logic 1 signal 15 } 
	{ inByte14_dout sc_in sc_lv 8 signal 16 } 
	{ inByte14_num_data_valid sc_in sc_lv 7 signal 16 } 
	{ inByte14_fifo_cap sc_in sc_lv 7 signal 16 } 
	{ inByte14_empty_n sc_in sc_logic 1 signal 16 } 
	{ inByte14_read sc_out sc_logic 1 signal 16 } 
	{ inByte15_dout sc_in sc_lv 8 signal 17 } 
	{ inByte15_num_data_valid sc_in sc_lv 7 signal 17 } 
	{ inByte15_fifo_cap sc_in sc_lv 7 signal 17 } 
	{ inByte15_empty_n sc_in sc_logic 1 signal 17 } 
	{ inByte15_read sc_out sc_logic 1 signal 17 } 
	{ init_value_2 sc_in sc_lv 32 signal 0 } 
	{ blocks_in_chunk sc_in sc_lv 29 signal 1 } 
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
	{ mask_1 sc_in sc_lv 32 signal 34 } 
	{ crc_out sc_out sc_lv 32 signal 35 } 
	{ crc_out_ap_vld sc_out sc_logic 1 outvld 35 } 
	{ ap_ext_blocking_n sc_out sc_logic 1 signal -1 } 
	{ ap_str_blocking_n sc_out sc_logic 1 signal -1 } 
	{ ap_int_blocking_n sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "inByte0_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte0", "role": "dout" }} , 
 	{ "name": "inByte0_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte0", "role": "num_data_valid" }} , 
 	{ "name": "inByte0_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte0", "role": "fifo_cap" }} , 
 	{ "name": "inByte0_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte0", "role": "empty_n" }} , 
 	{ "name": "inByte0_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte0", "role": "read" }} , 
 	{ "name": "inByte1_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1", "role": "dout" }} , 
 	{ "name": "inByte1_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1", "role": "num_data_valid" }} , 
 	{ "name": "inByte1_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1", "role": "fifo_cap" }} , 
 	{ "name": "inByte1_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1", "role": "empty_n" }} , 
 	{ "name": "inByte1_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1", "role": "read" }} , 
 	{ "name": "inByte2_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte2", "role": "dout" }} , 
 	{ "name": "inByte2_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte2", "role": "num_data_valid" }} , 
 	{ "name": "inByte2_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte2", "role": "fifo_cap" }} , 
 	{ "name": "inByte2_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte2", "role": "empty_n" }} , 
 	{ "name": "inByte2_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte2", "role": "read" }} , 
 	{ "name": "inByte3_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte3", "role": "dout" }} , 
 	{ "name": "inByte3_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte3", "role": "num_data_valid" }} , 
 	{ "name": "inByte3_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte3", "role": "fifo_cap" }} , 
 	{ "name": "inByte3_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte3", "role": "empty_n" }} , 
 	{ "name": "inByte3_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte3", "role": "read" }} , 
 	{ "name": "inByte4_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte4", "role": "dout" }} , 
 	{ "name": "inByte4_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte4", "role": "num_data_valid" }} , 
 	{ "name": "inByte4_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte4", "role": "fifo_cap" }} , 
 	{ "name": "inByte4_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte4", "role": "empty_n" }} , 
 	{ "name": "inByte4_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte4", "role": "read" }} , 
 	{ "name": "inByte5_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte5", "role": "dout" }} , 
 	{ "name": "inByte5_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte5", "role": "num_data_valid" }} , 
 	{ "name": "inByte5_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte5", "role": "fifo_cap" }} , 
 	{ "name": "inByte5_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte5", "role": "empty_n" }} , 
 	{ "name": "inByte5_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte5", "role": "read" }} , 
 	{ "name": "inByte6_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte6", "role": "dout" }} , 
 	{ "name": "inByte6_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte6", "role": "num_data_valid" }} , 
 	{ "name": "inByte6_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte6", "role": "fifo_cap" }} , 
 	{ "name": "inByte6_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte6", "role": "empty_n" }} , 
 	{ "name": "inByte6_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte6", "role": "read" }} , 
 	{ "name": "inByte7_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte7", "role": "dout" }} , 
 	{ "name": "inByte7_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte7", "role": "num_data_valid" }} , 
 	{ "name": "inByte7_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte7", "role": "fifo_cap" }} , 
 	{ "name": "inByte7_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte7", "role": "empty_n" }} , 
 	{ "name": "inByte7_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte7", "role": "read" }} , 
 	{ "name": "inByte8_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte8", "role": "dout" }} , 
 	{ "name": "inByte8_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte8", "role": "num_data_valid" }} , 
 	{ "name": "inByte8_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte8", "role": "fifo_cap" }} , 
 	{ "name": "inByte8_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte8", "role": "empty_n" }} , 
 	{ "name": "inByte8_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte8", "role": "read" }} , 
 	{ "name": "inByte9_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte9", "role": "dout" }} , 
 	{ "name": "inByte9_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte9", "role": "num_data_valid" }} , 
 	{ "name": "inByte9_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte9", "role": "fifo_cap" }} , 
 	{ "name": "inByte9_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte9", "role": "empty_n" }} , 
 	{ "name": "inByte9_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte9", "role": "read" }} , 
 	{ "name": "inByte10_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte10", "role": "dout" }} , 
 	{ "name": "inByte10_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte10", "role": "num_data_valid" }} , 
 	{ "name": "inByte10_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte10", "role": "fifo_cap" }} , 
 	{ "name": "inByte10_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte10", "role": "empty_n" }} , 
 	{ "name": "inByte10_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte10", "role": "read" }} , 
 	{ "name": "inByte11_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte11", "role": "dout" }} , 
 	{ "name": "inByte11_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte11", "role": "num_data_valid" }} , 
 	{ "name": "inByte11_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte11", "role": "fifo_cap" }} , 
 	{ "name": "inByte11_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte11", "role": "empty_n" }} , 
 	{ "name": "inByte11_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte11", "role": "read" }} , 
 	{ "name": "inByte12_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte12", "role": "dout" }} , 
 	{ "name": "inByte12_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte12", "role": "num_data_valid" }} , 
 	{ "name": "inByte12_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte12", "role": "fifo_cap" }} , 
 	{ "name": "inByte12_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte12", "role": "empty_n" }} , 
 	{ "name": "inByte12_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte12", "role": "read" }} , 
 	{ "name": "inByte13_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte13", "role": "dout" }} , 
 	{ "name": "inByte13_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte13", "role": "num_data_valid" }} , 
 	{ "name": "inByte13_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte13", "role": "fifo_cap" }} , 
 	{ "name": "inByte13_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte13", "role": "empty_n" }} , 
 	{ "name": "inByte13_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte13", "role": "read" }} , 
 	{ "name": "inByte14_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte14", "role": "dout" }} , 
 	{ "name": "inByte14_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte14", "role": "num_data_valid" }} , 
 	{ "name": "inByte14_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte14", "role": "fifo_cap" }} , 
 	{ "name": "inByte14_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte14", "role": "empty_n" }} , 
 	{ "name": "inByte14_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte14", "role": "read" }} , 
 	{ "name": "inByte15_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte15", "role": "dout" }} , 
 	{ "name": "inByte15_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte15", "role": "num_data_valid" }} , 
 	{ "name": "inByte15_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte15", "role": "fifo_cap" }} , 
 	{ "name": "inByte15_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte15", "role": "empty_n" }} , 
 	{ "name": "inByte15_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte15", "role": "read" }} , 
 	{ "name": "init_value_2", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "init_value_2", "role": "default" }} , 
 	{ "name": "blocks_in_chunk", "direction": "in", "datatype": "sc_lv", "bitwidth":29, "type": "signal", "bundle":{"name": "blocks_in_chunk", "role": "default" }} , 
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
 	{ "name": "mask_1", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "mask_1", "role": "default" }} , 
 	{ "name": "crc_out", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crc_out", "role": "default" }} , 
 	{ "name": "crc_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "crc_out", "role": "ap_vld" }} , 
 	{ "name": "ap_ext_blocking_n", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ap_ext_blocking_n", "role": "default" }} , 
 	{ "name": "ap_str_blocking_n", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ap_str_blocking_n", "role": "default" }} , 
 	{ "name": "ap_int_blocking_n", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "ap_int_blocking_n", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1"],
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
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
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
		crc_out {Type O LastRead -1 FirstWrite 1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "2", "Max" : "134217730"}
	, {"Name" : "Interval", "Min" : "2", "Max" : "134217730"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	init_value_2 { ap_none {  { init_value_2 in_data 0 32 } } }
	blocks_in_chunk { ap_none {  { blocks_in_chunk in_data 0 29 } } }
	inByte0 { ap_fifo {  { inByte0_dout fifo_port_we 0 8 }  { inByte0_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte0_fifo_cap fifo_update 0 7 }  { inByte0_empty_n fifo_status 0 1 }  { inByte0_read fifo_data 1 1 } } }
	inByte1 { ap_fifo {  { inByte1_dout fifo_port_we 0 8 }  { inByte1_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte1_fifo_cap fifo_update 0 7 }  { inByte1_empty_n fifo_status 0 1 }  { inByte1_read fifo_data 1 1 } } }
	inByte2 { ap_fifo {  { inByte2_dout fifo_port_we 0 8 }  { inByte2_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte2_fifo_cap fifo_update 0 7 }  { inByte2_empty_n fifo_status 0 1 }  { inByte2_read fifo_data 1 1 } } }
	inByte3 { ap_fifo {  { inByte3_dout fifo_port_we 0 8 }  { inByte3_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte3_fifo_cap fifo_update 0 7 }  { inByte3_empty_n fifo_status 0 1 }  { inByte3_read fifo_data 1 1 } } }
	inByte4 { ap_fifo {  { inByte4_dout fifo_port_we 0 8 }  { inByte4_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte4_fifo_cap fifo_update 0 7 }  { inByte4_empty_n fifo_status 0 1 }  { inByte4_read fifo_data 1 1 } } }
	inByte5 { ap_fifo {  { inByte5_dout fifo_port_we 0 8 }  { inByte5_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte5_fifo_cap fifo_update 0 7 }  { inByte5_empty_n fifo_status 0 1 }  { inByte5_read fifo_data 1 1 } } }
	inByte6 { ap_fifo {  { inByte6_dout fifo_port_we 0 8 }  { inByte6_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte6_fifo_cap fifo_update 0 7 }  { inByte6_empty_n fifo_status 0 1 }  { inByte6_read fifo_data 1 1 } } }
	inByte7 { ap_fifo {  { inByte7_dout fifo_port_we 0 8 }  { inByte7_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte7_fifo_cap fifo_update 0 7 }  { inByte7_empty_n fifo_status 0 1 }  { inByte7_read fifo_data 1 1 } } }
	inByte8 { ap_fifo {  { inByte8_dout fifo_port_we 0 8 }  { inByte8_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte8_fifo_cap fifo_update 0 7 }  { inByte8_empty_n fifo_status 0 1 }  { inByte8_read fifo_data 1 1 } } }
	inByte9 { ap_fifo {  { inByte9_dout fifo_port_we 0 8 }  { inByte9_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte9_fifo_cap fifo_update 0 7 }  { inByte9_empty_n fifo_status 0 1 }  { inByte9_read fifo_data 1 1 } } }
	inByte10 { ap_fifo {  { inByte10_dout fifo_port_we 0 8 }  { inByte10_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte10_fifo_cap fifo_update 0 7 }  { inByte10_empty_n fifo_status 0 1 }  { inByte10_read fifo_data 1 1 } } }
	inByte11 { ap_fifo {  { inByte11_dout fifo_port_we 0 8 }  { inByte11_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte11_fifo_cap fifo_update 0 7 }  { inByte11_empty_n fifo_status 0 1 }  { inByte11_read fifo_data 1 1 } } }
	inByte12 { ap_fifo {  { inByte12_dout fifo_port_we 0 8 }  { inByte12_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte12_fifo_cap fifo_update 0 7 }  { inByte12_empty_n fifo_status 0 1 }  { inByte12_read fifo_data 1 1 } } }
	inByte13 { ap_fifo {  { inByte13_dout fifo_port_we 0 8 }  { inByte13_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte13_fifo_cap fifo_update 0 7 }  { inByte13_empty_n fifo_status 0 1 }  { inByte13_read fifo_data 1 1 } } }
	inByte14 { ap_fifo {  { inByte14_dout fifo_port_we 0 8 }  { inByte14_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte14_fifo_cap fifo_update 0 7 }  { inByte14_empty_n fifo_status 0 1 }  { inByte14_read fifo_data 1 1 } } }
	inByte15 { ap_fifo {  { inByte15_dout fifo_port_we 0 8 }  { inByte15_num_data_valid fifo_status_num_data_valid 0 7 }  { inByte15_fifo_cap fifo_update 0 7 }  { inByte15_empty_n fifo_status 0 1 }  { inByte15_read fifo_data 1 1 } } }
	crcTables_0 { ap_memory {  { crcTables_0_address0 mem_address 1 8 }  { crcTables_0_ce0 mem_ce 1 1 }  { crcTables_0_q0 in_data 0 32 } } }
	crcTables_1 { ap_memory {  { crcTables_1_address0 mem_address 1 8 }  { crcTables_1_ce0 mem_ce 1 1 }  { crcTables_1_q0 in_data 0 32 } } }
	crcTables_2 { ap_memory {  { crcTables_2_address0 mem_address 1 8 }  { crcTables_2_ce0 mem_ce 1 1 }  { crcTables_2_q0 in_data 0 32 } } }
	crcTables_3 { ap_memory {  { crcTables_3_address0 mem_address 1 8 }  { crcTables_3_ce0 mem_ce 1 1 }  { crcTables_3_q0 in_data 0 32 } } }
	crcTables_4 { ap_memory {  { crcTables_4_address0 mem_address 1 8 }  { crcTables_4_ce0 mem_ce 1 1 }  { crcTables_4_q0 in_data 0 32 } } }
	crcTables_5 { ap_memory {  { crcTables_5_address0 mem_address 1 8 }  { crcTables_5_ce0 mem_ce 1 1 }  { crcTables_5_q0 in_data 0 32 } } }
	crcTables_6 { ap_memory {  { crcTables_6_address0 mem_address 1 8 }  { crcTables_6_ce0 mem_ce 1 1 }  { crcTables_6_q0 in_data 0 32 } } }
	crcTables_7 { ap_memory {  { crcTables_7_address0 mem_address 1 8 }  { crcTables_7_ce0 mem_ce 1 1 }  { crcTables_7_q0 in_data 0 32 } } }
	crcTables_8 { ap_memory {  { crcTables_8_address0 mem_address 1 8 }  { crcTables_8_ce0 mem_ce 1 1 }  { crcTables_8_q0 in_data 0 32 } } }
	crcTables_9 { ap_memory {  { crcTables_9_address0 mem_address 1 8 }  { crcTables_9_ce0 mem_ce 1 1 }  { crcTables_9_q0 in_data 0 32 } } }
	crcTables_10 { ap_memory {  { crcTables_10_address0 mem_address 1 8 }  { crcTables_10_ce0 mem_ce 1 1 }  { crcTables_10_q0 in_data 0 32 } } }
	crcTables_11 { ap_memory {  { crcTables_11_address0 mem_address 1 8 }  { crcTables_11_ce0 mem_ce 1 1 }  { crcTables_11_q0 in_data 0 32 } } }
	crcTables_12 { ap_memory {  { crcTables_12_address0 mem_address 1 8 }  { crcTables_12_ce0 mem_ce 1 1 }  { crcTables_12_q0 in_data 0 32 } } }
	crcTables_13 { ap_memory {  { crcTables_13_address0 mem_address 1 8 }  { crcTables_13_ce0 mem_ce 1 1 }  { crcTables_13_q0 in_data 0 32 } } }
	crcTables_14 { ap_memory {  { crcTables_14_address0 mem_address 1 8 }  { crcTables_14_ce0 mem_ce 1 1 }  { crcTables_14_q0 in_data 0 32 } } }
	crcTables_15 { ap_memory {  { crcTables_15_address0 mem_address 1 8 }  { crcTables_15_ce0 mem_ce 1 1 }  { crcTables_15_q0 in_data 0 32 } } }
	mask_1 { ap_none {  { mask_1 in_data 0 32 } } }
	crc_out { ap_vld {  { crc_out out_data 1 32 }  { crc_out_ap_vld out_vld 1 1 } } }
}
