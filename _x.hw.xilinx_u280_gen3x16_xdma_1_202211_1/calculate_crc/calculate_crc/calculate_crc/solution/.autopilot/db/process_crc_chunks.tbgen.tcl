set moduleName process_crc_chunks
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
set C_modelName {process_crc_chunks}
set C_modelType { void 0 }
set C_modelArgList {
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
	{ crc_size int 32 regular  }
	{ init_value int 32 regular  }
	{ outStream33 int 32 regular {fifo 1 volatile }  }
	{ numChunks int 32 regular  }
	{ chunkSize int 32 regular  }
	{ numChunks_c int 32 regular {fifo 1}  }
}
set C_modelArgMapList {[ 
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
 	{ "Name" : "crc_size", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "init_value", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "outStream33", "interface" : "fifo", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "numChunks", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "chunkSize", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "numChunks_c", "interface" : "fifo", "bitwidth" : 32, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 149
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_continue sc_in sc_logic 1 continue -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ crcTables_0_address0 sc_out sc_lv 8 signal 0 } 
	{ crcTables_0_ce0 sc_out sc_logic 1 signal 0 } 
	{ crcTables_0_q0 sc_in sc_lv 32 signal 0 } 
	{ crcTables_1_address0 sc_out sc_lv 8 signal 1 } 
	{ crcTables_1_ce0 sc_out sc_logic 1 signal 1 } 
	{ crcTables_1_q0 sc_in sc_lv 32 signal 1 } 
	{ crcTables_2_address0 sc_out sc_lv 8 signal 2 } 
	{ crcTables_2_ce0 sc_out sc_logic 1 signal 2 } 
	{ crcTables_2_q0 sc_in sc_lv 32 signal 2 } 
	{ crcTables_3_address0 sc_out sc_lv 8 signal 3 } 
	{ crcTables_3_ce0 sc_out sc_logic 1 signal 3 } 
	{ crcTables_3_q0 sc_in sc_lv 32 signal 3 } 
	{ crcTables_4_address0 sc_out sc_lv 8 signal 4 } 
	{ crcTables_4_ce0 sc_out sc_logic 1 signal 4 } 
	{ crcTables_4_q0 sc_in sc_lv 32 signal 4 } 
	{ crcTables_5_address0 sc_out sc_lv 8 signal 5 } 
	{ crcTables_5_ce0 sc_out sc_logic 1 signal 5 } 
	{ crcTables_5_q0 sc_in sc_lv 32 signal 5 } 
	{ crcTables_6_address0 sc_out sc_lv 8 signal 6 } 
	{ crcTables_6_ce0 sc_out sc_logic 1 signal 6 } 
	{ crcTables_6_q0 sc_in sc_lv 32 signal 6 } 
	{ crcTables_7_address0 sc_out sc_lv 8 signal 7 } 
	{ crcTables_7_ce0 sc_out sc_logic 1 signal 7 } 
	{ crcTables_7_q0 sc_in sc_lv 32 signal 7 } 
	{ crcTables_8_address0 sc_out sc_lv 8 signal 8 } 
	{ crcTables_8_ce0 sc_out sc_logic 1 signal 8 } 
	{ crcTables_8_q0 sc_in sc_lv 32 signal 8 } 
	{ crcTables_9_address0 sc_out sc_lv 8 signal 9 } 
	{ crcTables_9_ce0 sc_out sc_logic 1 signal 9 } 
	{ crcTables_9_q0 sc_in sc_lv 32 signal 9 } 
	{ crcTables_10_address0 sc_out sc_lv 8 signal 10 } 
	{ crcTables_10_ce0 sc_out sc_logic 1 signal 10 } 
	{ crcTables_10_q0 sc_in sc_lv 32 signal 10 } 
	{ crcTables_11_address0 sc_out sc_lv 8 signal 11 } 
	{ crcTables_11_ce0 sc_out sc_logic 1 signal 11 } 
	{ crcTables_11_q0 sc_in sc_lv 32 signal 11 } 
	{ crcTables_12_address0 sc_out sc_lv 8 signal 12 } 
	{ crcTables_12_ce0 sc_out sc_logic 1 signal 12 } 
	{ crcTables_12_q0 sc_in sc_lv 32 signal 12 } 
	{ crcTables_13_address0 sc_out sc_lv 8 signal 13 } 
	{ crcTables_13_ce0 sc_out sc_logic 1 signal 13 } 
	{ crcTables_13_q0 sc_in sc_lv 32 signal 13 } 
	{ crcTables_14_address0 sc_out sc_lv 8 signal 14 } 
	{ crcTables_14_ce0 sc_out sc_logic 1 signal 14 } 
	{ crcTables_14_q0 sc_in sc_lv 32 signal 14 } 
	{ crcTables_15_address0 sc_out sc_lv 8 signal 15 } 
	{ crcTables_15_ce0 sc_out sc_logic 1 signal 15 } 
	{ crcTables_15_q0 sc_in sc_lv 32 signal 15 } 
	{ inByte017_dout sc_in sc_lv 8 signal 16 } 
	{ inByte017_num_data_valid sc_in sc_lv 7 signal 16 } 
	{ inByte017_fifo_cap sc_in sc_lv 7 signal 16 } 
	{ inByte017_empty_n sc_in sc_logic 1 signal 16 } 
	{ inByte017_read sc_out sc_logic 1 signal 16 } 
	{ inByte118_dout sc_in sc_lv 8 signal 17 } 
	{ inByte118_num_data_valid sc_in sc_lv 7 signal 17 } 
	{ inByte118_fifo_cap sc_in sc_lv 7 signal 17 } 
	{ inByte118_empty_n sc_in sc_logic 1 signal 17 } 
	{ inByte118_read sc_out sc_logic 1 signal 17 } 
	{ inByte219_dout sc_in sc_lv 8 signal 18 } 
	{ inByte219_num_data_valid sc_in sc_lv 7 signal 18 } 
	{ inByte219_fifo_cap sc_in sc_lv 7 signal 18 } 
	{ inByte219_empty_n sc_in sc_logic 1 signal 18 } 
	{ inByte219_read sc_out sc_logic 1 signal 18 } 
	{ inByte320_dout sc_in sc_lv 8 signal 19 } 
	{ inByte320_num_data_valid sc_in sc_lv 7 signal 19 } 
	{ inByte320_fifo_cap sc_in sc_lv 7 signal 19 } 
	{ inByte320_empty_n sc_in sc_logic 1 signal 19 } 
	{ inByte320_read sc_out sc_logic 1 signal 19 } 
	{ inByte421_dout sc_in sc_lv 8 signal 20 } 
	{ inByte421_num_data_valid sc_in sc_lv 7 signal 20 } 
	{ inByte421_fifo_cap sc_in sc_lv 7 signal 20 } 
	{ inByte421_empty_n sc_in sc_logic 1 signal 20 } 
	{ inByte421_read sc_out sc_logic 1 signal 20 } 
	{ inByte522_dout sc_in sc_lv 8 signal 21 } 
	{ inByte522_num_data_valid sc_in sc_lv 7 signal 21 } 
	{ inByte522_fifo_cap sc_in sc_lv 7 signal 21 } 
	{ inByte522_empty_n sc_in sc_logic 1 signal 21 } 
	{ inByte522_read sc_out sc_logic 1 signal 21 } 
	{ inByte623_dout sc_in sc_lv 8 signal 22 } 
	{ inByte623_num_data_valid sc_in sc_lv 7 signal 22 } 
	{ inByte623_fifo_cap sc_in sc_lv 7 signal 22 } 
	{ inByte623_empty_n sc_in sc_logic 1 signal 22 } 
	{ inByte623_read sc_out sc_logic 1 signal 22 } 
	{ inByte724_dout sc_in sc_lv 8 signal 23 } 
	{ inByte724_num_data_valid sc_in sc_lv 7 signal 23 } 
	{ inByte724_fifo_cap sc_in sc_lv 7 signal 23 } 
	{ inByte724_empty_n sc_in sc_logic 1 signal 23 } 
	{ inByte724_read sc_out sc_logic 1 signal 23 } 
	{ inByte825_dout sc_in sc_lv 8 signal 24 } 
	{ inByte825_num_data_valid sc_in sc_lv 7 signal 24 } 
	{ inByte825_fifo_cap sc_in sc_lv 7 signal 24 } 
	{ inByte825_empty_n sc_in sc_logic 1 signal 24 } 
	{ inByte825_read sc_out sc_logic 1 signal 24 } 
	{ inByte926_dout sc_in sc_lv 8 signal 25 } 
	{ inByte926_num_data_valid sc_in sc_lv 7 signal 25 } 
	{ inByte926_fifo_cap sc_in sc_lv 7 signal 25 } 
	{ inByte926_empty_n sc_in sc_logic 1 signal 25 } 
	{ inByte926_read sc_out sc_logic 1 signal 25 } 
	{ inByte1027_dout sc_in sc_lv 8 signal 26 } 
	{ inByte1027_num_data_valid sc_in sc_lv 7 signal 26 } 
	{ inByte1027_fifo_cap sc_in sc_lv 7 signal 26 } 
	{ inByte1027_empty_n sc_in sc_logic 1 signal 26 } 
	{ inByte1027_read sc_out sc_logic 1 signal 26 } 
	{ inByte1128_dout sc_in sc_lv 8 signal 27 } 
	{ inByte1128_num_data_valid sc_in sc_lv 7 signal 27 } 
	{ inByte1128_fifo_cap sc_in sc_lv 7 signal 27 } 
	{ inByte1128_empty_n sc_in sc_logic 1 signal 27 } 
	{ inByte1128_read sc_out sc_logic 1 signal 27 } 
	{ inByte1229_dout sc_in sc_lv 8 signal 28 } 
	{ inByte1229_num_data_valid sc_in sc_lv 7 signal 28 } 
	{ inByte1229_fifo_cap sc_in sc_lv 7 signal 28 } 
	{ inByte1229_empty_n sc_in sc_logic 1 signal 28 } 
	{ inByte1229_read sc_out sc_logic 1 signal 28 } 
	{ inByte1330_dout sc_in sc_lv 8 signal 29 } 
	{ inByte1330_num_data_valid sc_in sc_lv 7 signal 29 } 
	{ inByte1330_fifo_cap sc_in sc_lv 7 signal 29 } 
	{ inByte1330_empty_n sc_in sc_logic 1 signal 29 } 
	{ inByte1330_read sc_out sc_logic 1 signal 29 } 
	{ inByte1431_dout sc_in sc_lv 8 signal 30 } 
	{ inByte1431_num_data_valid sc_in sc_lv 7 signal 30 } 
	{ inByte1431_fifo_cap sc_in sc_lv 7 signal 30 } 
	{ inByte1431_empty_n sc_in sc_logic 1 signal 30 } 
	{ inByte1431_read sc_out sc_logic 1 signal 30 } 
	{ inByte1532_dout sc_in sc_lv 8 signal 31 } 
	{ inByte1532_num_data_valid sc_in sc_lv 7 signal 31 } 
	{ inByte1532_fifo_cap sc_in sc_lv 7 signal 31 } 
	{ inByte1532_empty_n sc_in sc_logic 1 signal 31 } 
	{ inByte1532_read sc_out sc_logic 1 signal 31 } 
	{ crc_size sc_in sc_lv 32 signal 32 } 
	{ init_value sc_in sc_lv 32 signal 33 } 
	{ outStream33_din sc_out sc_lv 32 signal 34 } 
	{ outStream33_num_data_valid sc_in sc_lv 7 signal 34 } 
	{ outStream33_fifo_cap sc_in sc_lv 7 signal 34 } 
	{ outStream33_full_n sc_in sc_logic 1 signal 34 } 
	{ outStream33_write sc_out sc_logic 1 signal 34 } 
	{ numChunks sc_in sc_lv 32 signal 35 } 
	{ chunkSize sc_in sc_lv 32 signal 36 } 
	{ numChunks_c_din sc_out sc_lv 32 signal 37 } 
	{ numChunks_c_num_data_valid sc_in sc_lv 2 signal 37 } 
	{ numChunks_c_fifo_cap sc_in sc_lv 2 signal 37 } 
	{ numChunks_c_full_n sc_in sc_logic 1 signal 37 } 
	{ numChunks_c_write sc_out sc_logic 1 signal 37 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_continue", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "continue", "bundle":{"name": "ap_continue", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
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
 	{ "name": "crc_size", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crc_size", "role": "default" }} , 
 	{ "name": "init_value", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "init_value", "role": "default" }} , 
 	{ "name": "outStream33_din", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "outStream33", "role": "din" }} , 
 	{ "name": "outStream33_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "outStream33", "role": "num_data_valid" }} , 
 	{ "name": "outStream33_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "outStream33", "role": "fifo_cap" }} , 
 	{ "name": "outStream33_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "outStream33", "role": "full_n" }} , 
 	{ "name": "outStream33_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "outStream33", "role": "write" }} , 
 	{ "name": "numChunks", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "numChunks", "role": "default" }} , 
 	{ "name": "chunkSize", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "chunkSize", "role": "default" }} , 
 	{ "name": "numChunks_c_din", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "numChunks_c", "role": "din" }} , 
 	{ "name": "numChunks_c_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "numChunks_c", "role": "num_data_valid" }} , 
 	{ "name": "numChunks_c_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "numChunks_c", "role": "fifo_cap" }} , 
 	{ "name": "numChunks_c_full_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "numChunks_c", "role": "full_n" }} , 
 	{ "name": "numChunks_c_write", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "numChunks_c", "role": "write" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "4"],
		"CDFG" : "process_crc_chunks",
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
			{"Name" : "crcTables_0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "crcTables_0", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_0", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_1", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_2", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_3", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_4", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_5", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_6", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_7", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_8", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_9", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_9", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_10", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_10", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_11", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_11", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_12", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_12", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_13", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_13", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_14", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_14", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crcTables_15", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "crcTables_15", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte017", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte017", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte017", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte118", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte118", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte118", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte219", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte219", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte219", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte320", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte320", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte320", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte421", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte421", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte421", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte522", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte522", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte522", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte623", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte623", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte623", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte724", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte724", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte724", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte825", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte825", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte825", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte926", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte926", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte926", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte1027", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte1027", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte1027", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte1128", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte1128", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte1128", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte1229", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte1229", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte1229", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte1330", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte1330", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte1330", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte1431", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte1431", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte1431", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "inByte1532", "Type" : "Fifo", "Direction" : "I", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"SubConnect" : [
					{"ID" : "4", "SubInstance" : "grp_crc_process_tail_bytes_fu_268", "Port" : "inByte1532", "Inst_start_state" : "6", "Inst_end_state" : "7"},
					{"ID" : "1", "SubInstance" : "grp_crc_process_full_blocks_fu_197", "Port" : "inByte1532", "Inst_start_state" : "4", "Inst_end_state" : "5"}]},
			{"Name" : "crc_size", "Type" : "None", "Direction" : "I"},
			{"Name" : "init_value", "Type" : "None", "Direction" : "I"},
			{"Name" : "outStream33", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "64", "DependentChanType" : "0",
				"BlockSignal" : [
					{"Name" : "outStream33_blk_n", "Type" : "RtlSignal"}]},
			{"Name" : "numChunks", "Type" : "None", "Direction" : "I"},
			{"Name" : "chunkSize", "Type" : "None", "Direction" : "I"},
			{"Name" : "numChunks_c", "Type" : "Fifo", "Direction" : "O", "DependentProc" : ["0"], "DependentChan" : "0", "DependentChanDepth" : "2", "DependentChanType" : "2",
				"BlockSignal" : [
					{"Name" : "numChunks_c_blk_n", "Type" : "RtlSignal"}]}],
		"Loop" : [
			{"Name" : "chunk_loop", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "7", "FirstState" : "ap_ST_fsm_state3", "LastState" : ["ap_ST_fsm_state7"], "QuitState" : ["ap_ST_fsm_state3"], "PreState" : ["ap_ST_fsm_state2"], "PostState" : ["ap_ST_fsm_state1"], "OneDepthLoop" : "0", "OneStateBlock": ""}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_crc_process_full_blocks_fu_197", "Parent" : "0", "Child" : ["2"],
		"CDFG" : "crc_process_full_blocks",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "crcTables_0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_0", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_1", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_2", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_3", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_3", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_4", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_4", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_5", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_5", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_6", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_6", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_7", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_7", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_8", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_8", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_9", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_9", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_10", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_10", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_11", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_11", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_12", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_12", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_13", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_13", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_14", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_14", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "crcTables_15", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "crcTables_15", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte017", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte017", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte118", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte118", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte219", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte219", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte320", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte320", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte421", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte421", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte522", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte522", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte623", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte623", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte724", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte724", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte825", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte825", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte926", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte926", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte1027", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte1027", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte1128", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte1128", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte1229", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte1229", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte1330", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte1330", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte1431", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte1431", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "inByte1532", "Type" : "Fifo", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "2", "SubInstance" : "grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Port" : "inByte1532", "Inst_start_state" : "1", "Inst_end_state" : "2"}]},
			{"Name" : "blocks_in_chunk", "Type" : "None", "Direction" : "I"},
			{"Name" : "crc", "Type" : "None", "Direction" : "I"},
			{"Name" : "mask", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "2", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_crc_process_full_blocks_fu_197.grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131", "Parent" : "1", "Child" : ["3"],
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
	{"ID" : "3", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_crc_process_full_blocks_fu_197.grp_crc_process_full_blocks_Pipeline_full_block_loop_fu_131.flow_control_loop_pipe_sequential_init_U", "Parent" : "2"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_crc_process_tail_bytes_fu_268", "Parent" : "0", "Child" : ["5"],
		"CDFG" : "crc_process_tail_bytes",
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
			{"Name" : "crcTables_0", "Type" : "Memory", "Direction" : "I"},
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
			{"Name" : "tail_bytes", "Type" : "None", "Direction" : "I"},
			{"Name" : "crc", "Type" : "None", "Direction" : "I"},
			{"Name" : "mask", "Type" : "None", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "tail_byte_loop", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter3", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter3", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "5", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_crc_process_tail_bytes_fu_268.flow_control_loop_pipe_sequential_init_U", "Parent" : "4"}]}


set ArgLastReadFirstWriteLatency {
	process_crc_chunks {
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
		crc_size {Type I LastRead 1 FirstWrite -1}
		init_value {Type I LastRead 1 FirstWrite -1}
		outStream33 {Type O LastRead -1 FirstWrite 6}
		numChunks {Type I LastRead 0 FirstWrite -1}
		chunkSize {Type I LastRead 0 FirstWrite -1}
		numChunks_c {Type O LastRead -1 FirstWrite 0}}
	crc_process_full_blocks {
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
		blocks_in_chunk {Type I LastRead 0 FirstWrite -1}
		crc {Type I LastRead 0 FirstWrite -1}
		mask {Type I LastRead 0 FirstWrite -1}}
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
		p_out {Type O LastRead -1 FirstWrite 2}}
	crc_process_tail_bytes {
		crcTables_0 {Type I LastRead 2 FirstWrite -1}
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
		tail_bytes {Type I LastRead 0 FirstWrite -1}
		crc {Type I LastRead 0 FirstWrite -1}
		mask {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "-1", "Max" : "-1"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	crcTables_0 { ap_memory {  { crcTables_0_address0 mem_address 1 8 }  { crcTables_0_ce0 mem_ce 1 1 }  { crcTables_0_q0 mem_dout 0 32 } } }
	crcTables_1 { ap_memory {  { crcTables_1_address0 mem_address 1 8 }  { crcTables_1_ce0 mem_ce 1 1 }  { crcTables_1_q0 mem_dout 0 32 } } }
	crcTables_2 { ap_memory {  { crcTables_2_address0 mem_address 1 8 }  { crcTables_2_ce0 mem_ce 1 1 }  { crcTables_2_q0 mem_dout 0 32 } } }
	crcTables_3 { ap_memory {  { crcTables_3_address0 mem_address 1 8 }  { crcTables_3_ce0 mem_ce 1 1 }  { crcTables_3_q0 mem_dout 0 32 } } }
	crcTables_4 { ap_memory {  { crcTables_4_address0 mem_address 1 8 }  { crcTables_4_ce0 mem_ce 1 1 }  { crcTables_4_q0 mem_dout 0 32 } } }
	crcTables_5 { ap_memory {  { crcTables_5_address0 mem_address 1 8 }  { crcTables_5_ce0 mem_ce 1 1 }  { crcTables_5_q0 mem_dout 0 32 } } }
	crcTables_6 { ap_memory {  { crcTables_6_address0 mem_address 1 8 }  { crcTables_6_ce0 mem_ce 1 1 }  { crcTables_6_q0 mem_dout 0 32 } } }
	crcTables_7 { ap_memory {  { crcTables_7_address0 mem_address 1 8 }  { crcTables_7_ce0 mem_ce 1 1 }  { crcTables_7_q0 mem_dout 0 32 } } }
	crcTables_8 { ap_memory {  { crcTables_8_address0 mem_address 1 8 }  { crcTables_8_ce0 mem_ce 1 1 }  { crcTables_8_q0 mem_dout 0 32 } } }
	crcTables_9 { ap_memory {  { crcTables_9_address0 mem_address 1 8 }  { crcTables_9_ce0 mem_ce 1 1 }  { crcTables_9_q0 mem_dout 0 32 } } }
	crcTables_10 { ap_memory {  { crcTables_10_address0 mem_address 1 8 }  { crcTables_10_ce0 mem_ce 1 1 }  { crcTables_10_q0 mem_dout 0 32 } } }
	crcTables_11 { ap_memory {  { crcTables_11_address0 mem_address 1 8 }  { crcTables_11_ce0 mem_ce 1 1 }  { crcTables_11_q0 mem_dout 0 32 } } }
	crcTables_12 { ap_memory {  { crcTables_12_address0 mem_address 1 8 }  { crcTables_12_ce0 mem_ce 1 1 }  { crcTables_12_q0 mem_dout 0 32 } } }
	crcTables_13 { ap_memory {  { crcTables_13_address0 mem_address 1 8 }  { crcTables_13_ce0 mem_ce 1 1 }  { crcTables_13_q0 mem_dout 0 32 } } }
	crcTables_14 { ap_memory {  { crcTables_14_address0 mem_address 1 8 }  { crcTables_14_ce0 mem_ce 1 1 }  { crcTables_14_q0 mem_dout 0 32 } } }
	crcTables_15 { ap_memory {  { crcTables_15_address0 mem_address 1 8 }  { crcTables_15_ce0 mem_ce 1 1 }  { crcTables_15_q0 mem_dout 0 32 } } }
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
	crc_size { ap_none {  { crc_size in_data 0 32 } } }
	init_value { ap_none {  { init_value in_data 0 32 } } }
	outStream33 { ap_fifo {  { outStream33_din fifo_port_we 1 32 }  { outStream33_num_data_valid fifo_status_num_data_valid 0 7 }  { outStream33_fifo_cap fifo_update 0 7 }  { outStream33_full_n fifo_status 0 1 }  { outStream33_write fifo_data 1 1 } } }
	numChunks { ap_none {  { numChunks in_data 0 32 } } }
	chunkSize { ap_none {  { chunkSize in_data 0 32 } } }
	numChunks_c { ap_fifo {  { numChunks_c_din fifo_port_we 1 32 }  { numChunks_c_num_data_valid fifo_status_num_data_valid 0 2 }  { numChunks_c_fifo_cap fifo_update 0 2 }  { numChunks_c_full_n fifo_status 0 1 }  { numChunks_c_write fifo_data 1 1 } } }
}
