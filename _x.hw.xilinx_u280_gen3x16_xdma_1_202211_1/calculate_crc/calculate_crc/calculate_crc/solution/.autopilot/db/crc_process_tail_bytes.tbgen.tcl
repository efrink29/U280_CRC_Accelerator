set moduleName crc_process_tail_bytes
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
set C_modelName {crc_process_tail_bytes}
set C_modelType { int 32 }
set C_modelArgList {
	{ crcTables_0 int 32 regular {array 256 { 1 } 1 1 }  }
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
	{ tail_bytes int 31 regular  }
	{ crc int 32 regular  }
	{ mask int 32 regular  }
}
set C_modelArgMapList {[ 
	{ "Name" : "crcTables_0", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
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
 	{ "Name" : "tail_bytes", "interface" : "wire", "bitwidth" : 31, "direction" : "READONLY"} , 
 	{ "Name" : "crc", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "mask", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "ap_return", "interface" : "wire", "bitwidth" : 32} ]}
# RTL Port declarations: 
set portNum 93
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ inByte1431_dout sc_in sc_lv 8 signal 15 } 
	{ inByte1431_num_data_valid sc_in sc_lv 7 signal 15 } 
	{ inByte1431_fifo_cap sc_in sc_lv 7 signal 15 } 
	{ inByte1431_empty_n sc_in sc_logic 1 signal 15 } 
	{ inByte1431_read sc_out sc_logic 1 signal 15 } 
	{ inByte1330_dout sc_in sc_lv 8 signal 14 } 
	{ inByte1330_num_data_valid sc_in sc_lv 7 signal 14 } 
	{ inByte1330_fifo_cap sc_in sc_lv 7 signal 14 } 
	{ inByte1330_empty_n sc_in sc_logic 1 signal 14 } 
	{ inByte1330_read sc_out sc_logic 1 signal 14 } 
	{ inByte1229_dout sc_in sc_lv 8 signal 13 } 
	{ inByte1229_num_data_valid sc_in sc_lv 7 signal 13 } 
	{ inByte1229_fifo_cap sc_in sc_lv 7 signal 13 } 
	{ inByte1229_empty_n sc_in sc_logic 1 signal 13 } 
	{ inByte1229_read sc_out sc_logic 1 signal 13 } 
	{ inByte1128_dout sc_in sc_lv 8 signal 12 } 
	{ inByte1128_num_data_valid sc_in sc_lv 7 signal 12 } 
	{ inByte1128_fifo_cap sc_in sc_lv 7 signal 12 } 
	{ inByte1128_empty_n sc_in sc_logic 1 signal 12 } 
	{ inByte1128_read sc_out sc_logic 1 signal 12 } 
	{ inByte1027_dout sc_in sc_lv 8 signal 11 } 
	{ inByte1027_num_data_valid sc_in sc_lv 7 signal 11 } 
	{ inByte1027_fifo_cap sc_in sc_lv 7 signal 11 } 
	{ inByte1027_empty_n sc_in sc_logic 1 signal 11 } 
	{ inByte1027_read sc_out sc_logic 1 signal 11 } 
	{ inByte926_dout sc_in sc_lv 8 signal 10 } 
	{ inByte926_num_data_valid sc_in sc_lv 7 signal 10 } 
	{ inByte926_fifo_cap sc_in sc_lv 7 signal 10 } 
	{ inByte926_empty_n sc_in sc_logic 1 signal 10 } 
	{ inByte926_read sc_out sc_logic 1 signal 10 } 
	{ inByte825_dout sc_in sc_lv 8 signal 9 } 
	{ inByte825_num_data_valid sc_in sc_lv 7 signal 9 } 
	{ inByte825_fifo_cap sc_in sc_lv 7 signal 9 } 
	{ inByte825_empty_n sc_in sc_logic 1 signal 9 } 
	{ inByte825_read sc_out sc_logic 1 signal 9 } 
	{ inByte724_dout sc_in sc_lv 8 signal 8 } 
	{ inByte724_num_data_valid sc_in sc_lv 7 signal 8 } 
	{ inByte724_fifo_cap sc_in sc_lv 7 signal 8 } 
	{ inByte724_empty_n sc_in sc_logic 1 signal 8 } 
	{ inByte724_read sc_out sc_logic 1 signal 8 } 
	{ inByte623_dout sc_in sc_lv 8 signal 7 } 
	{ inByte623_num_data_valid sc_in sc_lv 7 signal 7 } 
	{ inByte623_fifo_cap sc_in sc_lv 7 signal 7 } 
	{ inByte623_empty_n sc_in sc_logic 1 signal 7 } 
	{ inByte623_read sc_out sc_logic 1 signal 7 } 
	{ inByte522_dout sc_in sc_lv 8 signal 6 } 
	{ inByte522_num_data_valid sc_in sc_lv 7 signal 6 } 
	{ inByte522_fifo_cap sc_in sc_lv 7 signal 6 } 
	{ inByte522_empty_n sc_in sc_logic 1 signal 6 } 
	{ inByte522_read sc_out sc_logic 1 signal 6 } 
	{ inByte421_dout sc_in sc_lv 8 signal 5 } 
	{ inByte421_num_data_valid sc_in sc_lv 7 signal 5 } 
	{ inByte421_fifo_cap sc_in sc_lv 7 signal 5 } 
	{ inByte421_empty_n sc_in sc_logic 1 signal 5 } 
	{ inByte421_read sc_out sc_logic 1 signal 5 } 
	{ inByte320_dout sc_in sc_lv 8 signal 4 } 
	{ inByte320_num_data_valid sc_in sc_lv 7 signal 4 } 
	{ inByte320_fifo_cap sc_in sc_lv 7 signal 4 } 
	{ inByte320_empty_n sc_in sc_logic 1 signal 4 } 
	{ inByte320_read sc_out sc_logic 1 signal 4 } 
	{ inByte219_dout sc_in sc_lv 8 signal 3 } 
	{ inByte219_num_data_valid sc_in sc_lv 7 signal 3 } 
	{ inByte219_fifo_cap sc_in sc_lv 7 signal 3 } 
	{ inByte219_empty_n sc_in sc_logic 1 signal 3 } 
	{ inByte219_read sc_out sc_logic 1 signal 3 } 
	{ inByte118_dout sc_in sc_lv 8 signal 2 } 
	{ inByte118_num_data_valid sc_in sc_lv 7 signal 2 } 
	{ inByte118_fifo_cap sc_in sc_lv 7 signal 2 } 
	{ inByte118_empty_n sc_in sc_logic 1 signal 2 } 
	{ inByte118_read sc_out sc_logic 1 signal 2 } 
	{ inByte017_dout sc_in sc_lv 8 signal 1 } 
	{ inByte017_num_data_valid sc_in sc_lv 7 signal 1 } 
	{ inByte017_fifo_cap sc_in sc_lv 7 signal 1 } 
	{ inByte017_empty_n sc_in sc_logic 1 signal 1 } 
	{ inByte017_read sc_out sc_logic 1 signal 1 } 
	{ inByte1532_dout sc_in sc_lv 8 signal 16 } 
	{ inByte1532_num_data_valid sc_in sc_lv 7 signal 16 } 
	{ inByte1532_fifo_cap sc_in sc_lv 7 signal 16 } 
	{ inByte1532_empty_n sc_in sc_logic 1 signal 16 } 
	{ inByte1532_read sc_out sc_logic 1 signal 16 } 
	{ crcTables_0_address0 sc_out sc_lv 8 signal 0 } 
	{ crcTables_0_ce0 sc_out sc_logic 1 signal 0 } 
	{ crcTables_0_q0 sc_in sc_lv 32 signal 0 } 
	{ tail_bytes sc_in sc_lv 31 signal 17 } 
	{ crc sc_in sc_lv 32 signal 18 } 
	{ mask sc_in sc_lv 32 signal 19 } 
	{ ap_return sc_out sc_lv 32 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "inByte1431_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1431", "role": "dout" }} , 
 	{ "name": "inByte1431_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1431", "role": "num_data_valid" }} , 
 	{ "name": "inByte1431_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1431", "role": "fifo_cap" }} , 
 	{ "name": "inByte1431_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1431", "role": "empty_n" }} , 
 	{ "name": "inByte1431_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1431", "role": "read" }} , 
 	{ "name": "inByte1330_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1330", "role": "dout" }} , 
 	{ "name": "inByte1330_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1330", "role": "num_data_valid" }} , 
 	{ "name": "inByte1330_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1330", "role": "fifo_cap" }} , 
 	{ "name": "inByte1330_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1330", "role": "empty_n" }} , 
 	{ "name": "inByte1330_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1330", "role": "read" }} , 
 	{ "name": "inByte1229_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1229", "role": "dout" }} , 
 	{ "name": "inByte1229_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1229", "role": "num_data_valid" }} , 
 	{ "name": "inByte1229_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1229", "role": "fifo_cap" }} , 
 	{ "name": "inByte1229_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1229", "role": "empty_n" }} , 
 	{ "name": "inByte1229_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1229", "role": "read" }} , 
 	{ "name": "inByte1128_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1128", "role": "dout" }} , 
 	{ "name": "inByte1128_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1128", "role": "num_data_valid" }} , 
 	{ "name": "inByte1128_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1128", "role": "fifo_cap" }} , 
 	{ "name": "inByte1128_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1128", "role": "empty_n" }} , 
 	{ "name": "inByte1128_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1128", "role": "read" }} , 
 	{ "name": "inByte1027_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1027", "role": "dout" }} , 
 	{ "name": "inByte1027_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1027", "role": "num_data_valid" }} , 
 	{ "name": "inByte1027_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1027", "role": "fifo_cap" }} , 
 	{ "name": "inByte1027_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1027", "role": "empty_n" }} , 
 	{ "name": "inByte1027_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1027", "role": "read" }} , 
 	{ "name": "inByte926_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte926", "role": "dout" }} , 
 	{ "name": "inByte926_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte926", "role": "num_data_valid" }} , 
 	{ "name": "inByte926_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte926", "role": "fifo_cap" }} , 
 	{ "name": "inByte926_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte926", "role": "empty_n" }} , 
 	{ "name": "inByte926_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte926", "role": "read" }} , 
 	{ "name": "inByte825_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte825", "role": "dout" }} , 
 	{ "name": "inByte825_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte825", "role": "num_data_valid" }} , 
 	{ "name": "inByte825_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte825", "role": "fifo_cap" }} , 
 	{ "name": "inByte825_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte825", "role": "empty_n" }} , 
 	{ "name": "inByte825_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte825", "role": "read" }} , 
 	{ "name": "inByte724_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte724", "role": "dout" }} , 
 	{ "name": "inByte724_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte724", "role": "num_data_valid" }} , 
 	{ "name": "inByte724_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte724", "role": "fifo_cap" }} , 
 	{ "name": "inByte724_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte724", "role": "empty_n" }} , 
 	{ "name": "inByte724_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte724", "role": "read" }} , 
 	{ "name": "inByte623_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte623", "role": "dout" }} , 
 	{ "name": "inByte623_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte623", "role": "num_data_valid" }} , 
 	{ "name": "inByte623_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte623", "role": "fifo_cap" }} , 
 	{ "name": "inByte623_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte623", "role": "empty_n" }} , 
 	{ "name": "inByte623_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte623", "role": "read" }} , 
 	{ "name": "inByte522_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte522", "role": "dout" }} , 
 	{ "name": "inByte522_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte522", "role": "num_data_valid" }} , 
 	{ "name": "inByte522_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte522", "role": "fifo_cap" }} , 
 	{ "name": "inByte522_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte522", "role": "empty_n" }} , 
 	{ "name": "inByte522_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte522", "role": "read" }} , 
 	{ "name": "inByte421_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte421", "role": "dout" }} , 
 	{ "name": "inByte421_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte421", "role": "num_data_valid" }} , 
 	{ "name": "inByte421_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte421", "role": "fifo_cap" }} , 
 	{ "name": "inByte421_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte421", "role": "empty_n" }} , 
 	{ "name": "inByte421_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte421", "role": "read" }} , 
 	{ "name": "inByte320_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte320", "role": "dout" }} , 
 	{ "name": "inByte320_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte320", "role": "num_data_valid" }} , 
 	{ "name": "inByte320_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte320", "role": "fifo_cap" }} , 
 	{ "name": "inByte320_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte320", "role": "empty_n" }} , 
 	{ "name": "inByte320_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte320", "role": "read" }} , 
 	{ "name": "inByte219_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte219", "role": "dout" }} , 
 	{ "name": "inByte219_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte219", "role": "num_data_valid" }} , 
 	{ "name": "inByte219_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte219", "role": "fifo_cap" }} , 
 	{ "name": "inByte219_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte219", "role": "empty_n" }} , 
 	{ "name": "inByte219_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte219", "role": "read" }} , 
 	{ "name": "inByte118_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte118", "role": "dout" }} , 
 	{ "name": "inByte118_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte118", "role": "num_data_valid" }} , 
 	{ "name": "inByte118_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte118", "role": "fifo_cap" }} , 
 	{ "name": "inByte118_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte118", "role": "empty_n" }} , 
 	{ "name": "inByte118_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte118", "role": "read" }} , 
 	{ "name": "inByte017_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte017", "role": "dout" }} , 
 	{ "name": "inByte017_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte017", "role": "num_data_valid" }} , 
 	{ "name": "inByte017_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte017", "role": "fifo_cap" }} , 
 	{ "name": "inByte017_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte017", "role": "empty_n" }} , 
 	{ "name": "inByte017_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte017", "role": "read" }} , 
 	{ "name": "inByte1532_dout", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "inByte1532", "role": "dout" }} , 
 	{ "name": "inByte1532_num_data_valid", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1532", "role": "num_data_valid" }} , 
 	{ "name": "inByte1532_fifo_cap", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "inByte1532", "role": "fifo_cap" }} , 
 	{ "name": "inByte1532_empty_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1532", "role": "empty_n" }} , 
 	{ "name": "inByte1532_read", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inByte1532", "role": "read" }} , 
 	{ "name": "crcTables_0_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "crcTables_0", "role": "address0" }} , 
 	{ "name": "crcTables_0_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "crcTables_0", "role": "ce0" }} , 
 	{ "name": "crcTables_0_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crcTables_0", "role": "q0" }} , 
 	{ "name": "tail_bytes", "direction": "in", "datatype": "sc_lv", "bitwidth":31, "type": "signal", "bundle":{"name": "tail_bytes", "role": "default" }} , 
 	{ "name": "crc", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "crc", "role": "default" }} , 
 	{ "name": "mask", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "mask", "role": "default" }} , 
 	{ "name": "ap_return", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "ap_return", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1"],
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
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
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
	{"Name" : "Latency", "Min" : "5", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "5", "Max" : "-1"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	crcTables_0 { ap_memory {  { crcTables_0_address0 mem_address 1 8 }  { crcTables_0_ce0 mem_ce 1 1 }  { crcTables_0_q0 in_data 0 32 } } }
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
	tail_bytes { ap_none {  { tail_bytes in_data 0 31 } } }
	crc { ap_none {  { crc in_data 0 32 } } }
	mask { ap_none {  { mask in_data 0 32 } } }
}
