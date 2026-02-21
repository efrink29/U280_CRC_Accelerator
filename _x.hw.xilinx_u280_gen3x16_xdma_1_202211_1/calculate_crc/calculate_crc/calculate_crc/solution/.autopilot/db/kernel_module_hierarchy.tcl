set ModuleHierarchy {[{
"Name" : "calculate_crc","ID" : "0","Type" : "sequential",
"SubInsts" : [
	{"Name" : "grp_process_crc_fu_122","ID" : "1","Type" : "dataflow",
		"SubInsts" : [
		{"Name" : "process_crc_Loop_init_lut_proc_U0","ID" : "2","Type" : "sequential",
			"SubInsts" : [
			{"Name" : "grp_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1_fu_91","ID" : "3","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "init_lut_VITIS_LOOP_678_1","ID" : "4","Type" : "pipeline"},]},]},
		{"Name" : "read_input_U0","ID" : "5","Type" : "sequential",
			"SubInsts" : [
			{"Name" : "grp_read_input_Pipeline_mem_rd_fu_142","ID" : "6","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "mem_rd","ID" : "7","Type" : "pipeline"},]},]},
		{"Name" : "entry_proc_U0","ID" : "8","Type" : "sequential"},
		{"Name" : "process_blocks_U0","ID" : "9","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "chunk_loop","ID" : "10","Type" : "no",
			"SubInsts" : [
			{"Name" : "grp_process_blocks_Pipeline_block_loop_fu_199","ID" : "11","Type" : "sequential",
					"SubLoops" : [
					{"Name" : "block_loop","ID" : "12","Type" : "pipeline"},]},
			{"Name" : "grp_process_blocks_Pipeline_tail_loop_fu_271","ID" : "13","Type" : "sequential",
					"SubLoops" : [
					{"Name" : "tail_loop","ID" : "14","Type" : "pipeline"},]},]},]},
		{"Name" : "write_output_U0","ID" : "15","Type" : "sequential",
			"SubInsts" : [
			{"Name" : "grp_write_output_Pipeline_VITIS_LOOP_442_1_fu_58","ID" : "16","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "VITIS_LOOP_442_1","ID" : "17","Type" : "pipeline"},]},]},]},]
}]}