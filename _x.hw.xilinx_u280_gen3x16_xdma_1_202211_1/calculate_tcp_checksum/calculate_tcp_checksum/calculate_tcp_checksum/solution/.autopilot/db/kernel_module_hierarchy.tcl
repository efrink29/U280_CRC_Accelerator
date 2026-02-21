set ModuleHierarchy {[{
"Name" : "calculate_tcp_checksum","ID" : "0","Type" : "sequential",
"SubInsts" : [
	{"Name" : "grp_process_tcp_checksum_fu_86","ID" : "1","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "tcp_chunk_loop","ID" : "2","Type" : "no",
		"SubInsts" : [
		{"Name" : "grp_process_tcp_checksum_Pipeline_word_loop_fu_263","ID" : "3","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "word_loop","ID" : "4","Type" : "pipeline"},]},
		{"Name" : "grp_process_tcp_checksum_Pipeline_vec_word_loop_fu_274","ID" : "5","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "vec_word_loop","ID" : "6","Type" : "pipeline"},]},
		{"Name" : "grp_process_tcp_checksum_Pipeline_tail_word_loop_fu_283","ID" : "7","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "tail_word_loop","ID" : "8","Type" : "pipeline"},]},]},]},]
}]}