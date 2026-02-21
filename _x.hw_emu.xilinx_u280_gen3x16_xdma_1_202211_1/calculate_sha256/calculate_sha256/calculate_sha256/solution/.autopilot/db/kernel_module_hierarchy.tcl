set ModuleHierarchy {[{
"Name" : "calculate_sha256","ID" : "0","Type" : "sequential",
"SubLoops" : [
	{"Name" : "hash_chunk_loop","ID" : "1","Type" : "no",
	"SubInsts" : [
	{"Name" : "grp_sha256_process_chunk_fu_106","ID" : "2","Type" : "sequential",
			"SubInsts" : [
			{"Name" : "grp_sha256_process_chunk_Pipeline_clear_pad_block_fu_2520","ID" : "3","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "clear_pad_block","ID" : "4","Type" : "pipeline"},]},
			{"Name" : "grp_sha256_process_chunk_Pipeline_copy_remainder_fu_2525","ID" : "5","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "copy_remainder","ID" : "6","Type" : "pipeline"},]},
			{"Name" : "grp_sha256_process_chunk_Pipeline_clear_second_block_fu_2536","ID" : "7","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "clear_second_block","ID" : "8","Type" : "pipeline"},]},
			{"Name" : "grp_sha256_process_chunk_Pipeline_len_store_double_fu_2541","ID" : "9","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "len_store_double","ID" : "10","Type" : "pipeline"},]},
			{"Name" : "grp_sha256_process_chunk_Pipeline_len_store_single_fu_2547","ID" : "11","Type" : "sequential",
				"SubLoops" : [
				{"Name" : "len_store_single","ID" : "12","Type" : "pipeline"},]},],
			"SubLoops" : [
			{"Name" : "full_block_loop_vec","ID" : "13","Type" : "no"},
			{"Name" : "full_block_loop_scalar","ID" : "14","Type" : "no",
			"SubInsts" : [
			{"Name" : "grp_sha256_compress_block_fu_2497","ID" : "15","Type" : "sequential",
					"SubInsts" : [
					{"Name" : "grp_sha256_compress_block_Pipeline_init_words_fu_132","ID" : "16","Type" : "sequential",
						"SubLoops" : [
						{"Name" : "init_words","ID" : "17","Type" : "pipeline"},]},
					{"Name" : "grp_sha256_compress_block_Pipeline_expand_words_fu_144","ID" : "18","Type" : "sequential",
						"SubLoops" : [
						{"Name" : "expand_words","ID" : "19","Type" : "pipeline"},]},
					{"Name" : "grp_sha256_compress_block_Pipeline_round_loop_fu_151","ID" : "20","Type" : "sequential",
						"SubLoops" : [
						{"Name" : "round_loop","ID" : "21","Type" : "pipeline"},]},]},]},]},
	{"Name" : "grp_calculate_sha256_Pipeline_hash_write_digest_fu_116","ID" : "22","Type" : "sequential",
			"SubLoops" : [
			{"Name" : "hash_write_digest","ID" : "23","Type" : "pipeline"},]},]},]
}]}