set SynModuleInfo {
  {SRCNAME sha256_compress_block_Pipeline_round_loop MODELNAME sha256_compress_block_Pipeline_round_loop RTLNAME calculate_sha256_sha256_compress_block_Pipeline_round_loop
    SUBMODULES {
      {MODELNAME calculate_sha256_mux_164_32_1_1 RTLNAME calculate_sha256_mux_164_32_1_1 BINDTYPE op TYPE mux IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME calculate_sha256_mux_167_32_1_1 RTLNAME calculate_sha256_mux_167_32_1_1 BINDTYPE op TYPE mux IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME calculate_sha256_sha256_compress_block_Pipeline_round_loop_SHA256_K_ROM_AUTO_1R RTLNAME calculate_sha256_sha256_compress_block_Pipeline_round_loop_SHA256_K_ROM_AUTO_1R BINDTYPE storage TYPE rom IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME calculate_sha256_flow_control_loop_pipe_sequential_init RTLNAME calculate_sha256_flow_control_loop_pipe_sequential_init BINDTYPE interface TYPE internal_upc_flow_control INSTNAME calculate_sha256_flow_control_loop_pipe_sequential_init_U}
    }
  }
  {SRCNAME sha256_compress_block MODELNAME sha256_compress_block RTLNAME calculate_sha256_sha256_compress_block}
  {SRCNAME sha256_process_chunk_Pipeline_clear_pad_block MODELNAME sha256_process_chunk_Pipeline_clear_pad_block RTLNAME calculate_sha256_sha256_process_chunk_Pipeline_clear_pad_block}
  {SRCNAME sha256_process_chunk_Pipeline_copy_remainder MODELNAME sha256_process_chunk_Pipeline_copy_remainder RTLNAME calculate_sha256_sha256_process_chunk_Pipeline_copy_remainder}
  {SRCNAME sha256_process_chunk_Pipeline_clear_second_block MODELNAME sha256_process_chunk_Pipeline_clear_second_block RTLNAME calculate_sha256_sha256_process_chunk_Pipeline_clear_second_block}
  {SRCNAME sha256_process_chunk_Pipeline_len_store_double MODELNAME sha256_process_chunk_Pipeline_len_store_double RTLNAME calculate_sha256_sha256_process_chunk_Pipeline_len_store_double}
  {SRCNAME sha256_process_chunk_Pipeline_len_store_single MODELNAME sha256_process_chunk_Pipeline_len_store_single RTLNAME calculate_sha256_sha256_process_chunk_Pipeline_len_store_single}
  {SRCNAME sha256_process_chunk MODELNAME sha256_process_chunk RTLNAME calculate_sha256_sha256_process_chunk
    SUBMODULES {
      {MODELNAME calculate_sha256_sha256_process_chunk_pad_block_RAM_AUTO_1R1W RTLNAME calculate_sha256_sha256_process_chunk_pad_block_RAM_AUTO_1R1W BINDTYPE storage TYPE ram IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME sha256_process_and_write_chunk_Pipeline_write_digest MODELNAME sha256_process_and_write_chunk_Pipeline_write_digest RTLNAME calculate_sha256_sha256_process_and_write_chunk_Pipeline_write_digest
    SUBMODULES {
      {MODELNAME calculate_sha256_mux_84_32_1_1 RTLNAME calculate_sha256_mux_84_32_1_1 BINDTYPE op TYPE mux IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME sha256_process_and_write_chunk MODELNAME sha256_process_and_write_chunk RTLNAME calculate_sha256_sha256_process_and_write_chunk}
  {SRCNAME calculate_sha256 MODELNAME calculate_sha256 RTLNAME calculate_sha256 IS_TOP 1
    SUBMODULES {
      {MODELNAME calculate_sha256_gmem0_m_axi RTLNAME calculate_sha256_gmem0_m_axi BINDTYPE interface TYPE adapter IMPL m_axi}
      {MODELNAME calculate_sha256_control_s_axi RTLNAME calculate_sha256_control_s_axi BINDTYPE interface TYPE interface_s_axilite}
    }
  }
}
