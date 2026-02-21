set SynModuleInfo {
  {SRCNAME entry_proc MODELNAME entry_proc RTLNAME calculate_crc_entry_proc}
  {SRCNAME process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1 MODELNAME process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1 RTLNAME calculate_crc_process_crc_Loop_init_lut_proc_Pipeline_init_lut_VITIS_LOOP_678_1
    SUBMODULES {
      {MODELNAME calculate_crc_flow_control_loop_pipe_sequential_init RTLNAME calculate_crc_flow_control_loop_pipe_sequential_init BINDTYPE interface TYPE internal_upc_flow_control INSTNAME calculate_crc_flow_control_loop_pipe_sequential_init_U}
    }
  }
  {SRCNAME process_crc_Loop_init_lut_proc MODELNAME process_crc_Loop_init_lut_proc RTLNAME calculate_crc_process_crc_Loop_init_lut_proc}
  {SRCNAME read_input_Pipeline_mem_rd MODELNAME read_input_Pipeline_mem_rd RTLNAME calculate_crc_read_input_Pipeline_mem_rd}
  {SRCNAME read_input MODELNAME read_input RTLNAME calculate_crc_read_input
    SUBMODULES {
      {MODELNAME calculate_crc_mul_32s_32s_32_2_1 RTLNAME calculate_crc_mul_32s_32s_32_2_1 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME process_blocks_Pipeline_block_loop MODELNAME process_blocks_Pipeline_block_loop RTLNAME calculate_crc_process_blocks_Pipeline_block_loop}
  {SRCNAME process_blocks_Pipeline_tail_loop MODELNAME process_blocks_Pipeline_tail_loop RTLNAME calculate_crc_process_blocks_Pipeline_tail_loop}
  {SRCNAME process_blocks MODELNAME process_blocks RTLNAME calculate_crc_process_blocks}
  {SRCNAME write_output_Pipeline_VITIS_LOOP_442_1 MODELNAME write_output_Pipeline_VITIS_LOOP_442_1 RTLNAME calculate_crc_write_output_Pipeline_VITIS_LOOP_442_1}
  {SRCNAME write_output MODELNAME write_output RTLNAME calculate_crc_write_output}
  {SRCNAME process_crc MODELNAME process_crc RTLNAME calculate_crc_process_crc
    SUBMODULES {
      {MODELNAME calculate_crc_process_crc_crcTables_RAM_1P_BRAM_1R1W_memcore RTLNAME calculate_crc_process_crc_crcTables_RAM_1P_BRAM_1R1W_memcore BINDTYPE storage TYPE ram_1p IMPL bram LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME calculate_crc_process_crc_crcTables_RAM_1P_BRAM_1R1W RTLNAME calculate_crc_process_crc_crcTables_RAM_1P_BRAM_1R1W BINDTYPE storage TYPE ram_1p IMPL bram LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME calculate_crc_fifo_w64_d4_S RTLNAME calculate_crc_fifo_w64_d4_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME crc_out_c_U}
      {MODELNAME calculate_crc_fifo_w32_d3_S RTLNAME calculate_crc_fifo_w32_d3_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME crc_size_c_U}
      {MODELNAME calculate_crc_fifo_w32_d3_S RTLNAME calculate_crc_fifo_w32_d3_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME init_value_c_U}
      {MODELNAME calculate_crc_fifo_w8_d64_S RTLNAME calculate_crc_fifo_w8_d64_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME inByte0_U}
      {MODELNAME calculate_crc_fifo_w8_d64_S RTLNAME calculate_crc_fifo_w8_d64_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME inByte1_U}
      {MODELNAME calculate_crc_fifo_w8_d64_S RTLNAME calculate_crc_fifo_w8_d64_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME inByte2_U}
      {MODELNAME calculate_crc_fifo_w8_d64_S RTLNAME calculate_crc_fifo_w8_d64_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME inByte3_U}
      {MODELNAME calculate_crc_fifo_w8_d64_S RTLNAME calculate_crc_fifo_w8_d64_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME inByte4_U}
      {MODELNAME calculate_crc_fifo_w8_d64_S RTLNAME calculate_crc_fifo_w8_d64_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME inByte5_U}
      {MODELNAME calculate_crc_fifo_w8_d64_S RTLNAME calculate_crc_fifo_w8_d64_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME inByte6_U}
      {MODELNAME calculate_crc_fifo_w8_d64_S RTLNAME calculate_crc_fifo_w8_d64_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME inByte7_U}
      {MODELNAME calculate_crc_fifo_w8_d64_S RTLNAME calculate_crc_fifo_w8_d64_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME inByte8_U}
      {MODELNAME calculate_crc_fifo_w8_d64_S RTLNAME calculate_crc_fifo_w8_d64_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME inByte9_U}
      {MODELNAME calculate_crc_fifo_w8_d64_S RTLNAME calculate_crc_fifo_w8_d64_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME inByte10_U}
      {MODELNAME calculate_crc_fifo_w8_d64_S RTLNAME calculate_crc_fifo_w8_d64_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME inByte11_U}
      {MODELNAME calculate_crc_fifo_w8_d64_S RTLNAME calculate_crc_fifo_w8_d64_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME inByte12_U}
      {MODELNAME calculate_crc_fifo_w8_d64_S RTLNAME calculate_crc_fifo_w8_d64_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME inByte13_U}
      {MODELNAME calculate_crc_fifo_w8_d64_S RTLNAME calculate_crc_fifo_w8_d64_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME inByte14_U}
      {MODELNAME calculate_crc_fifo_w8_d64_S RTLNAME calculate_crc_fifo_w8_d64_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME inByte15_U}
      {MODELNAME calculate_crc_fifo_w32_d2_S RTLNAME calculate_crc_fifo_w32_d2_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME numChunks_c12_U}
      {MODELNAME calculate_crc_fifo_w32_d2_S RTLNAME calculate_crc_fifo_w32_d2_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME chunkSize_c_U}
      {MODELNAME calculate_crc_fifo_w32_d64_A RTLNAME calculate_crc_fifo_w32_d64_A BINDTYPE storage TYPE fifo IMPL memory ALLOW_PRAGMA 1 INSTNAME outStream_U}
      {MODELNAME calculate_crc_fifo_w32_d2_S RTLNAME calculate_crc_fifo_w32_d2_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME numChunks_c_U}
      {MODELNAME calculate_crc_start_for_write_output_U0 RTLNAME calculate_crc_start_for_write_output_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME start_for_write_output_U0_U}
    }
  }
  {SRCNAME calculate_crc MODELNAME calculate_crc RTLNAME calculate_crc IS_TOP 1
    SUBMODULES {
      {MODELNAME calculate_crc_gmem0_m_axi RTLNAME calculate_crc_gmem0_m_axi BINDTYPE interface TYPE adapter IMPL m_axi}
      {MODELNAME calculate_crc_gmem1_m_axi RTLNAME calculate_crc_gmem1_m_axi BINDTYPE interface TYPE adapter IMPL m_axi}
      {MODELNAME calculate_crc_control_s_axi RTLNAME calculate_crc_control_s_axi BINDTYPE interface TYPE interface_s_axilite}
    }
  }
}
