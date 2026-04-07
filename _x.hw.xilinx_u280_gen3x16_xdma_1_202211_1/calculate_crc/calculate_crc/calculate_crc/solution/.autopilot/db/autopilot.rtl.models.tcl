set SynModuleInfo {
  {SRCNAME load_crc_tables_Pipeline_load_lut_rows_load_lut_cols MODELNAME load_crc_tables_Pipeline_load_lut_rows_load_lut_cols RTLNAME calculate_crc_load_crc_tables_Pipeline_load_lut_rows_load_lut_cols
    SUBMODULES {
      {MODELNAME calculate_crc_flow_control_loop_pipe_sequential_init RTLNAME calculate_crc_flow_control_loop_pipe_sequential_init BINDTYPE interface TYPE internal_upc_flow_control INSTNAME calculate_crc_flow_control_loop_pipe_sequential_init_U}
    }
  }
  {SRCNAME load_crc_tables MODELNAME load_crc_tables RTLNAME calculate_crc_load_crc_tables}
  {SRCNAME entry_proc MODELNAME entry_proc RTLNAME calculate_crc_entry_proc}
  {SRCNAME read_input_Pipeline_mem_rd MODELNAME read_input_Pipeline_mem_rd RTLNAME calculate_crc_read_input_Pipeline_mem_rd}
  {SRCNAME read_input MODELNAME read_input RTLNAME calculate_crc_read_input
    SUBMODULES {
      {MODELNAME calculate_crc_mul_32s_32s_32_2_1 RTLNAME calculate_crc_mul_32s_32s_32_2_1 BINDTYPE op TYPE mul IMPL auto LATENCY 1 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME crc_process_full_blocks_Pipeline_full_block_loop MODELNAME crc_process_full_blocks_Pipeline_full_block_loop RTLNAME calculate_crc_crc_process_full_blocks_Pipeline_full_block_loop}
  {SRCNAME crc_process_full_blocks MODELNAME crc_process_full_blocks RTLNAME calculate_crc_crc_process_full_blocks}
  {SRCNAME crc_process_tail_bytes MODELNAME crc_process_tail_bytes RTLNAME calculate_crc_crc_process_tail_bytes}
  {SRCNAME process_crc_chunks MODELNAME process_crc_chunks RTLNAME calculate_crc_process_crc_chunks}
  {SRCNAME write_output_Pipeline_VITIS_LOOP_350_1 MODELNAME write_output_Pipeline_VITIS_LOOP_350_1 RTLNAME calculate_crc_write_output_Pipeline_VITIS_LOOP_350_1}
  {SRCNAME write_output MODELNAME write_output RTLNAME calculate_crc_write_output}
  {SRCNAME crc_dataflow_region MODELNAME crc_dataflow_region RTLNAME calculate_crc_crc_dataflow_region
    SUBMODULES {
      {MODELNAME calculate_crc_fifo_w64_d4_S RTLNAME calculate_crc_fifo_w64_d4_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME {$InstName}}
      {MODELNAME calculate_crc_fifo_w8_d64_S RTLNAME calculate_crc_fifo_w8_d64_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME {$InstName}}
      {MODELNAME calculate_crc_fifo_w32_d64_A RTLNAME calculate_crc_fifo_w32_d64_A BINDTYPE storage TYPE fifo IMPL memory ALLOW_PRAGMA 1 INSTNAME {$InstName}}
      {MODELNAME calculate_crc_fifo_w32_d2_S RTLNAME calculate_crc_fifo_w32_d2_S BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME {$InstName}}
      {MODELNAME calculate_crc_start_for_write_output_U0 RTLNAME calculate_crc_start_for_write_output_U0 BINDTYPE storage TYPE fifo IMPL srl ALLOW_PRAGMA 1 INSTNAME {$InstName}}
    }
  }
  {SRCNAME calculate_crc MODELNAME calculate_crc RTLNAME calculate_crc IS_TOP 1
    SUBMODULES {
      {MODELNAME calculate_crc_crcTables_RAM_1P_BRAM_1R1W RTLNAME calculate_crc_crcTables_RAM_1P_BRAM_1R1W BINDTYPE storage TYPE ram_1p IMPL bram LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME calculate_crc_gmem0_m_axi RTLNAME calculate_crc_gmem0_m_axi BINDTYPE interface TYPE adapter IMPL m_axi}
      {MODELNAME calculate_crc_gmem1_m_axi RTLNAME calculate_crc_gmem1_m_axi BINDTYPE interface TYPE adapter IMPL m_axi}
      {MODELNAME calculate_crc_control_s_axi RTLNAME calculate_crc_control_s_axi BINDTYPE interface TYPE interface_s_axilite}
    }
  }
}
