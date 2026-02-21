set SynModuleInfo {
  {SRCNAME process_tcp_checksum_Pipeline_word_loop MODELNAME process_tcp_checksum_Pipeline_word_loop RTLNAME calculate_tcp_checksum_process_tcp_checksum_Pipeline_word_loop
    SUBMODULES {
      {MODELNAME calculate_tcp_checksum_flow_control_loop_pipe_sequential_init RTLNAME calculate_tcp_checksum_flow_control_loop_pipe_sequential_init BINDTYPE interface TYPE internal_upc_flow_control INSTNAME calculate_tcp_checksum_flow_control_loop_pipe_sequential_init_U}
    }
  }
  {SRCNAME process_tcp_checksum_Pipeline_vec_word_loop MODELNAME process_tcp_checksum_Pipeline_vec_word_loop RTLNAME calculate_tcp_checksum_process_tcp_checksum_Pipeline_vec_word_loop}
  {SRCNAME process_tcp_checksum_Pipeline_tail_word_loop MODELNAME process_tcp_checksum_Pipeline_tail_word_loop RTLNAME calculate_tcp_checksum_process_tcp_checksum_Pipeline_tail_word_loop}
  {SRCNAME process_tcp_checksum MODELNAME process_tcp_checksum RTLNAME calculate_tcp_checksum_process_tcp_checksum}
  {SRCNAME calculate_tcp_checksum MODELNAME calculate_tcp_checksum RTLNAME calculate_tcp_checksum IS_TOP 1
    SUBMODULES {
      {MODELNAME calculate_tcp_checksum_gmem0_m_axi RTLNAME calculate_tcp_checksum_gmem0_m_axi BINDTYPE interface TYPE adapter IMPL m_axi}
      {MODELNAME calculate_tcp_checksum_control_s_axi RTLNAME calculate_tcp_checksum_control_s_axi BINDTYPE interface TYPE interface_s_axilite}
    }
  }
}
