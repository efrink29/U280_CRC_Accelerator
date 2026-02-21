# This script segment is generated automatically by AutoPilot

# clear list
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_begin
    cg_default_interface_gen_bundle_begin
    AESL_LIB_XILADAPTER::native_axis_begin
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 43 \
    name gmem0 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_gmem0 \
    op interface \
    ports { m_axi_gmem0_AWVALID { O 1 bit } m_axi_gmem0_AWREADY { I 1 bit } m_axi_gmem0_AWADDR { O 64 vector } m_axi_gmem0_AWID { O 1 vector } m_axi_gmem0_AWLEN { O 32 vector } m_axi_gmem0_AWSIZE { O 3 vector } m_axi_gmem0_AWBURST { O 2 vector } m_axi_gmem0_AWLOCK { O 2 vector } m_axi_gmem0_AWCACHE { O 4 vector } m_axi_gmem0_AWPROT { O 3 vector } m_axi_gmem0_AWQOS { O 4 vector } m_axi_gmem0_AWREGION { O 4 vector } m_axi_gmem0_AWUSER { O 1 vector } m_axi_gmem0_WVALID { O 1 bit } m_axi_gmem0_WREADY { I 1 bit } m_axi_gmem0_WDATA { O 128 vector } m_axi_gmem0_WSTRB { O 16 vector } m_axi_gmem0_WLAST { O 1 bit } m_axi_gmem0_WID { O 1 vector } m_axi_gmem0_WUSER { O 1 vector } m_axi_gmem0_ARVALID { O 1 bit } m_axi_gmem0_ARREADY { I 1 bit } m_axi_gmem0_ARADDR { O 64 vector } m_axi_gmem0_ARID { O 1 vector } m_axi_gmem0_ARLEN { O 32 vector } m_axi_gmem0_ARSIZE { O 3 vector } m_axi_gmem0_ARBURST { O 2 vector } m_axi_gmem0_ARLOCK { O 2 vector } m_axi_gmem0_ARCACHE { O 4 vector } m_axi_gmem0_ARPROT { O 3 vector } m_axi_gmem0_ARQOS { O 4 vector } m_axi_gmem0_ARREGION { O 4 vector } m_axi_gmem0_ARUSER { O 1 vector } m_axi_gmem0_RVALID { I 1 bit } m_axi_gmem0_RREADY { O 1 bit } m_axi_gmem0_RDATA { I 128 vector } m_axi_gmem0_RLAST { I 1 bit } m_axi_gmem0_RID { I 1 vector } m_axi_gmem0_RFIFONUM { I 9 vector } m_axi_gmem0_RUSER { I 1 vector } m_axi_gmem0_RRESP { I 2 vector } m_axi_gmem0_BVALID { I 1 bit } m_axi_gmem0_BREADY { O 1 bit } m_axi_gmem0_BRESP { I 2 vector } m_axi_gmem0_BID { I 1 vector } m_axi_gmem0_BUSER { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 44 \
    name sext_ln287 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln287 \
    op interface \
    ports { sext_ln287 { I 60 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 45 \
    name loop_count \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_loop_count \
    op interface \
    ports { loop_count { I 29 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 46 \
    name inByte0 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte0 \
    op interface \
    ports { inByte0_din { O 8 vector } inByte0_num_data_valid { I 7 vector } inByte0_fifo_cap { I 7 vector } inByte0_full_n { I 1 bit } inByte0_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 47 \
    name inByte1 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte1 \
    op interface \
    ports { inByte1_din { O 8 vector } inByte1_num_data_valid { I 7 vector } inByte1_fifo_cap { I 7 vector } inByte1_full_n { I 1 bit } inByte1_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 48 \
    name inByte2 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte2 \
    op interface \
    ports { inByte2_din { O 8 vector } inByte2_num_data_valid { I 7 vector } inByte2_fifo_cap { I 7 vector } inByte2_full_n { I 1 bit } inByte2_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 49 \
    name inByte3 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte3 \
    op interface \
    ports { inByte3_din { O 8 vector } inByte3_num_data_valid { I 7 vector } inByte3_fifo_cap { I 7 vector } inByte3_full_n { I 1 bit } inByte3_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 50 \
    name inByte4 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte4 \
    op interface \
    ports { inByte4_din { O 8 vector } inByte4_num_data_valid { I 7 vector } inByte4_fifo_cap { I 7 vector } inByte4_full_n { I 1 bit } inByte4_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 51 \
    name inByte5 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte5 \
    op interface \
    ports { inByte5_din { O 8 vector } inByte5_num_data_valid { I 7 vector } inByte5_fifo_cap { I 7 vector } inByte5_full_n { I 1 bit } inByte5_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 52 \
    name inByte6 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte6 \
    op interface \
    ports { inByte6_din { O 8 vector } inByte6_num_data_valid { I 7 vector } inByte6_fifo_cap { I 7 vector } inByte6_full_n { I 1 bit } inByte6_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 53 \
    name inByte7 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte7 \
    op interface \
    ports { inByte7_din { O 8 vector } inByte7_num_data_valid { I 7 vector } inByte7_fifo_cap { I 7 vector } inByte7_full_n { I 1 bit } inByte7_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 54 \
    name inByte8 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte8 \
    op interface \
    ports { inByte8_din { O 8 vector } inByte8_num_data_valid { I 7 vector } inByte8_fifo_cap { I 7 vector } inByte8_full_n { I 1 bit } inByte8_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 55 \
    name inByte9 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte9 \
    op interface \
    ports { inByte9_din { O 8 vector } inByte9_num_data_valid { I 7 vector } inByte9_fifo_cap { I 7 vector } inByte9_full_n { I 1 bit } inByte9_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 56 \
    name inByte10 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte10 \
    op interface \
    ports { inByte10_din { O 8 vector } inByte10_num_data_valid { I 7 vector } inByte10_fifo_cap { I 7 vector } inByte10_full_n { I 1 bit } inByte10_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 57 \
    name inByte11 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte11 \
    op interface \
    ports { inByte11_din { O 8 vector } inByte11_num_data_valid { I 7 vector } inByte11_fifo_cap { I 7 vector } inByte11_full_n { I 1 bit } inByte11_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 58 \
    name inByte12 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte12 \
    op interface \
    ports { inByte12_din { O 8 vector } inByte12_num_data_valid { I 7 vector } inByte12_fifo_cap { I 7 vector } inByte12_full_n { I 1 bit } inByte12_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 59 \
    name inByte13 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte13 \
    op interface \
    ports { inByte13_din { O 8 vector } inByte13_num_data_valid { I 7 vector } inByte13_fifo_cap { I 7 vector } inByte13_full_n { I 1 bit } inByte13_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 60 \
    name inByte14 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte14 \
    op interface \
    ports { inByte14_din { O 8 vector } inByte14_num_data_valid { I 7 vector } inByte14_fifo_cap { I 7 vector } inByte14_full_n { I 1 bit } inByte14_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 61 \
    name inByte15 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte15 \
    op interface \
    ports { inByte15_din { O 8 vector } inByte15_num_data_valid { I 7 vector } inByte15_fifo_cap { I 7 vector } inByte15_full_n { I 1 bit } inByte15_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id -1 \
    name ap_ctrl \
    type ap_ctrl \
    reset_level 1 \
    sync_rst true \
    corename ap_ctrl \
    op interface \
    ports { ap_start { I 1 bit } ap_ready { O 1 bit } ap_done { O 1 bit } ap_idle { O 1 bit } } \
} "
}


# Adapter definition:
set PortName ap_clk
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_clock] == "cg_default_interface_gen_clock"} {
eval "cg_default_interface_gen_clock { \
    id -2 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_clk \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-113\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}


# Adapter definition:
set PortName ap_rst
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_reset] == "cg_default_interface_gen_reset"} {
eval "cg_default_interface_gen_reset { \
    id -3 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_rst \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-114\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}



# merge
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_end
    cg_default_interface_gen_bundle_end
    AESL_LIB_XILADAPTER::native_axis_end
}


# flow_control definition:
set InstName calculate_crc_flow_control_loop_pipe_sequential_init_U
set CompName calculate_crc_flow_control_loop_pipe_sequential_init
set name flow_control_loop_pipe_sequential_init
if {${::AESL::PGuard_autocg_gen} && ${::AESL::PGuard_autocg_ipmgen}} {
if {[info proc ::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control] == "::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control"} {
eval "::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control { \
    name ${name} \
    prefix calculate_crc_ \
}"
} else {
puts "@W \[IMPL-107\] Cannot find ::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control, check your platform lib"
}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $CompName BINDTYPE interface TYPE internal_upc_flow_control INSTNAME $InstName
}


