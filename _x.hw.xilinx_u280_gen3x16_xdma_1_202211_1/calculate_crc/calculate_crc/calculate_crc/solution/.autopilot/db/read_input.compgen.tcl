# This script segment is generated automatically by AutoPilot

set name calculate_crc_mul_32s_32s_32_2_1
if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $name BINDTYPE {op} TYPE {mul} IMPL {auto} LATENCY 1 ALLOW_PRAGMA 1
}


# clear list
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_begin
    cg_default_interface_gen_bundle_begin
    AESL_LIB_XILADAPTER::native_axis_begin
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 64 \
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
    id 65 \
    name in_r \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_in_r \
    op interface \
    ports { in_r { I 64 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 66 \
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
    id 67 \
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
    id 68 \
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
    id 69 \
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
    id 70 \
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
    id 71 \
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
    id 72 \
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
    id 73 \
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
    id 74 \
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
    id 75 \
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
    id 76 \
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
    id 77 \
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
    id 78 \
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
    id 79 \
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
    id 80 \
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
    id 81 \
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
    id 82 \
    name numChunks \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_numChunks \
    op interface \
    ports { numChunks { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 83 \
    name chunkSize \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_chunkSize \
    op interface \
    ports { chunkSize { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 84 \
    name numChunks_c12 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_numChunks_c12 \
    op interface \
    ports { numChunks_c12_din { O 32 vector } numChunks_c12_num_data_valid { I 2 vector } numChunks_c12_fifo_cap { I 2 vector } numChunks_c12_full_n { I 1 bit } numChunks_c12_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 85 \
    name chunkSize_c \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_chunkSize_c \
    op interface \
    ports { chunkSize_c_din { O 32 vector } chunkSize_c_num_data_valid { I 2 vector } chunkSize_c_fifo_cap { I 2 vector } chunkSize_c_full_n { I 1 bit } chunkSize_c_write { O 1 bit } } \
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
    ports { ap_start { I 1 bit } ap_ready { O 1 bit } ap_done { O 1 bit } ap_idle { O 1 bit } ap_continue { I 1 bit } } \
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


