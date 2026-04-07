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
    id 60 \
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
    id 61 \
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
    id 62 \
    name inByte017 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte017 \
    op interface \
    ports { inByte017_din { O 8 vector } inByte017_num_data_valid { I 7 vector } inByte017_fifo_cap { I 7 vector } inByte017_full_n { I 1 bit } inByte017_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 63 \
    name inByte118 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte118 \
    op interface \
    ports { inByte118_din { O 8 vector } inByte118_num_data_valid { I 7 vector } inByte118_fifo_cap { I 7 vector } inByte118_full_n { I 1 bit } inByte118_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 64 \
    name inByte219 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte219 \
    op interface \
    ports { inByte219_din { O 8 vector } inByte219_num_data_valid { I 7 vector } inByte219_fifo_cap { I 7 vector } inByte219_full_n { I 1 bit } inByte219_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 65 \
    name inByte320 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte320 \
    op interface \
    ports { inByte320_din { O 8 vector } inByte320_num_data_valid { I 7 vector } inByte320_fifo_cap { I 7 vector } inByte320_full_n { I 1 bit } inByte320_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 66 \
    name inByte421 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte421 \
    op interface \
    ports { inByte421_din { O 8 vector } inByte421_num_data_valid { I 7 vector } inByte421_fifo_cap { I 7 vector } inByte421_full_n { I 1 bit } inByte421_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 67 \
    name inByte522 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte522 \
    op interface \
    ports { inByte522_din { O 8 vector } inByte522_num_data_valid { I 7 vector } inByte522_fifo_cap { I 7 vector } inByte522_full_n { I 1 bit } inByte522_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 68 \
    name inByte623 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte623 \
    op interface \
    ports { inByte623_din { O 8 vector } inByte623_num_data_valid { I 7 vector } inByte623_fifo_cap { I 7 vector } inByte623_full_n { I 1 bit } inByte623_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 69 \
    name inByte724 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte724 \
    op interface \
    ports { inByte724_din { O 8 vector } inByte724_num_data_valid { I 7 vector } inByte724_fifo_cap { I 7 vector } inByte724_full_n { I 1 bit } inByte724_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 70 \
    name inByte825 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte825 \
    op interface \
    ports { inByte825_din { O 8 vector } inByte825_num_data_valid { I 7 vector } inByte825_fifo_cap { I 7 vector } inByte825_full_n { I 1 bit } inByte825_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 71 \
    name inByte926 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte926 \
    op interface \
    ports { inByte926_din { O 8 vector } inByte926_num_data_valid { I 7 vector } inByte926_fifo_cap { I 7 vector } inByte926_full_n { I 1 bit } inByte926_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 72 \
    name inByte1027 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte1027 \
    op interface \
    ports { inByte1027_din { O 8 vector } inByte1027_num_data_valid { I 7 vector } inByte1027_fifo_cap { I 7 vector } inByte1027_full_n { I 1 bit } inByte1027_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 73 \
    name inByte1128 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte1128 \
    op interface \
    ports { inByte1128_din { O 8 vector } inByte1128_num_data_valid { I 7 vector } inByte1128_fifo_cap { I 7 vector } inByte1128_full_n { I 1 bit } inByte1128_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 74 \
    name inByte1229 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte1229 \
    op interface \
    ports { inByte1229_din { O 8 vector } inByte1229_num_data_valid { I 7 vector } inByte1229_fifo_cap { I 7 vector } inByte1229_full_n { I 1 bit } inByte1229_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 75 \
    name inByte1330 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte1330 \
    op interface \
    ports { inByte1330_din { O 8 vector } inByte1330_num_data_valid { I 7 vector } inByte1330_fifo_cap { I 7 vector } inByte1330_full_n { I 1 bit } inByte1330_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 76 \
    name inByte1431 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte1431 \
    op interface \
    ports { inByte1431_din { O 8 vector } inByte1431_num_data_valid { I 7 vector } inByte1431_fifo_cap { I 7 vector } inByte1431_full_n { I 1 bit } inByte1431_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 77 \
    name inByte1532 \
    type fifo \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte1532 \
    op interface \
    ports { inByte1532_din { O 8 vector } inByte1532_num_data_valid { I 7 vector } inByte1532_fifo_cap { I 7 vector } inByte1532_full_n { I 1 bit } inByte1532_write { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 78 \
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
    id 79 \
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


