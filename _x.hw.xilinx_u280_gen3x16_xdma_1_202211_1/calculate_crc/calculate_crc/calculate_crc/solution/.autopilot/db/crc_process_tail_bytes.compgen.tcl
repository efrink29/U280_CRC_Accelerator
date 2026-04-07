# This script segment is generated automatically by AutoPilot

# clear list
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_begin
    cg_default_interface_gen_bundle_begin
    AESL_LIB_XILADAPTER::native_axis_begin
}

# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 151 \
    name crcTables_0 \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename crcTables_0 \
    op interface \
    ports { crcTables_0_address0 { O 8 vector } crcTables_0_ce0 { O 1 bit } crcTables_0_q0 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'crcTables_0'"
}
}


# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 152 \
    name inByte017 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte017 \
    op interface \
    ports { inByte017_dout { I 8 vector } inByte017_num_data_valid { I 7 vector } inByte017_fifo_cap { I 7 vector } inByte017_empty_n { I 1 bit } inByte017_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 153 \
    name inByte118 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte118 \
    op interface \
    ports { inByte118_dout { I 8 vector } inByte118_num_data_valid { I 7 vector } inByte118_fifo_cap { I 7 vector } inByte118_empty_n { I 1 bit } inByte118_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 154 \
    name inByte219 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte219 \
    op interface \
    ports { inByte219_dout { I 8 vector } inByte219_num_data_valid { I 7 vector } inByte219_fifo_cap { I 7 vector } inByte219_empty_n { I 1 bit } inByte219_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 155 \
    name inByte320 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte320 \
    op interface \
    ports { inByte320_dout { I 8 vector } inByte320_num_data_valid { I 7 vector } inByte320_fifo_cap { I 7 vector } inByte320_empty_n { I 1 bit } inByte320_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 156 \
    name inByte421 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte421 \
    op interface \
    ports { inByte421_dout { I 8 vector } inByte421_num_data_valid { I 7 vector } inByte421_fifo_cap { I 7 vector } inByte421_empty_n { I 1 bit } inByte421_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 157 \
    name inByte522 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte522 \
    op interface \
    ports { inByte522_dout { I 8 vector } inByte522_num_data_valid { I 7 vector } inByte522_fifo_cap { I 7 vector } inByte522_empty_n { I 1 bit } inByte522_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 158 \
    name inByte623 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte623 \
    op interface \
    ports { inByte623_dout { I 8 vector } inByte623_num_data_valid { I 7 vector } inByte623_fifo_cap { I 7 vector } inByte623_empty_n { I 1 bit } inByte623_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 159 \
    name inByte724 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte724 \
    op interface \
    ports { inByte724_dout { I 8 vector } inByte724_num_data_valid { I 7 vector } inByte724_fifo_cap { I 7 vector } inByte724_empty_n { I 1 bit } inByte724_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 160 \
    name inByte825 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte825 \
    op interface \
    ports { inByte825_dout { I 8 vector } inByte825_num_data_valid { I 7 vector } inByte825_fifo_cap { I 7 vector } inByte825_empty_n { I 1 bit } inByte825_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 161 \
    name inByte926 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte926 \
    op interface \
    ports { inByte926_dout { I 8 vector } inByte926_num_data_valid { I 7 vector } inByte926_fifo_cap { I 7 vector } inByte926_empty_n { I 1 bit } inByte926_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 162 \
    name inByte1027 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte1027 \
    op interface \
    ports { inByte1027_dout { I 8 vector } inByte1027_num_data_valid { I 7 vector } inByte1027_fifo_cap { I 7 vector } inByte1027_empty_n { I 1 bit } inByte1027_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 163 \
    name inByte1128 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte1128 \
    op interface \
    ports { inByte1128_dout { I 8 vector } inByte1128_num_data_valid { I 7 vector } inByte1128_fifo_cap { I 7 vector } inByte1128_empty_n { I 1 bit } inByte1128_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 164 \
    name inByte1229 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte1229 \
    op interface \
    ports { inByte1229_dout { I 8 vector } inByte1229_num_data_valid { I 7 vector } inByte1229_fifo_cap { I 7 vector } inByte1229_empty_n { I 1 bit } inByte1229_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 165 \
    name inByte1330 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte1330 \
    op interface \
    ports { inByte1330_dout { I 8 vector } inByte1330_num_data_valid { I 7 vector } inByte1330_fifo_cap { I 7 vector } inByte1330_empty_n { I 1 bit } inByte1330_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 166 \
    name inByte1431 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte1431 \
    op interface \
    ports { inByte1431_dout { I 8 vector } inByte1431_num_data_valid { I 7 vector } inByte1431_fifo_cap { I 7 vector } inByte1431_empty_n { I 1 bit } inByte1431_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 167 \
    name inByte1532 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte1532 \
    op interface \
    ports { inByte1532_dout { I 8 vector } inByte1532_num_data_valid { I 7 vector } inByte1532_fifo_cap { I 7 vector } inByte1532_empty_n { I 1 bit } inByte1532_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 168 \
    name tail_bytes \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_tail_bytes \
    op interface \
    ports { tail_bytes { I 31 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 169 \
    name crc \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_crc \
    op interface \
    ports { crc { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 170 \
    name mask \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_mask \
    op interface \
    ports { mask { I 32 vector } } \
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

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id -2 \
    name ap_return \
    type ap_return \
    reset_level 1 \
    sync_rst true \
    corename ap_return \
    op interface \
    ports { ap_return { O 32 vector } } \
} "
}


# Adapter definition:
set PortName ap_clk
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_clock] == "cg_default_interface_gen_clock"} {
eval "cg_default_interface_gen_clock { \
    id -3 \
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
    id -4 \
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


