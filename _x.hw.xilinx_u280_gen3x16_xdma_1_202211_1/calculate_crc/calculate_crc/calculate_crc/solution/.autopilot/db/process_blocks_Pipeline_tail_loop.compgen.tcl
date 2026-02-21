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
    id 124 \
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
    id 122 \
    name crc_reload \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_crc_reload \
    op interface \
    ports { crc_reload { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 123 \
    name sub_ln329 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sub_ln329 \
    op interface \
    ports { sub_ln329 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 125 \
    name mask_1 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_mask_1 \
    op interface \
    ports { mask_1 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 126 \
    name inByte0 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte0 \
    op interface \
    ports { inByte0_dout { I 8 vector } inByte0_num_data_valid { I 7 vector } inByte0_fifo_cap { I 7 vector } inByte0_empty_n { I 1 bit } inByte0_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 127 \
    name inByte1 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte1 \
    op interface \
    ports { inByte1_dout { I 8 vector } inByte1_num_data_valid { I 7 vector } inByte1_fifo_cap { I 7 vector } inByte1_empty_n { I 1 bit } inByte1_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 128 \
    name inByte2 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte2 \
    op interface \
    ports { inByte2_dout { I 8 vector } inByte2_num_data_valid { I 7 vector } inByte2_fifo_cap { I 7 vector } inByte2_empty_n { I 1 bit } inByte2_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 129 \
    name inByte3 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte3 \
    op interface \
    ports { inByte3_dout { I 8 vector } inByte3_num_data_valid { I 7 vector } inByte3_fifo_cap { I 7 vector } inByte3_empty_n { I 1 bit } inByte3_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 130 \
    name inByte4 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte4 \
    op interface \
    ports { inByte4_dout { I 8 vector } inByte4_num_data_valid { I 7 vector } inByte4_fifo_cap { I 7 vector } inByte4_empty_n { I 1 bit } inByte4_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 131 \
    name inByte5 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte5 \
    op interface \
    ports { inByte5_dout { I 8 vector } inByte5_num_data_valid { I 7 vector } inByte5_fifo_cap { I 7 vector } inByte5_empty_n { I 1 bit } inByte5_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 132 \
    name inByte6 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte6 \
    op interface \
    ports { inByte6_dout { I 8 vector } inByte6_num_data_valid { I 7 vector } inByte6_fifo_cap { I 7 vector } inByte6_empty_n { I 1 bit } inByte6_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 133 \
    name inByte7 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte7 \
    op interface \
    ports { inByte7_dout { I 8 vector } inByte7_num_data_valid { I 7 vector } inByte7_fifo_cap { I 7 vector } inByte7_empty_n { I 1 bit } inByte7_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 134 \
    name inByte8 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte8 \
    op interface \
    ports { inByte8_dout { I 8 vector } inByte8_num_data_valid { I 7 vector } inByte8_fifo_cap { I 7 vector } inByte8_empty_n { I 1 bit } inByte8_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 135 \
    name inByte9 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte9 \
    op interface \
    ports { inByte9_dout { I 8 vector } inByte9_num_data_valid { I 7 vector } inByte9_fifo_cap { I 7 vector } inByte9_empty_n { I 1 bit } inByte9_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 136 \
    name inByte10 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte10 \
    op interface \
    ports { inByte10_dout { I 8 vector } inByte10_num_data_valid { I 7 vector } inByte10_fifo_cap { I 7 vector } inByte10_empty_n { I 1 bit } inByte10_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 137 \
    name inByte11 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte11 \
    op interface \
    ports { inByte11_dout { I 8 vector } inByte11_num_data_valid { I 7 vector } inByte11_fifo_cap { I 7 vector } inByte11_empty_n { I 1 bit } inByte11_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 138 \
    name inByte12 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte12 \
    op interface \
    ports { inByte12_dout { I 8 vector } inByte12_num_data_valid { I 7 vector } inByte12_fifo_cap { I 7 vector } inByte12_empty_n { I 1 bit } inByte12_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 139 \
    name inByte13 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte13 \
    op interface \
    ports { inByte13_dout { I 8 vector } inByte13_num_data_valid { I 7 vector } inByte13_fifo_cap { I 7 vector } inByte13_empty_n { I 1 bit } inByte13_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 140 \
    name inByte14 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte14 \
    op interface \
    ports { inByte14_dout { I 8 vector } inByte14_num_data_valid { I 7 vector } inByte14_fifo_cap { I 7 vector } inByte14_empty_n { I 1 bit } inByte14_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 141 \
    name inByte15 \
    type fifo \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_inByte15 \
    op interface \
    ports { inByte15_dout { I 8 vector } inByte15_num_data_valid { I 7 vector } inByte15_fifo_cap { I 7 vector } inByte15_empty_n { I 1 bit } inByte15_read { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 142 \
    name crc_2_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_crc_2_out \
    op interface \
    ports { crc_2_out { O 32 vector } crc_2_out_ap_vld { O 1 bit } } \
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


