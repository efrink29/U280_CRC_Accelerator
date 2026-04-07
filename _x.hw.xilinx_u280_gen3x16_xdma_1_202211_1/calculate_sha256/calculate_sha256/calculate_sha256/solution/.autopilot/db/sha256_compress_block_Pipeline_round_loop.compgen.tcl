# This script segment is generated automatically by AutoPilot

set id 1
set name calculate_sha256_mux_164_32_1_1
set corename simcore_mux
set op mux
set stage_num 1
set din0_width 32
set din0_signed 0
set din1_width 32
set din1_signed 0
set din2_width 32
set din2_signed 0
set din3_width 32
set din3_signed 0
set din4_width 32
set din4_signed 0
set din5_width 32
set din5_signed 0
set din6_width 32
set din6_signed 0
set din7_width 32
set din7_signed 0
set din8_width 32
set din8_signed 0
set din9_width 32
set din9_signed 0
set din10_width 32
set din10_signed 0
set din11_width 32
set din11_signed 0
set din12_width 32
set din12_signed 0
set din13_width 32
set din13_signed 0
set din14_width 32
set din14_signed 0
set din15_width 32
set din15_signed 0
set din16_width 4
set din16_signed 0
set dout_width 32
if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $name BINDTYPE {op} TYPE {mux} IMPL {auto} LATENCY 0 ALLOW_PRAGMA 1
}


set op mux
set corename Multiplexer
if {${::AESL::PGuard_autocg_gen} && ${::AESL::PGuard_autocg_ipmgen}} {
if {[info proc ::AESL_LIB_VIRTEX::xil_gen_pipemux] == "::AESL_LIB_VIRTEX::xil_gen_pipemux"} {
eval "::AESL_LIB_VIRTEX::xil_gen_pipemux { \
    id ${id} \
    name ${name} \
    corename ${corename} \
    op ${op} \
    reset_level 1 \
    sync_rst true \
    stage_num ${stage_num} \
    din0_width ${din0_width} \
    din0_signed ${din0_signed} \
    din1_width ${din1_width} \
    din1_signed ${din1_signed} \
    din2_width ${din2_width} \
    din2_signed ${din2_signed} \
    din3_width ${din3_width} \
    din3_signed ${din3_signed} \
    din4_width ${din4_width} \
    din4_signed ${din4_signed} \
    din5_width ${din5_width} \
    din5_signed ${din5_signed} \
    din6_width ${din6_width} \
    din6_signed ${din6_signed} \
    din7_width ${din7_width} \
    din7_signed ${din7_signed} \
    din8_width ${din8_width} \
    din8_signed ${din8_signed} \
    din9_width ${din9_width} \
    din9_signed ${din9_signed} \
    din10_width ${din10_width} \
    din10_signed ${din10_signed} \
    din11_width ${din11_width} \
    din11_signed ${din11_signed} \
    din12_width ${din12_width} \
    din12_signed ${din12_signed} \
    din13_width ${din13_width} \
    din13_signed ${din13_signed} \
    din14_width ${din14_width} \
    din14_signed ${din14_signed} \
    din15_width ${din15_width} \
    din15_signed ${din15_signed} \
    din16_width ${din16_width} \
    din16_signed ${din16_signed} \
    dout_width ${dout_width} \
}"
} else {
puts "@W \[IMPL-101\] Cannot find ::AESL_LIB_VIRTEX::xil_gen_pipemux, check your platform lib"
}
}


set id 5
set name calculate_sha256_mux_167_32_1_1
set corename simcore_mux
set op mux
set stage_num 1
set din0_width 32
set din0_signed 0
set din1_width 32
set din1_signed 0
set din2_width 32
set din2_signed 0
set din3_width 32
set din3_signed 0
set din4_width 32
set din4_signed 0
set din5_width 32
set din5_signed 0
set din6_width 32
set din6_signed 0
set din7_width 32
set din7_signed 0
set din8_width 32
set din8_signed 0
set din9_width 32
set din9_signed 0
set din10_width 32
set din10_signed 0
set din11_width 32
set din11_signed 0
set din12_width 32
set din12_signed 0
set din13_width 32
set din13_signed 0
set din14_width 32
set din14_signed 0
set din15_width 32
set din15_signed 0
set din16_width 7
set din16_signed 0
set dout_width 32
if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $name BINDTYPE {op} TYPE {mux} IMPL {auto} LATENCY 0 ALLOW_PRAGMA 1
}


set op mux
set corename Multiplexer
if {${::AESL::PGuard_autocg_gen} && ${::AESL::PGuard_autocg_ipmgen}} {
if {[info proc ::AESL_LIB_VIRTEX::xil_gen_pipemux] == "::AESL_LIB_VIRTEX::xil_gen_pipemux"} {
eval "::AESL_LIB_VIRTEX::xil_gen_pipemux { \
    id ${id} \
    name ${name} \
    corename ${corename} \
    op ${op} \
    reset_level 1 \
    sync_rst true \
    stage_num ${stage_num} \
    din0_width ${din0_width} \
    din0_signed ${din0_signed} \
    din1_width ${din1_width} \
    din1_signed ${din1_signed} \
    din2_width ${din2_width} \
    din2_signed ${din2_signed} \
    din3_width ${din3_width} \
    din3_signed ${din3_signed} \
    din4_width ${din4_width} \
    din4_signed ${din4_signed} \
    din5_width ${din5_width} \
    din5_signed ${din5_signed} \
    din6_width ${din6_width} \
    din6_signed ${din6_signed} \
    din7_width ${din7_width} \
    din7_signed ${din7_signed} \
    din8_width ${din8_width} \
    din8_signed ${din8_signed} \
    din9_width ${din9_width} \
    din9_signed ${din9_signed} \
    din10_width ${din10_width} \
    din10_signed ${din10_signed} \
    din11_width ${din11_width} \
    din11_signed ${din11_signed} \
    din12_width ${din12_width} \
    din12_signed ${din12_signed} \
    din13_width ${din13_width} \
    din13_signed ${din13_signed} \
    din14_width ${din14_width} \
    din14_signed ${din14_signed} \
    din15_width ${din15_width} \
    din15_signed ${din15_signed} \
    din16_width ${din16_width} \
    din16_signed ${din16_signed} \
    dout_width ${dout_width} \
}"
} else {
puts "@W \[IMPL-101\] Cannot find ::AESL_LIB_VIRTEX::xil_gen_pipemux, check your platform lib"
}
}


# Memory (RAM/ROM)  definition:
set ID 8
set hasByteEnable 0
set MemName calculate_sha256_sha256_compress_block_Pipeline_round_loop_SHA256_K_ROM_AUTO_1R
set CoreName ap_simcore_mem
set PortList { 1 }
set DataWd 32
set AddrRange 64
set AddrWd 6
set impl_style auto
set TrueReset 0
set IsROM 1
set ROMData { "01000010100010100010111110011000" "01110001001101110100010010010001" "10110101110000001111101111001111" "11101001101101011101101110100101" "00111001010101101100001001011011" "01011001111100010001000111110001" "10010010001111111000001010100100" "10101011000111000101111011010101" "11011000000001111010101010011000" "00010010100000110101101100000001" "00100100001100011000010110111110" "01010101000011000111110111000011" "01110010101111100101110101110100" "10000000110111101011000111111110" "10011011110111000000011010100111" "11000001100110111111000101110100" "11100100100110110110100111000001" "11101111101111100100011110000110" "00001111110000011001110111000110" "00100100000011001010000111001100" "00101101111010010010110001101111" "01001010011101001000010010101010" "01011100101100001010100111011100" "01110110111110011000100011011010" "10011000001111100101000101010010" "10101000001100011100011001101101" "10110000000000110010011111001000" "10111111010110010111111111000111" "11000110111000000000101111110011" "11010101101001111001000101000111" "00000110110010100110001101010001" "00010100001010010010100101100111" "00100111101101110000101010000101" "00101110000110110010000100111000" "01001101001011000110110111111100" "01010011001110000000110100010011" "01100101000010100111001101010100" "01110110011010100000101010111011" "10000001110000101100100100101110" "10010010011100100010110010000101" "10100010101111111110100010100001" "10101000000110100110011001001011" "11000010010010111000101101110000" "11000111011011000101000110100011" "11010001100100101110100000011001" "11010110100110010000011000100100" "11110100000011100011010110000101" "00010000011010101010000001110000" "00011001101001001100000100010110" "00011110001101110110110000001000" "00100111010010000111011101001100" "00110100101100001011110010110101" "00111001000111000000110010110011" "01001110110110001010101001001010" "01011011100111001100101001001111" "01101000001011100110111111110011" "01110100100011111000001011101110" "01111000101001010110001101101111" "10000100110010000111100000010100" "10001100110001110000001000001000" "10010000101111101111111111111010" "10100100010100000110110011101011" "10111110111110011010001111110111" "11000110011100010111100011110010" }
set HasInitializer 1
set Initializer $ROMData
set NumOfStage 2
set DelayBudget 0.699
if {${::AESL::PGuard_simmodel_gen}} {
if {[info proc ap_gen_simcore_mem] == "ap_gen_simcore_mem"} {
    eval "ap_gen_simcore_mem { \
    id ${ID} \
    name ${MemName} \
    corename ${CoreName}  \
    op mem  \
    hasByteEnable ${hasByteEnable} \
    reset_level 1 \
    sync_rst true \
    stage_num ${NumOfStage}  \
    port_num 1 \
    port_list \{${PortList}\} \
    data_wd ${DataWd} \
    addr_wd ${AddrWd} \
    addr_range ${AddrRange} \
    style ${impl_style} \
    true_reset ${TrueReset} \
    delay_budget ${DelayBudget} \
    HasInitializer ${HasInitializer} \
    rom_data \{${ROMData}\} \
 } "
} else {
    puts "@W \[IMPL-102\] Cannot find ap_gen_simcore_mem, check your platform lib"
}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $MemName BINDTYPE {storage} TYPE {rom} IMPL {auto} LATENCY 2 ALLOW_PRAGMA 1
}


set CoreName ROM
if {${::AESL::PGuard_autocg_gen} && ${::AESL::PGuard_autocg_ipmgen}} {
if {[info proc ::AESL_LIB_VIRTEX::xil_gen_ROM] == "::AESL_LIB_VIRTEX::xil_gen_ROM"} {
    eval "::AESL_LIB_VIRTEX::xil_gen_ROM { \
    id ${ID} \
    name ${MemName} \
    corename ${CoreName}  \
    op mem  \
    hasByteEnable ${hasByteEnable} \
    reset_level 1 \
    sync_rst true \
    stage_num ${NumOfStage}  \
    port_num 1 \
    port_list \{${PortList}\} \
    data_wd ${DataWd} \
    addr_wd ${AddrWd} \
    addr_range ${AddrRange} \
    style ${impl_style} \
    true_reset ${TrueReset} \
    delay_budget ${DelayBudget} \
    HasInitializer ${HasInitializer} \
    rom_data \{${ROMData}\} \
 } "
  } else {
    puts "@W \[IMPL-104\] Cannot find ::AESL_LIB_VIRTEX::xil_gen_ROM, check your platform lib"
  }
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
    id 9 \
    name wbuf_15 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_wbuf_15 \
    op interface \
    ports { wbuf_15 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 10 \
    name wbuf_14 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_wbuf_14 \
    op interface \
    ports { wbuf_14 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 11 \
    name wbuf_13 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_wbuf_13 \
    op interface \
    ports { wbuf_13 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 12 \
    name wbuf_12 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_wbuf_12 \
    op interface \
    ports { wbuf_12 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 13 \
    name wbuf_11 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_wbuf_11 \
    op interface \
    ports { wbuf_11 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 14 \
    name wbuf_10 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_wbuf_10 \
    op interface \
    ports { wbuf_10 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 15 \
    name wbuf_9 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_wbuf_9 \
    op interface \
    ports { wbuf_9 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 16 \
    name wbuf_8 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_wbuf_8 \
    op interface \
    ports { wbuf_8 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 17 \
    name wbuf_7 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_wbuf_7 \
    op interface \
    ports { wbuf_7 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 18 \
    name wbuf_6 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_wbuf_6 \
    op interface \
    ports { wbuf_6 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 19 \
    name wbuf_5 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_wbuf_5 \
    op interface \
    ports { wbuf_5 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 20 \
    name wbuf_4 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_wbuf_4 \
    op interface \
    ports { wbuf_4 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 21 \
    name wbuf_3 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_wbuf_3 \
    op interface \
    ports { wbuf_3 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 22 \
    name wbuf_2 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_wbuf_2 \
    op interface \
    ports { wbuf_2 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 23 \
    name wbuf_1 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_wbuf_1 \
    op interface \
    ports { wbuf_1 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 24 \
    name wbuf \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_wbuf \
    op interface \
    ports { wbuf { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 25 \
    name state_0_read \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_state_0_read \
    op interface \
    ports { state_0_read { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 26 \
    name state_1_read \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_state_1_read \
    op interface \
    ports { state_1_read { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 27 \
    name state_2_read \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_state_2_read \
    op interface \
    ports { state_2_read { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 28 \
    name state_3_read \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_state_3_read \
    op interface \
    ports { state_3_read { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 29 \
    name state_4_read \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_state_4_read \
    op interface \
    ports { state_4_read { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 30 \
    name state_5_read \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_state_5_read \
    op interface \
    ports { state_5_read { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 31 \
    name state_6_read \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_state_6_read \
    op interface \
    ports { state_6_read { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 32 \
    name state_7_read \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_state_7_read \
    op interface \
    ports { state_7_read { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 33 \
    name a_2_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_a_2_out \
    op interface \
    ports { a_2_out { O 32 vector } a_2_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 34 \
    name b_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_b_out \
    op interface \
    ports { b_out { O 32 vector } b_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 35 \
    name c_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_c_out \
    op interface \
    ports { c_out { O 32 vector } c_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 36 \
    name d_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_d_out \
    op interface \
    ports { d_out { O 32 vector } d_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 37 \
    name e_2_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_e_2_out \
    op interface \
    ports { e_2_out { O 32 vector } e_2_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 38 \
    name f_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_f_out \
    op interface \
    ports { f_out { O 32 vector } f_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 39 \
    name g_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_g_out \
    op interface \
    ports { g_out { O 32 vector } g_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 40 \
    name h_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_h_out \
    op interface \
    ports { h_out { O 32 vector } h_out_ap_vld { O 1 bit } } \
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
set InstName calculate_sha256_flow_control_loop_pipe_sequential_init_U
set CompName calculate_sha256_flow_control_loop_pipe_sequential_init
set name flow_control_loop_pipe_sequential_init
if {${::AESL::PGuard_autocg_gen} && ${::AESL::PGuard_autocg_ipmgen}} {
if {[info proc ::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control] == "::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control"} {
eval "::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control { \
    name ${name} \
    prefix calculate_sha256_ \
}"
} else {
puts "@W \[IMPL-107\] Cannot find ::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control, check your platform lib"
}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $CompName BINDTYPE interface TYPE internal_upc_flow_control INSTNAME $InstName
}


