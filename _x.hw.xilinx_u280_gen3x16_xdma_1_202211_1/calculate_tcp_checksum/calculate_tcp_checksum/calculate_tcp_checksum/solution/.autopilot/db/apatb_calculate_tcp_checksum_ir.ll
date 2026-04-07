; ModuleID = '/home/efrink/mled/split_cu/U280_CRC_Accelerator/_x.hw.xilinx_u280_gen3x16_xdma_1_202211_1/calculate_tcp_checksum/calculate_tcp_checksum/calculate_tcp_checksum/solution/.autopilot/db/a.g.ld.5.gdce.bc'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-n8:16:32:64-S128-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "fpga64-xilinx-none"

%"struct.ap_uint<512>" = type { %"struct.ap_int_base<512, false>" }
%"struct.ap_int_base<512, false>" = type { %"struct.ssdm_int<512, false>" }
%"struct.ssdm_int<512, false>" = type { i512 }

; Function Attrs: noinline
define void @apatb_calculate_tcp_checksum_ir(%"struct.ap_uint<512>"* noalias nocapture nonnull readonly %data_in, i32* noalias nocapture nonnull %crc_out, i32 %numChunks, i32 %chunkSize) local_unnamed_addr #0 {
entry:
  %malloccall = call i8* @malloc(i64 65536)
  %data_in_copy = bitcast i8* %malloccall to [1024 x i512]*
  %malloccall1 = tail call i8* @malloc(i64 4096)
  %crc_out_copy = bitcast i8* %malloccall1 to [1024 x i32]*
  %0 = bitcast %"struct.ap_uint<512>"* %data_in to [1024 x %"struct.ap_uint<512>"]*
  %1 = bitcast i32* %crc_out to [1024 x i32]*
  call fastcc void @copy_in([1024 x %"struct.ap_uint<512>"]* nonnull %0, [1024 x i512]* %data_in_copy, [1024 x i32]* nonnull %1, [1024 x i32]* %crc_out_copy)
  %2 = getelementptr [1024 x i512], [1024 x i512]* %data_in_copy, i32 0, i32 0
  %3 = getelementptr inbounds [1024 x i32], [1024 x i32]* %crc_out_copy, i32 0, i32 0
  call void @apatb_calculate_tcp_checksum_hw(i512* %2, i32* %3, i32 %numChunks, i32 %chunkSize)
  call void @copy_back([1024 x %"struct.ap_uint<512>"]* %0, [1024 x i512]* %data_in_copy, [1024 x i32]* %1, [1024 x i32]* %crc_out_copy)
  call void @free(i8* %malloccall)
  tail call void @free(i8* %malloccall1)
  ret void
}

declare noalias i8* @malloc(i64) local_unnamed_addr

; Function Attrs: argmemonly noinline norecurse
define internal fastcc void @copy_in([1024 x %"struct.ap_uint<512>"]* noalias readonly, [1024 x i512]* noalias, [1024 x i32]* noalias readonly, [1024 x i32]* noalias) unnamed_addr #1 {
entry:
  call fastcc void @"onebyonecpy_hls.p0a1024struct.ap_uint<512>"([1024 x i512]* %1, [1024 x %"struct.ap_uint<512>"]* %0)
  call fastcc void @onebyonecpy_hls.p0a1024i32([1024 x i32]* %3, [1024 x i32]* %2)
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal fastcc void @"onebyonecpy_hls.p0a1024struct.ap_uint<512>"([1024 x i512]* noalias, [1024 x %"struct.ap_uint<512>"]* noalias readonly) unnamed_addr #2 {
entry:
  %2 = icmp eq [1024 x i512]* %0, null
  %3 = icmp eq [1024 x %"struct.ap_uint<512>"]* %1, null
  %4 = or i1 %2, %3
  br i1 %4, label %ret, label %copy

copy:                                             ; preds = %entry
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %copy
  %for.loop.idx7 = phi i64 [ 0, %copy ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [1024 x %"struct.ap_uint<512>"], [1024 x %"struct.ap_uint<512>"]* %1, i64 0, i64 %for.loop.idx7, i32 0, i32 0, i32 0
  %5 = getelementptr [1024 x i512], [1024 x i512]* %0, i64 0, i64 %for.loop.idx7
  %6 = load i512, i512* %src.addr.0.0.05, align 64
  store i512 %6, i512* %5, align 64
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx7, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, 1024
  br i1 %exitcond, label %for.loop, label %ret

ret:                                              ; preds = %for.loop, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal fastcc void @onebyonecpy_hls.p0a1024i32([1024 x i32]* noalias, [1024 x i32]* noalias readonly) unnamed_addr #2 {
entry:
  %2 = icmp eq [1024 x i32]* %0, null
  %3 = icmp eq [1024 x i32]* %1, null
  %4 = or i1 %2, %3
  br i1 %4, label %ret, label %copy

copy:                                             ; preds = %entry
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %copy
  %for.loop.idx1 = phi i64 [ 0, %copy ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [1024 x i32], [1024 x i32]* %0, i64 0, i64 %for.loop.idx1
  %src.addr = getelementptr [1024 x i32], [1024 x i32]* %1, i64 0, i64 %for.loop.idx1
  %5 = load i32, i32* %src.addr, align 4
  store i32 %5, i32* %dst.addr, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx1, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, 1024
  br i1 %exitcond, label %for.loop, label %ret

ret:                                              ; preds = %for.loop, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal fastcc void @copy_out([1024 x %"struct.ap_uint<512>"]* noalias, [1024 x i512]* noalias readonly, [1024 x i32]* noalias, [1024 x i32]* noalias readonly) unnamed_addr #3 {
entry:
  call fastcc void @"onebyonecpy_hls.p0a1024struct.ap_uint<512>.6"([1024 x %"struct.ap_uint<512>"]* %0, [1024 x i512]* %1)
  call fastcc void @onebyonecpy_hls.p0a1024i32([1024 x i32]* %2, [1024 x i32]* %3)
  ret void
}

declare void @free(i8*) local_unnamed_addr

; Function Attrs: argmemonly noinline norecurse
define internal fastcc void @"onebyonecpy_hls.p0a1024struct.ap_uint<512>.6"([1024 x %"struct.ap_uint<512>"]* noalias, [1024 x i512]* noalias readonly) unnamed_addr #2 {
entry:
  %2 = icmp eq [1024 x %"struct.ap_uint<512>"]* %0, null
  %3 = icmp eq [1024 x i512]* %1, null
  %4 = or i1 %2, %3
  br i1 %4, label %ret, label %copy

copy:                                             ; preds = %entry
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %copy
  %for.loop.idx7 = phi i64 [ 0, %copy ], [ %for.loop.idx.next, %for.loop ]
  %5 = getelementptr [1024 x i512], [1024 x i512]* %1, i64 0, i64 %for.loop.idx7
  %dst.addr.0.0.06 = getelementptr [1024 x %"struct.ap_uint<512>"], [1024 x %"struct.ap_uint<512>"]* %0, i64 0, i64 %for.loop.idx7, i32 0, i32 0, i32 0
  %6 = load i512, i512* %5, align 64
  store i512 %6, i512* %dst.addr.0.0.06, align 64
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx7, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, 1024
  br i1 %exitcond, label %for.loop, label %ret

ret:                                              ; preds = %for.loop, %entry
  ret void
}

declare void @apatb_calculate_tcp_checksum_hw(i512*, i32*, i32, i32)

; Function Attrs: argmemonly noinline norecurse
define internal fastcc void @copy_back([1024 x %"struct.ap_uint<512>"]* noalias, [1024 x i512]* noalias readonly, [1024 x i32]* noalias, [1024 x i32]* noalias readonly) unnamed_addr #3 {
entry:
  call fastcc void @onebyonecpy_hls.p0a1024i32([1024 x i32]* %2, [1024 x i32]* %3)
  ret void
}

define void @calculate_tcp_checksum_hw_stub_wrapper(i512*, i32*, i32, i32) #4 {
entry:
  %malloccall = tail call i8* @malloc(i64 65536)
  %4 = bitcast i8* %malloccall to [1024 x %"struct.ap_uint<512>"]*
  %5 = bitcast i512* %0 to [1024 x i512]*
  %6 = bitcast i32* %1 to [1024 x i32]*
  call void @copy_out([1024 x %"struct.ap_uint<512>"]* %4, [1024 x i512]* %5, [1024 x i32]* null, [1024 x i32]* %6)
  %7 = bitcast [1024 x %"struct.ap_uint<512>"]* %4 to %"struct.ap_uint<512>"*
  %8 = bitcast [1024 x i32]* %6 to i32*
  call void @calculate_tcp_checksum_hw_stub(%"struct.ap_uint<512>"* %7, i32* %8, i32 %2, i32 %3)
  call void @copy_in([1024 x %"struct.ap_uint<512>"]* %4, [1024 x i512]* %5, [1024 x i32]* null, [1024 x i32]* %6)
  ret void
}

declare void @calculate_tcp_checksum_hw_stub(%"struct.ap_uint<512>"*, i32*, i32, i32)

attributes #0 = { noinline "fpga.wrapper.func"="wrapper" }
attributes #1 = { argmemonly noinline norecurse "fpga.wrapper.func"="copyin" }
attributes #2 = { argmemonly noinline norecurse "fpga.wrapper.func"="onebyonecpy_hls" }
attributes #3 = { argmemonly noinline norecurse "fpga.wrapper.func"="copyout" }
attributes #4 = { "fpga.wrapper.func"="stub" }

!llvm.dbg.cu = !{}
!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!llvm.module.flags = !{!1, !2, !3}
!blackbox_cfg = !{!4}

!0 = !{!"clang version 7.0.0 "}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{}
