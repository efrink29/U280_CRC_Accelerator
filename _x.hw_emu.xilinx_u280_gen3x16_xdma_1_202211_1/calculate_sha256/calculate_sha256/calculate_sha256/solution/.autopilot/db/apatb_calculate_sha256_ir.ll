; ModuleID = '/users/arashs/U280_CRC_Accelerator/_x.hw_emu.xilinx_u280_gen3x16_xdma_1_202211_1/calculate_sha256/calculate_sha256/calculate_sha256/solution/.autopilot/db/a.g.ld.5.gdce.bc'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-n8:16:32:64-S128-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "fpga64-xilinx-none"

%"struct.ap_uint<512>" = type { %"struct.ap_int_base<512, false>" }
%"struct.ap_int_base<512, false>" = type { %"struct.ssdm_int<512, false>" }
%"struct.ssdm_int<512, false>" = type { i512 }

; Function Attrs: noinline
define void @apatb_calculate_sha256_ir(%"struct.ap_uint<512>"* noalias nocapture nonnull readonly "maxi" %data_in, i32* noalias nocapture nonnull "maxi" %crc_out, i32 %numChunks, i32 %chunkSize) local_unnamed_addr #0 {
entry:
  %malloccall = tail call i8* @malloc(i64 65536)
  %data_in_copy = bitcast i8* %malloccall to [1024 x %"struct.ap_uint<512>"]*
  %malloccall1 = tail call i8* @malloc(i64 4096)
  %crc_out_copy = bitcast i8* %malloccall1 to [1024 x i32]*
  %0 = bitcast %"struct.ap_uint<512>"* %data_in to [1024 x %"struct.ap_uint<512>"]*
  %1 = bitcast i32* %crc_out to [1024 x i32]*
  call fastcc void @copy_in([1024 x %"struct.ap_uint<512>"]* nonnull %0, [1024 x %"struct.ap_uint<512>"]* %data_in_copy, [1024 x i32]* nonnull %1, [1024 x i32]* %crc_out_copy)
  call void @apatb_calculate_sha256_hw([1024 x %"struct.ap_uint<512>"]* %data_in_copy, [1024 x i32]* %crc_out_copy, i32 %numChunks, i32 %chunkSize)
  call void @copy_back([1024 x %"struct.ap_uint<512>"]* %0, [1024 x %"struct.ap_uint<512>"]* %data_in_copy, [1024 x i32]* %1, [1024 x i32]* %crc_out_copy)
  tail call void @free(i8* %malloccall)
  tail call void @free(i8* %malloccall1)
  ret void
}

declare noalias i8* @malloc(i64) local_unnamed_addr

; Function Attrs: argmemonly noinline norecurse
define internal fastcc void @copy_in([1024 x %"struct.ap_uint<512>"]* noalias readonly, [1024 x %"struct.ap_uint<512>"]* noalias, [1024 x i32]* noalias readonly, [1024 x i32]* noalias) unnamed_addr #1 {
entry:
  call fastcc void @"onebyonecpy_hls.p0a1024struct.ap_uint<512>"([1024 x %"struct.ap_uint<512>"]* %1, [1024 x %"struct.ap_uint<512>"]* %0)
  call fastcc void @onebyonecpy_hls.p0a1024i32([1024 x i32]* %3, [1024 x i32]* %2)
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal fastcc void @"onebyonecpy_hls.p0a1024struct.ap_uint<512>"([1024 x %"struct.ap_uint<512>"]* noalias %dst, [1024 x %"struct.ap_uint<512>"]* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [1024 x %"struct.ap_uint<512>"]* %dst, null
  %1 = icmp eq [1024 x %"struct.ap_uint<512>"]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a1024struct.ap_uint<512>"([1024 x %"struct.ap_uint<512>"]* nonnull %dst, [1024 x %"struct.ap_uint<512>"]* nonnull %src, i64 1024)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define void @"arraycpy_hls.p0a1024struct.ap_uint<512>"([1024 x %"struct.ap_uint<512>"]* %dst, [1024 x %"struct.ap_uint<512>"]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [1024 x %"struct.ap_uint<512>"]* %src, null
  %1 = icmp eq [1024 x %"struct.ap_uint<512>"]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond7 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond7, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx8 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [1024 x %"struct.ap_uint<512>"], [1024 x %"struct.ap_uint<512>"]* %src, i64 0, i64 %for.loop.idx8, i32 0, i32 0, i32 0
  %dst.addr.0.0.06 = getelementptr [1024 x %"struct.ap_uint<512>"], [1024 x %"struct.ap_uint<512>"]* %dst, i64 0, i64 %for.loop.idx8, i32 0, i32 0, i32 0
  %3 = load i512, i512* %src.addr.0.0.05, align 64
  store i512 %3, i512* %dst.addr.0.0.06, align 64
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx8, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal fastcc void @onebyonecpy_hls.p0a1024i32([1024 x i32]* noalias %dst, [1024 x i32]* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [1024 x i32]* %dst, null
  %1 = icmp eq [1024 x i32]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a1024i32([1024 x i32]* nonnull %dst, [1024 x i32]* nonnull %src, i64 1024)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define void @arraycpy_hls.p0a1024i32([1024 x i32]* %dst, [1024 x i32]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [1024 x i32]* %src, null
  %1 = icmp eq [1024 x i32]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [1024 x i32], [1024 x i32]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [1024 x i32], [1024 x i32]* %src, i64 0, i64 %for.loop.idx2
  %3 = load i32, i32* %src.addr, align 4
  store i32 %3, i32* %dst.addr, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal fastcc void @copy_out([1024 x %"struct.ap_uint<512>"]* noalias, [1024 x %"struct.ap_uint<512>"]* noalias readonly, [1024 x i32]* noalias, [1024 x i32]* noalias readonly) unnamed_addr #4 {
entry:
  call fastcc void @"onebyonecpy_hls.p0a1024struct.ap_uint<512>"([1024 x %"struct.ap_uint<512>"]* %0, [1024 x %"struct.ap_uint<512>"]* %1)
  call fastcc void @onebyonecpy_hls.p0a1024i32([1024 x i32]* %2, [1024 x i32]* %3)
  ret void
}

declare void @free(i8*) local_unnamed_addr

declare void @apatb_calculate_sha256_hw([1024 x %"struct.ap_uint<512>"]*, [1024 x i32]*, i32, i32)

; Function Attrs: argmemonly noinline norecurse
define internal fastcc void @copy_back([1024 x %"struct.ap_uint<512>"]* noalias, [1024 x %"struct.ap_uint<512>"]* noalias readonly, [1024 x i32]* noalias, [1024 x i32]* noalias readonly) unnamed_addr #4 {
entry:
  call fastcc void @onebyonecpy_hls.p0a1024i32([1024 x i32]* %2, [1024 x i32]* %3)
  ret void
}

define void @calculate_sha256_hw_stub_wrapper([1024 x %"struct.ap_uint<512>"]*, [1024 x i32]*, i32, i32) #5 {
entry:
  call void @copy_out([1024 x %"struct.ap_uint<512>"]* null, [1024 x %"struct.ap_uint<512>"]* %0, [1024 x i32]* null, [1024 x i32]* %1)
  %4 = bitcast [1024 x %"struct.ap_uint<512>"]* %0 to %"struct.ap_uint<512>"*
  %5 = bitcast [1024 x i32]* %1 to i32*
  call void @calculate_sha256_hw_stub(%"struct.ap_uint<512>"* %4, i32* %5, i32 %2, i32 %3)
  call void @copy_in([1024 x %"struct.ap_uint<512>"]* null, [1024 x %"struct.ap_uint<512>"]* %0, [1024 x i32]* null, [1024 x i32]* %1)
  ret void
}

declare void @calculate_sha256_hw_stub(%"struct.ap_uint<512>"*, i32*, i32, i32)

attributes #0 = { noinline "fpga.wrapper.func"="wrapper" }
attributes #1 = { argmemonly noinline norecurse "fpga.wrapper.func"="copyin" }
attributes #2 = { argmemonly noinline norecurse "fpga.wrapper.func"="onebyonecpy_hls" }
attributes #3 = { argmemonly noinline norecurse "fpga.wrapper.func"="arraycpy_hls" }
attributes #4 = { argmemonly noinline norecurse "fpga.wrapper.func"="copyout" }
attributes #5 = { "fpga.wrapper.func"="stub" }

!llvm.dbg.cu = !{}
!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!llvm.module.flags = !{!1, !2, !3}
!blackbox_cfg = !{!4}

!0 = !{!"clang version 7.0.0 "}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{}
