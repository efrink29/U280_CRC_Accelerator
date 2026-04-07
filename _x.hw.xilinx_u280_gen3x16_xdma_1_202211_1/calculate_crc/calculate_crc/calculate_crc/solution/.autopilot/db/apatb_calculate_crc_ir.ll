; ModuleID = '/home/efrink/mled/split_cu/U280_CRC_Accelerator/_x.hw.xilinx_u280_gen3x16_xdma_1_202211_1/calculate_crc/calculate_crc/calculate_crc/solution/.autopilot/db/a.g.ld.5.gdce.bc'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-n8:16:32:64-S128-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "fpga64-xilinx-none"

; Function Attrs: noinline
define void @apatb_calculate_crc_ir(i8* noalias nocapture nonnull readonly %data_in, i32* noalias nocapture nonnull %crc_out, i32* noalias nocapture nonnull readonly %tables, i32 %numChunks, i32 %chunkSize, i32 %crc_size, i32 %init_value) local_unnamed_addr #0 {
entry:
  %data_in_copy = alloca [1024 x i8], align 512
  %malloccall = tail call i8* @malloc(i64 4096)
  %crc_out_copy = bitcast i8* %malloccall to [1024 x i32]*
  %malloccall1 = tail call i8* @malloc(i64 16384)
  %tables_copy = bitcast i8* %malloccall1 to [4096 x i32]*
  %0 = bitcast i8* %data_in to [1024 x i8]*
  %1 = bitcast i32* %crc_out to [1024 x i32]*
  %2 = bitcast i32* %tables to [4096 x i32]*
  call fastcc void @copy_in([1024 x i8]* nonnull %0, [1024 x i8]* nonnull align 512 %data_in_copy, [1024 x i32]* nonnull %1, [1024 x i32]* %crc_out_copy, [4096 x i32]* nonnull %2, [4096 x i32]* %tables_copy)
  %3 = getelementptr inbounds [1024 x i8], [1024 x i8]* %data_in_copy, i32 0, i32 0
  %4 = getelementptr inbounds [1024 x i32], [1024 x i32]* %crc_out_copy, i32 0, i32 0
  %5 = getelementptr inbounds [4096 x i32], [4096 x i32]* %tables_copy, i32 0, i32 0
  call void @apatb_calculate_crc_hw(i8* %3, i32* %4, i32* %5, i32 %numChunks, i32 %chunkSize, i32 %crc_size, i32 %init_value)
  call void @copy_back([1024 x i8]* %0, [1024 x i8]* %data_in_copy, [1024 x i32]* %1, [1024 x i32]* %crc_out_copy, [4096 x i32]* %2, [4096 x i32]* %tables_copy)
  tail call void @free(i8* %malloccall)
  tail call void @free(i8* %malloccall1)
  ret void
}

declare noalias i8* @malloc(i64) local_unnamed_addr

; Function Attrs: argmemonly noinline norecurse
define internal fastcc void @copy_in([1024 x i8]* noalias readonly, [1024 x i8]* noalias align 512, [1024 x i32]* noalias readonly, [1024 x i32]* noalias, [4096 x i32]* noalias readonly, [4096 x i32]* noalias) unnamed_addr #1 {
entry:
  call fastcc void @onebyonecpy_hls.p0a1024i8([1024 x i8]* align 512 %1, [1024 x i8]* %0)
  call fastcc void @onebyonecpy_hls.p0a1024i32([1024 x i32]* %3, [1024 x i32]* %2)
  call fastcc void @onebyonecpy_hls.p0a4096i32([4096 x i32]* %5, [4096 x i32]* %4)
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal fastcc void @onebyonecpy_hls.p0a1024i8([1024 x i8]* noalias align 512, [1024 x i8]* noalias readonly) unnamed_addr #2 {
entry:
  %2 = icmp eq [1024 x i8]* %0, null
  %3 = icmp eq [1024 x i8]* %1, null
  %4 = or i1 %2, %3
  br i1 %4, label %ret, label %copy

copy:                                             ; preds = %entry
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %copy
  %for.loop.idx1 = phi i64 [ 0, %copy ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [1024 x i8], [1024 x i8]* %0, i64 0, i64 %for.loop.idx1
  %src.addr = getelementptr [1024 x i8], [1024 x i8]* %1, i64 0, i64 %for.loop.idx1
  %5 = load i8, i8* %src.addr, align 1
  store i8 %5, i8* %dst.addr, align 1
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx1, 1
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
define internal fastcc void @onebyonecpy_hls.p0a4096i32([4096 x i32]* noalias, [4096 x i32]* noalias readonly) unnamed_addr #2 {
entry:
  %2 = icmp eq [4096 x i32]* %0, null
  %3 = icmp eq [4096 x i32]* %1, null
  %4 = or i1 %2, %3
  br i1 %4, label %ret, label %copy

copy:                                             ; preds = %entry
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %copy
  %for.loop.idx1 = phi i64 [ 0, %copy ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [4096 x i32], [4096 x i32]* %0, i64 0, i64 %for.loop.idx1
  %src.addr = getelementptr [4096 x i32], [4096 x i32]* %1, i64 0, i64 %for.loop.idx1
  %5 = load i32, i32* %src.addr, align 4
  store i32 %5, i32* %dst.addr, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx1, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, 4096
  br i1 %exitcond, label %for.loop, label %ret

ret:                                              ; preds = %for.loop, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal fastcc void @copy_out([1024 x i8]* noalias, [1024 x i8]* noalias readonly align 512, [1024 x i32]* noalias, [1024 x i32]* noalias readonly, [4096 x i32]* noalias, [4096 x i32]* noalias readonly) unnamed_addr #3 {
entry:
  call fastcc void @onebyonecpy_hls.p0a1024i8([1024 x i8]* %0, [1024 x i8]* align 512 %1)
  call fastcc void @onebyonecpy_hls.p0a1024i32([1024 x i32]* %2, [1024 x i32]* %3)
  call fastcc void @onebyonecpy_hls.p0a4096i32([4096 x i32]* %4, [4096 x i32]* %5)
  ret void
}

declare void @free(i8*) local_unnamed_addr

declare void @apatb_calculate_crc_hw(i8*, i32*, i32*, i32, i32, i32, i32)

; Function Attrs: argmemonly noinline norecurse
define internal fastcc void @copy_back([1024 x i8]* noalias, [1024 x i8]* noalias readonly align 512, [1024 x i32]* noalias, [1024 x i32]* noalias readonly, [4096 x i32]* noalias, [4096 x i32]* noalias readonly) unnamed_addr #3 {
entry:
  call fastcc void @onebyonecpy_hls.p0a1024i32([1024 x i32]* %2, [1024 x i32]* %3)
  ret void
}

define void @calculate_crc_hw_stub_wrapper(i8*, i32*, i32*, i32, i32, i32, i32) #4 {
entry:
  %7 = bitcast i8* %0 to [1024 x i8]*
  %8 = bitcast i32* %1 to [1024 x i32]*
  %9 = bitcast i32* %2 to [4096 x i32]*
  call void @copy_out([1024 x i8]* null, [1024 x i8]* %7, [1024 x i32]* null, [1024 x i32]* %8, [4096 x i32]* null, [4096 x i32]* %9)
  %10 = bitcast [1024 x i8]* %7 to i8*
  %11 = bitcast [1024 x i32]* %8 to i32*
  %12 = bitcast [4096 x i32]* %9 to i32*
  call void @calculate_crc_hw_stub(i8* %10, i32* %11, i32* %12, i32 %3, i32 %4, i32 %5, i32 %6)
  call void @copy_in([1024 x i8]* null, [1024 x i8]* %7, [1024 x i32]* null, [1024 x i32]* %8, [4096 x i32]* null, [4096 x i32]* %9)
  ret void
}

declare void @calculate_crc_hw_stub(i8*, i32*, i32*, i32, i32, i32, i32)

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
