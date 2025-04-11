module {
  llvm.func @foo(%arg0: i32 {llvm.noundef}, ...) -> (i32 {llvm.noundef}) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.undef : i32
    %6 = llvm.mlir.constant(-2 : i32) : i32
    %7 = llvm.mlir.constant(41 : i32) : i32
    %8 = llvm.mlir.constant(8 : i64) : i64
    %9 = llvm.mlir.constant(8 : i32) : i32
    %10 = llvm.alloca %0 x !llvm.array<1 x struct<"struct.__va_list_tag", (i32, i32, ptr, ptr)>> {alignment = 16 : i64} : (i32) -> !llvm.ptr
    llvm.intr.vastart %10 : !llvm.ptr
    %11 = llvm.icmp "sgt" %arg0, %1 : i32
    llvm.cond_br %11, ^bb1, ^bb8(%1 : i32)
  ^bb1:  // pred: ^bb0
    %12 = llvm.load %10 {alignment = 16 : i64} : !llvm.ptr -> i32
    %13 = llvm.getelementptr inbounds %10[%2, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.__va_list_tag", (i32, i32, ptr, ptr)>
    %14 = llvm.getelementptr inbounds %10[%2, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.__va_list_tag", (i32, i32, ptr, ptr)>
    %15 = llvm.load %14 {alignment = 16 : i64} : !llvm.ptr -> !llvm.ptr
    %16 = llvm.load %13 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %17 = llvm.and %arg0, %0  : i32
    %18 = llvm.icmp "eq" %arg0, %0 : i32
    llvm.cond_br %18, ^bb3(%5, %16, %1, %12 : i32, !llvm.ptr, i32, i32), ^bb2
  ^bb2:  // pred: ^bb1
    %19 = llvm.and %arg0, %6  : i32
    llvm.br ^bb9(%16, %1, %12, %1 : !llvm.ptr, i32, i32, i32)
  ^bb3(%20: i32, %21: !llvm.ptr, %22: i32, %23: i32):  // 2 preds: ^bb1, ^bb15
    %24 = llvm.icmp "eq" %17, %1 : i32
    llvm.cond_br %24, ^bb8(%20 : i32), ^bb4
  ^bb4:  // pred: ^bb3
    %25 = llvm.icmp "ult" %23, %7 : i32
    llvm.cond_br %25, ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    %26 = llvm.getelementptr %21[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %26, %13 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb7(%21 : !llvm.ptr)
  ^bb6:  // pred: ^bb4
    %27 = llvm.zext %23 : i32 to i64
    %28 = llvm.getelementptr %15[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %29 = llvm.add %23, %9  : i32
    llvm.store %29, %10 {alignment = 16 : i64} : i32, !llvm.ptr
    llvm.br ^bb7(%28 : !llvm.ptr)
  ^bb7(%30: !llvm.ptr):  // 2 preds: ^bb5, ^bb6
    %31 = llvm.load %30 {alignment = 4 : i64} : !llvm.ptr -> i32
    %32 = llvm.add %31, %22  : i32
    llvm.br ^bb8(%32 : i32)
  ^bb8(%33: i32):  // 3 preds: ^bb0, ^bb3, ^bb7
    llvm.intr.vaend %10 : !llvm.ptr
    llvm.return %33 : i32
  ^bb9(%34: !llvm.ptr, %35: i32, %36: i32, %37: i32):  // 2 preds: ^bb2, ^bb15
    %38 = llvm.icmp "ult" %36, %7 : i32
    llvm.cond_br %38, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %39 = llvm.zext %36 : i32 to i64
    %40 = llvm.getelementptr %15[%39] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %41 = llvm.add %36, %9  : i32
    llvm.store %41, %10 {alignment = 16 : i64} : i32, !llvm.ptr
    llvm.br ^bb12(%34, %41, %40 : !llvm.ptr, i32, !llvm.ptr)
  ^bb11:  // pred: ^bb9
    %42 = llvm.getelementptr %34[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %42, %13 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb12(%42, %36, %34 : !llvm.ptr, i32, !llvm.ptr)
  ^bb12(%43: !llvm.ptr, %44: i32, %45: !llvm.ptr):  // 2 preds: ^bb10, ^bb11
    %46 = llvm.load %45 {alignment = 4 : i64} : !llvm.ptr -> i32
    %47 = llvm.add %46, %35  : i32
    %48 = llvm.icmp "ult" %44, %7 : i32
    llvm.cond_br %48, ^bb14, ^bb13
  ^bb13:  // pred: ^bb12
    %49 = llvm.getelementptr %43[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %49, %13 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb15(%49, %44, %43 : !llvm.ptr, i32, !llvm.ptr)
  ^bb14:  // pred: ^bb12
    %50 = llvm.zext %44 : i32 to i64
    %51 = llvm.getelementptr %15[%50] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %52 = llvm.add %44, %9  : i32
    llvm.store %52, %10 {alignment = 16 : i64} : i32, !llvm.ptr
    llvm.br ^bb15(%43, %52, %51 : !llvm.ptr, i32, !llvm.ptr)
  ^bb15(%53: !llvm.ptr, %54: i32, %55: !llvm.ptr):  // 2 preds: ^bb13, ^bb14
    %56 = llvm.load %55 {alignment = 4 : i64} : !llvm.ptr -> i32
    %57 = llvm.add %56, %47  : i32
    %58 = llvm.add %37, %3  : i32
    %59 = llvm.icmp "eq" %58, %19 : i32
    llvm.cond_br %59, ^bb3(%57, %53, %57, %54 : i32, !llvm.ptr, i32, i32), ^bb9(%53, %57, %54, %58 : !llvm.ptr, i32, i32, i32)
  }

  func.func @main() -> i32 {
    %const = llvm.mlir.constant(4 : i32) : i32
    %const2 = llvm.mlir.constant(25 : i32) : i32
    %a = llvm.call @foo(%const, %const2, %const2, %const2, %const2) : (i32, i32, i32, i32, i32) -> i32
    func.return %a : i32
  }
}
