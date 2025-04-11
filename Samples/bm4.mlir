module{
  func.func private @malloc(i64) -> !llvm.ptr
  llvm.func @sieve(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %6 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %arg1, %4 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    %8 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %9 = llvm.sitofp %8 : i32 to f32
    %10 = llvm.call @sqrtf(%9) : (f32) -> f32
    %11 = llvm.fptosi %10 : f32 to i32
    llvm.store %11, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %1, %6 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb9
    %12 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
    %13 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %14 = llvm.icmp "sle" %12, %13 : i32
    llvm.cond_br %14, ^bb2, ^bb10
  ^bb2:  // pred: ^bb1
    %15 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %16 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
    %17 = llvm.sext %16 : i32 to i64
    %18 = llvm.getelementptr inbounds %15[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %19 = llvm.load %18 {alignment = 4 : i64} : !llvm.ptr -> i32
    %20 = llvm.icmp "ne" %19, %2 : i32
    llvm.cond_br %20, ^bb3, ^bb8
  ^bb3:  // pred: ^bb2
    %21 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
    %22 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
    %23 = llvm.mul %21, %22  : i32
    llvm.store %23, %7 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb4
  ^bb4:  // 2 preds: ^bb3, ^bb6
    %24 = llvm.load %7 {alignment = 4 : i64} : !llvm.ptr -> i32
    %25 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %26 = llvm.icmp "sle" %24, %25 : i32
    llvm.cond_br %26, ^bb5, ^bb7
  ^bb5:  // pred: ^bb4
    %27 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %28 = llvm.load %7 {alignment = 4 : i64} : !llvm.ptr -> i32
    %29 = llvm.sext %28 : i32 to i64
    %30 = llvm.getelementptr inbounds %27[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %2, %30 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb6
  ^bb6:  // pred: ^bb5
    %31 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
    %32 = llvm.load %7 {alignment = 4 : i64} : !llvm.ptr -> i32
    %33 = llvm.add %32, %31  : i32
    llvm.store %33, %7 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb4
  ^bb7:  // pred: ^bb4
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb2, ^bb7
    llvm.br ^bb9
  ^bb9:  // pred: ^bb8
    %34 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
    %35 = llvm.add %34, %0  : i32
    llvm.store %35, %6 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb10:  // pred: ^bb1
    llvm.return
  }
  llvm.func @sqrtf(f32 {llvm.noundef}) -> (f32 {llvm.noundef})
  func.func @main(%n: i32) -> i32 {
      %4 = llvm.mlir.constant (4 : i32) : i32
      %one = llvm.mlir.constant (1 : i32) : i32
      %1 = llvm.mlir.constant (1 : i8) : i8
      %n1 = llvm.add %n, %one : i32
      %nsize_ = llvm.mul %n1, %4 : i32
      %nsize = llvm.zext %nsize_ : i32 to i64
      %data = func.call @malloc(%nsize) : (i64) -> !llvm.ptr
      "llvm.intr.memset"(%data, %1, %nsize) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
      llvm.call @sieve(%n, %data) : (i32, !llvm.ptr) -> ()
      %42 = llvm.mlir.constant (42 : i32) : i32
      %checkPtr = llvm.getelementptr %data[%42] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %r = llvm.load %checkPtr : !llvm.ptr -> i32
      func.return %r : i32
  }
}
