module {
  func.func @fib(%arg0: i32) -> i32 {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(-1 : i32) : i32
    %4 = llvm.mlir.constant(-2 : i32) : i32
    %5 = llvm.mlir.constant(3 : i32) : i32
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.icmp "slt" %arg0, %0 : i32
    llvm.cond_br %7, ^bb3(%1 : i32), ^bb1(%arg0, %2 : i32, i32)
  ^bb1(%8: i32, %9: i32):  // 2 preds: ^bb0, ^bb1
    %10 = llvm.add %8, %3  : i32
    %11 = func.call @fib(%10) : (i32) -> i32
    %12 = llvm.add %8, %4  : i32
    %13 = llvm.add %11, %9  : i32
    %14 = llvm.icmp "ult" %8, %5 : i32
    llvm.cond_br %14, ^bb2, ^bb1(%12, %13 : i32, i32)
  ^bb3(%16: i32):  // 2 preds: ^bb0, ^bb2
    func.return %16 : i32
  ^bb2:  // pred: ^bb1
    %15 = llvm.add %13, %6  : i32
    llvm.br ^bb3(%15 : i32)
  }
  func.func @main(%arg: i64) -> i32 {
    %0 = llvm.trunc %arg: i64 to i32
    %1 = func.call @fib(%0) : (i32) -> i32
    func.return %1 : i32
  }
}
