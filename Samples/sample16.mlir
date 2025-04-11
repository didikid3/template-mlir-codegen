func.func private @malloc(i64) -> !llvm.ptr<i64>
func.func @main(%arg: i64) -> i32 {
    %arg0 = arith.trunci %arg : i64 to i32
    %c4 = llvm.mlir.constant(4 : index) : i64
    %c8 = llvm.mlir.constant(8 : index) : i64
    %0 = llvm.mlir.constant(2 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(10000001 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.mlir.constant(false) : i1
    %6 = llvm.mlir.constant(0 : i64) : i64
    %7 = llvm.mlir.constant(1 : i32) : i32
    %8 = llvm.mlir.constant(0 : i32) : i32
    %size1 = llvm.mul %2, %c4 : i64
    %size2 = llvm.mul %2, %c8 : i64
    %tmp = func.call @malloc(%size1) : (i64) -> !llvm.ptr<i64>
    %9 = llvm.bitcast %tmp : !llvm.ptr<i64> to !llvm.ptr<i32>
    %10 = func.call @malloc(%size2) : (i64) -> !llvm.ptr<i64>
    %11 = func.call @malloc(%size2) : (i64) -> !llvm.ptr<i64>
    llvm.br ^bb1(%3 : i64)
  ^bb1(%12: i64):  // 2 preds: ^bb0, ^bb2
    %13 = llvm.icmp "slt" %12, %2 : i64
    llvm.cond_br %13, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %14 = llvm.getelementptr %9[%12] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    llvm.store %7, %14 : !llvm.ptr<i32>
    %15 = llvm.add %12, %1  : i64
    llvm.br ^bb1(%15 : i64)
  ^bb3:  // pred: ^bb1
    %16 = llvm.getelementptr %9[1] : (!llvm.ptr<i32>) -> !llvm.ptr<i32>
    llvm.store %8, %16 : !llvm.ptr<i32>
    llvm.store %8, %9 : !llvm.ptr<i32>
    %17 = llvm.sext %arg0 : i32 to i64
    %18 = llvm.sext %arg0 : i32 to i64
    llvm.br ^bb4(%0, %8 : i64, i32)
  ^bb4(%19: i64, %20: i32):  // 2 preds: ^bb3, ^bb21
    %21 = llvm.icmp "slt" %19, %17 : i64
    llvm.cond_br %21, ^bb5, ^bb22
  ^bb5:  // pred: ^bb4
    %22 = llvm.getelementptr %9[%19] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %23 = llvm.load %22 : !llvm.ptr<i32>
    %24 = llvm.icmp "ne" %23, %8 : i32
    llvm.cond_br %24, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %25 = llvm.sext %20 : i32 to i64
    %26 = llvm.getelementptr %10[%25] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %19, %26 : !llvm.ptr<i64>
    %27 = llvm.add %20, %7  : i32
    %28 = llvm.getelementptr %11[%19] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %19, %28 : !llvm.ptr<i64>
    llvm.br ^bb8(%27 : i32)
  ^bb7:  // pred: ^bb5
    llvm.br ^bb8(%20 : i32)
  ^bb8(%29: i32):  // 2 preds: ^bb6, ^bb7
    llvm.br ^bb9
  ^bb9:  // pred: ^bb8
    %30 = llvm.sext %29 : i32 to i64
    llvm.br ^bb10(%6 : i64)
  ^bb10(%31: i64):  // 2 preds: ^bb9, ^bb20
    %32 = llvm.icmp "slt" %31, %30 : i64
    llvm.cond_br %32, ^bb11, ^bb18(%5, %31 : i1, i64)
  ^bb11:  // pred: ^bb10
    %33 = llvm.getelementptr %10[%31] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %34 = llvm.load %33 : !llvm.ptr<i64>
    %35 = llvm.mul %19, %34  : i64
    %36 = llvm.icmp "slt" %35, %18 : i64
    llvm.cond_br %36, ^bb12, ^bb16(%5, %31 : i1, i64)
  ^bb12:  // pred: ^bb11
    %37 = llvm.getelementptr %11[%19] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    %38 = llvm.load %37 : !llvm.ptr<i64>
    %39 = llvm.icmp "sle" %34, %38 : i64
    llvm.cond_br %39, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %40 = llvm.getelementptr %9[%35] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    llvm.store %8, %40 : !llvm.ptr<i32>
    %41 = llvm.getelementptr %11[%35] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    llvm.store %34, %41 : !llvm.ptr<i64>
    %42 = llvm.add %31, %4  : i64
    llvm.br ^bb15(%42 : i64)
  ^bb14:  // pred: ^bb12
    llvm.br ^bb15(%31 : i64)
  ^bb15(%43: i64):  // 2 preds: ^bb13, ^bb14
    llvm.br ^bb16(%39, %43 : i1, i64)
  ^bb16(%44: i1, %45: i64):  // 2 preds: ^bb11, ^bb15
    llvm.br ^bb17(%44, %45 : i1, i64)
  ^bb17(%46: i1, %47: i64):  // pred: ^bb16
    llvm.br ^bb18(%46, %47 : i1, i64)
  ^bb18(%48: i1, %49: i64):  // 2 preds: ^bb10, ^bb17
    llvm.br ^bb19(%48, %49 : i1, i64)
  ^bb19(%50: i1, %51: i64):  // pred: ^bb18
    llvm.br ^bb20
  ^bb20:  // pred: ^bb19
    llvm.cond_br %50, ^bb10(%51 : i64), ^bb21
  ^bb21:  // pred: ^bb20
    %52 = llvm.add %19, %1  : i64
    llvm.br ^bb4(%52, %29 : i64, i32)
  ^bb22:  // pred: ^bb4
    func.return %20 : i32
}
