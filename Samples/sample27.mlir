module {
llvm.func @recurse(%arg0: i32 {llvm.noundef}) {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %2 {alignment = 4 : i64} : i32, !llvm.ptr
    %4 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
    %5 = llvm.icmp "eq" %4, %1 : i32
    llvm.cond_br %5, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.br ^bb5
  ^bb2:  // pred: ^bb0
    %6 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.store %6, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    %7 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
    %8 = llvm.add %7, %0  : i32
    llvm.call @recurse(%8) : (i32) -> ()
    %9 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %10 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
    %11 = llvm.icmp "eq" %9, %10 : i32
    llvm.cond_br %11, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    %12 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.call @recurse(%12) : (i32) -> ()
    llvm.br ^bb5
  ^bb5:  // 3 preds: ^bb1, ^bb3, ^bb4
    llvm.return
  }
  func.func @main() -> i32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    llvm.call @recurse(%zero) : (i32) -> ()
    func.return %zero : i32
  }
}
