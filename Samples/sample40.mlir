
module {
  llvm.func @bar(%arg0: !llvm.ptr, %arg1: i64) -> () {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %8 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %9 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %6 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg1, %7 {alignment = 8 : i64} : i64, !llvm.ptr
    %10 = llvm.load %7 {alignment = 8 : i64} : !llvm.ptr -> i64
    %11 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %12 = llvm.getelementptr inbounds %11[%1] : (!llvm.ptr, i32) -> !llvm.ptr, i64
    %13 = llvm.load %12 {alignment = 8 : i64} : !llvm.ptr -> i64
    %14 = llvm.srem %10, %13  : i64
    %16 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    llvm.return
  }
  func.func @main() -> i64 {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %const = llvm.mlir.constant(1 : i64) : i64
    %ptr = llvm.alloca %0 x i64 : (i32) -> !llvm.ptr
    llvm.store %const, %ptr : i64, !llvm.ptr
    llvm.call @bar(%ptr, %const) : (!llvm.ptr, i64) -> ()
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb3
    %72 = llvm.load %ptr {alignment = 8 : i64} : !llvm.ptr -> i64
    %73 = llvm.load %ptr {alignment = 4 : i64} : !llvm.ptr -> i32
    %74 = llvm.sdiv %0, %73  : i32
    %75 = llvm.add %74, %0  : i32
    %76 = llvm.add %75, %0  : i32
    %77 = llvm.sext %76 : i32 to i64
    func.return %72 : i64
  }
}
