module {
  func.func @main() -> i32 {
    %string = llvm.mlir.constant("abc") : !llvm.array<3 x i8>
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.constant(0 : i64) : i64
    %74 = llvm.add %0, %1  : i64
    %75 = llvm.trunc %74 : i64 to i32
    %76 = llvm.mul %75, %75  : i32
    func.return %76 : i32
  }
}
