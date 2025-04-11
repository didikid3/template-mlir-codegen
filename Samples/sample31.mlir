module {
  llvm.mlir.global internal @test() {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<42 x i32> {
    %0 = llvm.mlir.undef : !llvm.array<42 x i32>
    %1 = llvm.mlir.constant(123 : i32) : i32
    %2 = llvm.insertvalue %1, %0[30] : !llvm.array<42 x i32>
    llvm.return %2 : !llvm.array<42 x i32>
  }
  func.func @main() -> i64 {
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(2 : i64) : i64
    %3 = llvm.add %1, %2 : i64
    llvm.br ^bb1(%3 : i64)
^bb1(%arg0 : i64):
    func.return %arg0 : i64
  }
}
