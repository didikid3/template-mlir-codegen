module {
  func.func @main() -> i32 {
    %cArray = llvm.mlir.constant(dense<[[0, 4, 1, 2, 5, 8, 12, 9, 6, 3, 7, 10, 13, 14, 11, 15], [0, 1, 4, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]> : tensor<2x16xi8>) : !llvm.array<2 x array<16 x i8>>
    %0 = llvm.mlir.undef : !llvm.array<42 x i32>
    %1 = llvm.mlir.constant(123 : i32) : i32
    %2 = llvm.insertvalue %1, %0[30] : !llvm.array<42 x i32>
    %ret1 = llvm.extractvalue %2[30] : !llvm.array<42 x i32>
    %ret2 = llvm.extractvalue %cArray[0, 3] : !llvm.array<2 x array<16 x i8>>
    %ret3 = llvm.zext %ret2 : i8 to i32
    %ret = llvm.add %ret1, %ret3 : i32
    func.return %ret : i32
  }
}
