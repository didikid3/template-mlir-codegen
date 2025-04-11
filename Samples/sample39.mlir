
module {
  llvm.func internal @foo() {
      %8 = llvm.mlir.constant(0 : i64) : i64
      %22 = llvm.mlir.constant(0 : i32) : i32
      %7 = llvm.mlir.undef : vector<16xi32>
      %43 = llvm.insertelement %22, %7[%8 : i64] : vector<16xi32>
      llvm.return
  }
  llvm.func internal @bar(%arg0: i32, %arg1: i32, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32, %arg5: !llvm.ptr, %arg6: i32, %arg7: i32, %arg8: !llvm.ptr, %arg9: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant (1 : i32) : i32
    %13 = llvm.alloca %0 x !llvm.ptr: (i32) -> !llvm.ptr
    llvm.store %arg8, %13 : !llvm.ptr, !llvm.ptr
    %ret = llvm.load %13 : !llvm.ptr -> !llvm.ptr
    llvm.return %ret : !llvm.ptr
  }
  func.func @main() -> !llvm.ptr {
    %ci32 = llvm.mlir.constant (42 : i32) : i32
    %addr = llvm.mlir.constant (123456 : i64) : i64
    %ptr = llvm.inttoptr %addr : i64 to !llvm.ptr
    %res = llvm.call @bar(%ci32, %ci32, %ptr, %ci32, %ci32, %ptr, %ci32, %ci32, %ptr, %ci32) : (i32, i32, !llvm.ptr, i32, i32, !llvm.ptr, i32, i32, !llvm.ptr, i32) -> !llvm.ptr
    func.return %res : !llvm.ptr
  }
}

