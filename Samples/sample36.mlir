module {
  llvm.mlir.global external @spec_fd() : !llvm.ptr {
    %0 = llvm.mlir.null : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @spec_uncompress(%arg0: i32, %arg1: i32) -> i64 {
    %2 = llvm.mlir.addressof @spec_fd : !llvm.ptr
    %ext = llvm.zext %arg1 : i32 to i64
    llvm.return %ext : i64
  }
  func.func @main() -> i64 {
      %a = llvm.mlir.constant (0 : i32) : i32
      %b = llvm.mlir.constant (0 : i32) : i32
      %c = llvm.call @spec_uncompress(%a, %b) : (i32, i32)->(i64)
      func.return %c : i64
  }
}
