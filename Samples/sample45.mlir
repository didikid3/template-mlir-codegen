module {
  llvm.func @bar(%a1 : !llvm.ptr, %a2 : i32, %a3 : i32, %a4 : !llvm.ptr, %a5 : i64, %a6 : i32, %a7 : !llvm.ptr, %a8 : i32) -> !llvm.ptr {
    %r1 = llvm.inttoptr %a8 : i32 to !llvm.ptr
    llvm.return %r1 : !llvm.ptr
  }
  llvm.func @bar2(%a1 : !llvm.ptr, %a2 : i32, %a3 : i32, %a4 : !llvm.ptr, %a5 : i64, %a6 : i32, %a7 : !llvm.ptr, %a8 : f32) -> f32 {
    %0 = llvm.mlir.constant (3.0 : f32) : f32
    %1 = llvm.fadd %0, %a8 : f32
    llvm.return %1 : f32
  }

  llvm.func @foo(
    %a1 : !llvm.ptr,
    %a2 : i32,
    %a3 : i32,
    %a4 : !llvm.ptr,
    %a5 : i64,
    %a6 : i32,
    %a7a : !llvm.ptr,
    %a7b : !llvm.ptr,
    %a7c : !llvm.ptr,
    %a7d : !llvm.ptr,
    %a7e : !llvm.ptr,
    %a7f : !llvm.ptr,
    %a7g : !llvm.ptr,
    %a7h : !llvm.ptr,
    %a7i : !llvm.ptr,
    %a8 : i32) -> !llvm.ptr {
    %r1 = llvm.inttoptr %a8 : i32 to !llvm.ptr
    llvm.return %r1 : !llvm.ptr
  }

  func.func @main() -> i64 {
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.mlir.constant(3 : i32) : i32
    %4 = llvm.mlir.null : !llvm.ptr
    %0 = llvm.mlir.constant (3.0 : f32) : f32
    %r1 = llvm.call @bar2(%4, %2, %2, %4, %1, %2, %4, %0) : (!llvm.ptr, i32, i32, !llvm.ptr, i64, i32, !llvm.ptr, f32) -> f32
    %r = llvm.call @bar(%4, %2, %2, %4, %1, %2, %4, %3) : (!llvm.ptr, i32, i32, !llvm.ptr, i64, i32, !llvm.ptr, i32) -> !llvm.ptr
    %r2 = llvm.call @foo(%4, %2, %2, %4, %1, %2, %4, %4, %4, %4, %4, %4, %4, %4, %4, %3) : (!llvm.ptr, i32, i32, !llvm.ptr, i64, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
    %r3 = llvm.fptoui %r1 : f32 to i64
    func.return %r3 : i64
  }
}
