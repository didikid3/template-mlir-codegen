module {
  func.func @bar(%a1 : i64, %a2 : i64, %a3 : i64, %a4 : i64, %a5 : i64, %a6 : i64, %a7 : i64, %a8 : i64, %a9 : i64, %a10 : i64) -> i64 {
    %r1 = llvm.add %a1, %a2 : i64
    %r2 = llvm.add %r1, %a3 : i64
    %r3 = llvm.add %r2, %a4 : i64
    %r4 = llvm.add %r3, %a5 : i64
    %r5 = llvm.add %r4, %a6 : i64
    %r6 = llvm.add %r5, %a7 : i64
    %r7 = llvm.add %r6, %a8 : i64
    %r8 = llvm.add %r7, %a9 : i64
    %r9 = llvm.add %r8, %a10 : i64
    %1 = llvm.mlir.constant(1 : i32) : i32
    %bla = llvm.mlir.constant(1 : i64) : i64
    %s = llvm.alloca %1 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %blub = llvm.add %bla, %bla : i64
    llvm.store %bla, %s {alignment = 8 : i64} : i64, !llvm.ptr
    func.return %r9 : i64
  }

  func.func @main() -> i64 {
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(2 : i64) : i64
    %3 = llvm.add %1, %2 : i64
    %r = func.call @bar(%1, %1, %1, %1, %1, %2, %1, %1, %1, %3) : (i64,i64,i64,i64,i64,i64,i64,i64,i64,i64) -> i64
    func.return %r : i64
  }
}
