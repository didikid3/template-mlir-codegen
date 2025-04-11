llvm.func @main() -> i64 {
  %x = llvm.mlir.constant(1 : i1) : i1
  %a = llvm.mlir.constant(5 : i64) : i64
  %b = llvm.mlir.constant(3 : i64) : i64
  llvm.cond_br %x, ^bb1, ^bb2
^bb1:
  %c = llvm.add %a, %b : i64
  llvm.br ^bb3(%c : i64)

^bb2:
  %d = llvm.sub %a, %b : i64
  llvm.br ^bb3(%d : i64)

^bb3(%r : i64):
  llvm.return %r : i64
}
