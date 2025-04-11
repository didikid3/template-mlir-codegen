func.func @main() -> i64 {
  %x = llvm.mlir.constant(1 : i1) : i1
  %z = llvm.and %x, %x : i1
  %a = llvm.mlir.constant(5 : i64) : i64
  %b = llvm.mlir.constant(3 : i64) : i64
  llvm.cond_br %z, ^bb1(%a : i64), ^bb2(%b : i64)
^bb1(%e : i64):
  %c = llvm.add %a, %b : i64
  %g = llvm.add %c, %e : i64
  llvm.br ^bb3(%g : i64)

^bb2(%f : i64):
  %d = llvm.sub %a, %b : i64
  %h = llvm.sub %d, %f : i64
  llvm.br ^bb3(%h : i64)

^bb3(%r : i64):
  return %r : i64
}
