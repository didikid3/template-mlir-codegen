module {
  func.func @main() -> i64 {
      llvm.br ^bb2
    ^bb1:
      %add = llvm.add %c1_, %c2_ : i64
      llvm.br ^bbEnd
    ^bb2:
      %c1 = llvm.mlir.constant (1 : i64) : i64
      %c1_ = llvm.add %c1, %c1 : i64
      %c2 = llvm.mlir.constant (7 : i64) : i64
      %c2_ = llvm.add %c2, %c2 : i64
      llvm.br ^bb1
    ^bbEnd:
      func.return %add : i64
  }
}
