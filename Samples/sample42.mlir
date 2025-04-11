module {
  func.func @main() -> i64 {
      %c1 = llvm.mlir.constant (1 : i64) : i64
      %c2 = llvm.mlir.constant (7 : i64) : i64
      llvm.br ^bb1(%c1, %c2 : i64, i64)
    ^bb1(%a : i64, %b : i64):
      llvm.switch %a : i64, ^bb1(%b, %a : i64, i64) [
        7 : ^bbEnd(%a : i64)
      ]
    ^bbEnd(%x : i64):
      func.return %x : i64
  }
}
