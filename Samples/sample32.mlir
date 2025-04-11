module {
  func.func @foo(%arg0 : i64) -> i64 {
    llvm.switch %arg0 : i64, ^bb3 [
      1: ^bb2,
      2: ^bb2,
      3: ^bb2,
      4: ^bb2,
      5: ^bb2
    ]
^bb1:
    %a1 = llvm.mlir.constant(1 : i64) : i64
    llvm.br ^bbEnd(%a1 : i64)
^bb2:
    %b1 = llvm.mlir.constant(2 : i64) : i64
    llvm.br ^bbEnd(%b1 : i64)
^bb3:
    %c1 = llvm.mlir.constant(3 : i64) : i64
    llvm.br ^bbEnd(%c1 : i64)
^bbEnd(%a : i64):
    func.return %a : i64
  }
  func.func @bar(%arg0 : i64) -> i64 {
    llvm.switch %arg0 : i64, ^bb3 [
      1: ^bb5,
      13: ^bb2,
      42: ^bb2,
      128: ^bb4
    ]
^bb1:
    %a1 = llvm.mlir.constant(1 : i64) : i64
    llvm.br ^bbEnd(%a1 : i64)
^bb2:
    %b1 = llvm.mlir.constant(2 : i64) : i64
    llvm.br ^bbEnd(%b1 : i64)
^bb3:
    %c1 = llvm.mlir.constant(3 : i64) : i64
    llvm.br ^bbEnd(%c1 : i64)
^bb4:
    %d1 = llvm.mlir.constant(4 : i64) : i64
    llvm.br ^bbEnd(%d1 : i64)
^bb5:
    %e1 = llvm.mlir.constant(5 : i64) : i64
    llvm.br ^bbEnd(%e1 : i64)
^bbEnd(%a : i64):
    func.return %a : i64
  }
  func.func @main(%arg0 : i64) -> i64 {
    %const = llvm.mlir.constant(3 : i64) : i64
    %extra = func.call @foo(%const) : (i64) -> i64
    %a = func.call @foo(%arg0) : (i64) -> i64
    %b = func.call @bar(%arg0) : (i64) -> i64
    %c = llvm.add %a, %b : i64
    %d = llvm.add %c, %extra : i64
    func.return %d : i64
  }
}
