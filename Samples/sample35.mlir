module {
  func.func @main() -> i32 {
    %one = llvm.mlir.constant(1 : i32) : i32
    %val = llvm.mlir.constant(1 : i32) : i32
    %two = llvm.mlir.constant(2 : i32) : i32
    %three = llvm.mlir.constant(3 : i32) : i32
    %four = llvm.mlir.constant(4 : i32) : i32
    %29 = llvm.mlir.undef : vector<2xi64>
    llvm.switch %val : i32, ^bbend(%two : i32) [
      1: ^bb1,
      0: ^bb2,
      2: ^bb3
    ]
  ^bb1:
    llvm.br ^bbend(%one : i32)
  ^bb2:
    llvm.br ^bbend(%three : i32)
  ^bb3:
    llvm.br ^bbend(%four : i32)
  ^bbend(%arg : i32):
    func.return %arg : i32
  }
}
