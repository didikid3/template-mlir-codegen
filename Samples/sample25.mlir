module {
  llvm.mlir.global private constant @str("Hello World\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  func.func @main() -> i32 {
    %0 = llvm.mlir.addressof @str : !llvm.ptr<array<12 x i8>>
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = func.call @puts(%0) : (!llvm.ptr<array<12 x i8>>) -> i32
    func.return %1 : i32
  }
  func.func private @puts(!llvm.ptr<array<12 x i8>>) -> i32
}
