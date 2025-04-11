module {
    llvm.func @foo(%arg0: !llvm.ptr) -> i64
    {
      %25 = llvm.mlir.constant(0 : i64) : i64
      llvm.return %25 : i64
    }
    llvm.func @bar(%arg0: !llvm.ptr) -> i64
    {
      %29 = llvm.mlir.constant(1 : i64) : i64
      llvm.return %29 : i64
    }
    llvm.mlir.global external @__foo(@foo) {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.ptr
    llvm.func @main() -> i32 {
      %c = llvm.mlir.constant (4294967295 : i32) : i32
      llvm.br ^bb1
    ^bb1:
        llvm.switch %c : i32, ^bb1 [
            4294967295: ^bbEnd,
            265: ^bb1,
            266: ^bb1,
            63: ^bb1
        ]
    ^bbEnd:
      %24 = llvm.mlir.undef : vector<8xi32>
      %25 = llvm.mlir.constant(0 : i64) : i64
      %26 = llvm.mlir.constant(1 : i64) : i64
      %125 = llvm.insertelement %c, %24[%25 : i64] : vector<8xi32>
      %126 = llvm.insertelement %c, %125[%26 : i64] : vector<8xi32>
      %a = llvm.mlir.null : !llvm.ptr
      %b = llvm.mlir.null : !llvm.ptr
      %52 = llvm.call @foo(%a) : (!llvm.ptr) -> i64
      %54 = llvm.call @bar(%b) : (!llvm.ptr) -> i64
      llvm.return %c : i32
  }
}
