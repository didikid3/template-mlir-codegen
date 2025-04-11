module {
  // func.func @bar() -> memref<2x2xi32>{
  //   %B = memref.alloc () : memref<2x2xi32>
  //   func.return %B : memref<2x2xi32>
  // }
  func.func @main(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) -> i32 {
    %one = llvm.mlir.constant (1 : i32) : i32
    %r = llvm.add %one, %arg0 : i32
    func.return %r : i32
  }
}
