module {
  func.func @main() -> i32 {
    %const = llvm.mlir.constant (1 : i32) : i32
    %0 = llvm.mlir.constant (0 : i32) : i32
    %alloc = llvm.alloca %const x i32 : (i32) -> !llvm.ptr<i32>
    llvm.store %0, %alloc : !llvm.ptr<i32>
    %38 = llvm.sext %0 : i32 to i64
    %39 = llvm.getelementptr inbounds %alloc[%38] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %40 = llvm.load %39 {alignment = 8 : i64} : !llvm.ptr<i32>
    func.return %40 : i32
  }
}
