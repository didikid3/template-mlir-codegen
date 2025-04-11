module {
  func.func @main() -> !llvm.ptr {
      %a = llvm.mlir.null : !llvm.ptr
      %indx = llvm.mlir.constant(dense<[2, 1]> : vector<2xi64>) : vector<2xi64>
      %ptr = llvm.getelementptr inbounds %a[%indx] : (!llvm.ptr, vector<2xi64>) -> !llvm.vec<2 x ptr>, !llvm.array<5 x array<2 x i64>>
      %zero = llvm.mlir.constant (0 : i64) : i64
      %ptr0 = llvm.extractelement %ptr[%zero : i64] : !llvm.vec<2 x ptr>
      func.return %ptr0 : !llvm.ptr
  }
}
