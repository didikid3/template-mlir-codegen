module {
  llvm.mlir.global internal constant @global() {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.struct<(i64, ptr, ptr, ptr)> {
    %9 = llvm.mlir.addressof @bar : !llvm.ptr
    %10 = llvm.mlir.null : !llvm.ptr
    %46 = llvm.mlir.addressof @foo : !llvm.ptr
    %47 = llvm.mlir.constant(4 : i64) : i64
    %48 = llvm.mlir.undef : !llvm.struct<(i64, ptr, ptr, ptr)>
    %49 = llvm.insertvalue %47, %48[0] : !llvm.struct<(i64, ptr, ptr, ptr)>
    %50 = llvm.insertvalue %46, %49[1] : !llvm.struct<(i64, ptr, ptr, ptr)>
    %51 = llvm.insertvalue %10, %50[2] : !llvm.struct<(i64, ptr, ptr, ptr)>
    %52 = llvm.insertvalue %9, %51[3] : !llvm.struct<(i64, ptr, ptr, ptr)>
    llvm.return %52 : !llvm.struct<(i64, ptr, ptr, ptr)>
  }
  llvm.func @foo() {
      llvm.return
  }
  llvm.func @bar() {
      llvm.return
  }
  func.func @main() -> i1 {
    %addr = llvm.mlir.addressof @global : !llvm.ptr
    %constZero = llvm.mlir.constant (0 : i64) : i64
    %nullPtr = llvm.mlir.null : !llvm.ptr
    %ptrptr = llvm.getelementptr inbounds %addr[%constZero, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i64, ptr, ptr, ptr)>
    %ptr = llvm.load %ptrptr : !llvm.ptr -> !llvm.ptr
    %res = llvm.icmp "ne" %nullPtr, %ptr : !llvm.ptr
    func.return %res : i1
  }
}
