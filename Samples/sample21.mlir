module {
  func.func @main(%arg0: i64) -> i32 {
    %arg32 = llvm.trunc %arg0 : i64 to i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(8 : i32) : i32
    %4 = llvm.mlir.constant(0 : i64) : i64
    %5 = llvm.mlir.constant(32 : i32) : i32
    %6 = llvm.mlir.constant(0 : i64) : i64
    %7 = llvm.mlir.constant(4294967264 : i64) : i64
    %8 = llvm.mlir.constant(0 : i64) : i64
    %9 = llvm.mlir.constant(dense<1> : vector<16xi8>) : vector<16xi8>
    %10 = llvm.mlir.constant(16 : i64) : i64
    %11 = llvm.mlir.constant(dense<1> : vector<16xi8>) : vector<16xi8>
    %12 = llvm.mlir.constant(32 : i64) : i64
    %13 = llvm.mlir.constant(24 : i64) : i64
    %14 = llvm.mlir.constant(0 : i64) : i64
    %15 = llvm.mlir.constant(4294967288 : i64) : i64
    %16 = llvm.mlir.constant(dense<1> : vector<8xi8>) : vector<8xi8>
    %17 = llvm.mlir.constant(8 : i64) : i64
    %18 = llvm.mlir.constant(1 : i8) : i8
    %19 = llvm.mlir.constant(1 : i64) : i64
    %20 = llvm.mlir.constant(4 : i32) : i32
    %21 = llvm.mlir.constant(0 : i32) : i32
    %22 = llvm.mlir.constant(2 : i64) : i64
    %23 = llvm.mlir.constant(4 : i32) : i32
    %24 = llvm.mlir.constant(0 : i32) : i32
    %25 = llvm.mlir.constant(0 : i8) : i8
    %26 = llvm.mlir.constant(1 : i32) : i32
    %27 = llvm.mlir.constant(0 : i8) : i8
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.add %arg32, %0  : i32
    %30 = llvm.sext %29 : i32 to i64
    %31 = func.call @malloc(%30) : (i64) -> !llvm.ptr
    %32 = llvm.icmp "eq" %29, %1 : i32
    llvm.cond_br %32, ^bb13(%2 : i32), ^bb1
  ^bb1:  // pred: ^bb0
    %33 = llvm.zext %29 : i32 to i64
    %34 = llvm.icmp "ult" %29, %3 : i32
    llvm.cond_br %34, ^bb10(%4 : i64), ^bb2
  ^bb2:  // pred: ^bb1
    %35 = llvm.icmp "ult" %29, %5 : i32
    llvm.cond_br %35, ^bb7(%6 : i64), ^bb3
  ^bb3:  // pred: ^bb2
    %36 = llvm.and %33, %7  : i64
    llvm.br ^bb4(%8 : i64)
  ^bb4(%37: i64):  // 2 preds: ^bb3, ^bb4
    %38 = llvm.getelementptr %31[%37] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %9, %38 : vector<16xi8>, !llvm.ptr
    %39 = llvm.getelementptr %38[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %11, %39 : vector<16xi8>, !llvm.ptr
    %40 = llvm.add %37, %12  : i64
    %41 = llvm.icmp "eq" %40, %36 : i64
    llvm.cond_br %41, ^bb5, ^bb4(%40 : i64)
  ^bb5:  // pred: ^bb4
    %42 = llvm.icmp "eq" %36, %33 : i64
    llvm.cond_br %42, ^bb11, ^bb6
  ^bb6:  // pred: ^bb5
    %43 = llvm.and %33, %13  : i64
    %44 = llvm.icmp "eq" %43, %14 : i64
    llvm.cond_br %44, ^bb10(%36 : i64), ^bb7(%36 : i64)
  ^bb7(%45: i64):  // 2 preds: ^bb2, ^bb6
    %46 = llvm.and %33, %15  : i64
    llvm.br ^bb8(%45 : i64)
  ^bb8(%47: i64):  // 2 preds: ^bb7, ^bb8
    %48 = llvm.getelementptr %31[%47] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %16, %48 : vector<8xi8>, !llvm.ptr
    %49 = llvm.add %47, %17  : i64
    %50 = llvm.icmp "eq" %49, %46 : i64
    llvm.cond_br %50, ^bb9, ^bb8(%49 : i64)
  ^bb9:  // pred: ^bb8
    %51 = llvm.icmp "eq" %46, %33 : i64
    llvm.cond_br %51, ^bb11, ^bb10(%46 : i64)
  ^bb10(%52: i64):  // 3 preds: ^bb1, ^bb6, ^bb9
    llvm.br ^bb12(%52 : i64)
  ^bb11:  // 3 preds: ^bb5, ^bb9, ^bb12
    %53 = llvm.icmp "slt" %arg32, %20 : i32
    llvm.cond_br %53, ^bb13(%21 : i32), ^bb14(%22, %23, %24 : i64, i32, i32)
  ^bb12(%54: i64):  // 2 preds: ^bb10, ^bb12
    %55 = llvm.getelementptr %31[%54] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %18, %55 : i8, !llvm.ptr
    %56 = llvm.add %54, %19  : i64
    %57 = llvm.icmp "eq" %56, %33 : i64
    llvm.cond_br %57, ^bb11, ^bb12(%56 : i64)
  ^bb13(%58: i32):  // 3 preds: ^bb0, ^bb11, ^bb18
    func.return %58 : i32
  ^bb14(%59: i64, %60: i32, %61: i32):  // 2 preds: ^bb11, ^bb18
    %62 = llvm.getelementptr %31[%59] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %63 = llvm.load %62 : !llvm.ptr -> i8
    %64 = llvm.icmp "eq" %63, %25 : i8
    llvm.cond_br %64, ^bb18(%61 : i32), ^bb15
  ^bb15:  // pred: ^bb14
    %65 = llvm.add %61, %26  : i32
    %66 = llvm.icmp "sgt" %60, %arg32 : i32
    llvm.cond_br %66, ^bb18(%65 : i32), ^bb16
  ^bb16:  // pred: ^bb15
    %67 = llvm.zext %60 : i32 to i64
    llvm.br ^bb17(%67 : i64)
  ^bb17(%68: i64):  // 2 preds: ^bb16, ^bb17
    %69 = llvm.getelementptr %31[%68] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %27, %69 : i8, !llvm.ptr
    %70 = llvm.add %68, %59  : i64
    %71 = llvm.trunc %70 : i64 to i32
    %72 = llvm.icmp "sgt" %71, %arg32 : i32
    llvm.cond_br %72, ^bb18(%65 : i32), ^bb17(%70 : i64)
  ^bb18(%73: i32):  // 3 preds: ^bb14, ^bb15, ^bb17
    %74 = llvm.add %59, %28  : i64
    %75 = llvm.trunc %74 : i64 to i32
    %tmp = llvm.trunc %74 : i64 to i32
    %76 = llvm.mul %75, %tmp  : i32
    %77 = llvm.icmp "sgt" %76, %arg32 : i32
    llvm.cond_br %77, ^bb13(%73 : i32), ^bb14(%74, %76, %73 : i64, i32, i32)
  }
  func.func private @malloc(i64) -> !llvm.ptr
}
