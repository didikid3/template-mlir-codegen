module{
    llvm.func @fib(%arg0: i32 {llvm.noundef}) -> (i32 {llvm.noundef}) {
        %0 = llvm.mlir.constant(1 : i32) : i32
        %1 = llvm.mlir.constant(2 : i32) : i32
        %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
        %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
        llvm.store %arg0, %4 {alignment = 4 : i64} : i32, !llvm.ptr
        %5 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
        %6 = llvm.icmp "sle" %5, %1 : i32
        llvm.cond_br %6, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
        llvm.store %0, %3 {alignment = 4 : i64} : i32, !llvm.ptr
        llvm.br ^bb3
    ^bb2:  // pred: ^bb0
        %7 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
        %8 = llvm.sub %7, %0  : i32
        %9 = llvm.call @fib(%8) : (i32) -> i32
        %10 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
        %11 = llvm.sub %10, %1  : i32
        %12 = llvm.call @fib(%11) : (i32) -> i32
        %13 = llvm.add %9, %12  : i32
        llvm.store %13, %3 {alignment = 4 : i64} : i32, !llvm.ptr
        llvm.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
        %14 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
        llvm.return %14 : i32
    }

    func.func @main(%n : i32) -> i32{
        %r = llvm.call @fib(%n) : (i32) -> i32
        func.return %r : i32
    }
}
