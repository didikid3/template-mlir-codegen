module{
    llvm.func @malloc(i64) -> !llvm.ptr
    llvm.func @quick(%arg0: !llvm.ptr {llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}) {
        %0 = llvm.mlir.constant(1 : i32) : i32
        %1 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
        %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
        %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
        %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
        %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
        %6 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
        %7 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
        llvm.store %arg0, %1 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
        llvm.store %arg1, %2 {alignment = 4 : i64} : i32, !llvm.ptr
        llvm.store %arg2, %3 {alignment = 4 : i64} : i32, !llvm.ptr
        %8 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
        %9 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
        %10 = llvm.icmp "sge" %8, %9 : i32
        llvm.cond_br %10, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
        llvm.br ^bb9
    ^bb2:  // pred: ^bb0
        %11 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
        %12 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
        %13 = llvm.sext %12 : i32 to i64
        %14 = llvm.getelementptr inbounds %11[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %15 = llvm.load %14 {alignment = 4 : i64} : !llvm.ptr -> i32
        llvm.store %15, %4 {alignment = 4 : i64} : i32, !llvm.ptr
        %16 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
        llvm.store %16, %5 {alignment = 4 : i64} : i32, !llvm.ptr
        %17 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
        llvm.store %17, %6 {alignment = 4 : i64} : i32, !llvm.ptr
        llvm.br ^bb3
    ^bb3:  // 2 preds: ^bb2, ^bb7
        %18 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
        %19 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
        %20 = llvm.icmp "sle" %18, %19 : i32
        llvm.cond_br %20, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
        %21 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
        %22 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
        %23 = llvm.sext %22 : i32 to i64
        %24 = llvm.getelementptr inbounds %21[%23] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %25 = llvm.load %24 {alignment = 4 : i64} : !llvm.ptr -> i32
        %26 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
        %27 = llvm.icmp "slt" %25, %26 : i32
        llvm.cond_br %27, ^bb5, ^bb6
    ^bb5:  // pred: ^bb4
        %28 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
        %29 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
        %30 = llvm.sext %29 : i32 to i64
        %31 = llvm.getelementptr inbounds %28[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %32 = llvm.load %31 {alignment = 4 : i64} : !llvm.ptr -> i32
        llvm.store %32, %7 {alignment = 4 : i64} : i32, !llvm.ptr
        %33 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
        %34 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
        %35 = llvm.sext %34 : i32 to i64
        %36 = llvm.getelementptr inbounds %33[%35] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %37 = llvm.load %36 {alignment = 4 : i64} : !llvm.ptr -> i32
        %38 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
        %39 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
        %40 = llvm.sext %39 : i32 to i64
        %41 = llvm.getelementptr inbounds %38[%40] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        llvm.store %37, %41 {alignment = 4 : i64} : i32, !llvm.ptr
        %42 = llvm.load %7 {alignment = 4 : i64} : !llvm.ptr -> i32
        %43 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
        %44 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
        %45 = llvm.sext %44 : i32 to i64
        %46 = llvm.getelementptr inbounds %43[%45] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        llvm.store %42, %46 {alignment = 4 : i64} : i32, !llvm.ptr
        %47 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
        %48 = llvm.add %47, %0  : i32
        llvm.store %48, %5 {alignment = 4 : i64} : i32, !llvm.ptr
        llvm.br ^bb6
    ^bb6:  // 2 preds: ^bb4, ^bb5
        llvm.br ^bb7
    ^bb7:  // pred: ^bb6
        %49 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
        %50 = llvm.add %49, %0  : i32
        llvm.store %50, %6 {alignment = 4 : i64} : i32, !llvm.ptr
        llvm.br ^bb3
    ^bb8:  // pred: ^bb3
        %51 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
        %52 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
        %53 = llvm.sext %52 : i32 to i64
        %54 = llvm.getelementptr inbounds %51[%53] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %55 = llvm.load %54 {alignment = 4 : i64} : !llvm.ptr -> i32
        %56 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
        %57 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
        %58 = llvm.sext %57 : i32 to i64
        %59 = llvm.getelementptr inbounds %56[%58] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        llvm.store %55, %59 {alignment = 4 : i64} : i32, !llvm.ptr
        %60 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
        %61 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
        %62 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
        %63 = llvm.sext %62 : i32 to i64
        %64 = llvm.getelementptr inbounds %61[%63] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        llvm.store %60, %64 {alignment = 4 : i64} : i32, !llvm.ptr
        %65 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
        %66 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
        %67 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
        %68 = llvm.sub %67, %0  : i32
        llvm.call @quick(%65, %66, %68) : (!llvm.ptr, i32, i32) -> ()
        %69 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
        %70 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
        %71 = llvm.add %70, %0  : i32
        %72 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
        llvm.call @quick(%69, %71, %72) : (!llvm.ptr, i32, i32) -> ()
        llvm.br ^bb9
    ^bb9:  // 2 preds: ^bb1, ^bb8
        llvm.return
    }

    llvm.func @fill(%arg0: !llvm.ptr {llvm.noundef}, %arg1: i32 {llvm.noundef}) {
        %0 = llvm.mlir.constant(1 : i32) : i32
        %1 = llvm.mlir.constant(0 : i32) : i32
        %2 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
        %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
        %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
        llvm.store %arg0, %2 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
        llvm.store %arg1, %3 {alignment = 4 : i64} : i32, !llvm.ptr
        llvm.store %1, %4 {alignment = 4 : i64} : i32, !llvm.ptr
        llvm.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb3
        %5 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
        %6 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
        %7 = llvm.icmp "slt" %5, %6 : i32
        llvm.cond_br %7, ^bb2, ^bb4
    ^bb2:  // pred: ^bb1
        %8 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
        %9 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
        %10 = llvm.sub %8, %9  : i32
        %11 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
        %12 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
        %13 = llvm.sext %12 : i32 to i64
        %14 = llvm.getelementptr inbounds %11[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        llvm.store %10, %14 {alignment = 4 : i64} : i32, !llvm.ptr
        llvm.br ^bb3
    ^bb3:  // pred: ^bb2
        %15 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
        %16 = llvm.add %15, %0  : i32
        llvm.store %16, %4 {alignment = 4 : i64} : i32, !llvm.ptr
        llvm.br ^bb1
    ^bb4:  // pred: ^bb1
        llvm.return
    }

    llvm.func @main(%n: i32) -> i32 {
        %4 = llvm.mlir.constant (4 : i32) : i32
        %0 = llvm.mlir.constant (0 : i32) : i32
        %1 = llvm.mlir.constant (1 : i32) : i32
        %42 = llvm.mlir.constant (42 : i32) : i32
        %nsize = llvm.mul %n, %4 : i32
        %n1 = llvm.sub %n, %1 : i32
        %nsize_ = llvm.zext %nsize : i32 to i64
        %data = llvm.call @malloc(%nsize_) : (i64) -> !llvm.ptr
        llvm.call @fill(%data, %n1) : (!llvm.ptr, i32) -> ()
        llvm.call @quick(%data, %0, %n1) : (!llvm.ptr, i32, i32) -> ()
        %checkPtr = llvm.getelementptr %data[%42] : (!llvm.ptr, i32) -> !llvm.ptr, i32
        %r = llvm.load %checkPtr : !llvm.ptr -> i32
        llvm.return %r : i32
    }
}
