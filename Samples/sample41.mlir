module {
  func.func @main() -> i64 {
    %1 = llvm.mlir.constant(5 : i64) : i64
    %2 = llvm.mlir.constant(6 : i64) : i64
    %29 = llvm.mlir.constant(4 : i64) : i64
    %39 = llvm.mlir.constant(1 : i8) : i8
    %40 = llvm.mlir.constant(9 : i8) : i8
    %41 = llvm.mlir.constant(2 : i8) : i8
    %42 = llvm.mlir.constant(20180515 : i32) : i32
    %43 = llvm.mlir.undef : !llvm.ptr
    %44 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
    %31 = llvm.mlir.undef : !llvm.array<3 x i64>
    %32 = llvm.insertvalue %1, %31[0] : !llvm.array<3 x i64>
    %33 = llvm.insertvalue %1, %32[1] : !llvm.array<3 x i64>
    %34 = llvm.insertvalue %1, %33[2] : !llvm.array<3 x i64>
    %35 = llvm.insertvalue %2, %33[2] : !llvm.array<3 x i64>
    %36 = llvm.mlir.undef : !llvm.array<2 x array<3 x i64>>
    %37 = llvm.insertvalue %35, %36[0] : !llvm.array<2 x array<3 x i64>>
    %38 = llvm.insertvalue %34, %37[1] : !llvm.array<2 x array<3 x i64>>
    %45 = llvm.insertvalue %43, %44[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
    %46 = llvm.insertvalue %29, %45[1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
    %47 = llvm.insertvalue %42, %46[2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
    %48 = llvm.insertvalue %41, %47[3] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
    %49 = llvm.insertvalue %40, %48[4] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
    %50 = llvm.insertvalue %39, %49[5] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
    %51 = llvm.insertvalue %39, %50[6] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
    %52 = llvm.insertvalue %38, %51[7] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
    %ret = llvm.extractvalue %52[7, 0, 2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
    func.return %ret : i64
  }
}
