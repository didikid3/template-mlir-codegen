module{
    func.func @fib(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
        %c1_i32 = arith.constant 1 : i32
        %c2_i32 = arith.constant 2 : i32
        %0 = llvm.mlir.undef : i32
        %1 = arith.cmpi sle, %arg0, %c2_i32 : i32
        %2 = arith.cmpi sgt, %arg0, %c2_i32 : i32
        %3 = scf.if %2 -> (i32) {
            %4 = arith.subi %arg0, %c1_i32 : i32
            %5 = func.call @fib(%4) : (i32) -> i32
            %6 = arith.subi %arg0, %c2_i32 : i32
            %7 = func.call @fib(%6) : (i32) -> i32
            %8 = arith.addi %5, %7 : i32
            scf.yield %8 : i32
        } else {
            %4 = arith.select %1, %c1_i32, %0 : i32
            scf.yield %4 : i32
        }
        return %3 : i32
    }

    func.func @main(%n : i32) -> i32{
        %r = func.call @fib(%n) : (i32) -> i32
        func.return %r : i32
    }
}
