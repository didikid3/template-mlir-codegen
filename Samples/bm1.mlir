module{
    func.func @quick(%arg0: memref<?xi32>, %arg1: i32, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
        %c1 = arith.constant 1 : index
        %c-1_i32 = arith.constant -1 : i32
        %c1_i32 = arith.constant 1 : i32
        %0 = arith.index_cast %arg2 : i32 to index
        %1 = arith.cmpi slt, %arg1, %arg2 : i32
        scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<?xi32>
        %3 = arith.addi %arg2, %c1_i32 : i32
        %4 = arith.index_cast %3 : i32 to index
        %5 = arith.index_cast %arg1 : i32 to index
        %6 = scf.for %arg3 = %5 to %4 step %c1 iter_args(%arg4 = %arg1) -> (i32) {
            %11 = memref.load %arg0[%arg3] : memref<?xi32>
            %12 = arith.cmpi slt, %11, %2 : i32
            %13 = scf.if %12 -> (i32) {
            %14 = arith.index_cast %arg4 : i32 to index
            %15 = memref.load %arg0[%14] : memref<?xi32>
            %16 = memref.load %arg0[%arg3] : memref<?xi32>
            memref.store %16, %arg0[%14] : memref<?xi32>
            memref.store %15, %arg0[%arg3] : memref<?xi32>
            %17 = arith.addi %arg4, %c1_i32 : i32
            scf.yield %17 : i32
            } else {
            scf.yield %arg4 : i32
            }
            scf.yield %13 : i32
        }
        %7 = arith.index_cast %6 : i32 to index
        %8 = memref.load %arg0[%7] : memref<?xi32>
        memref.store %8, %arg0[%0] : memref<?xi32>
        memref.store %2, %arg0[%7] : memref<?xi32>
        %9 = arith.addi %6, %c-1_i32 : i32
        func.call @quick(%arg0, %arg1, %9) : (memref<?xi32>, i32, i32) -> ()
        %10 = arith.addi %6, %c1_i32 : i32
        func.call @quick(%arg0, %10, %arg2) : (memref<?xi32>, i32, i32) -> ()
        }
        return
    }
    func.func @fill(%arg0: memref<?xi32>, %arg1: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %0 = arith.index_cast %arg1 : i32 to index
        scf.for %arg2 = %c0 to %0 step %c1 {
            %1 = arith.index_cast %arg2 : index to i32
            %2 = arith.subi %arg1, %1 : i32
            memref.store %2, %arg0[%arg2] : memref<?xi32>
        }
        return
    }
    func.func @main(%n : i32) -> i32 {
        %0 = arith.constant 0 : i32
        %1 = arith.constant 1 : i32
        %n1 = arith.subi %n, %1 : i32
        %indx = arith.index_cast %n : i32 to index
        %data = memref.alloc (%indx) : memref<?xi32>
        func.call @fill(%data, %n1) : (memref<?xi32>, i32) -> ()
        func.call @quick(%data, %0, %n1) : (memref<?xi32>, i32, i32) -> ()
        %42 = arith.constant 42 : index
        %r = memref.load %data[%42] : memref<?xi32>
        func.return %r : i32
    }
}
