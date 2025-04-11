func.func private @PRINT_STACKFRAME()
func.func @main(%arg: i64) -> i32 {
    %arg0 = arith.trunci %arg : i64 to i32
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c10000001 = arith.constant 10000001 : index
    %c0 = arith.constant 0 : index
    %c1_i64 = arith.constant 1 : i64
    %false = arith.constant false
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<10000001xi32>
    %alloc_0 = memref.alloc() : memref<10000001xi64>
    %alloc_1 = memref.alloc() : memref<10000001xi64>
    scf.for %arg1 = %c0 to %c10000001 step %c1 {
      memref.store %c1_i32, %alloc[%arg1] : memref<10000001xi32>
    }
    memref.store %c0_i32, %alloc[%c1] : memref<10000001xi32>
    memref.store %c0_i32, %alloc[%c0] : memref<10000001xi32>
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.extsi %arg0 : i32 to i64
    %2 = scf.for %arg1 = %c2 to %0 step %c1 iter_args(%arg2 = %c0_i32) -> (i32) {
      %3 = arith.index_cast %arg1 : index to i64
      %4 = memref.load %alloc[%arg1] : memref<10000001xi32>
      %5 = arith.cmpi ne, %4, %c0_i32 : i32
      %6 = scf.if %5 -> (i32) {
        %9 = arith.index_cast %arg2 : i32 to index
        memref.store %3, %alloc_0[%9] : memref<10000001xi64>
        %10 = arith.addi %arg2, %c1_i32 : i32
        memref.store %3, %alloc_1[%arg1] : memref<10000001xi64>
        scf.yield %10 : i32
      } else {
        scf.yield %arg2 : i32
      }
      %7 = arith.extsi %6 : i32 to i64
      %8 = scf.while (%arg3 = %c0_i64) : (i64) -> i64 {
        %9 = arith.cmpi slt, %arg3, %7 : i64
        %10:2 = scf.if %9 -> (i1, i64) {
          %11 = arith.index_cast %arg3 : i64 to index
          %12 = memref.load %alloc_0[%11] : memref<10000001xi64>
          %13 = arith.muli %3, %12 : i64
          %14 = arith.cmpi slt, %13, %1 : i64
          %15:2 = scf.if %14 -> (i1, i64) {
            %16 = memref.load %alloc_1[%arg1] : memref<10000001xi64>
            %17 = arith.cmpi sle, %12, %16 : i64
            %18 = scf.if %17 -> (i64) {
              %19 = arith.index_cast %13 : i64 to index
              memref.store %c0_i32, %alloc[%19] : memref<10000001xi32>
              memref.store %12, %alloc_1[%19] : memref<10000001xi64>
              %20 = arith.addi %arg3, %c1_i64 : i64
              scf.yield %20 : i64
            } else {
              scf.yield %arg3 : i64
            }
            scf.yield %17, %18 : i1, i64
          } else {
            scf.yield %false, %arg3 : i1, i64
          }
          scf.yield %15#0, %15#1 : i1, i64
        } else {
          scf.yield %false, %arg3 : i1, i64
        }
        scf.condition(%10#0) %10#1 : i64
      } do {
      ^bb0(%arg3: i64):
        scf.yield %arg3 : i64
      }
      scf.yield %6 : i32
    }
    return %2 : i32
}
