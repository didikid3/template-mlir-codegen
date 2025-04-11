module{
    func.func @sieve(%arg0: i32, %arg1: memref<?xi32>) {
        %c1_i32 = arith.constant 1 : i32
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c0_i32 = arith.constant 0 : i32
        %0 = arith.sitofp %arg0 : i32 to f32
        %1 = math.sqrt %0 : f32
        %2 = arith.fptosi %1 : f32 to i32
        %3 = arith.addi %2, %c1_i32 : i32
        %4 = arith.index_cast %3 : i32 to index
        %5 = arith.addi %arg0, %c1_i32 : i32
        %6 = arith.index_cast %5 : i32 to index
        scf.for %arg2 = %c2 to %4 step %c1 {
            %7 = arith.index_cast %arg2 : index to i32
            %8 = memref.load %arg1[%arg2] : memref<?xi32>
            %9 = arith.cmpi ne, %8, %c0_i32 : i32
            scf.if %9 {
                %10 = arith.muli %7, %7 : i32
                %11 = arith.index_cast %10 : i32 to index
                scf.for %arg3 = %11 to %6 step %arg2 {
                %12 = arith.subi %arg3, %11 : index
                %13 = arith.divui %12, %arg2 : index
                %14 = arith.muli %13, %arg2 : index
                %15 = arith.addi %11, %14 : index
                memref.store %c0_i32, %arg1[%15] : memref<?xi32>
                }
            }
        }
        return
    }

    func.func @main(%n : index) -> i32 {
        %1 = arith.constant 1 : i32
        %index1 = index.constant 1
        %n1 = index.add %n, %index1
        %data = memref.alloc (%n1) : memref<?xi32>
        linalg.fill ins(%1 : i32) outs(%data : memref<?xi32>)
        %ni32 = arith.index_cast %n : index to i32
        func.call @sieve(%ni32, %data) : (i32, memref<?xi32>) -> ()
        %42 = arith.constant 42 : index
        %r = memref.load %data[%42] : memref<?xi32>
        func.return %r : i32
    }
}
