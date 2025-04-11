func.func @main() -> index {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 1 : index
    %true = arith.constant true

    %arr = memref.alloc() : memref<8xi64>

    scf.for %arg1 = %c0 to %c8 step %c2 {
        %c1 = arith.constant 1 : i64
        memref.store %c1, %arr[%arg1] : memref<8xi64>
    }

    scf.for %arg1 = %c0 to %c8 step %c2 {
        %x = memref.load %arr[%arg1] : memref<8xi64>
    }

    %c4 = arith.constant 4 : index
    func.return %c4 : index
}
