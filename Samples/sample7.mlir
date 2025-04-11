func.func @main() -> index {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %A = memref.alloc (%c2, %c3) : memref<?x?xi32>
    return %c2 : index
}
