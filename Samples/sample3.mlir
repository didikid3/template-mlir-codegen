func.func @main() -> index {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index

    %A = memref.alloc (%c2, %c4) : memref<?x?xi32>
    return %c2 : index
}
