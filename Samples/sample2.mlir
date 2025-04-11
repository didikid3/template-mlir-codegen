func.func @main() -> index {
    %A = memref.alloc () : memref<1x1xi32>

    %cf1 = arith.constant 1 : i32
    %c1 = arith.constant 0 : index

    linalg.fill ins(%cf1 : i32) outs(%A : memref<1x1xi32>)
    return %c1 : index
}
