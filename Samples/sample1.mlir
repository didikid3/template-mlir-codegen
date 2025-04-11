func.func @main() -> index {
    %c1 = arith.constant 2 : index
    %c2 = arith.constant 4 : index

    %A = memref.alloc () : memref<2x2xi32>
    %B = memref.alloc () : memref<2x2xi32>
    %C = memref.alloc () : memref<2x2xi32>

    %cf1 = arith.constant 1 : i32

    linalg.matmul ins(%A, %B : memref<2x2xi32>, memref<2x2xi32>) outs(%C: memref<2x2xi32>)
    return %c1 : index
}
