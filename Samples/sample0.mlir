func.func @main() -> index {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c-1_i8 = arith.constant -1 : i8
    return %c2 : index
}
