func.func @main() -> i32 {
    %c1 = arith.constant -1 : i32
    %c2 = arith.constant -2 : i32
    %c3 = arith.addi %c2, %c1 : i32
    func.return %c3 : i32
}
