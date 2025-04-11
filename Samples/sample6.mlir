func.func @main(%arg0: i64) -> i64 {
    %c2 = arith.constant 2 : i64
    %c3 = arith.subi %c2, %arg0 : i64
    func.return %c3 : i64
}
