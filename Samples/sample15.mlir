func.func @main() -> i64 {
    %c20 = arith.constant 20 : i64
    %c1 = arith.constant 1 : i64
    %x0 = arith.subi %c20, %c1 : i64
    %x1 = arith.subi %x0, %c1 : i64
    %x2 = arith.subi %x1, %c1 : i64
    %x3 = arith.subi %x2, %c1 : i64
    %x4 = arith.subi %x3, %c1 : i64
    %x5 = arith.subi %x4, %c1 : i64
    %x6 = arith.subi %x5, %c1 : i64
    %x7 = arith.subi %x6, %c1 : i64
    %x8 = arith.subi %x7, %c1 : i64
    func.return %x8 : i64
}
