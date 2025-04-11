func.func @foo(%arg0 : i64, %arg1: i64) -> i64 {
    %f = func.constant @adder : (i64, i64) -> i64
    %r = func.call_indirect %f(%arg0, %arg1) : (i64, i64) -> i64
    func.return %r : i64
}

func.func @adder(%arg0: i64, %arg1: i64) -> i64 {
    %r1 = arith.addi %arg0, %arg1 : i64
    func.return %r1 : i64
}

func.func @main() -> i64 {
    %c1 = arith.constant 64 : i64
    %c2 = arith.constant 64 : i64
    %r = func.call @foo(%c1, %c2) : (i64, i64) -> i64
    func.return %r : i64
}
