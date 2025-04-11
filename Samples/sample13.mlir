func.func @main() -> i64 {
    %cEnd = arith.constant 6 : i64
    %init = arith.constant 0 : i64
    %res = scf.while (%arg1 = %init) : (i64) -> i64 {
        %c1 = arith.constant 1 : i64
        %r1 = arith.addi %arg1, %c1 : i64
        %condition = arith.cmpi slt, %r1, %cEnd : i64
        scf.condition(%condition) %r1 : i64
    } do {
        ^bb0(%arg2: i64):
        %c3 = arith.constant 3 : i64
        %next = arith.addi %arg2, %c3 : i64
        scf.yield %next : i64
    }
    func.return %res : i64
}
