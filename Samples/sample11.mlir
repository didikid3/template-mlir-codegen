func.func @main() -> i64 {
    %c0 = arith.constant 0 : i64
    %c8 = arith.constant 8 : i64
    %c2 = arith.constant 2 : i64
    %c3 = arith.constant 3 : i64
    %true = arith.constant true

    %r1, %r2 = scf.if %true -> (i64, i64) {
        scf.yield %c0, %c8 : i64, i64
    }else{
        scf.yield %c2, %c3 : i64, i64
    }

    func.return %r1 : i64
}
