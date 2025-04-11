func.func @main() -> i32 {
    %b = arith.constant 1 : i1
    %x = scf.if %b -> i32 {
        %x_true = arith.constant 2 : i32
        scf.yield %x_true : i32
    } else {
        %x_false = arith.constant 3 : i32
        scf.yield %x_false : i32
    }
    return %x : i32
}
