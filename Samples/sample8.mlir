func.func @main() -> i32 {
    %b = arith.constant 1 : i1
    %x = arith.constant 2 : i32
    scf.if %b {
      %c = arith.constant 1 : i1
    }
    return %x : i32
}
