func.func private @PRINT_STACKFRAME()

func.func @main() -> i64 {
    %c = arith.constant 0 : i64
    call @PRINT_STACKFRAME() : () -> ()
    func.return %c : i64
}
