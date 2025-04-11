func.func @main() -> index {
    %arg0 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = arith.addi %arg0, %c2 : index
    %1 = scf.while (%arg1 = %0) : (index) -> index {
        %2 = arith.cmpi slt, %arg1, %c0 : index
        scf.condition(%2) %arg1 : index
    } do {
    ^bb0(%arg2 : index):
        scf.yield %c2 : index
    }
    func.return %1 : index
}
