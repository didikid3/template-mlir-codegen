func.func @main(%n: index) -> i64 {
  %arr = memref.alloc(%n) : memref<?xi64>
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c1_ = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  memref.store %c1_, %arr[%c0] : memref<?xi64>
  memref.store %c1_, %arr[%c1] : memref<?xi64>
  scf.for %i = %c2 to %n step %c1{
    %i1 = arith.subi %i, %c1 : index
    %i2 = arith.subi %i, %c2 : index
    %a = memref.load %arr[%i1] : memref<?xi64>
    %b = memref.load %arr[%i2] : memref<?xi64>
    %fib = arith.addi %a, %b : i64
    memref.store %fib, %arr[%i] : memref<?xi64>
  }

  %n1 = arith.subi %n, %c1 : index
  %ret = memref.load %arr[%n1] : memref<?xi64>
  func.return %ret : i64
}
