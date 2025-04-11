module {
  func.func @main() -> i1 {
    %false = llvm.mlir.constant(0 : i1) : i1
    %29 = llvm.mlir.undef : vector<2xi64>
    %30 = llvm.mlir.constant(dense<[4000, 0]> : vector<2xi64>) : vector<2xi64>
    %31 = llvm.mlir.constant(dense<0> : vector<2xi64>) : vector<2xi64>
    %32 = llvm.mlir.constant(dense<-8> : vector<2xi64>) : vector<2xi64>
    llvm.cond_br %false, ^bb7(%29, %29, %30, %31 : vector<2xi64>, vector<2xi64>, vector<2xi64>, vector<2xi64>), ^bb5
  ^bb5:
    llvm.cond_br %false, ^bb2, ^bb3
  ^bb2:
    llvm.br ^bbend
  ^bb3:
    llvm.br ^bbend
  ^bb7(%123: vector<2xi64>, %124: vector<2xi64>, %125: vector<2xi64>, %126: vector<2xi64>):
    llvm.br ^bbend
  ^bbend:
    func.return %false : i1
  }
}
