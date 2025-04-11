${COMPILER_BIN_DIR}/clang implicit.c mcf.c mcfutil.c output.c pbeampp.c pbla.c pflowup.c psimplex.c pstart.c readmin.c treeup.c spec_qsort/spec_qsort.c spec_qsort/spec_qsort.c -DSPEC -Ispec_qsort -c -emit-llvm -O0 -Xclang -disable-O0-optnone
${COMPILER_BIN_DIR}/llvm-link -S implicit.bc mcf.bc mcfutil.bc output.bc pbeampp.bc pbla.bc pflowup.bc psimplex.bc pstart.bc readmin.bc spec_qsort.bc treeup.bc -o spec505.ll
${COMPILER_BIN_DIR}/mlir-translate --import-llvm spec505.ll -o spec505.mlir
