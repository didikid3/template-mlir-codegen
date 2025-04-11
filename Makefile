../Polygeist:
	git clone https://github.com/llvm/Polygeist.git ../Polygeist
	cd ../Polygeist && git checkout fd4194b099efbf2b717bd9270541e240072f0b1b \
	&& git submodule init && git submodule update

../Polygeist/build-release: ../Polygeist
	mkdir ../Polygeist/build-release
	cmake -G "Ninja" ../Polygeist/llvm-project/llvm \
		-B../Polygeist/build-release \
        -DLLVM_ENABLE_PROJECTS="clang;mlir" \
        -DLLVM_EXTERNAL_PROJECTS="polygeist" \
        -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=../Polygeist \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DCMAKE_BUILD_TYPE=RELEASE

../Polygeist/build-release/bin/cgeist: ../Polygeist/build-release
	cmake --build ../Polygeist/build-release

../llvm-project:
	git clone https://github.com/llvm/llvm-project.git ../llvm-project
	cd ../llvm-project && git checkout 5d492766a8fbfadb39c9830ebb64137edddfe0e5
	cd ../llvm-project && git apply ../mlir-codegen/llvm.patch/*

../llvm-project/build-release: ../llvm-project
	mkdir ../llvm-project/build-release
	cmake -G "Ninja" ../llvm-project/llvm \
		   -B../llvm-project/build-release \
           -DLLVM_ENABLE_PROJECTS="mlir;clang;flang" \
           -DLLVM_USE_PERF=OFF \
           -DLLVM_BUILD_EXAMPLES=OFF \
           -DLLVM_TARGETS_TO_BUILD="host" \
           -DCMAKE_BUILD_TYPE=Release \
           -DLLVM_ENABLE_ASSERTIONS=OFF \
           -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
           -DLLVM_ENABLE_FFI=ON

../llvm-project/build-release/bin/mlir-opt: ../llvm-project/build-release
	cmake --build ../llvm-project/build-release

../coremark:
	git clone https://github.com/eembc/coremark.git ../coremark

../coremark/coremark.mlir: ../coremark ../llvm-project/build-release/bin/mlir-opt
	${PWD}/../llvm-project/build-release/bin/clang -working-directory=../coremark -O0 -fno-pie -fno-pic -S -emit-llvm \
		core_list_join.c core_main.c core_matrix.c core_state.c core_util.c posix/core_portme.c \
		-DPERFORMANCE_RUN=1 -DITERATIONS=100000 -DSEED_METHOD=SEED_VOLATILE -Iposix -I. -DFLAGS_STR="\"(unknown)\""
	${PWD}/../llvm-project/build-release/bin/llvm-link ../coremark/core_list_join.ll ../coremark/core_main.ll ../coremark/core_matrix.ll \
		../coremark/core_portme.ll ../coremark/core_state.ll ../coremark/core_util.ll -o ../coremark/coremark.ll
	${PWD}/../llvm-project/build-release/bin/mlir-translate ../coremark/coremark.ll -o ../coremark/coremark.mlir --import-llvm

../PolyBenchC-4.2.1:
	git clone https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1.git ../PolyBenchC-4.2.1
	cd ../PolyBenchC-4.2.1 && git apply ../mlir-codegen/utils/polybench.patch

../lingo-db/build/lingodb-release/include:
	cd ../lingo-db && . /opt/intel/oneapi/tbb/latest/env/vars.sh && make build-release; exit 0

../mlir-codegen-lingodb/build/libmlir-codegen.a: ../lingo-db/build/lingodb-release/include
	cmake -G "Ninja" ../mlir-codegen-lingodb \
		-B../mlir-codegen-lingodb/build \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CXX_FLAGS="-fPIC -DLINGODB_LLVM" \
		-DCUSTOM_INSERT_VALUE=ON \
		-DOPTIMIZED_VARIABLE_PASSING=ON \
		-DEVICTION_STRATEGY=1 \
		-DLINGODB=${PWD}/../lingo-db/include/ \
        -DLINGODB_BUILD=${PWD}/../lingo-db/build/lingodb-release/include/ \
		-DMLIR_DIR=${PWD}/../lingo-db/build/llvm-build/lib/cmake/mlir
	cmake --build ../mlir-codegen-lingodb/build

../lingo-db/build/lingodb-release/run-sql: ../mlir-codegen-lingodb/build/libmlir-codegen.a
	. /opt/intel/oneapi/tbb/latest/env/vars.sh && cmake --build ../lingo-db/build/lingodb-release --target run-sql

spec: ../llvm-project/build-release/bin/mlir-opt
	COMPILER_BIN_DIR=${PWD}/../llvm-project/build-release/bin/ ./utils/spec-build/build-all.sh

build/mlir-codegen:
	python3 utils/common.py

build-debug/mlir-codegen:
	BUILD_TYPE=Debug BUILD_DIR=${PWD}/build-debug python3 utils/common.py

# benchmarks
prepare-x86:../llvm-project/build-release/bin/mlir-opt \
	../Polygeist/build-release/bin/cgeist \
	../PolyBenchC-4.2.1 \
	../coremark/coremark.mlir \
	# ../lingo-db/build/lingodb-release/run-sql

prepare-aarch64:../llvm-project/build-release/bin/mlir-opt \
	../coremark/coremark.mlir

benchmark-x86:
	utils/run-all.sh POLYBENCH,COREMARK,MICROBM

benchmark-aarch64:
	utils/run-all.sh COREMARK

benchmark-spec:
	utils/run-all.sh SPEC

viz:
	utils/run-all.sh VIZ
