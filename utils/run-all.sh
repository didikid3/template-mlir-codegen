#!/bin/bash
DIR=${PWD}
RUNS=10
PYTHON=python3
export MLIR_DIR=${DIR}/../llvm-project/build-release/lib/cmake/mlir
export COMPILER_BIN_DIR=${DIR}/../llvm-project/build-release/bin
mkdir -p ${DIR}/results

# Setup llvm
cd ${DIR}
make ../llvm-project/build-release/bin/mlir-opt

# PolyBench
if [[ $1 == *"POLYBENCH"* ]]; then
echo "Setup Polygeist"
# Setup Polygeist
cd ${DIR}
make ../Polygeist/build-release/bin/cgeist
echo "Setup POLYBENCH"
cd ${DIR}
make ../PolyBenchC-4.2.1
echo "Compile mlir-codegen"
cd ${DIR}
${PYTHON} utils/common.py
echo "Run POLYBENCH"
cd ${DIR}/../PolyBenchC-4.2.1
rm ${DIR}/results/polybench-polygeist.log
rm ${DIR}/results/polybench-llvm.log
for run in `seq 1 $RUNS`; do
    rm -r ${DIR}/dummy
    POLYGEIST=1 perl utilities/makefile-gen.pl . -cfg
    COMPILER_BIN_DIR=${DIR}/../Polygeist/build-release/bin TEMPLATE_LIB=${DIR}/dummy MLIR_BUILD=${DIR}/build perl utilities/run-all.pl . ${DIR}/results/polybench-polygeist.log
    #######################
    rm -r ${DIR}/dummy
    perl utilities/makefile-gen.pl . -cfg
    # FLAGS="-Xclang -disable-O0-optnone" # enable to also measure mid-end optimizations
    FLAGS=
    EXTRA_FLAGS=${FLAGS} TEMPLATE_LIB=${DIR}/dummy MLIR_BUILD=${DIR}/build perl utilities/run-all.pl . ${DIR}/results/polybench-llvm.log
done
fi

# Coremark
if [[ $1 == *"COREMARK"* ]]; then
echo "Setup COREMARK"
cd ${DIR}
make ../coremark/coremark.mlir
echo "Run COREMARK"
cd ${DIR}/Samples
RUNS=${RUNS} COREMARK=True COREMARK_DIR=${DIR}/../coremark python3 ../utils/benchmark.py 2> ${DIR}/results/coremark.log
fi

# Microbm
if [[ $1 == *"MICROBM"* ]]; then
echo "Run MICROBM"
cd ${DIR}/Samples
RUNS=${RUNS} python3 ../utils/benchmark.py 2> ${DIR}/results/benchmark.log
fi

# LingoDB
if [[ $1 == *"LINGODB"* ]]; then
source /opt/intel/oneapi/tbb/latest/env/vars.sh
echo "Setup LINGODB"
mkdir -p ${DIR}/results/lingodb
cd ${DIR}
make ../lingo-db/build/lingodb-release/run-sql
cd ${DIR}/../lingo-db
echo "Run LINGODB"
BUILD=build/lingodb-release
LINGODB_EXECUTION_MODE=MLIR  LINGODB_PARALLELISM=OFF python3 tools/scripts/benchmark-tpch.py ${BUILD} tpch 2> results-mlir.log  > ${DIR}/results/lingodb/time-mlir.log
LINGODB_EXECUTION_MODE=MLIR0 LINGODB_PARALLELISM=OFF python3 tools/scripts/benchmark-tpch.py ${BUILD} tpch 2> results-mlir0.log > ${DIR}/results/lingodb/time-mlir0.log
LINGODB_EXECUTION_MODE=SPEED LINGODB_PARALLELISM=OFF python3 tools/scripts/benchmark-tpch.py ${BUILD} tpch 2> results-speed.log > ${DIR}/results/lingodb/time-speed.log
LINGODB_EXECUTION_MODE=EXTREME_CHEAP LINGODB_PARALLELISM=OFF python3 tools/scripts/benchmark-tpch.py ${BUILD} tpch 2> results-cheap.log > ${DIR}/results/lingodb/time-cheap.log
fi

# Spec
if [[ $1 == *"SPEC"* ]]; then
cd ${DIR}
echo "Run SPEC"
cd spec-data && SPEC_PROGRAMS=${DIR}/spec-programs/ RUNS=3 python3 ../utils/spec.py 2> ../results/spec.log
fi

# visualization
if [[ $1 == *"VIZ"* ]]; then
cd ${DIR}
${PYTHON} utils/viz.py
fi
echo "Done."
