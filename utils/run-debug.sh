#!/bin/bash
DIR=${PWD}

PYTHON=python3
export MLIR_DIR=${DIR}/../llvm-project/build-release/lib/cmake/mlir
export COMPILER_BIN_DIR=${DIR}/../llvm-project/build-release/bin
export BUILD_TYPE=Debug

COREMARK_DIR=${DIR}/../coremark ${PYTHON} utils/debug-benchmark.py