# environment variable SPEC_BENCHSPEC_FOLDER
DIR=${PWD}
rm -rf spec-programs
rm -rf spec-data
mkdir spec-programs
mkdir spec-data
cd ${SPEC_BENCHSPEC_FOLDER}/CPU/505.mcf_r/src
${DIR}/utils/spec-build/505-build.sh
cp spec505.mlir ${DIR}/spec-programs/spec505.mlir
cp ${SPEC_BENCHSPEC_FOLDER}/CPU/505.mcf_r/data/refrate/input/inp.in ${DIR}/spec-data/inp.in

cd ${SPEC_BENCHSPEC_FOLDER}/CPU/525.x264_r/src
${DIR}/utils/spec-build/525-build.sh
cp spec525.mlir ${DIR}/spec-programs/spec525.mlir
./simple-build-ldecod_r-525.sh
./ldecod_r -i ${SPEC_BENCHSPEC_FOLDER}/CPU/525.x264_r/data/refrate/input/BuckBunny.264 -o ${DIR}/spec-data/BuckBunny.yuv

cd ${SPEC_BENCHSPEC_FOLDER}/CPU/548.exchange2_r/src
python3 ${DIR}/utils/spec-build/548-build.py
cp spec548.mlir ${DIR}/spec-programs/spec548.mlir
cp ${SPEC_BENCHSPEC_FOLDER}/CPU/548.exchange2_r/data/all/input/puzzles.txt ${DIR}/spec-data/puzzles.txt

cd ${SPEC_BENCHSPEC_FOLDER}/CPU/557.xz_r/src
${DIR}/utils/spec-build/557-build.sh
cp spec557.mlir ${DIR}/spec-programs/spec557.mlir
cp ${SPEC_BENCHSPEC_FOLDER}/CPU/557.xz_r/data/all/input/* ${DIR}/spec-data/
