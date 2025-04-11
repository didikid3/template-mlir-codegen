FROM ubuntu:jammy

RUN apt-get update
RUN apt-get -y --no-install-recommends install \
    build-essential cmake git ninja-build python3 \
    python3-dev python3-pip libffi-dev libghc-terminfo-dev zlib1g-dev gpg wget
RUN git config --global http.sslverify false
RUN pip3 install matplotlib numpy pandas pybind11

# install intel tbb (only required for lingo-db, only on x86_64)
RUN uname -m | grep x86_64 > /dev/null  && \
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | tee /etc/apt/sources.list.d/oneAPI.list && apt update; \
    exit 0
RUN uname -m | grep x86_64 > /dev/null && \
    apt-get -y --no-install-recommends install intel-oneapi-tbb-2021.11 intel-oneapi-tbb-devel-2021.11; \
    exit 0

ENV CMAKE_BUILD_PARALLEL_LEVEL=2

WORKDIR /src/mlir-codegen

# {podman|docker|...} build . -t mlir-codegen-build
# {podman|docker|...} run -it --rm --mount type=bind,source=${PWD}/..,target=src mlir-codegen-build bash
# cd $(dirname "${PWD}") && utils/run-all.sh POLYBENCH,COREMARK,MICROBM,VIZ