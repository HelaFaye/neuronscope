# llama.cpp with NeuronScope's tools (cett-dump, activation-streaming llama-server),
# built with CUDA 12.9: the newest toolkit that still compiles for Maxwell, Pascal
# and Volta (CUDA 13 starts at sm_75). Covers a Tesla M10 (sm_50) by default.
#
#   docker build -f docker/llama-cuda.Dockerfile -t neuronscope-llama:cuda .
#   docker build -f docker/llama-cuda.Dockerfile --build-arg CUDA_ARCH="61-real;86-real" -t ... .
#   docker run --rm --gpus all -v ~/models:/models -p 8080:8080 neuronscope-llama:cuda \
#       llama-server -m /models/model.gguf -ngl 99 -sm layer --host 0.0.0.0 --port 8080
#
# Needs the NVIDIA Container Toolkit on the host, and a host driver that
# supports CUDA 12.9 (R575+; R580 is the last branch for Maxwell/Pascal/Volta).
ARG CUDA_VERSION=12.9.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build
ARG CUDA_ARCH="50-real"
ARG LLAMA_REF=""
ARG JOBS=4
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY llama-tools /src/neuronscope/llama-tools
COPY scripts/build_llama_tools.sh scripts/cuda_info.py /src/neuronscope/scripts/
# No driver in a build container: the script links for the GPU host (see it).
RUN /src/neuronscope/scripts/build_llama_tools.sh --backend cuda --cuda-arch "$CUDA_ARCH" --portable \
      --server-activations --dir /src/llama.cpp -j "$JOBS" ${LLAMA_REF:+--ref "$LLAMA_REF"}

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/llama.cpp/build/bin/ /opt/llama/bin/
ENV PATH=/opt/llama/bin:$PATH LD_LIBRARY_PATH=/opt/llama/bin
EXPOSE 8080
CMD ["llama-server", "--help"]
