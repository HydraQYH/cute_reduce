#!/bin/bash

set -ex
# -O3 -DNDEBUG
# -O2 -g -lineinfo -DNDEBUG
# -O0 -G -g

nvcc -forward-unknown-to-host-compiler \
  -O3 -DNDEBUG -std=c++17 \
  -I${CUTLASS_DIR}/cutlass/include \
  "--generate-code=arch=compute_90a,code=[sm_90a]" \
  "--generate-code=arch=compute_90a,code=[compute_90a]" \
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
  -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
  --expt-relaxed-constexpr \
  -ftemplate-backtrace-limit=0 \
  -o native_block_reduce \
  native_block_reduce.cu

nvcc -forward-unknown-to-host-compiler \
  -O3 -DNDEBUG -std=c++17 \
  -I${CUTLASS_DIR}/cutlass/include \
  "--generate-code=arch=compute_90a,code=[sm_90a]" \
  "--generate-code=arch=compute_90a,code=[compute_90a]" \
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
  -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
  --expt-relaxed-constexpr \
  -ftemplate-backtrace-limit=0 \
  -o tmp_pipe_block_reduce \
  tmp_pipe_block_reduce.cu
