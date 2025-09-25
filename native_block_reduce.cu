#include <cstdio>
#include <random>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>

#define CUDA_CHECK(cmd)                                             \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

constexpr int warp_size = 32;

template<typename T>
CUTLASS_HOST_DEVICE void print(T& t) {
  cute::print(t);
  printf("\n");
}

template<typename T>
CUTLASS_HOST_DEVICE void print(T&& t) {
  cute::print(t);
  printf("\n");
}

template<typename T, typename TensorS, typename TiledCopy>
__global__ void cute_reduce(TensorS tensor_s, TiledCopy tiled_copy, T* output) {
  __shared__ T warp_local_reduce_sum[32];
  using namespace cute;
  namespace cg = cooperative_groups;
  using Tiler_MN = typename TiledCopy::Tiler_MN;
  cg::grid_group g = cg::this_grid();
  cg::thread_block tb = cg::this_thread_block();

  auto shape_s = shape(tensor_s);
  Tensor tensor_c = make_identity_tensor(shape_s);
  Tensor tensor_p = cute::lazy::transform(tensor_c, [&](auto c) { return elem_less(c, shape_s); });

  auto tiled_tensor_s = cute::tiled_divide(tensor_s, Tiler_MN{}); // (Tiler_MN, tile_count)
  auto tiled_tensor_p = cute::tiled_divide(tensor_p, Tiler_MN{});

  auto tile_count = size<1>(tiled_tensor_s);
  T acc(0.0f);

#pragma unroll
  for (auto cta_id = g.block_rank(); cta_id < tile_count; cta_id += g.num_blocks()) {
    auto cta_tile_s = cute::tensor<0>(tiled_tensor_s(make_coord(_), cta_id));
    auto cta_tile_p = cute::tensor<0>(tiled_tensor_p(make_coord(_), cta_id));

    ThrCopy thr_copy = tiled_copy.get_thread_slice(tb.thread_rank());
    Tensor thr_tile_s = thr_copy.partition_S(cta_tile_s);
    Tensor thr_tile_p = thr_copy.partition_S(cta_tile_p);
    Tensor fragment = make_fragment_like(thr_tile_s);
    copy_if(tiled_copy, thr_tile_p, thr_tile_s, fragment);
#ifndef NDEBUG
    if (thread(64, 1)) {
      print(thr_tile_s);
      print_tensor(thr_tile_p);
    }
#endif
    for (size_t i = 0; i < size(fragment); i++) {
      acc += fragment(i);
    }
  }
  __syncthreads();
  
  // Warp Reduce
  auto warp = cg::tiled_partition<warp_size>(tb);
  T warp_sum = cg::reduce(warp, acc, cg::plus<T>());

  // CTA Reduce
  if (warp.thread_rank() == 0) {
    warp_local_reduce_sum[warp.meta_group_rank()] = warp_sum;
  }
  __syncthreads();
  if (warp.meta_group_rank() == 0) {
    int cta_sum = cg::reduce(warp, warp_local_reduce_sum[warp.thread_rank()], cg::plus<T>());
    // Global Reduce
    if (warp.thread_rank() == 0) {
      atomicAdd(output, cta_sum);
    }
  }
}

int main(void) {
  using Element = int;
  constexpr int max_vectorization_bits = 128;
  constexpr int threads_per_block = 1024;
  constexpr int eles_per_thread = max_vectorization_bits / cutlass::sizeof_bits<Element>::value;
  
  std::mt19937 engine(std::random_device{}());
  std::uniform_int_distribution<size_t> rand_int_generator(4096, 8192);
  size_t n = rand_int_generator(engine) * eles_per_thread + 132 * 2 * 1024 * 4 * 64;
  // size_t n = 132 * 2 * 1024 * 4 * 64;
  printf("Total Element Count: %lu\n", n);

  thrust::host_vector<Element> h_S(n);
  thrust::host_vector<Element> h_D(4096); // One page

  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = static_cast<Element>(1);
  }
  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;

  auto thr_layout = cute::make_layout(cute::make_shape(cute::Int<threads_per_block>{}));
  auto val_layout = cute::make_layout(cute::make_shape(cute::Int<eles_per_thread>{}));
  // using CopyOp = cute::UniversalCopy<cute::uint_byte_t<sizeof(Element) * cute::size(val_layout)>>;
  using CopyOp = cute::UniversalCopy<cutlass::AlignedArray<Element, cute::size(val_layout)>>;
  using CopyAtom = cute::Copy_Atom<CopyOp, Element>;
  auto tiled_copy = cute::make_tiled_copy(CopyAtom{}, thr_layout, val_layout);
  print(tiled_copy);

  using TiledCopy = decltype(tiled_copy);
  using Tiler_MN = typename TiledCopy::Tiler_MN;
  print(Tiler_MN{});
  
  auto tensor_s = cute::make_tensor(cute::make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), cute::make_layout(n));
  // auto _tile_s = cute::local_tile(tensor_s, Tiler_MN{}, cute::make_coord(0));
  auto tiled_tensor_s = cute::tiled_divide(tensor_s, Tiler_MN{}); // (Tiler_MN, tile_count)
  auto tile_count = cute::size<1>(tiled_tensor_s);
  print(tiled_tensor_s);
  print(tile_count);
  print(cute::tensor<0>(tiled_tensor_s(make_coord(cute::_), 0)));
  print(cute::tensor<0>(tiled_tensor_s(make_coord(cute::_), 8)));
#ifndef NDEBUG
  {
    printf("Debug print:\n");
    auto shape_s = shape(tensor_s);
    cute::Tensor tensor_c = cute::make_identity_tensor(shape_s);
    cute::Tensor tensor_p = cute::lazy::transform(tensor_c, [&](auto c) { return cute::elem_less(c, shape_s); });
    auto tiled_tensor_s = cute::tiled_divide(tensor_s, Tiler_MN{}); // (Tiler_MN, tile_count)
    auto tiled_tensor_p = cute::tiled_divide(tensor_p, Tiler_MN{});
    auto tile_count = cute::size<1>(tiled_tensor_s);
    auto cta_tile_s = cute::tensor<0>(tiled_tensor_s(cute::make_coord(cute::_), 1));
    auto cta_tile_p = cute::tensor<0>(tiled_tensor_p(cute::make_coord(cute::_), 1));
    // auto thr_copy = tiled_copy.get_thread_slice(63);
    auto thr_copy = tiled_copy.get_thread_slice(64);
    auto thr_tile_s = thr_copy.partition_S(cta_tile_s);
    auto thr_tile_p = thr_copy.partition_S(cta_tile_p);
    print(tensor_p);
    print(tiled_tensor_p);
    print(cta_tile_p);
    cute::print_tensor(thr_tile_p);
  }
#endif
  cudaDeviceProp prop;
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  int max_active_blocks = -1;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
    cute_reduce<Element, decltype(tensor_s), TiledCopy>, cute::size(tiled_copy), 0));
  
  dim3 grid(prop.multiProcessorCount * max_active_blocks, 1, 1);
  dim3 block(cute::size(tiled_copy), 1, 1);
  printf("Kernel Output:\n");
  cute_reduce<Element, decltype(tensor_s), TiledCopy><<<grid, block>>>(tensor_s, tiled_copy, d_D.data().get());
  CUDA_CHECK(cudaDeviceSynchronize());
  h_D = d_D;
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("Acc: %d\n", *h_D.data());
  return 0;
}
