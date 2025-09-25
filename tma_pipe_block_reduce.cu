#include <cstdio>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cutlass/cutlass.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/device_kernel.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
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

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

template<typename T>
CUTLASS_HOST_DEVICE void print_newline(T& t) {
  cute::print(t);
  printf("\n");
}

template<typename T>
CUTLASS_HOST_DEVICE void print_newline(T&& t) {
  cute::print(t);
  printf("\n");
}

template <typename T, typename SmemPipeLayout, int STAGE>
struct alignas(128) SharedStorage {
  using SmemPipeLayout_ = SmemPipeLayout;
  static_assert(cute::size<1>(SmemPipeLayout{}) == STAGE);
  alignas(128) cute::ArrayEngine<T, cute::cosize_v<SmemPipeLayout>> pipe_box;
  uint64_t load_barrier[STAGE];
};

template<typename T, typename STensor, typename TiledCopy>
__device__ T tile_reudce(STensor s, TiledCopy tiled_copy, uint8_t* shm_temp_storage) {
  using namespace cute;
  namespace cg = cooperative_groups;
  using Tiler_MN = typename TiledCopy::Tiler_MN;
  cg::thread_block tb = cg::this_thread_block();

  using BlockReduce = cub::BlockReduce<T, size(tiled_copy)>;
  typename BlockReduce::TempStorage& temp_storage = *reinterpret_cast<typename BlockReduce::TempStorage*>(shm_temp_storage);
  auto tiled_s = zipped_divide(s, Tiler_MN{});

  constexpr int iterations = size<1>(tiled_s);
  T acc(0);
  for (int i = 0; i < iterations; i++) {
    auto coord = make_coord(make_coord(_, _), i);
    auto tile = tiled_s(coord);

    ThrCopy thr_copy = tiled_copy.get_thread_slice(tb.thread_rank());
    Tensor thr_tile = thr_copy.partition_S(tile);
    Tensor fragment = make_fragment_like(thr_tile);
    copy(tiled_copy, thr_tile, fragment);

    for (int i = 0; i < size(fragment); i++) {
      acc += fragment(i);
    }
  }
  T block_acc = BlockReduce(temp_storage).Sum(acc);
#ifndef NDEBUG
  if (thread0()) {
    printf("Shm address: %p\n", &temp_storage);
    printf("Thread Acc: %d\n", acc);
  }
#endif
  return block_acc;
}

template<typename T, typename TMA, typename SharedStorage, typename TiledCopy, typename GMemOrder = cute::LayoutRight>
__global__ void tma_reduce(CUTLASS_GRID_CONSTANT TMA const tma, T* output, size_t row, size_t col, TiledCopy tiled_copy) {
  using namespace cute;
  extern __shared__ __align__(1024) uint8_t shared_memory[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);

  using SmemPipeLayout = typename SharedStorage::SmemPipeLayout_; // ((TMA_M,TMA_N),(STAGE_COUNT)):((TMA_N,_1),(TMA_M*TMA_N))
  Tensor double_buffer = make_tensor(make_smem_ptr(smem.pipe_box.begin()), SmemPipeLayout{});

  // Create coord tensor
  Tensor tensor_s = tma.get_tma_tensor(make_shape(row, col));
  // zipped_divide: ((TMA_M,TMA_N),(m, n))
  Tensor tiled_tensor_s = zipped_divide(tensor_s, shape<0>(SmemPipeLayout{}));
  auto total_blocks = size<1>(tiled_tensor_s);
#ifndef NDEBUG
  if (thread0()) {
    print_newline(tensor_s);
    print_newline(tiled_tensor_s);
    print_newline(double_buffer);
    printf("Total blocks: %lu\n", total_blocks);
  }
  __syncthreads();
#endif
  T acc(0);
  // Init mbarrier
  constexpr int stage = size<1>(SmemPipeLayout{});
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
  // Init all stages
  if ((warp_idx == 0) && lane_predicate) {
    for (int i = 0; i < stage; i++) {
      uint64_t* mbar = &smem.load_barrier[i];
      ProducerBarType::init(mbar, 1); // Just one thread arrive
    }
  }
  cluster_sync();

  auto issue_state = cutlass::PipelineState<stage>();
  auto complete_state = cutlass::PipelineState<stage>();

  decltype(total_blocks) blk_offset = blockIdx.x; 
  constexpr int tma_transaction_bytes = sizeof(make_tensor_like(double_buffer(make_coord(_, 0))));  // Per stage Bytes
  
  // Issue (STAGE - 1) TMA Load -- Cold Start
  for (; issue_state.count() < stage - 1; ++issue_state) {
    if (blk_offset >= total_blocks) {
      break;
    }
    if ((warp_idx == 0) && lane_predicate) {
      // Partation Src/Dst
      auto cta_tile_s = tiled_tensor_s(make_coord(_, blk_offset));
      auto buffer = double_buffer(make_coord(_, issue_state.index()));
      auto [t_cta_tile_s, t_buffer] = tma_partition(tma, Int<0>{}, Layout<_1>{}, buffer, cta_tile_s);
      uint64_t* mbar = &smem.load_barrier[issue_state.index()];
      ProducerBarType::arrive_and_expect_tx(mbar, tma_transaction_bytes);
      copy(tma.with(*mbar), t_cta_tile_s, t_buffer);
    }
    __syncthreads();
    blk_offset += gridDim.x;
  }

  // Pipeline
  CUTE_NO_UNROLL
  while (true) {
    int break_flag = 0;
    // Issue next stage if needed
    if (blk_offset < total_blocks) {
      if ((warp_idx == 0) && lane_predicate) {
        auto cta_tile_s_next = tiled_tensor_s(make_coord(_, blk_offset));
        auto buffer_next = double_buffer(make_coord(_, issue_state.index())); // Use another buffer
        auto [t_cta_tile_s_next, t_buffer_next] = tma_partition(tma, Int<0>{}, Layout<_1>{}, buffer_next, cta_tile_s_next);
        uint64_t* mbar = &smem.load_barrier[issue_state.index()];
        ProducerBarType::arrive_and_expect_tx(mbar, tma_transaction_bytes); // tma_transaction_bytes is constexpr
        copy(tma.with(*mbar), t_cta_tile_s_next, t_buffer_next);
      }
      __syncthreads();
      ++issue_state;
      blk_offset += gridDim.x;
    } else {
      break_flag = 1;
    }

    // First Wait current stage & ++complete_state
    if (issue_state.count() > complete_state.count()) {
      uint64_t* mbar = &smem.load_barrier[complete_state.index()];
      ProducerBarType::wait(mbar, complete_state.phase());
      // Do reduce
      auto smem_tile = tensor<0>(double_buffer(make_coord(_, complete_state.index())));
      T block_reduce = tile_reudce<T, decltype(smem_tile), TiledCopy>(smem_tile, tiled_copy, shared_memory + sizeof(SharedStorage));
      acc += block_reduce;
      // Must after reduce because we use origin complete_state to index shared memory buffer
      ++complete_state;  // Move to next complete_state
#ifndef NDEBUG
      if (thread0()) {
        print_newline(smem_tile);
        printf("Block Reduce: %d\n", acc);
      }
#endif
    }

    if (break_flag) break;
  }

  uint32_t tail_count = issue_state.count() - complete_state.count();
  for (uint32_t i = 0; i < tail_count; i++) {
    uint64_t* mbar = &smem.load_barrier[complete_state.index()];
    ProducerBarType::wait(mbar, complete_state.phase());
    // Do reduce
    auto smem_tile = tensor<0>(double_buffer(make_coord(_, complete_state.index())));
    T block_reduce = tile_reudce<T, decltype(smem_tile), TiledCopy>(smem_tile, tiled_copy, shared_memory + sizeof(SharedStorage));
    acc += block_reduce;
    ++complete_state;  // Move to next complete_state
  }

  if (threadIdx.x == 0) {
    atomicAdd(output, acc);
  }
}

template<typename T, int TMA_M = 128, int TMA_N = 128, bool ROW_MAJOR = true>
struct S2RCopyHelper {
  static constexpr int threads_per_block = 128;
  static constexpr int max_vectorization_bits = 128;
  static constexpr int eles_per_thread = max_vectorization_bits / cutlass::sizeof_bits<T>::value;

  using ValLayout = std::conditional_t<ROW_MAJOR,
    typename cute::Layout<cute::Shape<cute::_1, cute::Int<eles_per_thread>>, cute::Stride<cute::Int<eles_per_thread>, cute::_1>>,
    typename cute::Layout<cute::Shape<cute::Int<eles_per_thread>, cute::_1>, cute::Stride<cute::_1, cute::Int<eles_per_thread>>>
  >;
  
  static_assert(TMA_M % eles_per_thread == 0);
  static_assert(TMA_N % eles_per_thread == 0);
  static constexpr int col_major_threads_per_col = TMA_M / eles_per_thread;
  static constexpr int row_major_threads_per_row = TMA_N / eles_per_thread;

  static_assert(threads_per_block % col_major_threads_per_col == 0);
  static_assert(threads_per_block % row_major_threads_per_row == 0);

  static constexpr int col_major_cols = threads_per_block / col_major_threads_per_col;
  static constexpr int row_major_rows = threads_per_block / row_major_threads_per_row;

  using ThrLayout = std::conditional_t<ROW_MAJOR,
    typename cute::Layout<cute::Shape<cute::Int<row_major_rows>, cute::Int<row_major_threads_per_row>>, cute::Stride<cute::Int<row_major_threads_per_row>, cute::_1>>,
    typename cute::Layout<cute::Shape<cute::Int<col_major_threads_per_col>, cute::Int<col_major_cols>>, cute::Stride<cute::_1, cute::Int<col_major_threads_per_col>>>
  >;

  using CopyOp = cute::UniversalCopy<cutlass::AlignedArray<T, cute::size(ValLayout{})>>;
  using CopyAtom = cute::Copy_Atom<CopyOp, T>;
  using TiledCopy = decltype(cute::make_tiled_copy(CopyAtom{}, ThrLayout{}, ValLayout{}));
};

template<typename T, int TMA_M = 128, int TMA_N = 128, bool ROW_MAJOR = true, int STAGE = 2>
int launch_tma_reduce(T* array, T* reduce, size_t row, size_t col) {
  using namespace cute;

  assert(row % TMA_M == 0);
  assert(col % TMA_N == 0);
  using StrideOrder = std::conditional_t<ROW_MAJOR, typename cute::LayoutRight, typename cute::LayoutLeft>;
  Tensor array2D = make_tensor(array, make_layout(make_shape(row, col), StrideOrder{}));
  print_newline(array2D);
  auto shm_layout = make_layout(make_shape(Int<TMA_M>{}, Int<TMA_N>{}), StrideOrder{});
  print_newline(shm_layout);
  auto shm_pipe_layout = logical_product(shm_layout, make_layout(make_shape(Int<STAGE>{})));  // (TMA_M, TMA_N) -> ((TMA_M, TMA_N), STAGE)
  print_newline(shm_pipe_layout);
  auto tma = make_tma_copy(SM90_TMA_LOAD{}, array2D, shm_layout);
  print_newline(tma);
  printf("make_tma_copy CTA_Tiler: ");
  print_newline(product_each(shape(shm_layout))); // CTA_Tiler

  // Define S2R copy
  using S2RTiledCopy = typename S2RCopyHelper<T, TMA_M, TMA_N, ROW_MAJOR>::TiledCopy;
  auto tiled_copy = S2RTiledCopy{};
  print(tiled_copy);

  // Launch parameter setup
  using _SharedStorage = SharedStorage<T, decltype(shm_pipe_layout), STAGE>; // shm_layout * pipeline stage + barrier
  int smem_size = static_cast<int>(sizeof(_SharedStorage)); // TMA Box
  using BlockReduce = cub::BlockReduce<T, size(tiled_copy)>;
  smem_size += sizeof(typename BlockReduce::TempStorage);  // Reduce Buffer
  printf("Shared Memory per Block: %d\n", smem_size);

  void const* kernel_ptr = reinterpret_cast<void const*>(&tma_reduce<T, decltype(tma), _SharedStorage, S2RTiledCopy, StrideOrder>);
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  cudaDeviceProp prop;
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  int max_active_blocks = -1;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
    kernel_ptr, 128, smem_size));

  dim3 dimGrid(max_active_blocks * prop.multiProcessorCount, 1, 1);
  dim3 dimCluster(1, 1, 1);
  dim3 dimBlock(128, 1, 1);
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  // Kernel Launch
  CUTLASS_CHECK(cutlass::launch_kernel_on_cluster(params, kernel_ptr, tma, reduce, row, col, tiled_copy));
  CUTE_CHECK_LAST();
  return 0;
}

int main(void) {
  using Element = int;
  size_t row = 78 * 128;
  size_t col = 36 * 128;
  thrust::host_vector<Element> h_S(row * col);
  thrust::host_vector<Element> h_D(4096); // One page

  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = static_cast<Element>(1);
  }
  for (size_t i = 0; i < h_D.size(); ++i) {
    h_D[i] = static_cast<Element>(0);
  }
  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;

  launch_tma_reduce<Element, 64, 64>(d_S.data().get(), d_D.data().get(), row, col);
  h_D = d_D;
  printf("Target result: %lu, Final result: %d\n", h_S.size(), h_D[0]);
  return 0;
}
