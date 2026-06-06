#include "sort_parallel.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <climits>
#include <cstdio>

// ============================================================
// 通用常量
// ============================================================
#define BITONIC_BLOCK_SIZE  256    // Bitonic 排序的线程块大小 (必须是 2 的幂)
#define MERGE_BLOCK_SIZE    256    // 归并阶段的线程块大小
#define SCAN_BLOCK_SIZE     256    // 前缀和扫描的线程块大小
#define QSORT_BLOCK_SIZE    256    // 快排分区的线程块大小
#define SEL_BLOCK_SIZE      256    // 选择排序的线程块大小
#define SMALL_SORT_THRESHOLD 1024  // 小于此阈值切换到 Bitonic 排序

// ============================================================
// 辅助函数: 检查 CUDA 错误
// ============================================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
        } \
    } while (0)

// ============================================================
// 辅助设备函数
// ============================================================

// ---------- co_rank: 归并路径分界点搜索 ----------
// 给定两个已排序数组 A[0..m-1], B[0..n-1],
// 在归并结果中位置 k 之前, 应该从 A 中取多少个元素?
// 使用二分搜索, O(log(min(m, k)))
__device__ int co_rank(int k, const int* A, int m, const int* B, int n) {
    int low = max(0, k - n);
    int high = min(k, m);
    while (low < high) {
        int i = (low + high) / 2;
        int j = k - i;
        // 检查 A[i-1] <= B[j] 且 B[j-1] <= A[i]
        if (j > 0 && i < m && B[j - 1] > A[i]) {
            low = i + 1;       // 需要更多 A 的元素
        } else if (i > 0 && j < n && A[i - 1] > B[j]) {
            high = i - 1;      // 需要更少 A 的元素
        } else {
            return i;          // 找到正确的分界
        }
    }
    return low;
}


// ============================================================
// 1. 并行归并排序 (Parallel Merge Sort)
// ============================================================

// ---------- 阶段 1: Bitonic 排序核函数 ----------
// 每个线程块在共享内存中对一个 tile 进行双调排序
__global__ void bitonic_sort_tile_kernel(int* d_arr, int n) {
    extern __shared__ int s_tile[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * BITONIC_BLOCK_SIZE + tid;

    // 加载数据到共享内存, 越界位置填充 INT_MAX 以保证排序后越界元素在末尾
    s_tile[tid] = (gid < n) ? d_arr[gid] : INT_MAX;
    __syncthreads();

    // Batcher's Bitonic Sort 网络
    // 外层循环: 逐步增大已排序的 bitonic 序列长度
    for (int stage = 2; stage <= BITONIC_BLOCK_SIZE; stage <<= 1) {
        // 内层循环: 逐步减小比较步长
        for (int step = stage >> 1; step > 0; step >>= 1) {
            int partner = tid ^ step;           // 比较配对的线程索引
            if (partner > tid) {
                // 根据 stage 决定当前是递增还是递减序列
                bool ascending = ((tid & stage) == 0);
                int a = s_tile[tid];
                int b = s_tile[partner];

                if (ascending) {
                    // 递增序列: 较小的值留在 tid 位置
                    if (a > b) {
                        s_tile[tid] = b;
                        s_tile[partner] = a;
                    }
                } else {
                    // 递减序列: 较大的值留在 tid 位置
                    if (a < b) {
                        s_tile[tid] = b;
                        s_tile[partner] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    // 写回全局内存
    if (gid < n) {
        d_arr[gid] = s_tile[tid];
    }
}

// ---------- 阶段 2: 归并核函数 ----------
// 每对相邻已排序段 (大小 = seg_size) 被一个线程块归并为一个更大的段
__global__ void merge_kernel(int* d_dst, const int* d_src,
                              int seg_size, int n) {
    int pair_id = blockIdx.x;
    int base = pair_id * 2 * seg_size;

    if (base >= n) return;

    // 计算两个子段的实际大小 (处理末尾不完整段)
    int na = seg_size;
    if (base + na > n) {
        na = n - base;
    }
    int nb = seg_size;
    if (base + na + nb > n) {
        nb = n - base - na;
    }

    // 如果第二个子段为空, 直接拷贝第一个子段
    if (nb <= 0) {
        for (int i = threadIdx.x; i < na; i += blockDim.x) {
            d_dst[base + i] = d_src[base + i];
        }
        return;
    }

    const int* A = d_src + base;
    const int* B = d_src + base + na;
    int total = na + nb;

    // 将输出均匀分配给线程块中的线程
    int items_per_thread = (total + blockDim.x - 1) / blockDim.x;
    int out_start = min(threadIdx.x * items_per_thread, total);
    int out_end   = min(out_start + items_per_thread, total);

    if (out_start >= total) return;

    // 使用 co_rank 确定该线程在 A 和 B 中的起始位置
    int a_pos = co_rank(out_start, A, na, B, nb);
    int b_pos = out_start - a_pos;

    // 每个线程执行串行归并, 写入自己负责的输出区间
    int* dst_ptr = d_dst + base + out_start;
    int count = out_end - out_start;
    for (int k = 0; k < count; ++k) {
        if (a_pos < na && (b_pos >= nb || A[a_pos] <= B[b_pos])) {
            dst_ptr[k] = A[a_pos++];
        } else {
            dst_ptr[k] = B[b_pos++];
        }
    }
}

// ---------- 并行归并排序主机入口 ----------
void parallel_merge_sort(int* d_arr, size_t n) {
    if (n <= 1) return;

    // 分配临时缓冲区
    int* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(int)));

    // --- 阶段 1: 局部块内 Bitonic 排序 ---
    int num_blocks_phase1 = (int)((n + BITONIC_BLOCK_SIZE - 1) / BITONIC_BLOCK_SIZE);
    size_t shared_mem_bytes = BITONIC_BLOCK_SIZE * sizeof(int);

    bitonic_sort_tile_kernel<<<num_blocks_phase1, BITONIC_BLOCK_SIZE, shared_mem_bytes>>>(
        d_arr, (int)n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- 阶段 2: 多轮归并 ---
    // 标记当前数据在 d_arr 还是 d_temp 中
    int* d_src = d_arr;
    int* d_dst = d_temp;

    for (size_t seg_size = BITONIC_BLOCK_SIZE; seg_size < n; seg_size <<= 1) {
        size_t num_pairs = (n + 2 * seg_size - 1) / (2 * seg_size);
        int num_blocks_merge = (int)num_pairs;

        merge_kernel<<<num_blocks_merge, MERGE_BLOCK_SIZE>>>(
            d_dst, d_src, (int)seg_size, (int)n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 交换源和目标指针
        int* tmp = d_src;
        d_src = d_dst;
        d_dst = tmp;
    }

    // 如果最终结果在 d_temp 中, 拷贝回原数组
    if (d_src == d_temp) {
        CUDA_CHECK(cudaMemcpy(d_arr, d_temp, n * sizeof(int), cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(d_temp));
}


// ============================================================
// 2. 并行快速排序 (Parallel Quick Sort)
// ============================================================

// ---------- 前缀和辅助核函数 ----------

// --- Warp 级 (32 线程) 排他性前缀和 ---
__device__ int warp_exclusive_scan(int val) {
    // Kogge-Stone 风格的蝶形扫描 (5 轮)
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int tmp = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & 31) >= offset) {
            val += tmp;
        }
    }
    // 转换为 exclusive scan: 每个线程返回前 (lane-1) 个元素的和
    int result = __shfl_up_sync(0xffffffff, val, 1);
    if ((threadIdx.x & 31) == 0) result = 0;
    return result;
}

// --- 块级排他性前缀和 (单 block, 支持 up to SCAN_BLOCK_SIZE 元素) ---
__global__ void block_exclusive_scan_kernel(const int* d_input, int* d_output,
                                             int* d_block_sums, int n) {
    __shared__ int s_warp_sums[32];  // 每个 warp 的部分和

    int tid = threadIdx.x;
    int gid = blockIdx.x * SCAN_BLOCK_SIZE + tid;
    int val = (gid < n) ? d_input[gid] : 0;

    // 1. Warp 级 exclusive scan
    int warp_id = tid >> 5;   // tid / 32
    int lane_id = tid & 31;   // tid % 32
    int warp_scan = warp_exclusive_scan(val);

    // 2. 每个 warp 的最后一线程记录 warp 总和
    if (lane_id == 31) {
        s_warp_sums[warp_id] = warp_scan + val;  // 这是 inclusive sum
    }
    __syncthreads();

    // 3. 线程 0 对 warp sums 做串行 exclusive scan
    if (warp_id == 0 && lane_id < (blockDim.x + 31) / 32) {
        int sum = 0;
        for (int i = 0; i < lane_id; ++i) {
            sum += s_warp_sums[i];
        }
        s_warp_sums[lane_id] = sum;  // 现在 s_warp_sums[w] = warp 0..(w-1) 的和
    }
    __syncthreads();

    // 4. 组合结果: block_exclusive = warp_exclusive + warp_offset
    int warp_offset = s_warp_sums[warp_id];
    int block_exclusive = warp_scan + warp_offset;

    if (gid < n) {
        d_output[gid] = block_exclusive;
    }

    // 5. 写入块总和 (用于全局 scan)
    if (tid == blockDim.x - 1 && d_block_sums != nullptr) {
        d_block_sums[blockIdx.x] = block_exclusive + val;
    }
}

// --- 跨块偏移加法核函数 ---
// 将块级前缀和结果加上全局块偏移
// d_offsets[i] = 前 i 个块的元素总数 (exclusive scan of block sums)
// 对于块 block_id 中的元素, 需要加上 d_offsets[block_id]
__global__ void add_block_offset_kernel(int* d_data, const int* d_offsets, int n) {
    int gid = blockIdx.x * SCAN_BLOCK_SIZE + threadIdx.x;
    if (gid < n) {
        int block_id = gid / SCAN_BLOCK_SIZE;
        d_data[gid] += d_offsets[block_id];
    }
}

// --- 全局扫描: 在块级扫描基础上组合跨块偏移 ---
void full_exclusive_scan(int* d_input, int* d_output, int* d_temp, int n) {
    if (n <= 0) return;

    int num_blocks = (n + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;
    int* d_block_sums = d_temp;  // 复用临时缓冲区的前 num_blocks 个位置

    // Step 1: 块级扫描
    block_exclusive_scan_kernel<<<num_blocks, SCAN_BLOCK_SIZE>>>(
        d_input, d_output, d_block_sums, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: 对块总和做扫描 (如果多于 1 个块)
    if (num_blocks > 1) {
        // 对块总和做 exclusive scan (单 block 即可, 块数通常很少)
        block_exclusive_scan_kernel<<<1, SCAN_BLOCK_SIZE>>>(
            d_block_sums, d_block_sums, nullptr, num_blocks);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 3: 将块偏移加到每个块的输出上
        int add_blocks = (n + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;
        add_block_offset_kernel<<<add_blocks, SCAN_BLOCK_SIZE>>>(
            d_output, d_block_sums, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// ---------- 快速排序分区核函数 ----------

// 计算 flag 数组: 0 = 小于枢轴; 1 = 大于等于枢轴
__global__ void compute_partition_flags_kernel(const int* d_src, int* d_flags,
                                                int pivot, int seg_start, int n) {
    int gid = seg_start + blockIdx.x * QSORT_BLOCK_SIZE + threadIdx.x;
    if (gid < n) {
        d_flags[gid - seg_start] = (d_src[gid] < pivot) ? 0 : 1;
    }
}

// 根据左右分区位置将元素散射到正确位置
// 左元素 → 段起始 + left_pos
// 右元素 → 段起始 + left_count + right_pos
__global__ void scatter_partition_kernel(const int* d_src, int* d_dst,
                                          const int* d_left_pos,
                                          const int* d_right_pos,
                                          int left_count,
                                          int pivot, int seg_start, int seg_len) {
    int gid = seg_start + blockIdx.x * QSORT_BLOCK_SIZE + threadIdx.x;
    if (gid >= seg_start + seg_len) return;

    int local_idx = gid - seg_start;
    int val = d_src[gid];

    if (val < pivot) {
        d_dst[seg_start + d_left_pos[local_idx]] = val;
    } else {
        d_dst[seg_start + left_count + d_right_pos[local_idx]] = val;
    }
}

// --- 分段 Bitonic 排序 (用于小段) ---
__global__ void bitonic_sort_segment_kernel(int* d_arr, int seg_start,
                                              int seg_len) {
    extern __shared__ int s_tile[];

    int tid = threadIdx.x;
    int gid = seg_start + tid;

    // 加载 (越界填 INT_MAX)
    s_tile[tid] = (tid < seg_len) ? d_arr[seg_start + tid] : INT_MAX;
    __syncthreads();

    // Bitonic sort (假设 seg_len <= BITONIC_BLOCK_SIZE)
    for (int stage = 2; stage <= BITONIC_BLOCK_SIZE; stage <<= 1) {
        for (int step = stage >> 1; step > 0; step >>= 1) {
            int partner = tid ^ step;
            if (partner > tid) {
                bool ascending = ((tid & stage) == 0);
                int a = s_tile[tid];
                int b = s_tile[partner];
                if (ascending) {
                    if (a > b) { s_tile[tid] = b; s_tile[partner] = a; }
                } else {
                    if (a < b) { s_tile[tid] = b; s_tile[partner] = a; }
                }
            }
            __syncthreads();
        }
    }

    if (tid < seg_len) {
        d_arr[seg_start + tid] = s_tile[tid];
    }
}

// ---------- 并行快速排序主机入口 ----------
// 算法: CPU 维护待排序段队列, 迭代处理
//   大段 -> GPU 并行分区 (flag + scan + scatter)
//   小段 -> GPU Bitonic 排序直接完成
void parallel_quick_sort(int* d_arr, size_t n) {
    if (n <= 1) return;

    // 分配辅助缓冲区
    int* d_temp = nullptr;    // 分区后的临时数组
    int* d_flags = nullptr;   // 标记数组
    int* d_scan = nullptr;    // 前缀和数组
    int* d_aux = nullptr;     // 扫描辅助缓冲区 (块和)
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flags, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scan, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_aux, ((n + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE) * sizeof(int)));

    // 复制原始数据到临时数组
    CUDA_CHECK(cudaMemcpy(d_temp, d_arr, n * sizeof(int), cudaMemcpyDeviceToDevice));

    // 段队列 (用数组模拟栈, CPU 端)
    struct Segment {
        int start;
        int len;
    };
    Segment* h_stack = new Segment[64]; // 假设递归深度不超过 64
    int stack_top = 0;
    h_stack[stack_top++] = {0, (int)n};

    while (stack_top > 0) {
        Segment seg = h_stack[--stack_top];
        if (seg.len <= 1) continue;

        // --- 小段直接用 Bitonic Sort 完成 ---
        if (seg.len <= SMALL_SORT_THRESHOLD) {
            int block_size = 1;
            while (block_size < seg.len) block_size <<= 1;
            if (block_size > BITONIC_BLOCK_SIZE) block_size = BITONIC_BLOCK_SIZE;
            bitonic_sort_segment_kernel<<<1, block_size, block_size * sizeof(int)>>>(
                d_temp, seg.start, seg.len);
            CUDA_CHECK(cudaDeviceSynchronize());
            continue;
        }

        // --- 大段: 并行分区 ---
        // 1. 选取枢轴 (简单策略: 取段中间位置元素)
        int pivot;
        CUDA_CHECK(cudaMemcpy(&pivot,
                               d_temp + seg.start + seg.len / 2,
                               sizeof(int), cudaMemcpyDeviceToHost));

        // 2. 计算 flag
        int flag_blocks = (seg.len + QSORT_BLOCK_SIZE - 1) / QSORT_BLOCK_SIZE;
        compute_partition_flags_kernel<<<flag_blocks, QSORT_BLOCK_SIZE>>>(
            d_temp, d_flags, pivot, seg.start, seg.start + seg.len);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3. 对 flag 做排他性前缀和, 得到每个 "< pivot" 元素在左侧的位置
        full_exclusive_scan(d_flags, d_scan, d_aux, seg.len);

        // 计算左侧元素的总数及每个 ">= pivot" 元素在右侧的位置
        int left_count;
        CUDA_CHECK(cudaMemcpy(&left_count,
                               d_scan + seg.len - 1,
                               sizeof(int), cudaMemcpyDeviceToHost));
        int last_flag;
        CUDA_CHECK(cudaMemcpy(&last_flag,
                               d_flags + seg.len - 1,
                               sizeof(int), cudaMemcpyDeviceToHost));
        left_count += (last_flag == 0) ? 1 : 0;

// ---------- 计算右半部分偏移的核函数 ----------
// right_pos[i] = i - left_pos[i]
// 即 ">= pivot" 元素在右半区的局部偏移
__global__ void compute_right_pos_kernel(int* d_right_pos,
                                          const int* d_left_pos,
                                          int seg_len) {
    int i = blockIdx.x * QSORT_BLOCK_SIZE + threadIdx.x;
    if (i < seg_len) {
        d_right_pos[i] = i - d_left_pos[i];
    }
}
        compute_right_pos_kernel<<<flag_blocks, QSORT_BLOCK_SIZE>>>(
            d_flags, d_scan, seg.len); // 复用 d_flags 存储 right_pos
        CUDA_CHECK(cudaDeviceSynchronize());

        // 4. 散射: 将元素写入 d_arr (以 d_arr 作为输出目标)
        scatter_partition_kernel<<<flag_blocks, QSORT_BLOCK_SIZE>>>(
            d_temp, d_arr, d_scan, d_flags, left_count,
            pivot, seg.start, seg.len);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 5. 将 d_arr 中已分区的数据拷回 d_temp (保持一致性)
        CUDA_CHECK(cudaMemcpy(d_temp + seg.start, d_arr + seg.start,
                               seg.len * sizeof(int), cudaMemcpyDeviceToDevice));

        // 6. 将左右子段压入栈 (先右后左, 保证先处理左段)
        int right_len = seg.len - left_count;
        if (right_len > 0) {
            h_stack[stack_top++] = {seg.start + left_count, right_len};
        }
        if (left_count > 0) {
            h_stack[stack_top++] = {seg.start, left_count};
        }
    }

    // 最终结果已在 d_temp 中, 拷回 d_arr
    CUDA_CHECK(cudaMemcpy(d_arr, d_temp, n * sizeof(int), cudaMemcpyDeviceToDevice));

    // 清理
    delete[] h_stack;
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_scan));
    CUDA_CHECK(cudaFree(d_aux));
}


// ============================================================
// 3. 并行选择排序 (Parallel Selection Sort / Rank Sort)
// ============================================================

// ---------- 排名计算核函数 ----------
// 每个线程负责一个元素, 与所有其他元素比较, 确定其最终排名
// 使用平局规则: 值相同时, 索引小的排名靠前 (保证稳定性)
__global__ void compute_ranks_kernel(const int* d_input, int* d_ranks, int n) {
    int i = blockIdx.x * SEL_BLOCK_SIZE + threadIdx.x;
    if (i >= n) return;

    int my_val = d_input[i];
    int rank = 0;

    for (int j = 0; j < n; ++j) {
        int other = d_input[j];
        if (other < my_val || (other == my_val && j < i)) {
            ++rank;
        }
    }
    d_ranks[i] = rank;
}

// ---------- 散射核函数 ----------
// 根据排名将元素写入输出数组的正确位置
__global__ void scatter_by_rank_kernel(const int* d_input, const int* d_ranks,
                                         int* d_output, int n) {
    int i = blockIdx.x * SEL_BLOCK_SIZE + threadIdx.x;
    if (i >= n) return;
    d_output[d_ranks[i]] = d_input[i];
}

// ---------- 并行选择排序主机入口 ----------
void parallel_selection_sort(int* d_arr, size_t n) {
    if (n <= 1) return;

    // 分配辅助内存
    int* d_ranks = nullptr;
    int* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ranks, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(int)));

    // 1. 计算每个元素的排名
    int num_blocks = (int)((n + SEL_BLOCK_SIZE - 1) / SEL_BLOCK_SIZE);
    compute_ranks_kernel<<<num_blocks, SEL_BLOCK_SIZE>>>(d_arr, d_ranks, (int)n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2. 根据排名散射到输出数组
    scatter_by_rank_kernel<<<num_blocks, SEL_BLOCK_SIZE>>>(d_arr, d_ranks, d_output, (int)n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. 拷回原数组
    CUDA_CHECK(cudaMemcpy(d_arr, d_output, n * sizeof(int), cudaMemcpyDeviceToDevice));

    // 清理
    CUDA_CHECK(cudaFree(d_ranks));
    CUDA_CHECK(cudaFree(d_output));
}
