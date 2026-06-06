#ifndef SORT_PARALLEL_CUH
#define SORT_PARALLEL_CUH

#include <cstddef>

// ============================================================
// GPU 并行排序算法 (CUDA C 实现)
// 所有函数对 GPU 全局内存中的数组进行原地或辅助排序
// d_arr: 指向 GPU 全局内存中待排序数组的指针
// n: 元素个数
// ============================================================

// --- Parallel Merge Sort (并行归并排序) ---
// 自底向上: 块内 Bitonic Sort + 多轮 Merge Path 归并
// 不稳定, 工作复杂度 O(N log N), 步复杂度 O(log² N)
// 需要额外 O(N) 全局内存作为辅助缓冲区
void parallel_merge_sort(int* d_arr, size_t n);

// --- Parallel Quick Sort (并行快速排序) ---
// 分段并行分区 + CPU 迭代管理子问题
// 不稳定, 平均工作复杂度 O(N log N)
// 需要额外 O(N) 全局内存用于 scan 和临时数组
void parallel_quick_sort(int* d_arr, size_t n);

// --- Parallel Selection Sort (并行选择排序 / Rank Sort) ---
// 全对全比较确定每个元素的排名, 然后散射到正确位置
// 不稳定, 工作复杂度 O(N²), 步复杂度 O(N)
// 需要额外 O(N) 全局内存用于排名数组
void parallel_selection_sort(int* d_arr, size_t n);

#endif // SORT_PARALLEL_CUH
