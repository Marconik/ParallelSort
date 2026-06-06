#ifndef SORT_SERIAL_H
#define SORT_SERIAL_H

#include <cstddef>

// ============================================================
// 串行排序算法 (C++ 实现)
// 所有函数对 [begin, end) 范围内的元素进行原地或辅助排序
// 支持 int 类型, 可模板化适配其他类型
// ============================================================

// --- Merge Sort (归并排序) ---
// 自顶向下递归归并排序, 稳定, O(N log N)
// arr: 待排序数组, n: 元素个数
void serial_merge_sort(int* arr, size_t n);

// --- Quick Sort (快速排序) ---
// 原地快速排序 (三数取中 + 三向切分), 不稳定, 平均 O(N log N)
// arr: 待排序数组, n: 元素个数
void serial_quick_sort(int* arr, size_t n);

// --- Selection Sort (选择排序) ---
// 原地选择排序, 不稳定 (可改为稳定), O(N^2)
// arr: 待排序数组, n: 元素个数
void serial_selection_sort(int* arr, size_t n);

#endif // SORT_SERIAL_H
