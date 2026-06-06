#include "sort_serial.h"
#include <algorithm>
#include <cstring>
#include <utility>

// ============================================================
// Merge Sort 实现
// ============================================================

namespace {

// 合并两个已排序子数组 [l, m) 和 [m, r) 到临时数组, 再拷回原数组
void merge(int* arr, int* tmp, size_t l, size_t m, size_t r) {
    size_t i = l;
    size_t j = m;
    size_t k = l;

    while (i < m && j < r) {
        if (arr[i] <= arr[j]) {
            tmp[k++] = arr[i++];
        } else {
            tmp[k++] = arr[j++];
        }
    }
    while (i < m) tmp[k++] = arr[i++];
    while (j < r) tmp[k++] = arr[j++];

    // 拷贝回原数组
    std::memcpy(arr + l, tmp + l, (r - l) * sizeof(int));
}

// 递归归并排序
void merge_sort_recursive(int* arr, int* tmp, size_t l, size_t r) {
    if (r - l <= 1) return;
    size_t m = l + (r - l) / 2;
    merge_sort_recursive(arr, tmp, l, m);
    merge_sort_recursive(arr, tmp, m, r);
    merge(arr, tmp, l, m, r);
}

} // anonymous namespace

void serial_merge_sort(int* arr, size_t n) {
    if (n <= 1) return;
    int* tmp = new int[n];
    merge_sort_recursive(arr, tmp, 0, n);
    delete[] tmp;
}


// ============================================================
// Quick Sort 实现
// ============================================================

namespace {

// 三数取中: 选取 arr[lo], arr[mid], arr[hi-1] 的中位数作为枢轴
size_t median_of_three(int* arr, size_t lo, size_t hi) {
    size_t mid = lo + (hi - lo) / 2;
    int a = arr[lo];
    int b = arr[mid];
    int c = arr[hi - 1];

    if ((a <= b && b <= c) || (c <= b && b <= a)) return mid;
    if ((b <= a && a <= c) || (c <= a && a <= b)) return lo;
    return hi - 1;
}

// 三向切分 (Bentley-McIlroy 风格):
// 返回一对索引 (lt, gt), 使得
//   [lo, lt)   <  pivot
//   [lt, gt]   == pivot
//   (gt, hi)   >  pivot
std::pair<size_t, size_t> three_way_partition(int* arr, size_t lo, size_t hi) {
    // 枢轴放到最前面
    size_t p_idx = median_of_three(arr, lo, hi);
    int pivot = arr[p_idx];
    std::swap(arr[lo], arr[p_idx]);

    size_t lt = lo;       // arr[lo..lt-1] < pivot
    size_t gt = hi - 1;   // arr[gt+1..hi-1] > pivot
    size_t i  = lo + 1;   // 当前扫描位置

    // arr[lt+1..i-1] == pivot
    while (i <= gt) {
        if (arr[i] < pivot) {
            std::swap(arr[lt], arr[i]);
            ++lt;
            ++i;
        } else if (arr[i] > pivot) {
            std::swap(arr[i], arr[gt]);
            --gt;
        } else {
            ++i;
        }
    }
    return {lt, gt + 1}; // 返回 [lt, gt+1) 为 == pivot 区间
}

void quick_sort_recursive(int* arr, size_t lo, size_t hi) {
    while (hi - lo > 1) {
        auto [lt, gt] = three_way_partition(arr, lo, hi);

        // 先处理较小的分区以减少递归深度 (尾递归优化)
        if (lt - lo < hi - gt) {
            quick_sort_recursive(arr, lo, lt);
            lo = gt;  // 尾递归: 用循环处理较大分区
        } else {
            quick_sort_recursive(arr, gt, hi);
            hi = lt;
        }
    }
}

} // anonymous namespace

void serial_quick_sort(int* arr, size_t n) {
    if (n <= 1) return;
    quick_sort_recursive(arr, 0, n);
}


// ============================================================
// Selection Sort 实现
// ============================================================

void serial_selection_sort(int* arr, size_t n) {
    for (size_t i = 0; i + 1 < n; ++i) {
        // 在 [i, n) 中找到最小元素的位置
        size_t min_idx = i;
        for (size_t j = i + 1; j < n; ++j) {
            if (arr[j] < arr[min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            std::swap(arr[i], arr[min_idx]);
        }
    }
}
