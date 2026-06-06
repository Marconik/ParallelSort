// ============================================================
// main.cpp — 排序算法测试框架
// ============================================================
//
// 用法:
//   sort_test gen <N>             生成 N 个随机整数写入 data.txt 后退出
//   sort_test                     读取 data.txt, 运行全部排序测试
//   sort_test --save <algorithm>  指定输出到 sorted.txt 的算法
//                                  (可选: merge, quick, selection, 默认: merge)
//
// 编译:
//   串行: g++ -O2 -std=c++17 -Iinclude src/sort_serial.cpp src/main.cpp -o sort_test
//   并行: nvcc -O2 -std=c++17 -Iinclude src/sort_serial.cpp src/sort_parallel.cu
//              src/main.cpp -o sort_test
// ============================================================

#include "sort_serial.h"

// CUDA 支持: 编译时自动检测, 也可用 -DENABLE_CUDA 强制开启
#if defined(__CUDACC__) || defined(ENABLE_CUDA)
    #define HAS_CUDA 1
    #include "sort_parallel.cuh"
#else
    #define HAS_CUDA 0
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <chrono>

// ============================================================
// 类型别名
// ============================================================
using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::milliseconds;

// ============================================================
// 辅助函数
// ============================================================

// 检查数组是否升序
static bool is_sorted(const int* arr, size_t n) {
    for (size_t i = 1; i < n; ++i) {
        if (arr[i - 1] > arr[i]) return false;
    }
    return true;
}

// 深拷贝数组
static int* copy_array(const int* src, size_t n) {
    int* dst = new int[n];
    std::memcpy(dst, src, n * sizeof(int));
    return dst;
}

// 生成随机整数数组
static int* generate_random_array(size_t n) {
    int* arr = new int[n];
    for (size_t i = 0; i < n; ++i) {
        arr[i] = rand();
    }
    return arr;
}

// 从文件读取数组 (格式: 第一行 N, 接下来 N 行每行一个整数)
static int* read_array_from_file(const char* filename, size_t& out_n) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: cannot open file '%s'\n", filename);
        return nullptr;
    }

    int n = 0;
    if (fscanf(f, "%d", &n) != 1 || n <= 0) {
        fprintf(stderr, "Error: invalid array size in '%s'\n", filename);
        fclose(f);
        return nullptr;
    }

    int* arr = new int[n];
    for (int i = 0; i < n; ++i) {
        if (fscanf(f, "%d", &arr[i]) != 1) {
            fprintf(stderr, "Error: not enough data in '%s' (expected %d)\n", filename, n);
            delete[] arr;
            fclose(f);
            return nullptr;
        }
    }
    fclose(f);

    out_n = static_cast<size_t>(n);
    return arr;
}

// 将数组写入文件 (格式: 第一行 N, 接下来 N 行每行一个整数)
static void write_array_to_file(const char* filename, const int* arr, size_t n,
                                 const char* algorithm_name) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error: cannot write file '%s'\n", filename);
        return;
    }
    fprintf(f, "# Sorted by: %s\n", algorithm_name);
    fprintf(f, "%zu\n", n);
    for (size_t i = 0; i < n; ++i) {
        fprintf(f, "%d\n", arr[i]);
    }
    fclose(f);
    printf("  -> Sorted result written to '%s' (algorithm: %s)\n", filename, algorithm_name);
}

// 打印单条测试结果
static void print_result(const char* name, double ms, bool correct) {
    printf("  %-30s  %10.3f ms   %s\n",
           name, ms, correct ? "PASS" : "FAIL");
}

// ============================================================
// 主函数
// ============================================================
int main(int argc, char* argv[]) {
    // ---------------- 解析命令行参数 ----------------
    const char* save_algo = "merge";   // 默认输出归并排序结果
    bool gen_mode = false;
    size_t gen_n  = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "gen") == 0 && i + 1 < argc) {
            gen_mode = true;
            gen_n = static_cast<size_t>(std::atoll(argv[i + 1]));
            ++i;
        } else if (std::strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_algo = argv[i + 1];
            ++i;
        }
    }

    // ---------------- 生成模式: 创建 data.txt 后退出 ----------------
    if (gen_mode) {
        if (gen_n == 0) {
            fprintf(stderr, "Usage: %s gen <N>\n", argv[0]);
            return 1;
        }
        srand(static_cast<unsigned>(time(nullptr)));
        int* arr = generate_random_array(gen_n);
        printf("Generating %zu random integers to data.txt ...\n", gen_n);

        FILE* f = fopen("data.txt", "w");
        if (!f) {
            fprintf(stderr, "Error: cannot create data.txt\n");
            delete[] arr;
            return 1;
        }
        fprintf(f, "%zu\n", gen_n);
        for (size_t i = 0; i < gen_n; ++i) {
            fprintf(f, "%d\n", arr[i]);
        }
        fclose(f);
        delete[] arr;
        printf("Done. data.txt created.\n");
        return 0;
    }

    // ---------------- 测试模式: 从 data.txt 读取并运行排序测试 ----------------
    printf("=== Sorting Algorithm Test Harness ===\n\n");

    // 1. 读取测试数据
    size_t N = 0;
    int* original = read_array_from_file("data.txt", N);
    if (!original) {
        fprintf(stderr, "Please run '%s gen <N>' first to generate data.txt\n", argv[0]);
        return 1;
    }
    printf("Loaded %zu elements from data.txt\n\n", N);

    // 2. 存储各算法的排序后数组 (用于最终输出)
    int* saved_arr = nullptr;
    const char* saved_name = nullptr;

    // ============================================================
    // 串行排序测试 (C++)
    // ============================================================
    printf("--- Serial Sorting (C++) ---\n");

    // --- Merge Sort ---
    {
        int* arr = copy_array(original, N);
        auto t0 = Clock::now();
        serial_merge_sort(arr, N);
        auto t1 = Clock::now();
        bool ok = is_sorted(arr, N);
        double ms = static_cast<double>(
            std::chrono::duration_cast<Ms>(t1 - t0).count());
        print_result("serial_merge_sort", ms, ok);

        if (std::strcmp(save_algo, "merge") == 0) {
            saved_arr = arr;
            saved_name = "serial_merge_sort";
        } else {
            delete[] arr;
        }
    }

    // --- Quick Sort ---
    {
        int* arr = copy_array(original, N);
        auto t0 = Clock::now();
        serial_quick_sort(arr, N);
        auto t1 = Clock::now();
        bool ok = is_sorted(arr, N);
        double ms = static_cast<double>(
            std::chrono::duration_cast<Ms>(t1 - t0).count());
        print_result("serial_quick_sort", ms, ok);

        if (std::strcmp(save_algo, "quick") == 0) {
            saved_arr = arr;
            saved_name = "serial_quick_sort";
        } else {
            delete[] arr;
        }
    }

    // --- Selection Sort ---
    {
        if (N >= 1e5) {
            printf("  Skipping serial_selection_sort for large N (%zu) ...\n",  N);
        }
        else {
            int* arr = copy_array(original, N);
            auto t0 = Clock::now();
            serial_selection_sort(arr, N);
            auto t1 = Clock::now();
            bool ok = is_sorted(arr, N);
            double ms = static_cast<double>(
                std::chrono::duration_cast<Ms>(t1 - t0).count());
            print_result("serial_selection_sort", ms, ok);

            if (std::strcmp(save_algo, "selection") == 0) {
                saved_arr = arr;
                saved_name = "serial_selection_sort";
            } else {
                delete[] arr;
            }
        }
    }

    // ============================================================
    // GPU 并行排序测试 (CUDA)
    // ============================================================
#if HAS_CUDA
    printf("\n--- GPU Parallel Sorting (CUDA) ---\n");

    // --- Parallel Merge Sort ---
    {
        int* d_arr = nullptr;
        cudaMalloc(&d_arr, N * sizeof(int));
        cudaMemcpy(d_arr, original, N * sizeof(int), cudaMemcpyHostToDevice);

        auto t0 = Clock::now();
        parallel_merge_sort(d_arr, N);
        cudaDeviceSynchronize();
        auto t1 = Clock::now();

        int* result = new int[N];
        cudaMemcpy(result, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
        bool ok = is_sorted(result, N);
        double ms = static_cast<double>(
            std::chrono::duration_cast<Ms>(t1 - t0).count());
        print_result("parallel_merge_sort", ms, ok);

        if (std::strcmp(save_algo, "merge") == 0) {
            delete[] saved_arr;
            saved_arr = result;
            saved_name = "parallel_merge_sort";
        } else {
            delete[] result;
        }
        cudaFree(d_arr);
    }

    // --- Parallel Quick Sort ---
    {
        int* d_arr = nullptr;
        cudaMalloc(&d_arr, N * sizeof(int));
        cudaMemcpy(d_arr, original, N * sizeof(int), cudaMemcpyHostToDevice);

        auto t0 = Clock::now();
        parallel_quick_sort(d_arr, N);
        cudaDeviceSynchronize();
        auto t1 = Clock::now();

        int* result = new int[N];
        cudaMemcpy(result, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
        bool ok = is_sorted(result, N);
        double ms = static_cast<double>(
            std::chrono::duration_cast<Ms>(t1 - t0).count());
        print_result("parallel_quick_sort", ms, ok);

        if (std::strcmp(save_algo, "quick") == 0) {
            delete[] saved_arr;
            saved_arr = result;
            saved_name = "parallel_quick_sort";
        } else {
            delete[] result;
        }
        cudaFree(d_arr);
    }

    // --- Parallel Selection Sort ---
    {
            int* d_arr = nullptr;
            cudaMalloc(&d_arr, N * sizeof(int));
            cudaMemcpy(d_arr, original, N * sizeof(int), cudaMemcpyHostToDevice);

            auto t0 = Clock::now();
            parallel_selection_sort(d_arr, N);
            cudaDeviceSynchronize();
            auto t1 = Clock::now();

            int* result = new int[N];
            cudaMemcpy(result, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
            bool ok = is_sorted(result, N);
            double ms = static_cast<double>(
                std::chrono::duration_cast<Ms>(t1 - t0).count());
            print_result("parallel_selection_sort", ms, ok);

            if (std::strcmp(save_algo, "selection") == 0) {
                delete[] saved_arr;
                saved_arr = result;
                saved_name = "parallel_selection_sort";
            } else {
                delete[] result;
            }
            cudaFree(d_arr);
    }
#else
    printf("\n--- GPU Parallel Sorting (CUDA) ---\n");
    printf("  (CUDA not available — compile with nvcc or -DENABLE_CUDA)\n");
#endif

    // ============================================================
    // 3. 输出排序结果到 sorted.txt
    // ============================================================
    printf("\n");
    if (saved_arr) {
        write_array_to_file("sorted.txt", saved_arr, N, saved_name);
        delete[] saved_arr;
    } else {
        printf("  No result saved (algorithm '%s' not run)\n", save_algo);
    }

    // 清理
    delete[] original;
    printf("\n=== Done ===\n");
    return 0;
}

