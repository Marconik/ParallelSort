# ParallelSort

三种经典排序算法在 GPU 上的并行实现，包含 C++ 串行版本和 CUDA C 并行版本。

## 算法概览

| 算法 | 串行 (C++) | 并行 (CUDA) | 并行策略 |
|------|:---------:|:----------:|----------|
| **Merge Sort** | 自顶向下递归 | 块内 Bitonic Sort + 多轮 Merge Path 归并 | 分治 + co-rank 二分分界 |
| **Quick Sort** | 三数取中 + 三向切分 | 大段 CPU 分区 + 小段 GPU Bitonic Sort | 混合 CPU/GPU 分治 |
| **Selection Sort** | 原地选择 | 全对全比较排名 (Rank Sort) | 暴力并行比较 |

### Merge Sort（归并排序）

- **串行**：经典 $O(N \log N)$ 分治，递归二分到单元素后两两归并。
- **GPU 并行**：自底向上。阶段一：每个线程块在共享内存中用 **Batcher 双调排序网络** 对子数组局部排序；阶段二：多轮归并，利用 **Merge Path 对角线分解** 将归并任务均匀划分给各线程块，块间无通信。

### Quick Sort（快速排序）

- **串行**：三数取中选枢轴，三向切分（< pivot | == pivot | > pivot），尾递归优化。
- **GPU 并行**：混合策略。大段数据拷回主机做三向分区（保证正确性），中等段用 **Bitonic Sort** 在共享内存中搞定，小段直接 Bitonic 排序收尾。

### Selection Sort（选择排序）

- **串行**：经典 $O(N^2)$ 原地选择。
- **GPU 并行**：每个线程负责一个元素，与数组中所有其他元素比较确定其最终排名，然后一次性散射到输出数组的正确位置。适合小数据量。

## 构建与运行

```powershell
# 1. 编译 (需 CUDA Toolkit + VS Build Tools)
nvcc -O2 -std=c++17 -Iinclude src/sort_serial.cpp src/sort_parallel.cu src/main.cu -o sort_test.exe

# 2. 生成测试数据
./sort_test gen 65536

# 3. 运行所有排序算法并对比
./sort_test

# 4. 指定输出到 sorted.txt 的算法
./sort_test --save quick
```

## 依赖

- C++17 编译器
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 13.0+
- NVIDIA GPU（计算能力 ≥ 6.0）
- Windows 上需 Visual Studio Build Tools（提供 MSVC 宿主编译器）
