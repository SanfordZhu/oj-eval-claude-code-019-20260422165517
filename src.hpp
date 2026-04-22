#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto Q = rater.GetNextQuery();

    // Move Q to shared for computation
    gpu_sim.MoveMatrixToSharedMem(Q);

    // Build K_acc and V_acc by concatenating first (i+1) rows in HBM, then move to shared
    Matrix *K_acc = nullptr;
    Matrix *V_acc = nullptr;

    for (size_t j = 0; j <= i; ++j) {
      if (j == 0) {
        K_acc = matrix_memory_allocator.Allocate("K_acc_0");
        V_acc = matrix_memory_allocator.Allocate("V_acc_0");
        // Copy in HBM to initialize accumulators
        gpu_sim.Copy(keys[0], K_acc, kInGpuHbm);
        gpu_sim.Copy(values[0], V_acc, kInGpuHbm);
      } else {
        Matrix *new_K = matrix_memory_allocator.Allocate("K_acc_concat");
        Matrix *new_V = matrix_memory_allocator.Allocate("V_acc_concat");
        // Concatenate vertically along rows in HBM to minimize shared memory usage
        gpu_sim.Concat(K_acc, keys[j], new_K, 0, kInGpuHbm);
        gpu_sim.Concat(V_acc, values[j], new_V, 0, kInGpuHbm);
        gpu_sim.ReleaseMatrix(K_acc);
        gpu_sim.ReleaseMatrix(V_acc);
        K_acc = new_K;
        V_acc = new_V;
      }
    }

    // Transpose K_acc (still in HBM), then move both K_acc and V_acc to shared for matmuls
    gpu_sim.Transpose(K_acc, kInGpuHbm);

    gpu_sim.MoveMatrixToSharedMem(K_acc);
    gpu_sim.MoveMatrixToSharedMem(V_acc);

    // Compute scores = Q * K_acc (Q in shared, K_acc in shared)
    Matrix *scores = matrix_memory_allocator.Allocate("scores_QKt");
    gpu_sim.MatMul(Q, K_acc, scores);

    // Softmax per row on scores in shared
    Matrix *softmax_acc = nullptr;
    for (size_t r = 0; r < Q->GetRowNum(); ++r) {
      Matrix *row_r = matrix_memory_allocator.Allocate("row_scores");
      gpu_sim.GetRow(scores, r, row_r, kInSharedMemory);

      Matrix *row_exp = matrix_memory_allocator.Allocate("row_exp");
      gpu_sim.MatExp(row_r, row_exp);

      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_exp, row_sum);

      Matrix *row_soft = matrix_memory_allocator.Allocate("row_soft");
      gpu_sim.MatDiv(row_exp, row_sum, row_soft);

      if (r == 0) {
        softmax_acc = matrix_memory_allocator.Allocate("softmax_acc");
        gpu_sim.Copy(row_soft, softmax_acc, kInSharedMemory);
      } else {
        Matrix *new_soft = matrix_memory_allocator.Allocate("softmax_concat");
        gpu_sim.Concat(softmax_acc, row_soft, new_soft, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_acc);
        softmax_acc = new_soft;
      }
      gpu_sim.ReleaseMatrix(row_r);
      gpu_sim.ReleaseMatrix(row_exp);
      gpu_sim.ReleaseMatrix(row_sum);
      gpu_sim.ReleaseMatrix(row_soft);
    }

    // Final answer = softmax(QK^T) * V_acc
    Matrix *answer = matrix_memory_allocator.Allocate("answer");
    gpu_sim.MatMul(softmax_acc, V_acc, answer);

    // Cleanup temporaries
    gpu_sim.ReleaseMatrix(scores);
    gpu_sim.ReleaseMatrix(softmax_acc);
    gpu_sim.ReleaseMatrix(K_acc);
    gpu_sim.ReleaseMatrix(V_acc);

    // Move answer to HBM, run simulator, then commit
    gpu_sim.MoveMatrixToGpuHbm(answer);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
