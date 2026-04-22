#pragma once
#include "simulator.hpp"
#include <vector>
#include <string>
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  std::vector<Matrix *> keysT(keys.size(), nullptr);

  for (size_t i = 0; i < keys.size(); ++i) {
    Matrix *Q = rater.GetNextQuery();
    gpu_sim.MoveMatrixToSharedMem(Q);

    // Prepare keys^T (d x 1) and values (1 x d) in shared up to index i
    for (size_t j = 0; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(values[j]);
      if (keysT[j] == nullptr) {
        gpu_sim.MoveMatrixToSharedMem(keys[j]);
        Matrix *kj = matrix_memory_allocator.Allocate("keyT_" + std::to_string(j));
        gpu_sim.Copy(keys[j], kj, kInSharedMemory);
        gpu_sim.Transpose(kj, kInSharedMemory); // now d x 1
        keysT[j] = kj;
      }
    }

    Matrix *answer = nullptr;

    // For each row in Q: softmax over dot products with keys, then weighted sum of values
    for (size_t r = 0; r < Q->GetRowNum(); ++r) {
      Matrix *row_q = matrix_memory_allocator.Allocate("row_q");
      gpu_sim.GetRow(Q, r, row_q, kInSharedMemory); // 1 x d

      // Build scores_row = [ q·k_j ] as 1 x (i+1)
      Matrix *scores_row = nullptr;
      for (size_t j = 0; j <= i; ++j) {
        Matrix *dot = matrix_memory_allocator.Allocate("dot");
        gpu_sim.MatMul(row_q, keysT[j], dot); // 1x1
        if (j == 0) {
          scores_row = matrix_memory_allocator.Allocate("scores_row");
          gpu_sim.Copy(dot, scores_row, kInSharedMemory);
        } else {
          Matrix *new_scores = matrix_memory_allocator.Allocate("scores_row_cat");
          gpu_sim.Concat(scores_row, dot, new_scores, 1, kInSharedMemory);
          gpu_sim.ReleaseMatrix(scores_row);
          scores_row = new_scores;
        }
        gpu_sim.ReleaseMatrix(dot);
      }

      // Softmax(scores_row)
      Matrix *row_exp = matrix_memory_allocator.Allocate("row_exp");
      gpu_sim.MatExp(scores_row, row_exp);
      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_exp, row_sum);
      Matrix *row_soft = matrix_memory_allocator.Allocate("row_soft");
      gpu_sim.MatDiv(row_exp, row_sum, row_soft);
      gpu_sim.ReleaseMatrix(scores_row);
      gpu_sim.ReleaseMatrix(row_exp);
      gpu_sim.ReleaseMatrix(row_sum);

      // Weighted sum: row_soft (1 x (i+1)) * V_acc ( (i+1) x d ) => 1 x d
      // Build V_acc by vertical concat in shared for indices 0..i
      Matrix *V_acc = nullptr;
      for (size_t j = 0; j <= i; ++j) {
        if (j == 0) {
          V_acc = matrix_memory_allocator.Allocate("V_acc");
          gpu_sim.Copy(values[0], V_acc, kInSharedMemory);
        } else {
          Matrix *new_V = matrix_memory_allocator.Allocate("V_acc_cat");
          gpu_sim.Concat(V_acc, values[j], new_V, 0, kInSharedMemory);
          gpu_sim.ReleaseMatrix(V_acc);
          V_acc = new_V;
        }
      }

      Matrix *row_ans = matrix_memory_allocator.Allocate("row_ans");
      gpu_sim.MatMul(row_soft, V_acc, row_ans);
      gpu_sim.ReleaseMatrix(V_acc);
      gpu_sim.ReleaseMatrix(row_soft);
      gpu_sim.ReleaseMatrix(row_q);

      // Append to final answer
      if (r == 0) {
        answer = matrix_memory_allocator.Allocate("answer");
        gpu_sim.Copy(row_ans, answer, kInSharedMemory);
      } else {
        Matrix *new_answer = matrix_memory_allocator.Allocate("answer_cat");
        gpu_sim.Concat(answer, row_ans, new_answer, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(answer);
        answer = new_answer;
      }
      gpu_sim.ReleaseMatrix(row_ans);
    }

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
