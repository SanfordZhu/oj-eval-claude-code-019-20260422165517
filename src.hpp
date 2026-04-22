#pragma once
#include "simulator.hpp"
#include <vector>
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  std::vector<Matrix *> keysT(keys.size(), nullptr);

  for (size_t i = 0; i < keys.size(); ++i) {
    Matrix *Q = rater.GetNextQuery();
    gpu_sim.MoveMatrixToSharedMem(Q);

    // Prepare transposed keys up to i in shared
    for (size_t j = 0; j <= i; ++j) {
      if (keysT[j] == nullptr) {
        gpu_sim.MoveMatrixToSharedMem(keys[j]);
        Matrix *kj = matrix_memory_allocator.Allocate("keyT_" + std::to_string(j));
        gpu_sim.Copy(keys[j], kj, kInSharedMemory);
        gpu_sim.Transpose(kj, kInSharedMemory); // now d x 1
        keysT[j] = kj;
        gpu_sim.MoveMatrixToGpuHbm(keys[j]); // keep original in HBM
      }
    }

    Matrix *answer = nullptr;

    // For each row in Q, compute softmax over dot products, then weighted sum of values
    for (size_t r = 0; r < Q->GetRowNum(); ++r) {
      Matrix *row_q = matrix_memory_allocator.Allocate("row_q");
      gpu_sim.GetRow(Q, r, row_q, kInSharedMemory);

      // Build scores row: [ (row_q · key_j) ]_{j=0..i }
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

      // Softmax over scores_row
      Matrix *row_exp = matrix_memory_allocator.Allocate("row_exp");
      gpu_sim.MatExp(scores_row, row_exp);
      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_exp, row_sum);
      Matrix *row_soft = matrix_memory_allocator.Allocate("row_soft");
      gpu_sim.MatDiv(row_exp, row_sum, row_soft);
      gpu_sim.ReleaseMatrix(scores_row);
      gpu_sim.ReleaseMatrix(row_exp);
      gpu_sim.ReleaseMatrix(row_sum);

      // Weighted sum of values: sum_j soft[r,j] * V[j]
      Matrix *row_ans = nullptr;
      for (size_t j = 0; j <= i; ++j) {
        Matrix *factor = matrix_memory_allocator.Allocate("factor");
        gpu_sim.GetColumn(row_soft, j, factor, kInSharedMemory); // 1x1

        gpu_sim.MoveMatrixToSharedMem(values[j]);
        Matrix *scaled = matrix_memory_allocator.Allocate("scaled_v");
        gpu_sim.MatMulNum(values[j], factor, scaled); // 1xd

        if (j == 0) {
          row_ans = matrix_memory_allocator.Allocate("row_ans");
          gpu_sim.Copy(scaled, row_ans, kInSharedMemory);
        } else {
          Matrix *new_row_ans = matrix_memory_allocator.Allocate("row_ans_sum");
          gpu_sim.MatAdd(row_ans, scaled, new_row_ans);
          gpu_sim.ReleaseMatrix(row_ans);
          row_ans = new_row_ans;
        }
        gpu_sim.ReleaseMatrix(scaled);
        gpu_sim.ReleaseMatrix(factor);
        gpu_sim.MoveMatrixToGpuHbm(values[j]);
      }

      // Append row_ans to final answer
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
      gpu_sim.ReleaseMatrix(row_q);
      gpu_sim.ReleaseMatrix(row_soft);
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
