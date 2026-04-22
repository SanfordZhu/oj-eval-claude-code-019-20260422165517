#include "simulator.hpp"
#include "src.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
  sjtu::GpuSimulator gpu_sim;
  sjtu::MatrixMemoryAllocator matrix_memory_allocator;
  std::vector<sjtu::Matrix *> keys;
  std::vector<sjtu::Matrix *> values;
  std::vector<sjtu::Matrix *> queries;
  std::vector<sjtu::Matrix *> answers;

  const int n = 4;
  const int d = 8;

  for (int i = 0; i < n; ++i) {
    std::vector<float> k(d), v(d);
    for (int j = 0; j < d; ++j) {
      k[j] = static_cast<float>(rand()) / RAND_MAX;
      v[j] = static_cast<float>(rand()) / RAND_MAX;
    }
    keys.push_back(new sjtu::Matrix(1, d, k, gpu_sim));
    matrix_memory_allocator.Bind(keys.back(), "key_" + std::to_string(i));
    values.push_back(new sjtu::Matrix(1, d, v, gpu_sim));
    matrix_memory_allocator.Bind(values.back(), "value_" + std::to_string(i));
  }

  for (int i = 0; i < n; ++i) {
    std::vector<float> q((i + 1) * d), ans((i + 1) * d, 0.0f);
    for (int j = 0; j < (i + 1) * d; ++j) {
      q[j] = static_cast<float>(rand()) / RAND_MAX;
    }
    queries.push_back(new sjtu::Matrix(i + 1, d, q, gpu_sim));
    matrix_memory_allocator.Bind(queries.back(), "query_" + std::to_string(i));
    answers.push_back(new sjtu::Matrix(i + 1, d, ans, gpu_sim));
    matrix_memory_allocator.Bind(answers.back(), "answer_" + std::to_string(i));
  }

  sjtu::Rater rater(keys, values, queries, answers);
  sjtu::Test(rater, gpu_sim, matrix_memory_allocator);
  return 0;
}
