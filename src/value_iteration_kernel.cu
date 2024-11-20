#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

#include "value_iteration2/ValueIterator.h"

using namespace value_iteration2;

__device__ int toIndex(int ix, int iy, int it, int cell_num_x, int cell_num_t) {
  return it + ix * cell_num_t + iy * (cell_num_t * cell_num_x);
}

__device__ uint64_t actionCost(State& s, Action& a, State* states,
                               uint64_t max_cost, int num_states,
                               int cell_num_x, int cell_num_y, int cell_num_t,
                               uint64_t prob_base_bit) {
  uint64_t cost = 0;
  for (auto& tran : a._state_transitions[s.it_]) {
    int ix = s.ix_ + tran._dix;
    if (ix < 0 || ix >= cell_num_x) return max_cost;

    int iy = s.iy_ + tran._diy;
    if (iy < 0 || iy >= cell_num_y) return max_cost;

    int it = (tran._dit + cell_num_t) % cell_num_t;

    auto& after_s = states[toIndex(ix, iy, it, cell_num_x, cell_num_t)];
    if (!after_s.free_) return max_cost;

    cost += (after_s.total_cost_ + after_s.penalty_ + after_s.local_penalty_) *
            tran._prob;
  }

  return cost >> prob_base_bit;
}

__global__ void valueIterationKernel(State* states, Action* actions,
                                     int num_states, int num_actions,
                                     uint64_t max_cost, int cell_num_x,
                                     int cell_num_y, int cell_num_t,
                                     uint64_t prob_base_bit) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_states) {
    State& s = states[idx];
    if (!s.free_ || s.final_state_) return;

    uint64_t min_cost = max_cost;
    Action* min_action = nullptr;

    for (int a_idx = 0; a_idx < num_actions; ++a_idx) {
      Action& a = actions[a_idx];
      uint64_t cost = actionCost(s, a, states, max_cost, num_states, cell_num_x,
                                 cell_num_y, cell_num_t, prob_base_bit);

      if (cost < min_cost) {
        min_cost = cost;
        min_action = &a;
      }
    }

    s.total_cost_ = min_cost;
    s.optimal_action_ = min_action;
  }
}

__global__ void setStateKernel(State* states, nav_msgs::msg::OccupancyGrid map,
                               double safety_radius, int cell_num_x,
                               int cell_num_y, int cell_num_t) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < cell_num_x * cell_num_y * cell_num_t) {
    int ix = idx % cell_num_x;
    int iy = (idx / cell_num_x) % cell_num_y;
    int it = idx / (cell_num_x * cell_num_y);

    unsigned int cost = (unsigned int)(map.data[ix + cell_num_x * iy] & 0xFF);
    states[idx] = State(ix, iy, it, cost);
  }
}
