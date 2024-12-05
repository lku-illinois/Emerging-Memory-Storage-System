
#include "analysis.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include <algorithm>
#include <fstream>
#include <iostream>

#include "ast.h"
#include "simulationUtils.h"
#include "simulator.h"
#include "printUtils.h"
#include <ctime>
#include <tuple>
#include <cstdlib> // For std::system

using Simulator::TensorMovementHint;
using Simulator::TensorLocation;

extern std::string migration_policy_str;
extern std::string eviction_policy_str;

extern double GPU_frequency_GHz;
extern double GPU_memory_size_GB;
extern double CPU_PCIe_bandwidth_GBps;
extern double SSD_PCIe_bandwidth_GBps;
extern double GPU_malloc_uspB;
// extern double GPU_free_uspB;
extern int prefetch_degree;
extern int borden;
extern int is_transformer;
double CPU_memory_line_GB = -1;
double SSD_latency_us = -1;
double system_latency_us = -1;
double delta_parameter = -1;
double loosen_parameter = 1;

long long memory_offset_intermediate = 0;
long long memory_offset_weights = 0;
int kernel_index = 0;
int prefetch_optimize = 1;
std::vector<Tensor *> tensor_list;
std::vector<CUDAKernel> kernel_list;

// TODO: Important: The global list for all inactive tensor periods
std::vector<InactivePeriod *> inactive_periods_list;

std::vector<double> kernel_time_table;
std::vector<EvictionGuideEntry> EvictionGuideTable;
std::vector<long> GPU_resident_memory_estimation;
std::vector<long> CPU_resident_memory_estimation;

std::vector<TensorMovementHint> movement_hints;
std::vector<InactivePeriod *> offloaded_local_intervals;

string Tensor::name() const { return "tensor" + std::to_string(tensor_id); }

bool Tensor::is_alive(int current_kernel) const {
  return is_global_weight || (live_interval.second == -1 ? current_kernel == live_interval.first
                                                         : current_kernel >= live_interval.first &&
                                                               current_kernel < live_interval.second);
}

void Tensor::print() const {
  std::cout << "tensor" << tensor_id << " Is weight (global)?: " << this->is_global_weight << ", "
            << "Size in byte: " << size_in_byte << std::endl;
}

Tensor::Tensor(long long size, bool glob) {
  static int tensor_count = 0;
  tensor_id = tensor_count++;
  size_in_byte = size;
  raw_size_byte = size;
  is_global_weight = glob;
  if (glob) {
    address_offset = memory_offset_weights;
    // page-level alignment
    long N_pages = (size % 4096 == 0) ? (size / 4096) : ((size / 4096) + 1);
    memory_offset_weights += N_pages * 4096;
    size_in_byte = N_pages * 4096;
  } else {
    address_offset = memory_offset_intermediate;
    // page-level alignment
    long N_pages = (size % 4096 == 0) ? (size / 4096) : ((size / 4096) + 1);
    memory_offset_intermediate += N_pages * 4096;
    size_in_byte = N_pages * 4096;
  }
}

unsigned long Tensor::getGlobalOffset() {
  return address_offset + (is_global_weight ? 0 : memory_offset_weights);
}

CUDAKernel::CUDAKernel(int kernel_id, CUDAKernelType t, std::vector<Tensor *> input_tensor_list,
                       std::vector<Tensor *> output_tensor_list, Tensor *workspace_tensor) {
  this->kernel_id = kernel_id;
  this->type = t;
  this->inputs.insert(input_tensor_list.begin(), input_tensor_list.end());
  this->outputs.insert(output_tensor_list.begin(), output_tensor_list.end());
  this->workspace = workspace;
}

void CUDAKernel::print() {
  std::cout << "---------------------------------------------------------------"
               "---------------"
            << std::endl;
  std::cout << "Kernel ID: " << kernel_id << ", "
            << "Name: " << print_kerneltype_array[type] << std::endl;
  std::cout << "Execution Time:            " << execution_cycles << std::endl;
  std::cout << "Input Tensors:" << std::endl;
  for (auto it = inputs.begin(); it != inputs.end(); it++) {
    (*it)->print();
  }
  std::cout << "Output Tensors:" << std::endl;
  for (auto it = outputs.begin(); it != outputs.end(); it++) {
    (*it)->print();
  }
}

void CUDAKernel::getRequiredTensors(std::vector<Tensor *> &required_tensors) const {
  std::unordered_set<Tensor *> set;
  getRequiredTensors(set);
  for (Tensor *tensor : set) required_tensors.push_back(tensor);
}

void CUDAKernel::getRequiredTensors(std::unordered_set<Tensor *> &required_tensors) const {
  for (Tensor *tensor : inputs) required_tensors.insert(tensor);
  for (Tensor *tensor : outputs) required_tensors.insert(tensor);
}

void CUDAKernel::getRequiredTensors(std::vector<Tensor *> &required_tensors,
                                    std::vector<Tensor *> &required_input_tensors,
                                    std::vector<Tensor *> &required_output_tensors) const {
  std::unordered_set<Tensor *> set;
  for (Tensor *tensor : inputs) {
    set.insert(tensor);
    required_tensors.push_back(tensor);
    required_input_tensors.push_back(tensor);
  }
  for (Tensor *tensor : outputs) {
    if (set.find(tensor) == set.end()) {
      required_tensors.push_back(tensor);
      required_output_tensors.push_back(tensor);
    }
  }
}

/**
 * @brief this function is used to fill the liveness information of every tensor
 * @todo you should fill the field live_interval for each tensor in the tensor_list
 *       see descriptions of live_interval in Tensor::live_interval
 */
void tensor_first_pass_liveness_analysis() {
  const int tensor_num = tensor_list.size();
  const int kernel_num = kernel_list.size();

  for (int i = 0; i < tensor_num; i++) {
    Tensor *current_tensor = tensor_list[i];
    // TODO: complete liveness analysis
    if (!current_tensor->is_global_weight) {
      // This tensor is intermediate
      current_tensor->live_interval = {kernel_num, -1}; // Set to max values initially

      for (int j = 0; j < kernel_num; j++) {
        CUDAKernel current_kernel = kernel_list[j];

        // Check if the tensor is used in this kernel
        std::vector<Tensor*> required_tensors;
        current_kernel.getRequiredTensors(required_tensors);

        for (Tensor *tensor : required_tensors) {
          if (tensor == current_tensor) {
            // Update live_interval
            current_tensor->live_interval.first = std::min(current_tensor->live_interval.first, j);
            current_tensor->live_interval.second = std::max(current_tensor->live_interval.second, j+1);
          }
        }
      }

      if (current_tensor->live_interval.first == kernel_num) {
        current_tensor->live_interval = {-1, -1};
      }

    }
    // global tensors do not need this info
  }

}

void Tensor::print_liveness() {
  this->print();
  if (!this->is_global_weight) {
    std::cout << "Liveness: Birth: " << this->live_interval.first << ", Death: " << this->live_interval.second
              << "." << std::endl;
  } else {
    std::cout << "Liveness: Global" << std::endl;
  }
}

/**
 * @brief this function is used to fill the inactive period information of every tensor
 * @todo you should fill the field inactive_periods for each tensor in the tensor_list
 *       see descriptions of inactive_periods in Tensor::inactive_periods
 */
void tensor_second_pass_interval_formation() {
  const int tensor_num = tensor_list.size();
  const int kernel_num = kernel_list.size();

  for (int i = 0; i < tensor_num; i++) {
    Tensor *current_tensor = tensor_list[i];
    // TODO: complete inactive period analysis
    if (!current_tensor->is_global_weight) {
      // This tensor is intermediate

      // Clear any existing inactive periods
      current_tensor->inactive_periods.clear();

      int start = current_tensor->live_interval.first;
      int end = current_tensor->live_interval.second;

      // Add inactive period before the active range
      if (start > 0) {
        InactivePeriod *inactive_before = new InactivePeriod(current_tensor);
        inactive_before->kernelLevel_interval = {0, start};
        current_tensor->inactive_periods.push_back(inactive_before);
      }

      // Add inactive period after the active range
      if (end < kernel_num) {
        InactivePeriod *inactive_after = new InactivePeriod(current_tensor);
        inactive_after->kernelLevel_interval = {end, kernel_num};
        current_tensor->inactive_periods.push_back(inactive_after);
      }

    } else {
      // This tensor is global
      InactivePeriod *no_inactive = new InactivePeriod(current_tensor);
      no_inactive->kernelLevel_interval = {1, -1};
      no_inactive->is_looped = true;
      current_tensor->inactive_periods.push_back(no_inactive);
    }
  }
}

void Tensor::print_inactive_periods() {
  // print();
  std::cout << "Inactive Periods:" << std::endl;
  for (int i = 0; i < inactive_periods.size(); i++) {
    std::cout << "interval " << i << ": " << inactive_periods[i]->kernelLevel_interval.first << "--------"
              << inactive_periods[i]->kernelLevel_interval.second << std::endl;
    std::cout << "Estimated Time:" << inactive_periods[i]->time_estimated << std::endl;
  }
  std::cout << "_______________________________________________________________" << std::endl;
}

// A provided compiler pass to calculate the estimated execution time for every
// tensors' inactive period length(time)
void get_inactive_periods_time() {
  int kernel_num = kernel_list.size();

  // Setup a cumulative time list;
  double time = 0;
  kernel_time_table.push_back(0);
  for (int i = 0; i < kernel_num; i++) {
    time += (double)kernel_list[i].execution_cycles / (double)(GPU_frequency_GHz * 1000);
    kernel_time_table.push_back(time);
  }

  // Fill the looped extend kernel time table      0 - 2 * kernel_num
  std::vector<double> kernel_time_table_extended;
  kernel_time_table_extended.resize(kernel_num);
  for (int j = 0; j < kernel_num; j++) {
    kernel_time_table_extended[j] = kernel_time_table[j];
  }
  double last_time = kernel_time_table[kernel_num];
  kernel_time_table_extended.push_back(last_time);
  for (int j = 0; j < kernel_num; j++) {
    last_time += (double)kernel_list[j].execution_cycles / (double)(GPU_frequency_GHz * 1000);
    kernel_time_table_extended.push_back(last_time);
  }

  for (int i = 0; i < inactive_periods_list.size(); i++) {
    if (!inactive_periods_list[i]->is_looped) {
      assert(inactive_periods_list[i]->kernelLevel_interval.second >
             inactive_periods_list[i]->kernelLevel_interval.first);
      inactive_periods_list[i]->time_estimated =
          kernel_time_table[inactive_periods_list[i]->kernelLevel_interval.second] -
          kernel_time_table[inactive_periods_list[i]->kernelLevel_interval.first];
    } else {
      assert(inactive_periods_list[i]->kernelLevel_interval.second <
             inactive_periods_list[i]->kernelLevel_interval.first);
      int end = inactive_periods_list[i]->kernelLevel_interval.second;
      int start = inactive_periods_list[i]->kernelLevel_interval.first;
      end += kernel_num;
      inactive_periods_list[i]->time_estimated =
          kernel_time_table_extended[end] - kernel_time_table_extended[start];
    }
  }
}

void InactivePeriod::print() {
  std::cout << "interval " << ": " << kernelLevel_interval.first << "--------" << kernelLevel_interval.second
            << std::endl;
  std::cout << "Estimated Time:" << time_estimated << std::endl;
  std::cout << "Tensor: ";
  this->tensor_back_ptr->print();
  std::cout << "_______________________________________________________________" << std::endl;
}

void print_GPU_mem_really_in_use() {
  for (int i = 0; i < kernel_list.size(); i++) {
    std::vector<Tensor *> r;
    kernel_list[i].getRequiredTensors(r);
    long size_bite = 0;
    for (int j = 0; j < r.size(); j++) {
      size_bite += r[j]->size_in_byte;
    }
    std::cout << "Kernel " << i << ": " << size_bite << std::endl;
  }
}

/**
 * @brief fill this function to schedule your movement hints
 */
void scheduling_movement_hints() {
  // TODO: fill the data structure "std::vector<TensorMovementHint> movement_hints" with your own hints!


  // make sure the movement hints are sorted, the simulator depends on this
  std::sort(movement_hints.begin(), movement_hints.end());
}




//////////////////////////////////// CHAPTER 3 /////////////////////////////////////////////////
void print_liveness_and_inactive_periods(){
  const int tensor_num = tensor_list.size();

  for (int i = 0; i < tensor_num; i++) {
    Tensor *current_tensor = tensor_list[i];
    // Print live_interval for the current tensor
    current_tensor->print_liveness();
    current_tensor->print_inactive_periods();
  }
}




void calculate_minimum_memory_demand() {
  const int kernel_num = kernel_list.size();
  std::vector<long long> memory_usage(kernel_num, 0);

  for (int i = 0; i < tensor_list.size(); i++) {
    Tensor *tensor = tensor_list[i];
    if (tensor->live_interval.first == -1) {
        continue; // Skip unused tensors
    }

    // Add the tensor's memory to all kernels in its active interval
    for (int k = tensor->live_interval.first; k < tensor->live_interval.second; k++) {
        memory_usage[k] += tensor->size_in_byte;
    }
  }

  // Find the maximum memory usage and the corresponding kernel
  auto max_it = std::max_element(memory_usage.begin(), memory_usage.end());
  long long max_memory = *max_it;
  int max_kernel = std::distance(memory_usage.begin(), max_it); // Get the index of the max value

  std::cout << "Minimum Memory Demand: " << max_memory / (1024.0 * 1024.0) << " MB" << std::endl;
  std::cout << "Kernel with Maximum Memory Usage: " << max_kernel << std::endl;
  std::cout << "_______________________________________________________________" << std::endl;
}


void plot_tensor_size_vs_lifetime() {
  // size_byte(long long) vs livetime(int)
  std::vector<std::pair<long long, int>> data;

  for (int i = 0; i < tensor_list.size(); i++) {
      Tensor *tensor = tensor_list[i];
      if (tensor->live_interval.first == -1) {
          continue; // Skip unused tensors
      }

      int lifetime = tensor->live_interval.second - tensor->live_interval.first;
      data.push_back({tensor->size_in_byte, lifetime});
  }

  // Print data for plotting
  std::cout << "Plot Tensor Size vs Lifetime: " << std::endl;
  for (const auto &entry : data) {
      std::cout << "Size(MB): " << entry.first / (1024.0 * 1024.0) << ", Lifetime: " << entry.second << std::endl;
  }
  
  
  
  // Export data to a file or integrate with Python for plotting
  // Output data to a CSV file
  // Generate unique file name using timestamp
  std::time_t now = std::time(nullptr);
  std::tm *ltm = std::localtime(&now);
  std::string file_name = "tensor_size_vs_lifetime_" +
                        std::to_string(ltm->tm_hour) + "_" +
                        std::to_string(ltm->tm_min) + "_" +
                        std::to_string(ltm->tm_sec) + ".csv";

  // Output data to a unique CSV file
  std::ofstream file(file_name);
  if (!file.is_open()) {
      std::cerr << "Failed to open file for writing." << std::endl;
      return;
  }

  // Write header
  file << "Size_MB,Lifetime\n";

  // Write data
  for (const auto &entry : data) {
      file << entry.first / (1024.0 * 1024.0) << "," << entry.second << "\n";
  }

  file.close();

  std::cout << "Data exported to " << file_name << " for plotting." << std::endl;

  // new
  // Call the Python script to generate the plot
  std::string command = "python3 plot_tensor_vs_lifetime.py " + file_name;
  int result = std::system(command.c_str());

  if (result != 0) {
      std::cerr << "Failed to run the Python script. Please check the Python file." << std::endl;
  }

  std::cout << "_______________________________________________________________" << std::endl;
}


void plot_active_inactive_distribution() {
  // tensor id(int) vs active time(int) vs inactive time(int)
  std::vector<std::tuple<int,int,int>> data;

  for (int i = 0; i < tensor_list.size(); i++) {
    Tensor *tensor = tensor_list[i];
    if (tensor->live_interval.first == -1) {
        continue; // Skip unused tensors
    }

    int tensor_id_ = tensor->tensor_id;
    int active_time = tensor->live_interval.second - tensor->live_interval.first;
    int inactive_time = 0;
    for (const auto &period : tensor->inactive_periods) {
        inactive_time += period->kernelLevel_interval.second - period->kernelLevel_interval.first;
    }

    data.push_back(std::make_tuple(tensor_id_, active_time, inactive_time));

    std::cout << "Tensor " << tensor_id_ << ": Active Time: " << active_time
              << ", Inactive Time: " << inactive_time << std::endl;
  }
  

  // Export data to a file or integrate with Python for plotting
  // Output data to a CSV file
  // Generate unique file name using timestamp
  std::time_t now = std::time(nullptr);
  std::tm *ltm = std::localtime(&now);
  std::string file_name = "active_time_vs_inactive_time" +
                        std::to_string(ltm->tm_hour) + "_" +
                        std::to_string(ltm->tm_min) + "_" +
                        std::to_string(ltm->tm_sec) + ".csv";

  // Output data to a unique CSV file
  std::ofstream file(file_name);
  if (!file.is_open()) {
      std::cerr << "Failed to open file for writing." << std::endl;
      return;
  }

  // Write header
  file << "Tensor_id,ActiveTime,InactiveTime\n";

  // Write data
  for (const auto &entry : data) {
    file << std::get<0>(entry) << ","          // First element
         << std::get<1>(entry) << ","          // Second element
         << std::get<2>(entry) << "\n";        // Third element
  }

  file.close();

  std::cout << "Data exported to " << file_name << " for plotting." << std::endl;

  // new
  // Call the Python script to generate plots
  std::string command = "python3 plot_active_vs_inactive.py " + file_name;
  int result = std::system(command.c_str());

  if (result != 0) {
      std::cerr << "Failed to run the Python script. Please check the Python file." << std::endl;
  }

  std::cout << "_______________________________________________________________" << std::endl;
}


void calculate_memory_with_swapping() {
  const int kernel_num = kernel_list.size();
  const int tensor_num = tensor_list.size();
  std::vector<long long> memory_usage(kernel_num, 0);

  for (int k = 0; k < kernel_num; k++) {
    std::vector<Tensor*> required_tensors;
    CUDAKernel current_kernel = kernel_list[k];

    // Get tensors required for this kernel
    current_kernel.getRequiredTensors(required_tensors);

    for (int i = 0; i < tensor_num; i++) {
        Tensor *current_tensor = tensor_list[i];

        // Check if the tensor is within its live interval
        if (current_tensor->live_interval.first <= k && current_tensor->live_interval.second > k) {
            // Check if the tensor is required in this kernel
            if (std::find(required_tensors.begin(), required_tensors.end(), current_tensor) != required_tensors.end()) {
                // Tensor is actively used
                memory_usage[k] += current_tensor->size_in_byte;
            }
        }
    }
  }

  // Find the peak memory usage with swapping
  long long max_memory = *std::max_element(memory_usage.begin(), memory_usage.end());
  std::cout << "Memory Demand with Swapping: " << max_memory / (1024.0 * 1024.0) << " MB" << std::endl;
  std::cout << "_______________________________________________________________" << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////