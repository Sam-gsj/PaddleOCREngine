// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/base/base_pipeline.h"
#include "thread_pool.h"

template <typename Pipeline, typename PipelineParams, typename PipelineResult>
class AutoParallelSimpleInferencePipeline : public BasePipeline {
private:
  struct InferenceInstance {
    std::shared_ptr<BasePipeline> pipeline;
    std::queue<std::function<void()>> task_queue;
    std::mutex queue_mutex;
    std::atomic<bool> is_busy{false};
    int instance_id;
  };

public:
  AutoParallelSimpleInferencePipeline(const PipelineParams &params);
  absl::Status Init();

  template <typename PipelineInput>
  std::future<PipelineResult> PredictAsync(const PipelineInput &input);

  template <typename PipelineInput>
  absl::Status PredictThread(const PipelineInput &input);
  absl::StatusOr<PipelineResult> GetResult();

  virtual ~AutoParallelSimpleInferencePipeline();

private:
  void ProcessInstanceTasks(int instance_id);
  PipelineParams params_;
  int thread_num_;

  std::atomic<int> round_robin_index_{0};
  std::unique_ptr<PaddlePool::ThreadPool> pool_;
  std::vector<std::unique_ptr<InferenceInstance>> instances_;

  std::queue<std::future<PipelineResult>> legacy_results_;
  std::mutex legacy_results_mutex_;
};

template <typename Pipeline, typename PipelineParams, typename PipelineResult>
AutoParallelSimpleInferencePipeline<Pipeline, PipelineParams, PipelineResult>::
    AutoParallelSimpleInferencePipeline(const PipelineParams &params)
    : BasePipeline(), params_(params), thread_num_(params.thread_num) {
  if (thread_num_ > 1) {
    auto status = Init();
    if (!status.ok()) {
      INFOE("Pipeline pool init error : %s", status.ToString().c_str());
      exit(-1);
    }
  }
}

template <typename Pipeline, typename PipelineParams, typename PipelineResult>
absl::Status
AutoParallelSimpleInferencePipeline<Pipeline, PipelineParams, PipelineResult>::Init() {
  try {
    pool_ = std::unique_ptr<PaddlePool::ThreadPool>(
        new PaddlePool::ThreadPool(thread_num_));

    for (int i = 0; i < thread_num_; i++) {
      auto instance =
          std::unique_ptr<InferenceInstance>(new InferenceInstance());
      instance->instance_id = i;

      instance->pipeline = std::shared_ptr<BasePipeline>(new Pipeline(params_));

      instances_.push_back(std::move(instance));
    }
  } catch (const std::bad_alloc &e) {
    return absl::ResourceExhaustedError(std::string("Out of memory: ") +
                                        e.what());
  } catch (const std::exception &e) {
    return absl::InternalError(std::string("Init failed: ") + e.what());
  }
  return absl::OkStatus();
}

template <typename Pipeline, typename PipelineParams, typename PipelineResult>
template <typename PipelineInput>
std::future<PipelineResult> AutoParallelSimpleInferencePipeline<
    Pipeline, PipelineParams, PipelineResult>::PredictAsync(const PipelineInput &input) {
      
  int instance_id = round_robin_index_.fetch_add(1) % thread_num_;
  auto &instance = instances_[instance_id];

  auto promise = std::make_shared<std::promise<PipelineResult>>();
  auto future = promise->get_future();

  auto task = [this, instance_id, input, promise]() {
    try {
      // auto* base_pipeline_ptr = this->instances_[instance_id]->pipeline.get();
      // auto* concrete_pipeline = static_cast<Pipeline*>(base_pipeline_ptr);

      auto result = this->instances_[instance_id]->pipeline->Predict(input);

      promise->set_value(std::move(result));

    } catch (...) {
      promise->set_exception(std::current_exception());
    }
  };

  {
    std::lock_guard<std::mutex> lock(instance->queue_mutex);
    instance->task_queue.push(task);
  }

  bool expected = false;
  if (instance->is_busy.compare_exchange_strong(expected, true)) {
    pool_->submit([this, instance_id]() { this->ProcessInstanceTasks(instance_id); });
  }

  return future;
}

template <typename Pipeline, typename PipelineParams, typename PipelineResult>
void AutoParallelSimpleInferencePipeline<
    Pipeline, PipelineParams, PipelineResult>::ProcessInstanceTasks(int instance_id) {
  auto &instance = instances_[instance_id];

  while (true) {
    std::function<void()> task;
    {
      std::lock_guard<std::mutex> lock(instance->queue_mutex);
      if (instance->task_queue.empty()) {
        instance->is_busy = false;
        break; 
      }
      task = std::move(instance->task_queue.front());
      instance->task_queue.pop();
    } 

    if (task) {
      task(); 
    }
  }
}

template <typename Pipeline, typename PipelineParams, typename PipelineResult>
template <typename PipelineInput>
absl::Status AutoParallelSimpleInferencePipeline<
    Pipeline, PipelineParams, PipelineResult>::PredictThread(const PipelineInput &input) {
  try {
    auto future = PredictAsync(input);

    std::lock_guard<std::mutex> lock(legacy_results_mutex_);
    legacy_results_.push(std::move(future));

    return absl::OkStatus();
  } catch (const std::exception &e) {
    return absl::InternalError(std::string("Failed to submit inference: ") +
                               e.what());
  }
}

template <typename Pipeline, typename PipelineParams, typename PipelineResult>
absl::StatusOr<PipelineResult>
AutoParallelSimpleInferencePipeline<Pipeline, PipelineParams, PipelineResult>::GetResult() {
  std::lock_guard<std::mutex> lock(legacy_results_mutex_);

  if (legacy_results_.empty())
    return absl::NotFoundError("No inference result available");

  try {
    auto future = std::move(legacy_results_.front());
    legacy_results_.pop();

    PipelineResult result = future.get();
    return result;
  } catch (const std::exception &e) {
    return absl::InternalError(std::string("Failed to get inference result: ") +
                               e.what());
  }
}

template <typename Pipeline, typename PipelineParams, typename PipelineResult>
AutoParallelSimpleInferencePipeline<
    Pipeline, PipelineParams, PipelineResult>::~AutoParallelSimpleInferencePipeline() {
  while (!legacy_results_.empty()) {
    try {
      legacy_results_.front().get();
    } catch (...) {
    }
    legacy_results_.pop();
  }

  for (auto &instance : instances_) {
    while (instance->is_busy.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}
