// Copyright (c) 2026 badguy Authors. All Rights Reserved.
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
#ifdef USE_ONNX
#pragma once

#include "src/base/base_infer.h"
#include "src/utils/ilogger.h"
#include "src/utils/pp_option.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// ONNX Runtime C++ API
#include <onnxruntime_cxx_api.h>

class ONNXInfer : public BaseInfer {
public:
  explicit ONNXInfer(const std::string &model_name,
                     const std::string &model_dir,
                     const PaddlePredictorOption &option);
  ~ONNXInfer() override;

  absl::StatusOr<std::vector<cv::Mat>>
  Apply(const std::vector<cv::Mat> &x) override;

private:
  bool Init(const std::string &onnx_path);

  void InitNodes();

private:
  PaddlePredictorOption option_;

  std::shared_ptr<Ort::Env> env_;
  std::shared_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;

  std::vector<std::string> input_node_names_;
  std::vector<const char *> input_node_names_ptr_;

  std::vector<std::string> output_node_names_;
  std::vector<const char *> output_node_names_ptr_;

  std::shared_ptr<Ort::MemoryInfo> memory_info_;
};

#endif // USE_ONNX
