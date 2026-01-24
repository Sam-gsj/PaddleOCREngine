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
#ifdef USE_RKNN
#pragma once

#include "src/base/base_infer.h"
#include "src/utils/ilogger.h"
#include "src/utils/pp_option.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// RKNN C API
#include "rknn_api.h"

class RKNNInfer : public BaseInfer {
public:
  explicit RKNNInfer(const std::string &model_name,
                     const std::string &model_dir,
                     const PaddlePredictorOption &option);
  ~RKNNInfer() override;

  absl::StatusOr<std::vector<cv::Mat>>
  Apply(const std::vector<cv::Mat> &x) override;

private:
  // 加载模型并初始化上下文
  bool Init(const std::string &rknn_path);

  // 查询输入输出 tensor 的属性
  void QueryModelInfo();

private:
  PaddlePredictorOption option_;

  rknn_context ctx_ = 0; // RKNN 上下文句柄

  // 输入输出参数缓存
  rknn_input_output_num io_num_;
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;

  // 内部缓存，减少重复分配
  std::vector<rknn_input> inputs_structs_;
  std::vector<rknn_output> outputs_structs_;
  unsigned char *model_data_ = nullptr; // 模型文件内存
};

#endif // USE_RKNN
