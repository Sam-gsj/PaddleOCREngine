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
#ifdef USE_NCNN
#pragma once

#include "src/base/base_infer.h"
#include "src/utils/ilogger.h"
#include "src/utils/pp_option.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

// NCNN核心头文件
#include <ncnn/allocator.h>
#include <ncnn/mat.h>
#include <ncnn/net.h>

class NCNNInfer : public BaseInfer {
public:
  explicit NCNNInfer(const std::string &model_name,
                     const std::string &model_dir,
                     const PaddlePredictorOption &option);
  ~NCNNInfer() = default;

  absl::StatusOr<std::vector<cv::Mat>>
  Apply(const std::vector<cv::Mat> &x) override;

private:
  bool Init(const std::string &param_path, const std::string &bin_path);

  void InitBlobs();

private:
  PaddlePredictorOption option_;

  std::unique_ptr<ncnn::Net> net_;

  struct BlobInfo {
    std::string name;
    int w = 0, h = 0, c = 0;
    bool is_input = false;
  };

  std::vector<BlobInfo> blobs_;
  std::vector<BlobInfo *> input_blobs_;
  std::vector<BlobInfo *> output_blobs_;
};

#endif
