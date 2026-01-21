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
#ifdef USE_TensorRT
#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "src/base/base_infer.h"
#include "src/utils/ilogger.h"
#include "src/utils/pp_option.h"

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class TensorRTInfer : public BaseInfer {
public:
    // 修改：构造函数直接接受 engine 文件路径
    explicit TensorRTInfer(const std::string& engine_file_path,
                           const PaddlePredictorOption& option);
    ~TensorRTInfer();

    absl::StatusOr<std::vector<cv::Mat>>
    Apply(const std::vector<cv::Mat>& x) override;

private:
    bool Init(const std::string& engine_path);
    void GetBindings();

private:
    PaddlePredictorOption option_; // 保留 Option 以获取 DeviceId

    std::unique_ptr<TRTLogger> logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    cudaStream_t stream_ = nullptr;

    struct TensorInfo {
        std::string name;
        int index; // TRT 8.x 核心: Binding Index
        nvinfer1::DataType dtype;
        nvinfer1::Dims dims; 
        size_t size_bytes = 0;
        void* device_buffer = nullptr; 
        bool is_input = false;
    };

    std::vector<TensorInfo> tensors_;
    std::vector<void*> bindings_ptrs_; // 用于 enqueueV2
};

#endif