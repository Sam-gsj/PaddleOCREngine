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
#include "tensorrt_infer.h"
#include <fstream>
#include <iostream>
#include <numeric>

#define CHECK_CUDA(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            std::cerr << "Cuda failure: " << ret << " at line " << __LINE__ << std::endl; \
            abort(); \
        } \
    } while (0)

void TRTLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << "[TRT] " << msg << std::endl;
    }
}

TensorRTInfer::TensorRTInfer(const std::string& engine_file_path,
                             const PaddlePredictorOption& option)
    : option_(option) {
    
    logger_ = std::make_unique<TRTLogger>();
    
    // 设置 Device
    int device_id = option_.DeviceId() >= 0 ? option_.DeviceId() : 0;
    CHECK_CUDA(cudaSetDevice(device_id));

    // 直接使用传入的路径进行初始化
    if (!Init(engine_file_path)) {
        INFOE("Failed to initialize TensorRT engine from: %s", engine_file_path.c_str());
        exit(-1);
    }
}

TensorRTInfer::~TensorRTInfer() {
    // 析构时切换到对应的 Device，防止释放错误 Context
    if (option_.DeviceId() >= 0) {
        cudaSetDevice(option_.DeviceId());
    }
    for (auto& t : tensors_) {
        if (t.device_buffer) {
            cudaFree(t.device_buffer);
        }
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

bool TensorRTInfer::Init(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        INFOE("Read engine file failed: %s", engine_path.c_str());
        return false;
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    runtime_.reset(nvinfer1::createInferRuntime(*logger_));
    if (!runtime_) return false;

    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) return false;

    context_.reset(engine_->createExecutionContext());
    if (!context_) return false;

    CHECK_CUDA(cudaStreamCreate(&stream_));

    GetBindings();

    return true;
}

void TensorRTInfer::GetBindings() {
    int nb_bindings = engine_->getNbBindings();
    bindings_ptrs_.resize(nb_bindings, nullptr);

    for (int i = 0; i < nb_bindings; ++i) {
        TensorInfo info;
        info.index = i;
        info.name = engine_->getBindingName(i);
        info.is_input = engine_->bindingIsInput(i);
        info.dtype = engine_->getBindingDataType(i);
        // 获取构建时的维度
        info.dims = engine_->getBindingDimensions(i); 

        tensors_.push_back(info);
        
        INFO("Tensor: %s, Index: %d, Input: %d", info.name.c_str(), i, info.is_input);
    }
}

absl::StatusOr<std::vector<cv::Mat>>
TensorRTInfer::Apply(const std::vector<cv::Mat>& x) {
    if (x.empty()) return absl::InvalidArgumentError("Input data is empty");

    std::vector<TensorInfo*> input_ptrs;
    std::vector<TensorInfo*> output_ptrs;
    for (auto& t : tensors_) {
        if (t.is_input) input_ptrs.push_back(&t);
        else output_ptrs.push_back(&t);
    }

    if (x.size() != input_ptrs.size()) {
        return absl::InvalidArgumentError("Input size mismatch with model inputs.");
    }

    int device_id = option_.DeviceId() >= 0 ? option_.DeviceId() : 0;
    CHECK_CUDA(cudaSetDevice(device_id));

    // =========================================================================
    // 1. 输入处理
    // =========================================================================
    for (size_t i = 0; i < x.size(); ++i) {
        auto* tensor_info = input_ptrs[i];

        // 1.1 构造输入维度 (TRT 8.x + Paddle逻辑)
        nvinfer1::Dims trt_dims;
        trt_dims.nbDims = x[0].dims;
        for (int d = 0; d < x[0].dims; ++d) {
            trt_dims.d[d] = x[0].size[d];
        }

        // 1.2 设置维度 (TRT 8.x 使用 Binding Index)
        if (!context_->setBindingDimensions(tensor_info->index, trt_dims)) {
             return absl::InternalError("Set binding dimensions failed for tensor index: " + std::to_string(tensor_info->index));
        }
        tensor_info->dims = trt_dims;

        // 1.3 显存分配
        size_t vol = 1;
        for(int d=0; d<trt_dims.nbDims; ++d) vol *= trt_dims.d[d];
        size_t bytes = vol * sizeof(float);

        if (bytes > tensor_info->size_bytes) {
            if (tensor_info->device_buffer) cudaFree(tensor_info->device_buffer);
            CHECK_CUDA(cudaMalloc(&tensor_info->device_buffer, bytes));
            tensor_info->size_bytes = bytes;
        }

        // 1.4 更新绑定数组
        bindings_ptrs_[tensor_info->index] = tensor_info->device_buffer;

        // 1.5 H2D 拷贝 (Raw copy)
        CHECK_CUDA(cudaMemcpyAsync(tensor_info->device_buffer, 
                                   x[i].data, 
                                   bytes, 
                                   cudaMemcpyHostToDevice, 
                                   stream_));
    }

    // =========================================================================
    // 2. 准备输出显存
    // =========================================================================
    for (auto* out_tensor : output_ptrs) {
        // TRT 8.x: 获取推导后的输出维度
        nvinfer1::Dims out_dims = context_->getBindingDimensions(out_tensor->index);
        
        size_t vol = 1;
        for(int d=0; d<out_dims.nbDims; ++d) vol *= out_dims.d[d];
        size_t bytes = vol * sizeof(float);

        if (bytes > out_tensor->size_bytes) {
            if (out_tensor->device_buffer) cudaFree(out_tensor->device_buffer);
            CHECK_CUDA(cudaMalloc(&out_tensor->device_buffer, bytes));
            out_tensor->size_bytes = bytes;
        }
        
        bindings_ptrs_[out_tensor->index] = out_tensor->device_buffer;
    }

    // =========================================================================
    // 3. 执行推理 (enqueueV2)
    // =========================================================================
    if (!context_->enqueueV2(bindings_ptrs_.data(), stream_, nullptr)) {
        return absl::InternalError("TensorRT enqueueV2 failed.");
    }

    // =========================================================================
    // 4. 输出处理
    // =========================================================================
    std::vector<std::vector<float>> outputs;
    std::vector<int> output_shape_vec = {};

    for (auto* out_tensor : output_ptrs) {
        nvinfer1::Dims dims = context_->getBindingDimensions(out_tensor->index);
        std::vector<int> current_shape;
        size_t numel = 1;
        for (int d = 0; d < dims.nbDims; ++d) {
            current_shape.push_back(static_cast<int>(dims.d[d]));
            numel *= dims.d[d];
        }
        output_shape_vec = current_shape;

        std::vector<float> out_data(numel);
        CHECK_CUDA(cudaMemcpyAsync(out_data.data(), 
                                   out_tensor->device_buffer, 
                                   numel * sizeof(float), 
                                   cudaMemcpyDeviceToHost, 
                                   stream_));
        outputs.push_back(std::move(out_data));
    }

    CHECK_CUDA(cudaStreamSynchronize(stream_));

    if (outputs.empty()) return std::vector<cv::Mat>();

    // =========================================================================
    // 5. 封装结果
    // =========================================================================
    cv::Mat pred(output_shape_vec.size(), output_shape_vec.data(), CV_32F);
    memcpy(pred.ptr<float>(), outputs[0].data(),
           outputs[0].size() * sizeof(float));

    std::vector<cv::Mat> pred_outputs = {pred};
    return pred_outputs;
}
#endif