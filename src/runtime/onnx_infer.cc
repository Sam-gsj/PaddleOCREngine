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
#include "onnx_infer.h"
#include "src/utils/utility.h"

ONNXInfer::ONNXInfer(const std::string &model_name,
                     const std::string &model_dir,
                     const PaddlePredictorOption &option)
    : option_(option) {

  auto onnx_file = Utility::FindFileWithSuffix(model_dir, "onnx");
  if (!onnx_file.ok()) {
    INFOE("Can not find ONNX model file in %s", model_dir.c_str());
    exit(-1);
  }

  if (!Init(onnx_file.value())) {
    INFOE("Failed to initialize ONNX Runtime session");
    exit(-1);
  }

  InitNodes();
}

ONNXInfer::~ONNXInfer() {}

bool ONNXInfer::Init(const std::string &onnx_path) {
  try {
    env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNXInfer");

    Ort::SessionOptions session_options;

    if (option_.CpuThreads() > 0) {
      session_options.SetIntraOpNumThreads(option_.CpuThreads());
    }

    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

#if _WIN32
    std::wstring w_model_path =
        std::wstring(onnx_path.begin(), onnx_path.end());
    session_ = std::make_shared<Ort::Session>(*env_, w_model_path.c_str(),
                                              session_options);
#else
    session_ = std::make_shared<Ort::Session>(*env_, onnx_path.c_str(),
                                              session_options);
#endif

    allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();
    memory_info_ = std::make_shared<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    return true;

  } catch (const Ort::Exception &e) {
    INFOE("ONNX Runtime Init Exception: %s", e.what());
    return false;
  }
}

void ONNXInfer::InitNodes() {
  size_t num_input_nodes = session_->GetInputCount();
  input_node_names_.resize(num_input_nodes);
  input_node_names_ptr_.resize(num_input_nodes);

  for (size_t i = 0; i < num_input_nodes; i++) {
    auto name_ptr = session_->GetInputNameAllocated(i, *allocator_);
    input_node_names_[i] = name_ptr.get();                   // 拷贝到 string
    input_node_names_ptr_[i] = input_node_names_[i].c_str(); // 保存 c_str

    INFO("ONNX Input [%d]: %s", (int)i, input_node_names_[i].c_str());
  }

  size_t num_output_nodes = session_->GetOutputCount();
  output_node_names_.resize(num_output_nodes);
  output_node_names_ptr_.resize(num_output_nodes);

  for (size_t i = 0; i < num_output_nodes; i++) {
    auto name_ptr = session_->GetOutputNameAllocated(i, *allocator_);
    output_node_names_[i] = name_ptr.get();
    output_node_names_ptr_[i] = output_node_names_[i].c_str();

    INFO("ONNX Output [%d]: %s", (int)i, output_node_names_[i].c_str());
  }
}

absl::StatusOr<std::vector<cv::Mat>>
ONNXInfer::Apply(const std::vector<cv::Mat> &x) {

  if (x.empty())
    return absl::InvalidArgumentError("Input data empty");
  if (session_ == nullptr)
    return absl::InternalError("Session not initialized");

  const cv::Mat &nchw_mat = x[0]; // 假设输入已经是 NCHW 且是 float 类型

  if (!nchw_mat.isContinuous()) {
    return absl::InvalidArgumentError("Input Mat must be continuous");
  }
  if (nchw_mat.type() != CV_32F) {
    return absl::InvalidArgumentError("Input Mat must be CV_32F");
  }

  std::vector<int64_t> input_shape;
  if (nchw_mat.dims == 4) {
    input_shape = {nchw_mat.size[0], nchw_mat.size[1], nchw_mat.size[2],
                   nchw_mat.size[3]};
  } else {
    for (int i = 0; i < nchw_mat.dims; ++i)
      input_shape.push_back(nchw_mat.size[i]);
  }

  size_t input_tensor_size = nchw_mat.total();
  float *input_data_ptr = (float *)nchw_mat.data;

  std::vector<Ort::Value> input_tensors;
  try {

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        *memory_info_, input_data_ptr, input_tensor_size, input_shape.data(),
        input_shape.size());
    input_tensors.push_back(std::move(input_tensor));

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr}, input_node_names_ptr_.data(),
        input_tensors.data(), input_tensors.size(),
        output_node_names_ptr_.data(), output_node_names_ptr_.size());

    std::vector<cv::Mat> result_mats;

    for (size_t i = 0; i < output_tensors.size(); ++i) {
      float *floatarr = output_tensors[i].GetTensorMutableData<float>();

      Ort::TensorTypeAndShapeInfo type_info =
          output_tensors[i].GetTensorTypeAndShapeInfo();
      std::vector<int64_t> output_shape = type_info.GetShape();
      size_t total_len = type_info.GetElementCount();

      INFO("ONNX Output [%d] Shape: [", (int)i);
      std::vector<int> cv_dims;
      for (auto dim : output_shape) {

        printf("%ld ", dim);
        cv_dims.push_back((int)dim);
      }
      printf("]\n");

      cv::Mat out_mat(cv_dims.size(), cv_dims.data(), CV_32F);

      if (out_mat.total() != total_len) {
        INFOE("Output shape mismatch: Mat total %zu != ORT total %zu",
              out_mat.total(), total_len);
        return absl::InternalError("Output shape mismatch");
      }

      memcpy(out_mat.data, floatarr, total_len * sizeof(float));
      result_mats.push_back(out_mat);
    }
    return result_mats;

  } catch (const Ort::Exception &e) {
    INFOE("ONNX Run Exception: %s", e.what());
    return absl::InternalError(std::string("ONNX Runtime error: ") + e.what());
  }
}

#endif
