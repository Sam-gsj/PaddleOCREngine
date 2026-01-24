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
#include "rknn_infer.h"
#include "src/utils/utility.h"
#include <cstring>
#include <fstream>

RKNNInfer::RKNNInfer(const std::string &model_name,
                     const std::string &model_dir,
                     const PaddlePredictorOption &option)
    : option_(option) {

  auto rknn_file = Utility::FindFileWithSuffix(model_dir, "rknn");
  if (!rknn_file.ok()) {
    INFOE("Can not find RKNN model file in %s", model_dir.c_str());
    exit(-1);
  }
  INFO("RKNN model file: %s", rknn_file.value().c_str());

  if (!Init(rknn_file.value())) {
    INFOE("Failed to initialize RKNN context");
    exit(-1);
  }

  QueryModelInfo();
}

RKNNInfer::~RKNNInfer() {
  if (ctx_ > 0) {
    rknn_destroy(ctx_);
  }
  if (model_data_) {
    delete[] model_data_;
  }
}

bool RKNNInfer::Init(const std::string &rknn_path) {
  // 1. 读取模型文件到内存
  std::ifstream ifs(rknn_path, std::ios::binary);
  if (!ifs.is_open()) {
    INFOE("Open RKNN file failed: %s", rknn_path.c_str());
    return false;
  }

  ifs.seekg(0, std::ios::end);
  int size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  model_data_ = new unsigned char[size];
  ifs.read((char *)model_data_, size);
  ifs.close();

  // 2. 初始化 RKNN
  // RKNN_FLAG_PRIOR_MEDIUM: 中等优先级
  int ret = rknn_init(&ctx_, model_data_, size, 0, NULL);
  if (ret < 0) {
    INFOE("rknn_init failed! ret=%d", ret);
    return false;
  }

  // 3. (可选) 绑定核心，例如 rknn_set_core_mask(ctx_, RKNN_NPU_CORE_0);
  return true;
}

void RKNNInfer::QueryModelInfo() {
  // 1. 查询输入输出数量
  int ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
  if (ret != RKNN_SUCC) {
    INFOE("rknn_query io_num failed! ret=%d", ret);
    exit(-1);
  }
  INFO("RKNN model inputs: %d, outputs: %d", io_num_.n_input, io_num_.n_output);

  // 2. 查询输入属性
  input_attrs_.resize(io_num_.n_input);
  inputs_structs_.resize(io_num_.n_input); // 预分配
  for (int i = 0; i < io_num_.n_input; i++) {
    input_attrs_[i].index = i;
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]),
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      INFOE("rknn_query input attr %d failed", i);
      exit(-1);
    }
    INFO("RKNN Input[%d]: name=%s, dims=[%d,%d,%d,%d], fmt=%s, type=%s", i,
         input_attrs_[i].name, input_attrs_[i].dims[0], input_attrs_[i].dims[1],
         input_attrs_[i].dims[2], input_attrs_[i].dims[3],
         get_format_string(input_attrs_[i].fmt),
         get_type_string(input_attrs_[i].type));
  }

  // 3. 查询输出属性
  output_attrs_.resize(io_num_.n_output);
  for (int i = 0; i < io_num_.n_output; i++) {
    output_attrs_[i].index = i;
    ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]),
                     sizeof(rknn_tensor_attr));
    INFO("RKNN Output[%d]: name=%s, dims=[%d,%d,%d,%d]", i,
         output_attrs_[i].name, output_attrs_[i].dims[0],
         output_attrs_[i].dims[1], output_attrs_[i].dims[2],
         output_attrs_[i].dims[3]);
  }
}

absl::StatusOr<std::vector<cv::Mat>>
RKNNInfer::Apply(const std::vector<cv::Mat> &x) {
  INFO("===== RKNN Infer Apply Start =====");
  if (x.empty())
    return absl::InvalidArgumentError("Input data empty");

  // 假设 batch size = 1，或者我们需要手动循环处理 batch
  // RKNN 通常不擅长动态 Batch，所以我们这里对 batch 进行循环调用

  std::vector<cv::Mat> final_results;
  const cv::Mat &input_mat = x[0]; // 假设这是 NCHW 的 float Mat

  // NCHW parsing
  int N = input_mat.size[0];
  int C = input_mat.size[1];
  int H = input_mat.size[2];
  int W = input_mat.size[3];
  size_t single_sample_size = C * H * W * sizeof(float); // 假设输入是 Float

  // 为了简化，我们假设模型只有一个输入。如果有多个，逻辑类似
  if (io_num_.n_input > 1 && x.size() != io_num_.n_input) {
    return absl::InvalidArgumentError("Input num mismatch");
  }

  // 循环处理每一个 Batch (因为 NPU 推理通常很快，且驱动对多Batch支持各异)
  // 实际上更优的做法是将所有 Batch 拼成一个内存块传给 RKNN
  // (如果模型是多Batch编译的) 但通用做法是 loop

  for (int n = 0; n < N; ++n) {
    // 1. 设置输入
    // 我们必须告诉 RKNN 输入数据的真实属性
    memset(inputs_structs_.data(), 0, sizeof(rknn_input) * io_num_.n_input);

    // 获取当前样本的数据指针
    const float *current_data = input_mat.ptr<float>() + n * C * H * W;

    inputs_structs_[0].index = 0;
    inputs_structs_[0].type = RKNN_TENSOR_FLOAT32; // 输入 Mat 是 float32
    inputs_structs_[0].size = single_sample_size;
    inputs_structs_[0].fmt = RKNN_TENSOR_NCHW; // 输入 Mat 是 NCHW 布局
    inputs_structs_[0].buf = (void *)current_data;
    // pass_through = 0: 让驱动根据模型的需要（如 Int8 NHWC）自动转换输入数据
    inputs_structs_[0].pass_through = 0;

    int ret = rknn_inputs_set(ctx_, io_num_.n_input, inputs_structs_.data());
    if (ret < 0) {
      INFOE("rknn_inputs_set failed! ret=%d", ret);
      return absl::InternalError("RKNN inputs set failed");
    }

    // 2. 执行推理
    ret = rknn_run(ctx_, NULL);
    if (ret < 0) {
      INFOE("rknn_run failed! ret=%d", ret);
      return absl::InternalError("RKNN run failed");
    }

    // 3. 获取输出
    std::vector<rknn_output> outputs(io_num_.n_output);
    memset(outputs.data(), 0, sizeof(rknn_output) * io_num_.n_output);

    // request float output (让驱动帮我们反量化回 float)
    for (int i = 0; i < io_num_.n_output; ++i) {
      outputs[i].want_float = 1;
    }

    ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), NULL);
    if (ret < 0) {
      INFOE("rknn_outputs_get failed! ret=%d", ret);
      return absl::InternalError("RKNN outputs get failed");
    }

    // 4. 将 RKNN 输出转换为 OpenCV Mat
    for (int i = 0; i < io_num_.n_output; ++i) {
      // 获取维度
      // 注意：rknn_outputs_get 返回的维度可能不是模型原始维度，特别是当
      // want_float=1 时 我们最好参考 output_attrs_

      // 这里简单处理：假设输出是 1D 或 2D 的
      // 实际上对于 PaddleOCR，输出通常是 [1, Time, Class]
      // 如果 RKNN 返回的是 [1, Class, Time] (NHWC转换导致)，需要注意

      // 为了安全，我们按元素个数拷贝
      size_t elem_count = outputs[i].size / sizeof(float); // 因为 want_float=1

      // 尝试恢复形状
      std::vector<int> out_shape;
      for (int d = 0; d < output_attrs_[i].n_dims; ++d) {
        out_shape.push_back(output_attrs_[i].dims[d]);
      }
      // 修正 shape: 如果 batch 是动态的，RKNN 可能返回 0 或 1
      if (out_shape[0] == 0)
        out_shape[0] = 1;

      cv::Mat out_mat(out_shape.size(), out_shape.data(), CV_32F);

      // 检查大小是否匹配
      if (out_mat.total() != elem_count) {
        // 如果形状不匹配，回退到 1D
        out_mat = cv::Mat(1, elem_count, CV_32F);
      }

      memcpy(out_mat.data, outputs[i].buf, outputs[i].size);

      // ================== 特殊处理 Rec 模型转置问题 ==================
      // 类似 NCNN，RKNN 也可能导致 Rec 输出转置 [Class, Time]
      if (out_mat.dims >= 2) {
        int h = out_mat.size[out_mat.dims - 2];
        int w = out_mat.size[out_mat.dims - 1];
        // 如果是 2D [Class, Time] (e.g. 6625, 80) -> 需要转置
        // 这里逻辑同 NCNN 修复
        if (h > w && h > 500) {
          cv::Mat transposed;
          // 注意：RKNN 输出是连续内存，可以直接 cv::transpose 吗？
          // 如果是高维 Mat，OpenCV transpose 只能处理 2D。
          // 简化：如果 total 没变，我们假设它是 2D
          cv::Mat temp_2d(h, w, CV_32F, out_mat.data);
          cv::transpose(temp_2d, transposed);
          out_mat = transposed.clone(); // Deep copy
        }
      }
      // ============================================================

      final_results.push_back(out_mat);
    }

    // 5. 释放 RKNN 输出内存 (必须！)
    rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());
  }

  INFO("===== RKNN Infer Apply End =====");
  return final_results;
}

#endif // USE_RKNN
