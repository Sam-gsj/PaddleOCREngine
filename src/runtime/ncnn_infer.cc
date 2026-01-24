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
#include "ncnn_infer.h"
#include "src/utils/utility.h"
#include <algorithm>
#include <cstring>
#include <fstream>

NCNNInfer::NCNNInfer(const std::string &model_name,
                     const std::string &model_dir,
                     const PaddlePredictorOption &option)
    : option_(option) {
  auto param_file = Utility::FindFileWithSuffix(model_dir, "param");
  auto bin_file = Utility::FindFileWithSuffix(model_dir, "bin");
  if (!param_file.ok() || !bin_file.ok()) {
    INFOE("Can not find NCNN model files (param/bin) in %s", model_dir.c_str());
    exit(-1);
  }

  INFO("NCNN param file: %s", param_file.value().c_str());
  INFO("NCNN bin file: %s", bin_file.value().c_str());

  if (!Init(param_file.value(), bin_file.value())) {
    INFOE("Failed to initialize NCNN net");
    exit(-1);
  }

  InitBlobs();
}

NCNNInfer::~NCNNInfer() {
  if (net_) {
    net_->clear();
    net_.reset();
  }
}

bool NCNNInfer::Init(const std::string &param_path,
                     const std::string &bin_path) {
  net_ = std::make_unique<ncnn::Net>();
  if (!net_) {
    INFOE("Create NCNN Net failed");
    return false;
  }

  // net_->opt.use_gpu = false;
  net_->opt.use_vulkan_compute = false;

  if (net_->load_param(param_path.c_str()) != 0) {
    INFOE("Load NCNN param failed: %s", param_path.c_str());
    return false;
  }

  if (net_->load_model(bin_path.c_str()) != 0) {
    INFOE("Load NCNN bin failed: %s", bin_path.c_str());
    return false;
  }

  INFO("NCNN init success (CPU only)");
  return true;
}

void NCNNInfer::InitBlobs() {
  std::vector<const char *> input_names_c = net_->input_names();
  std::vector<const char *> output_names_c = net_->output_names();

  blobs_.clear();
  blobs_.reserve(input_names_c.size() + output_names_c.size());

  for (const char *name : input_names_c) {
    BlobInfo info;
    info.name = name;
    info.is_input = true;
    blobs_.push_back(info);
  }

  for (const char *name : output_names_c) {
    BlobInfo info;
    info.name = name;
    info.is_input = false;
    blobs_.push_back(info);
  }

  input_blobs_.clear();
  output_blobs_.clear();

  for (size_t i = 0; i < blobs_.size(); ++i) {

    BlobInfo *p_info = &blobs_[i];

    if (p_info->is_input) {
      input_blobs_.push_back(p_info);
    } else {
      output_blobs_.push_back(p_info);
    }
  }

  for (const auto &blob : blobs_) {
    INFO("NCNN Blob: %s, Input: %d", blob.name.c_str(), blob.is_input);
  }

  if (input_blobs_.empty()) {
    INFOE("NCNN model has no input blobs");
    exit(-1);
  }
  if (output_blobs_.empty()) {
    INFOE("NCNN model has no output blobs");
    exit(-1);
  }
}

absl::StatusOr<std::vector<cv::Mat>>
NCNNInfer::Apply(const std::vector<cv::Mat> &x) {
  if (!net_)
    return absl::InternalError("NCNN net is null");
  if (x.empty())
    return absl::InvalidArgumentError("Input empty");

  const auto *input_blob = input_blobs_[0];
  const auto *output_blob = output_blobs_[0];
  const cv::Mat &nchw_mat = x[0];

  int N = nchw_mat.size[0];
  int C = nchw_mat.size[1];
  int H = nchw_mat.size[2];
  int W = nchw_mat.size[3];

  const float *check_ptr = nchw_mat.ptr<float>();

  size_t chw_elem_num = (size_t)C * H * W;
  const float *nchw_base_ptr = nchw_mat.ptr<float>();

  std::vector<float> all_output_data;
  int out_c = 0, out_h = 0, out_w = 0;

  for (int n = 0; n < N; ++n) {
    const float *current_src = nchw_base_ptr + n * chw_elem_num;

    ncnn::Mat in_mat(W, H, C); // w, h, c
    int plane_size = W * H;

    for (int c = 0; c < C; c++) {
      float *dst = in_mat.channel(c);
      const float *src = current_src + c * plane_size;
      memcpy(dst, src, plane_size * sizeof(float));
    }

    ncnn::Extractor ex = net_->create_extractor();

    ex.set_light_mode(true);

    ex.input(input_blob->name.c_str(), in_mat);

    ncnn::Mat out_mat;
    ex.extract(output_blob->name.c_str(), out_mat);

    if (out_mat.empty()) {
      return absl::InternalError("NCNN extract empty");
    }

    ncnn::Mat final_out;

    if (out_mat.h > out_mat.w && out_mat.h > 500) {

      final_out.create(out_mat.h, out_mat.w, sizeof(float));

      for (int i = 0; i < out_mat.h; i++) { // Class
        const float *src_ptr = out_mat.row(i);
        for (int j = 0; j < out_mat.w; j++) { // Time

          float *dst_ptr = final_out.row(j);
          dst_ptr[i] = src_ptr[j];
        }
      }
    } else {
      final_out = out_mat;
    }

    if (n == 0) {
      out_c = final_out.c;
      out_h = final_out.h;
      out_w = final_out.w;
    }

    int plane_bytes = final_out.w * final_out.h * sizeof(float);
    const float *ptr = (const float *)final_out.data;
    all_output_data.insert(all_output_data.end(), ptr,
                           ptr + final_out.w * final_out.h);
  }

  std::vector<int> out_shape = {N, out_c, out_h, out_w};

  if (out_c == 1) {
    out_shape = {N, out_h, out_w}; // [N, 80, 6625]
  }

  cv::Mat output_mat(out_shape.size(), out_shape.data(), CV_32F);
  memcpy(output_mat.ptr<float>(), all_output_data.data(),
         all_output_data.size() * sizeof(float));
  return {{output_mat}};
}

#endif
