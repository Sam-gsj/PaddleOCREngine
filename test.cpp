
#include "src/api/pipelines/doc_preprocessor.h"
#include "src/api/pipelines/ocr.h"

int main(){
    PaddleOCRParams params;
    params.text_detection_model_dir = "models/PP-OCRv5_mobile_det_infer"; // 文本检测模型路径
    params.text_recognition_model_dir = "models/PP-OCRv5_mobile_rec_infer"; // 文本识别模型路径

    // params.device = "gpu"; // 推理时使用GPU。请确保编译时添加 -DWITH_GPU=ON 选项，否则使用CPU。
    params.use_doc_orientation_classify = false;  // 不使用文档方向分类模型。
    params.use_doc_unwarping = false; // 不使用文本图像矫正模型。
    params.use_textline_orientation = false; // 不使用文本行方向分类模型。
    params.text_detection_model_name = "PP-OCRv5_mobile_det"; // 使用 PP-OCRv5_mobile_det 模型进行检测。
    params.text_recognition_model_name = "PP-OCRv5_mobile_rec"; // 使用 PP-OCRv5_mobile_rec 模型进行识别。
    params.vis_font_dir = "your_vis_font_dir"; // 当编译时添加 -DUSE_FREETYPE=ON 选项，必须提供相应 ttf 字体文件路径。

    auto infer = PaddleOCR(params);
    auto input = cv::imread("./general_ocr_002.png");
    std::vector<cv::Mat> inputs = {input,input,input};
    auto outputs = infer.Predict(inputs);
    for (auto& output : outputs) {
      output->Print();
      output->SaveToImg("./output/");
      output->SaveToJson("./output/");
    }

    DocPreprocessorParams params_doc;
    params_doc.doc_orientation_classify_model_dir = "models/PP-LCNet_x1_0_doc_ori_infer"; // 文档方向分类模型路径。
    params_doc.doc_unwarping_model_dir = "models/UVDoc_infer"; // 文本图像矫正模型路径。
    auto infer_doc = DocPreprocessor(params_doc);
    auto input_doc = cv::imread("./general_ocr_002.png");
    std::vector<cv::Mat> inputs_doc = {input_doc,input_doc,input_doc};
    auto outputs_doc = infer_doc.Predict(inputs_doc);
    for (auto& output_doc : outputs_doc) {
      output_doc->Print();
      output_doc->SaveToImg("./output/");
      output_doc->SaveToJson("./output/");
    }

}