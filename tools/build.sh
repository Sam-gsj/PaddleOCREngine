OPENCV_DIR=/home/badguys/PaddleOCR/deploy/cpp_infer/opencv-4.7.0/opencv4
LIB_DIR=/home/badguys/PaddleOCR/deploy/cpp_infer/paddle_inference
CUDA_LIB_DIR=your_cuda_lib_dir
CUDNN_LIB_DIR=your_cudnn_lib_dir

BUILD_DIR=build
# rm -rf ${BUILD_DIR}
# mkdir ${BUILD_DIR}
cd ${BUILD_DIR}

cmake .. \
    -DPADDLE_LIB=/home/badguys/PaddleOCR/deploy/cpp_infer/paddle_inference \
    -DWITH_MKL=ON \
    -DWITH_GPU=OFF \
    -DWITH_STATIC_LIB=OFF \
    -DWITH_TENSORRT=OFF \
    -DOPENCV_DIR=/home/badguys/PaddleOCR/deploy/cpp_infer/opencv-4.7.0/opencv4 \
    -DCUDNN_LIB=${CUDNN_LIB_DIR} \
    -DCUDA_LIB=${CUDA_LIB_DIR} \
    -DUSE_FREETYPE=OFF

make -j4


./build/ppocr ocr --input ./general_ocr_002.png --save_path ./output/  --text_detection_model_dir models/PP-OCRv5_mobile_det_infer --text_recognition_model_dir models/PP-OCRv5_mobile_rec_infer --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False --device cpu  --text_recognition_model_name PP-OCRv5_mobile_rec --text_detection_model_name PP-OCRv5_mobile_det 