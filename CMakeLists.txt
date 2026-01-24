cmake_minimum_required(VERSION 3.14)
project(ppocr CXX)

set(DEMO_NAME "ppocr")


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 后端开关
option(WITH_PADDLE      "Enable Paddle Inference support"   ON)
option(WITH_TensorRT    "Enable TensorRT support"           OFF)
option(WITH_NCNN        "Enable NCNN support (CPU only)"    OFF)
option(WITH_ONNXRUNTIME "Enable ONNX Runtime support"       OFF)
option(WITH_RKNN        "Enable RKNN support"               OFF) 

# 功能开关
option(WITH_MKL         "Compile with MKL/OpenBlas (Paddle only)"  ON)
option(WITH_GPU         "Compile with GPU support"                 OFF)
option(WITH_STATIC_LIB  "Compile with static library"              ON)
option(USE_FREETYPE     "Enable FreeType support"                  OFF)


SET(OPENCV_DIR "" CACHE PATH "Location of OpenCV")

# Paddle 相关路径 (仅在 WITH_PADDLE=ON 时生效)
SET(PADDLE_LIB "/home/badguy/PaddleOCREngine/paddle_inference" CACHE PATH "Location of Paddle libraries")

# TensorRT 相关路径
SET(TENSORRT_DIR "/usr/local/TensorRT" CACHE PATH "Location of TensorRT")
SET(CUDA_INCLUDE "/usr/local/cuda-11.8/include" CACHE PATH "Location of CUDA Headers")
SET(CUDA_LIB "" CACHE PATH "Location of CUDA Libs")
SET(CUDNN_LIB "" CACHE PATH "Location of CUDNN Libs")

# NCNN 相关路径
SET(NCNN_DIR "/usr/local/ncnn" CACHE PATH "Location of compiled NCNN library (CPU only)")

# ONNXRuntime 相关路径
SET(ONNXRUNTIME_DIR "/path/to/onnxruntime" CACHE PATH "Location of ONNX Runtime")


macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${${flag_var}} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif()
    endforeach(flag_var)
endmacro()

function(download_and_decompress url filename decompress_dir)
  if(NOT EXISTS "${filename}" AND NOT EXISTS "${decompress_dir}")
    message("Downloading file from ${url} to ${filename} ...")
    file(DOWNLOAD ${url} "${filename}.tmp" SHOW_PROGRESS)
    file(RENAME "${filename}.tmp" ${filename})
  endif()
  if(NOT EXISTS ${decompress_dir})
    file(MAKE_DIRECTORY ${decompress_dir})
    message("Decompress file ${filename} ...")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${filename} WORKING_DIRECTORY ${decompress_dir})
  endif()
endfunction()


if(NOT DEFINED OPENCV_DIR)
    message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/path/opencv")
endif()

if (WIN32)
  set(OpenCV_DIR "${OPENCV_DIR}/x64/vc16/lib") 
else()
  set(OpenCV_DIR "/usr/local/opencv4/lib64/cmake/opencv4") 
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(DEPS ${DEPS} ${OpenCV_LIBS})

if(USE_FREETYPE)
    if(NOT "opencv_freetype" IN_LIST OpenCV_LIBS)
      message(FATAL_ERROR "OpenCV was not compiled with the freetype module (opencv_freetype) !")
    endif()
    add_definitions(-DUSE_FREETYPE)
endif()


if (WIN32)
    add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
    if(WITH_MKL)
        set(FLAG_OPENMP "/openmp")
    endif()
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    if (WITH_STATIC_LIB)
        safe_set_static_flag()
        add_definitions(-DSTATIC_LIB)
        add_definitions(-DYAML_CPP_STATIC_DEFINE)
    endif()
else()
    if(WITH_MKL)
        set(FLAG_OPENMP "-fopenmp")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 ${FLAG_OPENMP}")
endif()


set(THIRD_PARTY_PATH ${CMAKE_CURRENT_LIST_DIR}/third_party)
set(PACKAGE_LIST abseil-cpp clipper_ver6.4.2 nlohmann)
foreach(PKG ${PACKAGE_LIST})
    set(PKG_URL "https://paddle-model-ecology.bj.bcebos.com/paddlex/cpp/libs/${PKG}.tgz")
    set(PKG_TGZ_PATH "${CMAKE_CURRENT_BINARY_DIR}/${PKG}.tgz")
    set(PKG_DST_PATH "${THIRD_PARTY_PATH}/${PKG}")
    download_and_decompress(${PKG_URL} ${PKG_TGZ_PATH} ${PKG_DST_PATH})
endforeach()

add_subdirectory(third_party/abseil-cpp)
add_subdirectory(third_party/clipper_ver6.4.2/cpp)
include_directories(${POLYCLIPPING_INCLUDE_DIR})

set(DEPS ${DEPS} absl::statusor polyclipping)


if(WITH_PADDLE)
    message(STATUS "Backend: Paddle Inference ENABLED")
    add_definitions(-DUSE_PADDLE)
    
    if(NOT DEFINED PADDLE_LIB)
        message(FATAL_ERROR "Please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
    endif()

    if (WITH_MKL)
        ADD_DEFINITIONS(-DUSE_MKL)
    endif()

    if (WIN32)
        include_directories("${PADDLE_LIB}/paddle/include")
        link_directories("${PADDLE_LIB}/paddle/lib")
    else()
        include_directories("${PADDLE_LIB}/paddle/include")
        link_directories("${PADDLE_LIB}/paddle/lib")
    endif()

    # Paddle 第三方依赖包含 (glog, gflags, protobuf, etc.)
    include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
    include_directories("${PADDLE_LIB}/third_party/install/glog/include")
    include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
    include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
    include_directories("${PADDLE_LIB}/third_party/install/zlib/include")
    include_directories("${PADDLE_LIB}/third_party/install/paddle2onnx/include")
    include_directories("${PADDLE_LIB}/third_party/install/yaml-cpp/include")
    include_directories("${PADDLE_LIB}/third_party/boost")
    include_directories("${PADDLE_LIB}/third_party/eigen3")

    link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")
    link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
    link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
    link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
    link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
    link_directories("${PADDLE_LIB}/third_party/install/paddle2onnx/lib")
    link_directories("${PADDLE_LIB}/third_party/install/yaml-cpp/lib")
    
    # MKL / OpenBLAS 配置
    if(WITH_MKL)
      include_directories("${PADDLE_LIB}/third_party/install/mklml/include")
      if (WIN32)
        set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.lib
                ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5md.lib)
      else ()
        set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
                ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
        execute_process(COMMAND cp -r ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX} /usr/lib)
      endif ()
      set(MKLDNN_PATH "${PADDLE_LIB}/third_party/install/onednn")
      if(EXISTS ${MKLDNN_PATH})
        include_directories("${MKLDNN_PATH}/include")
        if (WIN32)
          set(MKLDNN_LIB ${MKLDNN_PATH}/lib/mkldnn.lib)
        else ()
          set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libdnnl.so.3)
        endif ()
      endif()
    else()
      if (WIN32)
        set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
      else ()
        set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
      endif ()
    endif()

    # GPU 配置
    if (WITH_GPU)
        if (NOT DEFINED CUDA_LIB OR ${CUDA_LIB} STREQUAL "")
            message(FATAL_ERROR "please set CUDA_LIB with -DCUDA_LIB=/path/cuda-8.0/lib64")
        endif()
        if (NOT WIN32)
            if (NOT DEFINED CUDNN_LIB)
                message(FATAL_ERROR "please set CUDNN_LIB with -DCUDNN_LIB=/path/cudnn_v7.4/cuda/lib64")
            endif()
            add_definitions(-DWITH_GPU)
            set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
            set(DEPS ${DEPS} ${CUDNN_LIB}/libcudnn${CMAKE_SHARED_LIBRARY_SUFFIX})
        else()
             add_definitions(-DWITH_GPU)
             set(DEPS ${DEPS} ${CUDA_LIB}/cudart${CMAKE_STATIC_LIBRARY_SUFFIX})
             set(DEPS ${DEPS} ${CUDA_LIB}/cublas${CMAKE_STATIC_LIBRARY_SUFFIX})
             set(DEPS ${DEPS} ${CUDNN_LIB}/cudnn${CMAKE_STATIC_LIBRARY_SUFFIX})
        endif()
    endif()

    # Paddle Inference 库链接
    if(WITH_STATIC_LIB)
      if(WIN32)
        set(DEPS ${DEPS} ${PADDLE_LIB}/paddle/lib/paddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
      else()
        set(DEPS ${DEPS} ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
      endif()
    else()
      if(WIN32)
        set(DEPS ${DEPS} ${PADDLE_LIB}/paddle/lib/paddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
      else()
        set(DEPS ${DEPS} ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
      endif()
    endif()

    # 链接 Paddle 依赖库
    if (NOT WIN32)
        set(DEPS ${DEPS} ${MATH_LIB} ${MKLDNN_LIB} glog gflags protobuf z xxhash)
        if(EXISTS "${PADDLE_LIB}/third_party/install/snappystream/lib")
            set(DEPS ${DEPS} snappystream)
        endif()
        if (EXISTS "${PADDLE_LIB}/third_party/install/snappy/lib")
            set(DEPS ${DEPS} snappy)
        endif()
        if (EXISTS "${PADDLE_LIB}/third_party/install/yaml-cpp/lib")
            set(DEPS ${DEPS} yaml-cpp)
        endif()
    else()
        set(DEPS ${DEPS} ${MATH_LIB} ${MKLDNN_LIB} glog gflags_static libprotobuf xxhash libcmt shlwapi)
        if (EXISTS "${PADDLE_LIB}/third_party/install/snappy/lib")
            set(DEPS ${DEPS} snappy)
        endif()
        if(EXISTS "${PADDLE_LIB}/third_party/install/snappystream/lib")
            set(DEPS ${DEPS} snappystream)
        endif()
        if (EXISTS "${PADDLE_LIB}/third_party/install/yaml-cpp/lib")
            set(DEPS ${DEPS} yaml-cpp)
        endif()
    endif()

else()
    message(STATUS "Backend: Paddle Inference DISABLED")
endif(WITH_PADDLE)


if(WITH_TensorRT)
    message(STATUS "Backend: TensorRT ENABLED")
    add_definitions(-DUSE_TensorRT)
    
    include_directories("${CUDA_INCLUDE}")
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_DIR} ${TENSORRT_DIR}/include /usr/include /usr/include/x86_64-linux-gnu
    )
    find_library(TENSORRT_LIBRARY nvinfer
        HINTS ${TENSORRT_DIR} ${TENSORRT_DIR}/lib /usr/lib/x86_64-linux-gnu
    )

    if (TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
        message(STATUS "Found TensorRT Header: ${TENSORRT_INCLUDE_DIR}")
        message(STATUS "Found TensorRT Lib: ${TENSORRT_LIBRARY}")
        include_directories(${TENSORRT_INCLUDE_DIR})
        set(DEPS ${DEPS} ${TENSORRT_LIBRARY})
    else()
        message(FATAL_ERROR "Could NOT find TensorRT. Please set -DTENSORRT_DIR=/path/to/TensorRT")
    endif()
endif()


if(WITH_NCNN)
    message(STATUS "Backend: NCNN ENABLED")
    add_definitions(-DUSE_NCNN)
    
    if (WIN32)
        find_path(NCNN_INCLUDE_DIR ncnn/net.h HINTS ${NCNN_DIR} ${NCNN_DIR}/include)
        find_library(NCNN_LIBRARY ncnn HINTS ${NCNN_DIR} ${NCNN_DIR}/lib)
    else()
        find_path(NCNN_INCLUDE_DIR ncnn/net.h HINTS ${NCNN_DIR} ${NCNN_DIR}/include /usr/local/ncnn/include)
        find_library(NCNN_LIBRARY ncnn HINTS ${NCNN_DIR} ${NCNN_DIR}/lib /usr/local/ncnn/lib)
    endif()

    if (NCNN_INCLUDE_DIR AND NCNN_LIBRARY)
        message(STATUS "Found NCNN Header: ${NCNN_INCLUDE_DIR}")
        message(STATUS "Found NCNN Lib: ${NCNN_LIBRARY}")
        include_directories(${NCNN_INCLUDE_DIR})
        set(DEPS ${DEPS} ${NCNN_LIBRARY})
    else()
        message(FATAL_ERROR "Could NOT find NCNN. Please set -DNCNN_DIR=/path/to/ncnn")
    endif()

    if(UNIX)
        find_package(Iconv REQUIRED)
        set(DEPS ${DEPS} ${Iconv_LIBRARIES})
    endif()
endif()


if(WITH_ONNXRUNTIME)
    message(STATUS "Backend: ONNX Runtime ENABLED")
    add_definitions(-DUSE_ONNX)
    
    find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h HINTS ${ONNXRUNTIME_DIR}/include)
    find_library(ORT_LIB onnxruntime HINTS ${ONNXRUNTIME_DIR}/lib)

    if (ORT_INCLUDE_DIR AND ORT_LIB)
        message(STATUS "Found ORT Header: ${ORT_INCLUDE_DIR}")
        message(STATUS "Found ORT Lib: ${ORT_LIB}")
        include_directories(${ORT_INCLUDE_DIR})
        set(DEPS ${DEPS} ${ORT_LIB})
        
        get_filename_component(ORT_LIB_DIR ${ORT_LIB} DIRECTORY)
        if(WIN32)
            set(ORT_DLL "${ORT_LIB_DIR}/onnxruntime.dll")
        else()
            set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH};${ORT_LIB_DIR}")
        endif()
    else()
        message(FATAL_ERROR "Could NOT find ONNX Runtime. Please set -DONNXRUNTIME_DIR=...")
    endif()
endif()


if (NOT WIN32)
    set(EXTERNAL_LIB "-ldl -lrt -lgomp -lz -lm -lpthread")
    set(DEPS ${DEPS} ${EXTERNAL_LIB})
endif()

include_directories("${CMAKE_SOURCE_DIR}/")

file(GLOB_RECURSE SRC_LIST "./src/*.cc")
set(SRCS test.cpp)

add_executable(${DEMO_NAME} ${SRCS} ${SRC_LIST})
target_link_libraries(${DEMO_NAME} ${DEPS})


if (WIN32)
    # Paddle MKL DLLs
    if (WITH_PADDLE AND WITH_MKL)
        add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.dll ./mklml.dll
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5md.dll ./libiomp5md.dll
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/onednn/lib/mkldnn.dll ./mkldnn.dll
        )
    endif()

    # ONNX Runtime DLL
    if(WITH_ONNXRUNTIME)
        add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_DLL} $<TARGET_FILE_DIR:${DEMO_NAME}>
        )
    endif()
endif()
