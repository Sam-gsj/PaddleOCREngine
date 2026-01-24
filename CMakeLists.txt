project(ppocr CXX)
cmake_minimum_required(VERSION 3.14)

set(DEMO_NAME "ppocr")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       ON)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    OFF)
option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   ON)
option(WITH_TensorRT   "Enable TensorRT support"   OFF)
# ================= NCNN新增1：添加WITH_NCNN编译选项 =================
option(WITH_NCNN       "Enable NCNN support (CPU only)"   OFF)
# ==================================================================
option(USE_FREETYPE    "Enable FreeType support" OFF)

SET(PADDLE_LIB "/home/badguy/PaddleOCREngine/paddle_inference" CACHE PATH "Location of libraries")
SET(OPENCV_DIR "dsfds" CACHE PATH "Location of libraries")
SET(CUDA_LIB "" CACHE PATH "Location of libraries")
SET(CUDNN_LIB "" CACHE PATH "Location of libraries")
SET(TENSORRT_DIR "/usr/local/TensorRT" CACHE PATH "Location of TensorRT")
SET(CUDA_INCLUDE "/usr/local/cuda-11.8/include" CACHE PATH "Location of TensorRT")
# ================= NCNN新增2：添加NCNN_DIR缓存变量，指定NCNN库根目录 =================
SET(NCNN_DIR "/usr/local/ncnn" CACHE PATH "Location of compiled NCNN library (CPU only)")
# ====================================================================================

macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${${flag_var}} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif()
    endforeach(flag_var)
endmacro()

if (WITH_MKL)
    ADD_DEFINITIONS(-DUSE_MKL)
endif()

if(NOT DEFINED PADDLE_LIB)
  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()

if(NOT DEFINED OPENCV_DIR)
    message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/path/opencv")
endif()

if (WIN32)
  include_directories("${PADDLE_LIB}/paddle/include")
  link_directories("${PADDLE_LIB}/paddle/lib")
  set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
  set(OpenCV_DIR "${OPENCV_DIR}/x64/vc16/lib") 
  find_package(OpenCV REQUIRED )
  if(USE_FREETYPE)
    if(NOT "opencv_freetype" IN_LIST OpenCV_LIBS)
      message(FATAL_ERROR "OpenCV was not compiled with the freetype module (opencv_freetype) !")
    endif()
    add_definitions(-DUSE_FREETYPE)
  endif()  

  # ================= NCNN新增3：Windows系统NCNN配置（与Linux对齐，纯CPU）=================
  if(WITH_NCNN)
    add_definitions(-DUSE_NCNN)
    # 查找NCNN头文件
    find_path(NCNN_INCLUDE_DIR ncnn/net.h
        HINTS ${NCNN_DIR} ${NCNN_DIR}/include /usr/include /usr/include/ncnn
    )
    # 查找NCNN静态库（纯CPU编译的ncnn库，默认静态libncnn.a/libncnn.lib）
    find_library(NCNN_LIBRARY ncnn
        HINTS ${NCNN_DIR} ${NCNN_DIR}/lib ${NCNN_DIR}/build/lib /usr/lib /usr/lib/x86_64-linux-gnu
    )
    # 校验NCNN是否找到
    if (NCNN_INCLUDE_DIR AND NCNN_LIBRARY)
        message(STATUS "Found NCNN Header: ${NCNN_INCLUDE_DIR}")
        message(STATUS "Found NCNN Lib: ${NCNN_LIBRARY}")
        include_directories(${NCNN_INCLUDE_DIR})
        set(DEPS ${DEPS} ${NCNN_LIBRARY})
    else()
        message(FATAL_ERROR "Could NOT find NCNN. Please set -DNCNN_DIR=/path/to/compiled/ncnn")
    endif()
  endif()
  # ====================================================================================

else ()
  set(OpenCV_DIR "/usr/local/opencv4/lib64/cmake/opencv4") 
  find_package(OpenCV REQUIRED)
  if(USE_FREETYPE)
    if(NOT "opencv_freetype" IN_LIST OpenCV_LIBS)
      message(FATAL_ERROR "OpenCV was not compiled with the freetype module (opencv_freetype) !")
    endif()
    add_definitions(-DUSE_FREETYPE)
  endif()

  # ================= TensorRT配置（原有逻辑，保持不变）=================
  if(WITH_TensorRT)
    add_definitions(-DUSE_TensorRT)
    
    # 查找头文件
    include_directories("${CUDA_INCLUDE}")
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_DIR} ${TENSORRT_DIR}/include /usr/include /usr/include/x86_64-linux-gnu
    )
    
    # 查找库文件 (nvinfer)
    find_library(TENSORRT_LIBRARY nvinfer
        HINTS ${TENSORRT_DIR} ${TENSORRT_DIR}/lib /usr/lib/x86_64-linux-gnu
    )

    if (TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
        message(STATUS "Found TensorRT Header: ${TENSORRT_INCLUDE_DIR}")
        message(STATUS "Found TensorRT Lib: ${TENSORRT_LIBRARY}")
        include_directories(${TENSORRT_INCLUDE_DIR})
        # 将 TensorRT 库加入依赖列表
        set(DEPS ${DEPS} ${TENSORRT_LIBRARY})
    else()
        message(FATAL_ERROR "Could NOT find TensorRT. Please set -DTENSORRT_DIR=/path/to/TensorRT")
    endif()
  endif()  
  # ====================================================================

  # ================= NCNN新增4：Linux系统NCNN核心配置（纯CPU，重点）=================
  if(WITH_NCNN)
    add_definitions(-DUSE_NCNN)
    # 查找NCNN核心头文件（ncnn/net.h），适配NCNN默认安装/自定义编译路径
    find_path(NCNN_INCLUDE_DIR ncnn/net.h
        HINTS ${NCNN_DIR} ${NCNN_DIR}/include /usr/local/ncnn/include /usr/include/ncnn
    )
    # 查找NCNN静态库（纯CPU编译的libncnn.a，NCNN默认编译为静态库，与你的需求匹配）
    # 适配NCNN build/lib/ / install/lib/ / 系统库路径
    find_library(NCNN_LIBRARY ncnn
        HINTS ${NCNN_DIR} ${NCNN_DIR}/lib ${NCNN_DIR}/build/lib /usr/local/ncnn/lib /usr/lib/x86_64-linux-gnu
    )
    # 强制校验NCNN头文件和库是否找到，找不到直接抛错，提示配置NCNN_DIR
    if (NCNN_INCLUDE_DIR AND NCNN_LIBRARY)
        message(STATUS "Found NCNN Header: ${NCNN_INCLUDE_DIR}")
        message(STATUS "Found NCNN Lib: ${NCNN_LIBRARY}")
        # 添加NCNN头文件路径到工程
        include_directories(${NCNN_INCLUDE_DIR})
        # 将NCNN库添加到链接依赖DEPS中，后续统一链接
        set(DEPS ${DEPS} ${NCNN_LIBRARY})
    else()
        message(FATAL_ERROR "Could NOT find NCNN (CPU only). Please set -DNCNN_DIR=/path/to/your/compiled/ncnn")
    endif()
  endif()
  # ==================================================================================

  include_directories("${PADDLE_LIB}/paddle/include")
  link_directories("${PADDLE_LIB}/paddle/lib")
endif ()

include_directories(${OpenCV_INCLUDE_DIRS})

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
    # 原有C++11改为C++14，与NCNN/Abseil兼容（NCNN编译建议C++14+）
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 ${FLAG_OPENMP} -std=c++14")
    set(CMAKE_STATIC_LIBRARY_PREFIX "")
    message("cmake cxx flags: " ${CMAKE_CXX_FLAGS})
endif()

if (WITH_GPU)
    if (NOT DEFINED CUDA_LIB OR ${CUDA_LIB} STREQUAL "")
        message(FATAL_ERROR "please set CUDA_LIB with -DCUDA_LIB=/path/cuda-8.0/lib64")
    endif()
    if (NOT WIN32)
        if (NOT DEFINED CUDNN_LIB)
            message(FATAL_ERROR "please set CUDNN_LIB with -DCUDNN_LIB=/path/cudnn_v7.4/cuda/lib64")
        endif()
    add_definitions(-DWITH_GPU)
    endif(NOT WIN32)
endif()

include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
include_directories("${PADDLE_LIB}/third_party/install/zlib/include")
include_directories("${PADDLE_LIB}/third_party/install/onnxruntime/include")
include_directories("${PADDLE_LIB}/third_party/install/paddle2onnx/include")
include_directories("${PADDLE_LIB}/third_party/install/yaml-cpp/include")
include_directories("${PADDLE_LIB}/third_party/install/openvino/include")
include_directories("${PADDLE_LIB}/third_party/install/tbb/include")
include_directories("${PADDLE_LIB}/third_party/boost")
include_directories("${PADDLE_LIB}/third_party/eigen3")
include_directories("${PADDLE_LIB}/paddle/include/") 
include_directories("${CMAKE_SOURCE_DIR}/")

link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")
link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
link_directories("${PADDLE_LIB}/third_party/install/onnxruntime/lib")
link_directories("${PADDLE_LIB}/third_party/install/paddle2onnx/lib")
link_directories("${PADDLE_LIB}/third_party/install/yaml-cpp/lib")
link_directories("${PADDLE_LIB}/third_party/install/openvino/intel64")
link_directories("${PADDLE_LIB}/third_party/install/tbb/lib")
link_directories("${PADDLE_LIB}/paddle/lib")

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

# Note: libpaddle_inference_api.so/a must put before libpaddle_inference.so/a
if(WITH_STATIC_LIB)
  if(WIN32)
    set(DEPS ${DEPS} 
        ${PADDLE_LIB}/paddle/lib/paddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(DEPS ${DEPS} 
        ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
  endif()
else()
  if(WIN32)
    set(DEPS ${DEPS} 
        ${PADDLE_LIB}/paddle/lib/paddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    set(DEPS ${DEPS} 
        ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
endif(WITH_STATIC_LIB)

if (NOT WIN32)
    set(DEPS ${DEPS} 
        ${MATH_LIB} ${MKLDNN_LIB}
        glog gflags protobuf z xxhash
        )
    if(EXISTS "${PADDLE_LIB}/third_party/install/snappystream/lib")
        set(DEPS ${DEPS} snappystream)
    endif()
    if (EXISTS "${PADDLE_LIB}/third_party/install/snappy/lib")
        set(DEPS ${DEPS} snappy)
    endif()
else()
    set(DEPS ${DEPS}
        ${MATH_LIB} ${MKLDNN_LIB}
        glog gflags_static libprotobuf xxhash)
    set(DEPS ${DEPS} libcmt shlwapi)
    if (EXISTS "${PADDLE_LIB}/third_party/install/snappy/lib")
        set(DEPS ${DEPS} snappy)
    endif()
    if(EXISTS "${PADDLE_LIB}/third_party/install/snappystream/lib")
        set(DEPS ${DEPS} snappystream)
    endif()
endif(NOT WIN32)

if (EXISTS "${PADDLE_LIB}/third_party/install/yaml-cpp/lib")
  set(DEPS ${DEPS} yaml-cpp)
endif()

if(WITH_GPU)
  if(NOT WIN32)
    set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(DEPS ${DEPS} ${CUDNN_LIB}/libcudnn${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    set(DEPS ${DEPS} ${CUDA_LIB}/cudart${CMAKE_STATIC_LIBRARY_SUFFIX} )
    message($DEPS)
    set(DEPS ${DEPS} ${CUDA_LIB}/cublas${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDNN_LIB}/cudnn${CMAKE_STATIC_LIBRARY_SUFFIX})
  endif()
endif()

if (NOT WIN32)
    set(EXTERNAL_LIB "-ldl -lrt -lgomp -lz -lm -lpthread")
    set(DEPS ${DEPS} ${EXTERNAL_LIB})
endif()

set(THIRD_PARTY_PATH ${CMAKE_CURRENT_LIST_DIR}/third_party)
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

set(DEPS ${DEPS} ${OpenCV_LIBS})
set(DEPS ${DEPS} absl::statusor)
set(DEPS ${DEPS} polyclipping)

if(UNIX)
  find_package(Iconv REQUIRED) 
  # ================= NCNN新增5：链接Iconv库（NCNN纯CPU编译依赖Iconv，Linux必加）=================
  set(DEPS ${DEPS} ${Iconv_LIBRARIES})
  # ==============================================================================================
endif()

file(GLOB_RECURSE SRC_LIST "./src/*.cc")
set(SRCS test.cpp )
add_executable(${DEMO_NAME} ${SRCS} ${SRC_LIST} )
# 最后链接所有 DEPS，其中已包含NCNN/TensorRT/OpenCV等所有依赖
target_link_libraries(${DEMO_NAME} ${DEPS} )

if (WIN32 AND WITH_MKL)
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.dll ./mklml.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5md.dll ./libiomp5md.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/onednn/lib/mkldnn.dll ./mkldnn.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.dll ./release/mklml.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5md.dll ./release/libiomp5md.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/onednn/lib/mkldnn.dll ./release/mkldnn.dll
    )
endif()
