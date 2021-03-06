# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)

project(tritononnxruntimebackend LANGUAGES C CXX)

#
# Options
#
# To build the ONNX Runtime backend you must either:
#
#   - Point to an already built ONNX Runtime using
#     TRITON_ONNXRUNTIME_INCLUDE_PATHS and
#     TRITON_ONNXRUNTIME_LIB_PATHS
#
#   or:
#
#   - Set TRITON_BUILD_ONNXRUNTIME_VERSION to the version of ONNX
#     Runtime that you want to be built for the backend. Set
#     TRITON_BUILD_CONTAINER_VERSION to the Triton container version
#     that you want to target with the build.
#   - If you want TensorRT support set
#     TRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON.
#   - If you want OpenVINO support set
#     TRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON and set
#     TRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION to the OpenVino
#     version that is compatible with the specified version of ONNX
#     Runtime.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
option(TRITON_ENABLE_ONNXRUNTIME_TENSORRT
  "Enable TensorRT execution provider for ONNXRuntime backend in server" OFF)
option(TRITON_ENABLE_ONNXRUNTIME_OPENVINO
  "Enable OpenVINO execution provider for ONNXRuntime backend in server" OFF)
set(TRITON_BUILD_CONTAINER_VERSION "" CACHE STRING "Triton container version to build for")
set(TRITON_BUILD_ONNXRUNTIME_VERSION "" CACHE STRING "ONNXRuntime version to build")
set(TRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION "" CACHE STRING "ONNXRuntime OpenVINO version to build")
set(TRITON_ONNXRUNTIME_INCLUDE_PATHS "" CACHE PATH "Paths to ONNXRuntime includes")
set(TRITON_ONNXRUNTIME_LIB_PATHS "" CACHE PATH "Paths to ONNXRuntime libraries")

set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if("${TRITON_BUILD_ONNXRUNTIME_VERSION}" STREQUAL "")
  find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS ${TRITON_ONNXRUNTIME_LIB_PATHS})
  if(${TRITON_ENABLE_ONNXRUNTIME_OPENVINO})
    find_library(TBB_LIBRARY NAMES tbb PATHS ${TRITON_ONNXRUNTIME_LIB_PATHS})
    find_library(INFERENCE_ENGINE_LIBRARY
      NAMES inference_engine
      PATHS ${TRITON_ONNXRUNTIME_LIB_PATHS})
  endif() # TRITON_ENABLE_ONNXRUNTIME_OPENVINO

else()

  if(NOT TRITON_BUILD_CONTAINER_VERSION)
    message(FATAL_ERROR "TRITON_BUILD_ONNXRUNTIME_VERSION requires TRITON_BUILD_CONTAINER_VERSION")
  endif()

  set(TRITON_ONNXRUNTIME_INCLUDE_PATHS "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/include")
  set(TRITON_ONNXRUNTIME_LIB_PATHS "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/lib")
  set(TRITON_ONNXRUNTIME_DOCKER_IMAGE "tritonserver_onnxruntime")
  set(ONNXRUNTIME_LIBRARY "libonnxruntime.so.${TRITON_BUILD_ONNXRUNTIME_VERSION}")
  if(${TRITON_ENABLE_ONNXRUNTIME_OPENVINO})
    set(TBB_LIBRARY "libtbb.so")
    set(INFERENCE_ENGINE_LIBRARY "libinference_engine.so")
  endif() # TRITON_ENABLE_ONNXRUNTIME_OPENVINO
endif()

#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
endif() # TRITON_ENABLE_GPU

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_onnxruntime.ldscript libtriton_onnxruntime.ldscript COPYONLY)

add_library(
  triton-onnxruntime-backend SHARED
  src/onnxruntime.cc
  src/onnxruntime_loader.cc
  src/onnxruntime_loader.h
  src/onnxruntime_utils.cc
  src/onnxruntime_utils.h
)

add_library(
  TritonOnnxRuntimeBackend::triton-onnxruntime-backend ALIAS triton-onnxruntime-backend
)

target_include_directories(
  triton-onnxruntime-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TRITON_ONNXRUNTIME_INCLUDE_PATHS}
)

target_compile_features(triton-onnxruntime-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-onnxruntime-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc>
)

if(${TRITON_ENABLE_GPU})
  target_compile_definitions(
    triton-onnxruntime-backend
    PRIVATE TRITON_ENABLE_GPU=1
  )
endif() # TRITON_ENABLE_GPU
if(${TRITON_ENABLE_ONNXRUNTIME_TENSORRT})
  target_compile_definitions(
    triton-onnxruntime-backend
    PRIVATE TRITON_ENABLE_ONNXRUNTIME_TENSORRT=1
  )
endif() # TRITON_ENABLE_ONNXRUNTIME_TENSORRT
if(${TRITON_ENABLE_ONNXRUNTIME_OPENVINO})
  target_compile_definitions(
    triton-onnxruntime-backend
    PRIVATE TRITON_ENABLE_ONNXRUNTIME_OPENVINO=1
  )
endif() # TRITON_ENABLE_ONNXRUNTIME_OPENVINO

if (WIN32)
set_target_properties(
  triton-onnxruntime-backend
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_onnxruntime
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}"
)
else ()
set_target_properties(
  triton-onnxruntime-backend
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_onnxruntime
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}"
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_onnxruntime.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_onnxruntime.ldscript"
)
endif()

FOREACH(p ${TRITON_ONNXRUNTIME_LIB_PATHS})
  set(TRITON_ONNXRUNTIME_LDFLAGS ${TRITON_ONNXRUNTIME_LDFLAGS} "-L${p}")
ENDFOREACH(p)

target_link_libraries(
  triton-onnxruntime-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
    ${TRITON_ONNXRUNTIME_LDFLAGS}
    ${ONNXRUNTIME_LIBRARY}
)

if(${TRITON_ENABLE_GPU})
  target_link_libraries(
    triton-onnxruntime-backend
    PRIVATE
      CUDA::cudart
  )
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_ONNXRUNTIME_OPENVINO})
  target_link_libraries(
    triton-onnxruntime-backend
    PRIVATE
      ${TBB_LIBRARY}
      ${INFERENCE_ENGINE_LIBRARY}
  )
endif() # TRITON_ENABLE_ONNXRUNTIME_OPENVINO

#
# Build the ONNX Runtime libraries using docker.
#
if(NOT ("${TRITON_BUILD_ONNXRUNTIME_VERSION}" STREQUAL ""))
  set(_GEN_FLAGS "")
  if(${TRITON_ENABLE_ONNXRUNTIME_TENSORRT})
    set(_GEN_FLAGS ${_GEN_FLAGS} "--ort-tensorrt")
  endif() # TRITON_ENABLE_ONNXRUNTIME_TENSORRT
  if(${TRITON_ENABLE_ONNXRUNTIME_OPENVINO})
    set(_GEN_FLAGS ${_GEN_FLAGS} "--ort-openvino=${TRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION}")
  endif() # TRITON_ENABLE_ONNXRUNTIME_OPENVINO
  add_custom_command(
    OUTPUT
      onnxruntime/lib/${ONNXRUNTIME_LIBRARY}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_ort_dockerfile.py --triton-container="nvcr.io/nvidia/tritonserver:${TRITON_BUILD_CONTAINER_VERSION}-py3-min" --ort-version="${TRITON_BUILD_ONNXRUNTIME_VERSION}" ${_GEN_FLAGS} --output=Dockerfile.ort
    COMMAND docker build --cache-from=tritonserver_onnxruntime --cache-from=tritonserver_onnxruntime_cache0 --cache-from=tritonserver_onnxruntime_cache1 -t ${TRITON_ONNXRUNTIME_DOCKER_IMAGE} -f ./Dockerfile.ort ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND docker rm onnxruntime_backend_ort || echo "error ignored..." || true
    COMMAND docker create --name onnxruntime_backend_ort ${TRITON_ONNXRUNTIME_DOCKER_IMAGE}
    COMMAND rm -fr onnxruntime
    COMMAND docker cp onnxruntime_backend_ort:/opt/onnxruntime onnxruntime
    COMMAND docker rm onnxruntime_backend_ort
    COMMENT "Building ONNX Runtime"
  )
  add_custom_target(ort_target DEPENDS onnxruntime/lib/${ONNXRUNTIME_LIBRARY})
  add_library(onnxruntime-library SHARED IMPORTED GLOBAL)
  add_dependencies(onnxruntime-library ort_target)
  add_dependencies(triton-onnxruntime-backend onnxruntime-library)
  set_target_properties(
    onnxruntime-library
    PROPERTIES
      IMPORTED_LOCATION onnxruntime/lib/${ONNXRUNTIME_LIBRARY}
  )
endif() # TRITON_ONNXRUNTIME_DOCKER_BUILD

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonOnnxRuntimeBackend)

install(
  TARGETS
    triton-onnxruntime-backend
  EXPORT
    triton-onnxruntime-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/onnxruntime
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/onnxruntime
)

install(
  DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/onnxruntime
  PATTERN *lib EXCLUDE
  PATTERN *bin EXCLUDE
  PATTERN *include EXCLUDE
  PATTERN *test EXCLUDE
)

install(
  DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/bin/
    ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/lib/
  USE_SOURCE_PERMISSIONS
  DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/onnxruntime
)

install(
  DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/test
  USE_SOURCE_PERMISSIONS
  DESTINATION ${CMAKE_INSTALL_PREFIX}
)

install(
  EXPORT
    triton-onnxruntime-backend-targets
  FILE
    TritonOnnxRuntimeBackendTargets.cmake
  NAMESPACE
    TritonOnnxRuntimeBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonOnnxRuntimeBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonOnnxRuntimeBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonOnnxRuntimeBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-onnxruntime-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonOnnxRuntimeBackendTargets.cmake
  NAMESPACE TritonOnnxRuntimeBackend::
)

export(PACKAGE TritonOnnxRuntimeBackend)
