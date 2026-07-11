cmake_minimum_required(VERSION 3.18)
project(IntentPredictor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75;86)     # Change once you confirm which GPU you are using
if(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "$(CMAKE_CXX_FLAGS) $(TORCH_CXX_FLAGS)")

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

add_executable(intent_predictor
    src/main.cpp
    src/buffer.cpp
    src/pose_receiver.cpp
    src.predictor.cpp
    src/adapter.cpp
    kernels/traj_scorer.cpp
    kernels/adapter_grad.cpp
)

target_include_directories(intent_predictor PRIVATE
    include
    ${ZMQ_INCLUDE_DIRS}
)
target_link_directories(intent_predictor PRIVATE
    ${TORCH_LIBRARIES}
    ${ZMQ_LINK_LIBRARIES}
)

add_executable(test_inference tests/test_inference.cpp src/predictor.cpp)
target_include_directories(test_inference PRIVATE include)
target_link_libraries(test_inference PRIVATE ${TORCH_LIBRARIES})