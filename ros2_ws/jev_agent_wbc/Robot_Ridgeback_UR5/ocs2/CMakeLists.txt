cmake_minimum_required(VERSION 3.16)
project(jev_ridgeback_ocs2 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
set(OCS2_SOURCE_ROOT "" CACHE PATH "Pinned official OCS2 checkout; no framework emulation")
if(NOT EXISTS "${OCS2_SOURCE_ROOT}/ocs2_core/src/dynamics/SystemDynamicsBase.cpp")
  message(FATAL_ERROR "Set OCS2_SOURCE_ROOT to the pinned official source checkout")
endif()
# A real, explicitly limited OCS2 core build. This is not an MPC solver build.
add_library(ocs2_dynamics_slice STATIC
  ${OCS2_SOURCE_ROOT}/ocs2_core/src/Types.cpp
  ${OCS2_SOURCE_ROOT}/ocs2_core/src/misc/LinearAlgebra.cpp
  ${OCS2_SOURCE_ROOT}/ocs2_core/src/integration/OdeBase.cpp
  ${OCS2_SOURCE_ROOT}/ocs2_core/src/dynamics/ControlledSystemBase.cpp
  ${OCS2_SOURCE_ROOT}/ocs2_core/src/dynamics/SystemDynamicsBase.cpp)
target_include_directories(ocs2_dynamics_slice PUBLIC ${OCS2_SOURCE_ROOT}/ocs2_core/include)
target_link_libraries(ocs2_dynamics_slice PUBLIC Eigen3::Eigen)
add_executable(dynamics_check src/dynamics_check.cpp)
target_include_directories(dynamics_check PRIVATE include)
target_compile_definitions(dynamics_check PRIVATE JEV_OMNI_BASE=1)
target_link_libraries(dynamics_check PRIVATE ocs2_dynamics_slice)
enable_testing()
add_test(NAME actual_ocs2_dynamics COMMAND dynamics_check)

option(JEV_BUILD_MPC "Build the genuine OCS2 SLQ camera prediction probe" OFF)
if(JEV_BUILD_MPC)
  find_package(Boost REQUIRED COMPONENTS system filesystem log_setup log)
  find_package(ament_cmake REQUIRED)
  find_package(ocs2_core REQUIRED)
  find_package(ocs2_oc REQUIRED)
  find_package(ocs2_mpc REQUIRED)
  find_package(ocs2_ddp REQUIRED)
  add_executable(camera_mpc_probe src/camera_mpc_probe.cpp)
  target_include_directories(camera_mpc_probe PRIVATE include)
  target_link_libraries(camera_mpc_probe Eigen3::Eigen)
  ament_target_dependencies(camera_mpc_probe ocs2_core ocs2_oc ocs2_mpc ocs2_ddp)
  add_executable(camera_mpc_worker src/camera_mpc_worker.cpp)
  target_include_directories(camera_mpc_worker PRIVATE include)
  target_link_libraries(camera_mpc_worker Eigen3::Eigen)
  ament_target_dependencies(camera_mpc_worker ocs2_core ocs2_oc ocs2_mpc ocs2_ddp)
endif()

# Measured-body safety guard shares the exact hull-distance implementation.
add_library(measured_collision_guard SHARED src/measured_collision_guard.cpp)
target_include_directories(measured_collision_guard PRIVATE include)
target_link_libraries(measured_collision_guard PRIVATE Eigen3::Eigen)

add_executable(acceleration_check src/acceleration_check.cpp)
target_include_directories(acceleration_check PRIVATE include)
target_link_libraries(acceleration_check PRIVATE ocs2_dynamics_slice)
add_test(NAME actual_ocs2_acceleration_dynamics COMMAND acceleration_check)
