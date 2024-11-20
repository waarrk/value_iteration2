cmake_minimum_required(VERSION 3.8)
project(value_iteration2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(CUDA REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(grid_map_msgs REQUIRED)

# Set CUDA architecture
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_86")

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
ament_auto_find_build_dependencies()

# Specify CUDA source files
set(CUDA_SOURCES
  src/value_iteration_kernel.cu
)

# Add include directories
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${rclcpp_INCLUDE_DIRS}
  ${nav_msgs_INCLUDE_DIRS}
  ${rcl_interfaces_INCLUDE_DIRS}
  ${grid_map_msgs_INCLUDE_DIRS}
)

# Create CUDA library
cuda_add_library(value_iteration_kernel STATIC ${CUDA_SOURCES})

# Add executable for the node
ament_auto_add_executable(vi_node
  src/Action.cpp
  src/SweepWorkerStatus.cpp
  src/ValueIterator.cpp
  src/ValueIteratorLocal.cpp
  src/State.cpp
  src/StateTransition.cpp
  src/vi_node.cpp
)

# Link CUDA library and dependencies to the executable
target_link_libraries(vi_node
  value_iteration_kernel
  rclcpp::rclcpp
)

ament_target_dependencies(vi_node
  rclcpp
  nav_msgs
  rcl_interfaces
  grid_map_msgs
)

# Install targets
install(TARGETS vi_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY maps DESTINATION share/${PROJECT_NAME})

# Optional: Setup for testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)

  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
