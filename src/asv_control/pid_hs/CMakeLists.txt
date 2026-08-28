cmake_minimum_required(VERSION 3.8)
project(pid_hs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(fau_msgs REQUIRED)
include_directories(include)
add_executable(pid_hs_node
  src/pid_hs.cpp
  src/pid_hs_main.cpp
)

ament_target_dependencies(pid_hs_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  tf2 
  tf2_ros 
  nav_msgs 
  tf2_geometry_msgs
  fau_msgs
)

install(TARGETS
  pid_hs_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
