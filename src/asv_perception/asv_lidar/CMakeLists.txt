cmake_minimum_required(VERSION 3.8)
project(asv_lidar)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(fau_msgs REQUIRED)
find_package(lidar_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(image_transport REQUIRED)
find_package(yolov26_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# Install launch and config directories
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

add_executable(pyramid_buoy_detector src/pyramid_buoy_detector.cpp)
ament_target_dependencies(pyramid_buoy_detector rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs pcl_conversions pcl_ros PCL fau_msgs lidar_msgs yolov26_msgs)


install(TARGETS pyramid_buoy_detector
  DESTINATION lib/${PROJECT_NAME}
)

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
