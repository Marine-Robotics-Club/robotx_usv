cmake_minimum_required(VERSION 3.5)
project(vision_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

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

include_directories(${OpenCV_INCLUDE_DIRS})

# Add all executables
add_executable(vision_obstacle_avoidance_node src/vision_obstacle_avoidance.cpp)
ament_target_dependencies(vision_obstacle_avoidance_node rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs pcl_conversions pcl_ros PCL fau_msgs lidar_msgs yolov26_msgs)

add_executable(test_cpp src/test_cpp.cpp)
ament_target_dependencies(test_cpp rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs pcl_conversions pcl_ros PCL fau_msgs OpenCV cv_bridge image_transport lidar_msgs yolov26_msgs)

add_executable(buoy_detector src/buoy_detector.cpp)
ament_target_dependencies(buoy_detector rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs pcl_conversions pcl_ros PCL fau_msgs lidar_msgs yolov26_msgs)

add_executable(light_buoy_detector src/light_buoy_detector.cpp)
ament_target_dependencies(light_buoy_detector rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs pcl_conversions pcl_ros PCL fau_msgs lidar_msgs yolov26_msgs)

add_executable(ball_detector src/ball_detector.cpp)
ament_target_dependencies(ball_detector rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs pcl_conversions pcl_ros PCL fau_msgs lidar_msgs cv_bridge OpenCV image_transport yolov26_msgs)

add_executable(sensor_fusion src/sensor_fusion.cpp)
ament_target_dependencies(sensor_fusion rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs pcl_conversions pcl_ros PCL fau_msgs lidar_msgs cv_bridge OpenCV image_transport yolov26_msgs)

add_executable(dock_detector src/dock_detector.cpp)
ament_target_dependencies(dock_detector rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs pcl_conversions pcl_ros PCL fau_msgs lidar_msgs cv_bridge OpenCV image_transport yolov26_msgs)

add_executable(dock_bay_detector src/dock_bay_detector.cpp)
ament_target_dependencies(dock_bay_detector rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs pcl_conversions pcl_ros PCL fau_msgs lidar_msgs cv_bridge OpenCV image_transport yolov26_msgs)

add_executable(boat_detector src/boat_detector.cpp)
ament_target_dependencies(boat_detector rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs pcl_conversions pcl_ros PCL fau_msgs lidar_msgs cv_bridge OpenCV image_transport yolov26_msgs)

add_executable(sensor_fusion_left src/sensor_fusion_left.cpp)
ament_target_dependencies(sensor_fusion_left rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs pcl_conversions pcl_ros PCL fau_msgs lidar_msgs cv_bridge OpenCV image_transport yolov26_msgs)

add_executable(sensor_fusion_right src/sensor_fusion_right.cpp)
ament_target_dependencies(sensor_fusion_right rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs pcl_conversions pcl_ros PCL fau_msgs lidar_msgs cv_bridge OpenCV image_transport yolov26_msgs)

add_executable(dock_vertices src/dock_vertices.cpp)
ament_target_dependencies(dock_vertices rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs pcl_conversions pcl_ros PCL fau_msgs lidar_msgs cv_bridge OpenCV image_transport yolov26_msgs visualization_msgs)

add_executable(Camera_Estimater_RIGHT src/Camera_Estimater_RIGHT.cpp)
ament_target_dependencies(Camera_Estimater_RIGHT rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs pcl_conversions pcl_ros PCL fau_msgs lidar_msgs OpenCV cv_bridge image_transport yolov26_msgs)

add_executable(Camera_Estimater_LEFT src/Camera_Estimater_LEFT.cpp)
ament_target_dependencies(Camera_Estimater_LEFT rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs pcl_conversions pcl_ros PCL fau_msgs lidar_msgs OpenCV cv_bridge image_transport yolov26_msgs)
# Install all executables

add_executable(Camera_Estimater_CENTER src/Camera_Estimater_CENTER.cpp)
ament_target_dependencies(Camera_Estimater_CENTER rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs pcl_conversions pcl_ros PCL fau_msgs lidar_msgs OpenCV cv_bridge image_transport yolov26_msgs)

add_executable(task_2 src/task_2.cpp)
ament_target_dependencies(task_2 rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs pcl_conversions pcl_ros PCL fau_msgs lidar_msgs OpenCV cv_bridge image_transport yolov26_msgs)

add_executable(task_1 src/task_1.cpp)
ament_target_dependencies(task_1 rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs pcl_conversions pcl_ros PCL fau_msgs lidar_msgs OpenCV cv_bridge image_transport yolov26_msgs)

install(TARGETS vision_obstacle_avoidance_node test_cpp task_1 buoy_detector light_buoy_detector ball_detector boat_detector sensor_fusion dock_detector dock_bay_detector sensor_fusion_left sensor_fusion_right dock_vertices Camera_Estimater_RIGHT Camera_Estimater_LEFT Camera_Estimater_CENTER task_2
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

