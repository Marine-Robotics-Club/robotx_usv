cmake_minimum_required(VERSION 3.8)
project(fau_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PropulsionSystem.msg"
  "msg/Vision.msg"
  "msg/NedAcoustic.msg"
  "msg/NedObjects.msg"
  "msg/State.msg"
  "msg/ObjectInfo.msg"
  "msg/ObjectPositionArray.msg"
  "msg/ObjectPosition.msg"
  "msg/OccupancyGridInfo.msg"
  "msg/MotorCmds.msg"
  "msg/TeensyStatus.msg"
  "msg/AnchorCmds.msg"
  "msg/DroneCmds.msg"
  "msg/PepCmds.msg"
  "msg/PepStatus.msg"
  "msg/DroneStatus.msg"
  "msg/PtoStatus.msg"
  "msg/YoloDetection.msg"
  "msg/PidGains.msg"
  "msg/TaskInfo.msg"
  "msg/ControlEffort.msg"

  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
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
