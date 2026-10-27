cmake_minimum_required(VERSION 3.8)
project(line_follower)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ai_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_core
  src/line_follower_node.cpp
)
ament_target_dependencies(${PROJECT_NAME}_core
  rclcpp
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  OpenCV
  std_msgs
  ai_msgs
  nav_msgs
)

add_executable(line_follower_node src/main.cpp)
target_link_libraries(line_follower_node ${PROJECT_NAME}_core)

add_executable(detection_turn_controller src/detection_turn_controller.cpp)
ament_target_dependencies(detection_turn_controller
  rclcpp
  ai_msgs
  geometry_msgs
)

add_executable(cmd_mux_node src/cmd_mux_node.cpp)
ament_target_dependencies(cmd_mux_node
  rclcpp
  geometry_msgs
)

add_executable(start_toggle_node src/start_toggle_node.cpp)
ament_target_dependencies(start_toggle_node
  rclcpp
  std_msgs
)

install(TARGETS ${PROJECT_NAME}_core line_follower_node
  DESTINATION lib/${PROJECT_NAME})
install(TARGETS detection_turn_controller
  DESTINATION lib/${PROJECT_NAME})
install(TARGETS cmd_mux_node
  DESTINATION lib/${PROJECT_NAME})
install(TARGETS start_toggle_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
