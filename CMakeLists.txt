cmake_minimum_required(VERSION 3.0.2)
project(chef_robot_assistant)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  message_generation
  rospy
  sensor_msgs
  std_msgs
)

add_service_files(
  FILES
  DetectIngredients.srv
  GenerateDishImage.srv
  GenerateRecipe.srv
  SpeakText.srv
  TranscribeSpeech.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    cv_bridge
    message_runtime
    rospy
    sensor_msgs
    std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/interaction_manager.py
  scripts/media_node.py
  scripts/recipe_node.py
  scripts/speech_node.py
  scripts/vision_node.py
  scripts/webcam_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
