# Install script for directory: /home/rm/rm_aloha/src/rm_msgs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/rm/rm_aloha/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/rm_msgs/msg" TYPE FILE FILES
    "/home/rm/rm_aloha/src/rm_msgs/msg/Arm_Analog_Output.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Arm_Digital_Output.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Arm_Joint_Speed_Max.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Arm_IO_State.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/JointPos.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/MoveC.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/MoveJ.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/MoveJ_P.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/MoveL.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Tool_Analog_Output.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Tool_Digital_Output.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Tool_IO_State.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Plan_State.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Cabinet.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/ChangeTool_State.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/ChangeTool_Name.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/ChangeWorkFrame_State.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/ChangeWorkFrame_Name.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Arm_Current_State.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/GetArmState_Command.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Stop.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Joint_Teach.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Pos_Teach.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Ort_Teach.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Stop_Teach.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Gripper_Set.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Gripper_Pick.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Joint_Enable.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Joint_Max_Speed.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/IO_Update.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Turtle_Driver.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Socket_Command.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Start_Multi_Drag_Teach.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Set_Force_Position.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Force_Position_Move_Joint.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Force_Position_Move_Pose.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Force_Position_State.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Six_Force.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Manual_Set_Force_Pose.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/CartePos.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Lift_Height.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Lift_Speed.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Joint_Current.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Joint_Step.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/ArmState.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Hand_Posture.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Hand_Seq.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Hand_Speed.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Hand_Force.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Hand_Angle.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/LiftState.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Servo_GetAngle.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Servo_Move.msg"
    "/home/rm/rm_aloha/src/rm_msgs/msg/Set_Realtime_Push.msg"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/rm_msgs/cmake" TYPE FILE FILES "/home/rm/rm_aloha/build/rm_msgs/catkin_generated/installspace/rm_msgs-msg-paths.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/rm/rm_aloha/devel/include/rm_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/rm/rm_aloha/devel/share/roseus/ros/rm_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/rm/rm_aloha/devel/share/common-lisp/ros/rm_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/rm/rm_aloha/devel/share/gennodejs/ros/rm_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python3.8" -m compileall "/home/rm/rm_aloha/devel/lib/python3/dist-packages/rm_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages" TYPE DIRECTORY FILES "/home/rm/rm_aloha/devel/lib/python3/dist-packages/rm_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/rm/rm_aloha/build/rm_msgs/catkin_generated/installspace/rm_msgs.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/rm_msgs/cmake" TYPE FILE FILES "/home/rm/rm_aloha/build/rm_msgs/catkin_generated/installspace/rm_msgs-msg-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/rm_msgs/cmake" TYPE FILE FILES
    "/home/rm/rm_aloha/build/rm_msgs/catkin_generated/installspace/rm_msgsConfig.cmake"
    "/home/rm/rm_aloha/build/rm_msgs/catkin_generated/installspace/rm_msgsConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/rm_msgs" TYPE FILE FILES "/home/rm/rm_aloha/src/rm_msgs/package.xml")
endif()

