from robotic_arm_package.robotic_arm import *

arm_left = Arm(RM65, "192.168.1.18")
_,state = arm_left.Get_Gripper_State()
print(state.actpos)