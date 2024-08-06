import time
import numpy as np
import collections
import matplotlib.pyplot as plt
import dm_env
import socket
from pyquaternion import Quaternion
import json
# from constants import DT, START_ARM_POSE, MASTER_GRIPPER_JOINT_NORMALIZE_FN, PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN
# from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
# from constants import PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
from robot_utils import Recorder, ImageRecorder
from camera_utils import MultiCameraReader
# from robot_utils import setup_master_bot, setup_puppet_bot, move_arms, move_grippers
# from interbotix_xs_modules.arm import InterbotixManipulatorXS
# from interbotix_xs_msgs.msg import JointSingleCommand
import pyrealsense2 as rs
import pyagxrobots
# from dynamixel_client import DynamixelClient
from robotic_arm_package.robotic_arm import *
import IPython

e = IPython.embed


class RealEnv:
    """
    Environment for real robot bi-manual manipulation
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam_low": (480x640x3),         # h, w, c, dtype='uint8'
                                   "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
    """

    def __init__(self, arm_left_ip, arm_right_ip, serial_numbers):

        self.arm_left = socket.socket()
        self.arm_left.connect((arm_left_ip, 8080))
        self.arm_right = socket.socket()
        self.arm_right.connect((arm_right_ip, 8080))

        self.cmd_get_joint_degree = '{"command":"get_joint_degree"}\r\n'
        self.cmd_get_gripper_state = '{"command":"get_gripper_state"}\r\n'
        self.cmd_set_gripper_release = '{"command": "set_gripper_release", "speed": 500, "block": true}\r\n'
        self.cmd_set_gripper_route = '{"command":"set_gripper_route","min":0,"max":1000}\r\n'

        # self.recorder_left = Recorder('l', init_node=True)
        # time.sleep(1.0)
        # self.recorder_right = Recorder('r', init_node=True)
        # time.sleep(1.0)
        # q 应该需要创建ros节点 初始化函数的时候已经覆盖了
        # self.image_recorder = ImageRecorder(init_node=True)
        # serial_numbers = ['242322076532', '152122077968', '150622070125']
        print("================================Camera_Init========================================")
        self.camera_reader = MultiCameraReader(serial_numbers)
        time.sleep(1)

        # at 初始化的时候需要设置夹爪的状态
        self.arm_left.send(self.cmd_set_gripper_route.encode('utf-8'))
        _ = self.arm_left.recv(1024)
        self.arm_right.send(self.cmd_set_gripper_route.encode('utf-8'))
        _ = self.arm_right.recv(1024)

        # at 需要创建一个变量来保存 上一个时间步的位姿信息，方便计算得到速度——速度采用的是位置差分
        # at 每次获取环境信息的时候 会先获取位姿信息 ==> 在求解速度的时候 不需要重复去获取位姿信息 直接读之前采集的信息即可
        self.pre_arm_qpos = None
        self.cur_arm_qpos = None

    def get_rm_pose(self):
        left_states = self.arm_left.Get_Joint_Degree()
        right_states = self.arm_right.Get_Joint_Degree()
        return left_states, right_states

    def json_list(self, byte_data, key):
        # 将字节串解码为字符串
        str_data = byte_data.decode('utf-8')
        # 使用 json.loads 解析 JSON 字符串并直接提取 joint 列表
        joint_list = json.loads(str_data)[key]
        # 将列表转换为 NumPy 数组并进行缩放
        date_np = np.array(joint_list, dtype=float)
        return date_np

    def tojson(self, data, cmd):
        if cmd == 'joint':
            np.floor(data * 1000).astype(int)
            joint_str = json.dumps(data)
            cmd = '{"command":"movej", "joint":' + joint_str + ',"v":40,"r":0}\r\n'
        else:
            gripper_data = json.dumps(data)
            cmd = '{"command":"set_gripper_position","position":' + gripper_data + ' ,"block":true}\r\n'
        return cmd

    # 6个关节角度  * 2 + 末端夹爪的姿态
    def get_qpos(self):
        t0 = time.time()
        # _, left_qpos_raw = self.arm_left.Get_Joint_Degree()
        # _, right_qpos_raw = self.arm_right.Get_Joint_Degree()
        # joint_right = self.arm_right.recv(1024)
        # joint_right = self.arm_right.recv(1024)

        self.arm_left.send(self.cmd_get_joint_degree.encode('utf-8'))
        joint_left_qpos = self.arm_left.recv(1024)
        # print("left:", joint_left_qpos)
        self.arm_right.send(self.cmd_get_joint_degree.encode('utf-8'))
        joint_right_qpos = self.arm_right.recv(1024)
        # print("right:", joint_right_qpos)
        t1 = time.time()
        print("Joint_Degree_time:", t1 - t0)
        left_arm_qpos = self.json_list(joint_left_qpos, 'joint') * 0.001
        right_arm_qpos = self.json_list(joint_right_qpos, 'joint') * 0.001

        # 夹爪的角度/速度 状态
        t0 = time.time()
        self.arm_left.send(self.cmd_get_gripper_state.encode('utf-8'))
        left_gripper_qpos = self.arm_left.recv(1024)
        print('left_gripper_qpos:', left_gripper_qpos)
        self.arm_right.send(self.cmd_get_gripper_state.encode('utf-8'))
        right_gripper_qpos = self.arm_right.recv(1024)
        print('right_gripper_qpos:', right_gripper_qpos)

        # _, self.left_gripper_qpos = self.arm_left.Get_Gripper_State()
        # _, self.right_gripper_qpos = self.arm_right.Get_Gripper_State()
        t1 = time.time()
        print("Gripper_State_time:", t1 - t0)
        # Q actpos ---> 夹爪开口度 并无数据   temperature表示有数字显示---> 相对度量角度 范围是[0,1000]
        left_actpos = self.json_list(left_gripper_qpos, "actpos")  # 开合度，软件写的temperature
        right_actpos = self.json_list(right_gripper_qpos, "actpos")

        # at 将夹爪的角度添加到 qpos中去  6 + 6 + 2*夹爪
        left_arm_qpos = np.append(left_arm_qpos, left_actpos)
        right_arm_qpos = np.append(right_arm_qpos, right_actpos)
        arm_qpos = np.concatenate([left_arm_qpos, right_arm_qpos])
        # print(arm_qpos)

        #  at 在每次读取qpos的时候就把变量存入
        if self.pre_arm_qpos is None:
            self.pre_arm_qpos = arm_qpos
            self.cur_arm_qpos = arm_qpos
        else:
            self.pre_arm_qpos = self.cur_arm_qpos
            self.cur_arm_qpos = arm_qpos

        return arm_qpos

    def get_qvel(self):
        # TODO 修改速度获取
        #  q calc_velocity_from_diff_position(last_pos, pos, dT: float)
        #  Q 速度的获取需要上一个时间步的位姿 和当前的位姿 相当于差速计算
        # at 修改如下： 把夹爪的角度也放入了qpos当中去了 这样可以直接计算6个关节 + 2夹爪
        arm_qvel = calc_velocity_from_diff_position(self.pre_arm_qpos, self.cur_arm_qpos, 0.02)
        return arm_qvel

    def get_effort(self):
        # TODO  填充力相关函数
        # 先要获取电流，由电流--->力 def current_to_torque(current: list)
        t0 = time.time()
        _, left_current_raw = self.arm_left.Get_Joint_Current()
        _, right_current_raw = self.arm_right.Get_Joint_Current()
        t1 = time.time()
        print("Joint_Current_time", t1 - t0)
        # q 需要知道维度 机械臂是6自由度 6+6
        left_current_raw = left_current_raw[:6]
        right_current_raw = right_current_raw[:6]

        left_robot_effort = current_to_torque(left_current_raw)
        right_robot_effort = current_to_torque(right_current_raw)

        # todo 加入夹爪的力
        # Q 已完成！！！ 6+6+1+1=14
        left_force = self.left_gripper_qpos.current_force
        right_force = self.right_gripper_qpos.current_force
        left_robot_effort.append(left_force)
        right_robot_effort.append(right_force)

        left_robot_effort = np.array(left_robot_effort)
        right_robot_effort = np.array(right_robot_effort)
        # q 根据前面数据的要求是7+7=14
        return np.concatenate([left_robot_effort, right_robot_effort])
        # return np.zeros(14)

    def get_images(self):
        # return self.image_recorder.get_images()
        # return get_camera(self.camera)
        return self.camera_reader.get_frames()

    def get_base_vel_t265(self):
        raise NotImplementedError
        frames = self.pipeline.wait_for_frames()
        pose_frame = frames.get_pose_frame()
        pose = pose_frame.get_pose_data()

        q1 = Quaternion(w=pose.rotation.w, x=pose.rotation.x, y=pose.rotation.y, z=pose.rotation.z)
        rotation = -np.array(q1.yaw_pitch_roll)[0]
        rotation_vec = np.array([np.cos(rotation), np.sin(rotation)])
        linear_vel_vec = np.array([pose.velocity.z, pose.velocity.x])
        is_forward = rotation_vec.dot(linear_vel_vec) > 0

        base_linear_vel = np.sqrt(pose.velocity.z ** 2 + pose.velocity.x ** 2) * (1 if is_forward else -1)
        base_angular_vel = pose.angular_velocity.y
        return np.array([base_linear_vel, base_angular_vel])

    def set_gripper_pose(self, left_gripper_desired_pos_normalized, right_gripper_desired_pos_normalized):
        left_gripper_desired_joint = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(left_gripper_desired_pos_normalized)
        self.gripper_command.cmd = left_gripper_desired_joint
        self.puppet_bot_left.gripper.core.pub_single.publish(self.gripper_command)

        right_gripper_desired_joint = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(right_gripper_desired_pos_normalized)
        self.gripper_command.cmd = right_gripper_desired_joint
        self.puppet_bot_right.gripper.core.pub_single.publish(self.gripper_command)

    def _reset_joints(self):
        reset_position = START_ARM_POSE[:6]
        move_arms([self.puppet_bot_left, self.puppet_bot_right], [reset_position, reset_position], move_time=1)

    def _reset_gripper(self):
        """Set to position mode and do position resets: first open then close. Then change back to PWM mode"""
        move_grippers([self.puppet_bot_left, self.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)
        move_grippers([self.puppet_bot_left, self.puppet_bot_right], [PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=1)

    def get_observation(self, get_tracer_vel=False):
        obs = collections.OrderedDict()
        t0 = time.time()
        obs['images'] = self.get_images()
        t1 = time.time()
        print("image_time:", t1 - t0)
        obs['qpos'] = self.get_qpos()
        # obs['qvel'] = self.get_qvel()
        # obs['effort'] = self.get_effort()

        return obs

    def get_reward(self):
        return 0

    def reset(self, fake=False):
        # if not fake:
        #     # Reboot puppet robot gripper motors
        #     self.puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
        #     self.puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
        #     self._reset_joints()
        #     self._reset_gripper()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def set_gripper_pose(self, left_gripper_desired_pos_normalized, right_gripper_desired_pos_normalized):

        self.arm_left.Set_Gripper_Position(int(1 + left_gripper_desired_pos_normalized * 999), block=False)
        self.arm_right.Set_Gripper_Position(int(1 + right_gripper_desired_pos_normalized * 999), block=False)

    def step_rm(self, action, base_action=None, get_tracer_vel=False, get_obs=True):
        # state_len = int(len(action) / 2)
        left_action = action[:7]
        right_action = action[7:]
        # self.arm_left.Movej_Cmd(left_action[:6], v=50, block=False)
        # self.arm_right.Movej_Cmd(right_action[:6], v=50, block=False)
        # self.set_gripper_pose(left_action[-1], right_action[-1])
        self.arm_left.send(self.tojson(left_action[:6], 'joint').encode('utf-8'))
        left_joint = self.arm_left.recv(1024)
        self.arm_right.send(self.tojson(right_action[:6], 'joint').encode('utf-8'))
        right_joint = self.arm_right.recv(1024)
        self.arm_left.send(self.tojson(left_action[-1], 'gripper').encode('utf-8'))
        left_gripper = self.arm_left.recv(1024)
        self.arm_right.send(self.tojson(right_action[-1], 'gripper').encode('utf-8'))
        right_gripper = self.arm_right.recv(1024)
        if base_action is not None:
            # linear_vel_limit = 1.5
            # angular_vel_limit = 1.5
            # base_action_linear = np.clip(base_action[0], -linear_vel_limit, linear_vel_limit)
            # base_action_angular = np.clip(base_action[1], -angular_vel_limit, angular_vel_limit)
            base_action_linear, base_action_angular = base_action
        # time.sleep(DT)
        if get_obs:
            obs = self.get_observation(get_tracer_vel)
        else:
            obs = None
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs)

    def step(self, base_action=None, get_tracer_vel=False, get_obs=True):
        # state_len = int(len(action) / 2)
        # left_action = action[:state_len]
        # right_action = action[state_len:]
        # self.puppet_bot_left.arm.set_joint_positions(left_action[:6], blocking=False)
        # self.puppet_bot_right.arm.set_joint_positions(right_action[:6], blocking=False)
        # self.set_gripper_pose(left_action[-1], right_action[-1])
        # if base_action is not None:
        #     # linear_vel_limit = 1.5
        #     # angular_vel_limit = 1.5
        #     # base_action_linear = np.clip(base_action[0], -linear_vel_limit, linear_vel_limit)
        #     # base_action_angular = np.clip(base_action[1], -angular_vel_limit, angular_vel_limit)
        #     base_action_linear, base_action_angular = base_action
        # # time.sleep(DT)
        if get_obs:
            obs = self.get_observation(get_tracer_vel)
        else:
            obs = None
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs)


# 由电流获取力
def current_to_torque(current: list):
    """
    current: A
    torque: Nm
    """
    ki_rm65 = [7, 7, 7, 3, 3, 3]
    torque = current
    for i in range(len(current)):
        torque[i] = current[i] * ki_rm65[i]

    return torque


# 获取速度
def calc_velocity_from_diff_position(last_pos, pos, dT: float):
    """
    last_pos: 上一时刻的位置, 若为第一次, 则last_pos = pos
    pos: 当前时刻的位置
    dT: 时间间隔, 单位: 秒
    """
    # 检查pos的数据类型 是否为 int float
    type_flag = isinstance(pos, (int, float))
    if not type_flag:
        vel = pos
        for i in range(len(pos)):
            vel[i] = (pos[i] - last_pos[i]) / dT

        return vel
    else:
        return (pos - last_pos) / dT


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)  # 6 joint + 1 gripper, for two arms
    # Arm actions
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7 + 6] = master_bot_right.dxl.joint_states.position[:6]
    # Gripper actions
    action[6] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_left.dxl.joint_states.position[6])
    action[7 + 6] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])

    return action


# def get_base_action():


def make_real_env(arm_left_ip, arm_right_ip, serial_numbers):
    env = RealEnv(arm_left_ip, arm_right_ip, serial_numbers)
    return env


def test_real_teleop():
    """
    Test bimanual teleoperation and show image observations onscreen.
    It first reads joint poses from both master arms.
    Then use it as actions to step the environment.
    The environment returns full observations including images.

    An alternative approach is to have separate scripts for teleoperation and observation recording.
    This script will result in higher fidelity (obs, action) pairs
    """

    onscreen_render = True
    render_cam = 'cam_left_wrist'

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    setup_master_bot(master_bot_left)
    setup_master_bot(master_bot_right)

    # setup the environment
    env = make_real_env(init_node=True)
    ts = env.reset(fake=True)
    episode = [ts]
    # setup visualization
    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam])
        plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        if onscreen_render:
            plt_img.set_data(ts.observation['images'][render_cam])
            plt.pause(DT)
        else:
            time.sleep(DT)


if __name__ == '__main__':
    test_real_teleop()
