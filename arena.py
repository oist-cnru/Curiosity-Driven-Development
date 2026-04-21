#%%
# ============================
# IMPORTS
# ============================

# Standard libraries
import os
import math
from math import pi, sin, cos, tan, radians, degrees, sqrt, isnan
from time import sleep
from random import uniform, shuffle
from copy import deepcopy
import threading
import statistics

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
from skimage.transform import resize

# Local modules
from utils import (
    shape_map, color_map, task_map, Goal, empty_goal, relative_to,
    opposite_relative_to, duration, wait_for_button_press,
    get_goal_from_digits, exceptions_dict, print
)
from arena_navigator import run_tk


# ============================
# PyBullet PHYSICS SETUP
# ============================

def get_physics(GUI, args, w=10, h=10):
    """
    Set up PyBullet physics engine and environment.
    """
    if GUI:
        physicsClient = p.connect(p.GUI)
        start_cam = (1, 90, -89, (w / 2, h / 2, w))
        p.resetDebugVisualizerCamera(1, 90, -89, (w / 2, h / 2, w), physicsClientId=physicsClient)

        tk_thread = threading.Thread(target=run_tk, args=(physicsClient, start_cam))
        tk_thread.daemon = True
        tk_thread.start()
    else:
        physicsClient = p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=physicsClient)

    p.setAdditionalSearchPath('pybullet_data')
    p.setGravity(0, 0, args.gravity, physicsClientId=physicsClient)
    p.setTimeStep(args.time_step, physicsClientId=physicsClient)
    p.setPhysicsEngineParameter(
        numSolverIterations=args.numSolverIterations,
        numSubSteps=args.numSubSteps,
        physicsClientId=physicsClient
    )
    return physicsClient


# ============================
# JOINT UTILITIES
# ============================

def get_joint_index(body_id, joint_name, physicsClient):
    """
    Get the index of a named joint from a body.
    """
    num_joints = p.getNumJoints(body_id, physicsClientId=physicsClient)
    for i in range(num_joints):
        info = p.getJointInfo(body_id, i, physicsClientId=physicsClient)
        if info[1].decode() == joint_name:
            return i
    return -1


def get_joint_indices(body_id, physicsClient):
    """
    Return a mapping from joint number to joint index for all valid joints.
    """
    num_joints = p.getNumJoints(body_id, physicsClientId=physicsClient)
    joint_indices = {}

    for i in range(num_joints):
        info = p.getJointInfo(body_id, i, physicsClientId=physicsClient)
        joint_name = info[1].decode()
        parts = joint_name.split('_')[:-1]

        if 'sensor' not in joint_name and 'joint' in parts:
            if parts[-2] == 'joint':
                joint_id = int(parts[-1])
                joint_indices[joint_id] = i

    return joint_indices



def find_key_by_value(my_dict, target_value):
    """
    Search for the key associated with a value in a dictionary.
    """
    for key, value in my_dict.items():
        if value == target_value:
            return key
    return None



# ============================
# FOV AND MAP PARAMETERS
# ============================

fov_x_deg = 90
fov_y_deg = 90
fov_x_rad = radians(fov_x_deg)
fov_y_rad = radians(fov_y_deg)

near = 0.91
far = 9

right = near * tan(fov_x_rad / 2)
left = -right
top = near * tan(fov_y_rad / 2)
bottom = -top

agent_upper_starting_pos = 2.02
object_upper_starting_pos = 1.12      # Objects in use
object_lower_starting_pos = -8.85     # Objects waiting for use




# ============================
# CLASS: Arena
# ============================

class Arena:
    """
    Class representing the physical simulation environment.
    Handles robot setup, object placement, and episode lifecycle.
    """

    def __init__(self, GUI, args):
        self.args = args
        self.physicsClient = get_physics(GUI=GUI, args=self.args)

        # Objects currently in play
        self.objects_in_play = {}

        # Task-specific durations of robot behavior
        self.durations = {
            'watch': {}, 'be_near': {}, 'top': {},
            'push': {}, 'left': {}, 'right': {}, 'except': {}
        }

        # Motor command history
        self.history_of_actions = {
            'left_wheel_speed': [],
            'right_wheel_speed': [],
            'joint_target_velocities': {}
        }

        # Create planes (floor, lower level)
        plane_positions = [[0, 0]]
        for position in plane_positions:
            for z_offset in [0, -10]:
                plane_id = p.loadURDF(
                    'pybullet_data/plane.urdf',
                    position + [z_offset],
                    globalScaling=2,
                    useFixedBase=True,
                    physicsClientId=self.physicsClient
                )

        # Load robot
        self.default_orn = p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.physicsClient)
        robot_urdf_path = f'pybullet_data/robots/{self.args.robot_name}.urdf'

        self.robot_index = p.loadURDF(
            robot_urdf_path,
            (0, 0, agent_upper_starting_pos),
            self.default_orn,
            useFixedBase=False,
            globalScaling=self.args.body_size,
            physicsClientId=self.physicsClient
        )

        self.wheel_accelerations = [0, 0]
        self.joint_indices = get_joint_indices(self.robot_index, physicsClient=self.physicsClient)
        self.joint_accelerations = {key: 0 for key in self.joint_indices.keys()}

        # Visual and dynamic setup
        p.changeVisualShape(self.robot_index, -1, rgbaColor=(0.5, 0.5, 0.5, 1), physicsClientId=self.physicsClient)
        p.changeDynamics(self.robot_index, -1, maxJointVelocity=10000)

        self.sensors = {}
        self.wheels = []
    
        for link_index in range(p.getNumJoints(self.robot_index, physicsClientId=self.physicsClient)):
            joint_info = p.getJointInfo(self.robot_index, link_index, physicsClientId=self.physicsClient)
            link_name = joint_info[12].decode('utf-8')

            p.changeDynamics(self.robot_index, link_index, maxJointVelocity=10000)

            if 'wheel' in link_name:
                self.wheels.append((link_index, link_name))
            if 'sensor' in link_name:
                self.sensors[link_name] = link_index
                p.changeVisualShape(self.robot_index, link_index, rgbaColor=(1, 0, 0, 0), physicsClientId=self.physicsClient)
            elif 'spoke' in link_name or 'outline' in link_name:
                p.changeVisualShape(self.robot_index, link_index, rgbaColor=(1, 1, 1, 1), physicsClientId=self.physicsClient)
            elif 'camera_2' in link_name:
                p.changeVisualShape(self.robot_index, link_index, rgbaColor=(1, 1, 1, 0.3), physicsClientId=self.physicsClient)
            elif 'camera_3' in link_name:
                p.changeVisualShape(self.robot_index, link_index, rgbaColor=(1, 1, 1, 0.1), physicsClientId=self.physicsClient)
            else:
                p.changeVisualShape(self.robot_index, link_index, rgbaColor=(0, 0, 0, 1), physicsClientId=self.physicsClient)

        # Load objects on lower level
        linear_damping = 1
        angular_damping = 100
        self.loaded = {key: [] for key in shape_map.keys()}
        self.object_indexs = []

        for i, shape in shape_map.items():
            for j in range(2):
                pos = (5 * i, 5 * j, object_lower_starting_pos)
                shape_urdf_file = f'pybullet_data/shapes/{shape.file_name}'

                object_index = p.loadURDF(
                    shape_urdf_file,
                    pos,
                    p.getQuaternionFromEuler([0, 0, pi / 2]),
                    useFixedBase=False,
                    globalScaling=self.args.object_size,
                    physicsClientId=self.physicsClient
                )

                p.changeDynamics(object_index, -1, maxJointVelocity=10000,
                                 linearDamping=linear_damping, angularDamping=angular_damping)

                for link_index in range(p.getNumJoints(object_index, physicsClientId=self.physicsClient)):
                    p.changeDynamics(object_index, link_index, maxJointVelocity=10000,
                                     linearDamping=linear_damping, angularDamping=angular_damping)

                self.loaded[i].append((object_index, (pos[0], pos[1], object_lower_starting_pos)))
                self.object_indexs.append(object_index)

    # ----------------------------
    # UTILITIES
    # ----------------------------

    def change_physicsClient(self):
        """
        Update physics engine parameters after reconnecting or resuming.
        """
        p.setGravity(0, 0, self.args.gravity, physicsClientId=self.physicsClient)
        p.setTimeStep(self.args.time_step, physicsClientId=self.physicsClient)
        p.setPhysicsEngineParameter(
            numSolverIterations=self.args.numSolverIterations,
            numSubSteps=self.args.numSubSteps,
            physicsClientId=self.physicsClient
        )

    def end(self):
        """
        Reset object positions after an episode ends.
        """
        for (_, _, idle_pos), object_index in self.objects_in_play.items():
            p.resetBasePositionAndOrientation(object_index, idle_pos, self.default_orn, physicsClientId=self.physicsClient)

    def stop(self):
        """
        Clean up and disconnect simulation.
        """
        p.disconnect(physicsClientId=self.physicsClient)
        
        
    
    def begin(self, objects, goal, parenting, set_positions = None):
        """
        Begin episode with given objects and goals.
        """
        self.set_pos()
        self.set_yaw()
        self.set_wheel_speeds()
        self.wheel_accelerations = [0, 0]
        joint_angles = {joint_num: (getattr(self.args, f'max_joint_{joint_num}_angle') + getattr(self.args, f'min_joint_{joint_num}_angle')) / 2 for joint_num in self.joint_indices.keys()}
        self.set_joint_angles(joint_angles)
        self.set_joint_target_velocities()
        self.joint_accelerations = {key: 0 for key in self.joint_indices.keys()}
        self.goal = goal
        
        # If there are exceptions, adjust goal. 
        self.real_goal = goal   
        self.supposed_to_be_exception = False
        
        exception_list_a, exception_list_b = exceptions_dict[self.args.exceptions]
            
        if(self.goal.digits in exception_list_a):
            self.supposed_to_be_exception = True 
            goal_index = exception_list_a.index(self.goal.digits)
            self.real_goal = get_goal_from_digits(exception_list_b[goal_index])
            objects[0] = (self.real_goal.color, self.real_goal.shape)
        
        self.parenting = parenting
        self.objects_in_play = {}
        self.durations = {"watch" : {}, "be_near" : {}, "top" : {}, "push" : {}, "left" : {}, "right" : {}, "except" : {}}
        already_in_play = {key : 0 for key in shape_map.keys()}
        # If custom positions not set, generate positions.
        if(set_positions == None):
            set_positions = self.generate_positions(len(objects))
        # Position objects in use.
        for i, (color, shape) in enumerate(objects):
            color_index = find_key_by_value(color_map, color)
            shape_index = find_key_by_value(shape_map, shape)
            object_index, idle_pos = self.loaded[shape_index][already_in_play[shape_index]]
            already_in_play[shape_index] += 1
            x, y = set_positions[i]
            p.resetBasePositionAndOrientation(object_index, (x, y, object_upper_starting_pos), (0, 0, 0, 1), physicsClientId = self.physicsClient)
            self.object_faces_up(object_index)
            # Color object.
            link_name = p.getBodyInfo(object_index)[0].decode('utf-8')
            if "white" not in link_name.lower():
                p.changeVisualShape(object_index, -1, rgbaColor = color.rgba, physicsClientId = self.physicsClient)
            else:
                p.changeVisualShape(object_index, -1, rgbaColor = (1, 1, 1, 1), physicsClientId = self.physicsClient)
            for i in range(p.getNumJoints(object_index)):
                joint_info = p.getJointInfo(object_index, i, physicsClientId=self.physicsClient)
                joint_name = joint_info[1].decode("utf-8") 
                if "white" not in joint_name.lower(): 
                    p.changeVisualShape(object_index, i, rgbaColor = color.rgba, physicsClientId=self.physicsClient)
                        
            # Track objects in play, and how long the agent has performed actions on them.
            self.objects_in_play[(color_index, shape_index, idle_pos)] = object_index
            for task in ["watch", "be_near", "top", "push", "left", "right", "except"]:
                self.durations[task][object_index] = 0
            
        # Position robot.
        _, _, _, _, yaw = self.get_pos_spe_rpy(self.robot_index)
        self.robot_start_yaw = yaw
        start_agent_pos, start_agent_orn = p.getBasePositionAndOrientation(self.robot_index)

        # Track objects positions, and whether or not the robot is touching them.
        self.objects_start = self.get_object_positions()
        self.objects_end = self.get_object_positions()
        self.objects_touch = self.touching_any_object()
        for object_index, touch_dict in self.objects_touch.items():
           for body_part in touch_dict.keys():
               touch_dict[body_part] = 0 
        self.objects_local_pos_start = {}
        self.objects_local_pos_end = {}
        self.objects_angle_start = {}
        self.objects_angle_end = {}
        for object_index in self.objects_in_play.values():
            self.objects_local_pos_start[object_index] = self.get_local_position_of_object(object_index, start_agent_pos, start_agent_orn)
            self.objects_local_pos_end[object_index] = self.get_local_position_of_object(object_index, start_agent_pos, start_agent_orn)
            self.objects_angle_start[object_index] = self.get_object_angle(object_index)
            self.objects_angle_end[object_index] = self.get_object_angle(object_index)
            
            
            
    def step(self, left_wheel_speed, right_wheel_speed, joint_target_velocities, verbose = False, sleep_time = None, waiting = False):
        
        """
        One step in an episode.
        """
        
        # Check beginning stats regarding objects.
        _, _, _, _, yaw = self.get_pos_spe_rpy(self.robot_index)
        self.robot_start_yaw = yaw
        self.objects_start = self.get_object_positions()

        self.objects_local_pos_start = {}
        start_agent_pos, start_agent_orn = p.getBasePositionAndOrientation(self.robot_index)
        for object_index in self.objects_in_play.values():
            self.objects_local_pos_start[object_index] = self.get_local_position_of_object(object_index, start_agent_pos, start_agent_orn)
            self.objects_angle_start[object_index] = self.get_object_angle(object_index)
        
        touching = self.touching_any_object()
        for object_index, touch_dict in touching.items():
           for body_part in touch_dict.keys():
               touch_dict[body_part] = 0 
               
        # If user is investigating robot behavior, user may pause the robot.
        if(waiting): 
            WAITING = wait_for_button_press()
            
        # For linear interpolation, find change-rates for wheel and arm-joint velocities. 
        left_wheel_speed_end = relative_to(left_wheel_speed, -self.args.max_wheel_speed, self.args.max_wheel_speed)
        right_wheel_speed_end = relative_to(right_wheel_speed, -self.args.max_wheel_speed, self.args.max_wheel_speed)
        left_wheel_speed_start, right_wheel_speed_start = self.get_wheel_speeds()
        
        change_in_left_wheel = left_wheel_speed_end - left_wheel_speed_start
        change_in_left_wheel_per_step = change_in_left_wheel / self.args.steps_per_step
        change_in_right_wheel = right_wheel_speed_end - right_wheel_speed_start
        change_in_right_wheel_per_step = change_in_right_wheel / self.args.steps_per_step   
        
        joint_target_velocities_start = self.get_joint_speeds()
        joint_target_velocities_end = {}
        for key, value in joint_target_velocities.items():
            velocity_end = relative_to(
                joint_target_velocities[key], 
                -getattr(self.args, f'max_joint_speed'), 
                getattr(self.args, f'max_joint_speed'))
            joint_target_velocities[key] = velocity_end
            joint_target_velocities_end[key] = velocity_end
        change_in_joint_velocities_per_step = {}
        for key, value in joint_target_velocities.items():
            change_in_velocity = joint_target_velocities_end[key] - joint_target_velocities_start[key]
            change_in_joint_velocities_per_step[key] = change_in_velocity / self.args.steps_per_step  
        joint_target_velocities_step = {}
        for step in range(self.args.steps_per_step):
            joint_target_velocities_step[step] = {}
            for key, value in joint_target_velocities.items():
                joint_target_velocities_step[step][key] = joint_target_velocities_start[key] + change_in_joint_velocities_per_step[key] * (step + 1)   
            
        # This step consisted of multiple substeps in the physics simulator.
        for step in range(self.args.steps_per_step):   
            # Adjust wheel velocities.
            left_wheel_step = left_wheel_speed_start + change_in_left_wheel_per_step * (step + 1)
            right_wheel_step = right_wheel_speed_start + change_in_right_wheel_per_step * (step + 1)
            self.set_wheel_speeds(left_wheel_step, right_wheel_step) 
            
            # Check if joint-angles are valid. Then adjust angle velocities.
            joint_target_velocities_step_fixed = self.fix_joints(deepcopy(joint_target_velocities_step[step])) 
            for key, value in joint_target_velocities_step_fixed.items():
                if(joint_target_velocities_step[step][key] != value):
                    for substep in joint_target_velocities:
                        joint_target_velocities_step[substep][key] = value
            self.set_joint_target_velocities(joint_target_velocities_step_fixed)         
            
            # For continuous appearence in user-testing, substeps can be delayed.
            if(sleep_time != None):
                sleep(sleep_time / self.args.steps_per_step)
                
            # Run substep in physics simulator.
            p.stepSimulation(physicsClientId = self.physicsClient)
                                                                                            
            # Track how often the robot's sensors touch objects.        
            touching_now = self.touching_any_object()
            for object_index, touch_dict in touching_now.items():
                for body_part, value in touch_dict.items():
                    if(value):
                        touching[object_index][body_part] += 1/self.args.steps_per_step
                        if(touching[object_index][body_part]) > 1:
                            touching[object_index][body_part] = 1
                             
        # Check ending stats regarding objects.                               
        self.objects_end = self.get_object_positions()
        self.objects_touch = touching
        self.objects_local_pos_end = {}
        stop_agent_pos, stop_agent_orn = p.getBasePositionAndOrientation(self.robot_index)
        for object_index in self.objects_in_play.values():
            self.objects_local_pos_end[object_index] = self.get_local_position_of_object(object_index, stop_agent_pos, stop_agent_orn)
            self.objects_angle_end[object_index] = self.get_object_angle(object_index)
        
        
        
    # -----------------------------
    # Object-related Functions
    # -----------------------------

    def generate_positions(self, n):
        """
        Generate randomized object positions arranged in a circular formation.
        """
        distance = uniform(self.args.min_object_distance, self.args.max_object_distance)
        base_angle = uniform(0, 2 * pi)
        x1 = distance * cos(base_angle)
        y1 = distance * sin(base_angle)
        angle_step = (2 * pi) / n
        positions = [(x1, y1)]
        for i in range(1, n):
            distance = uniform(self.args.min_object_distance, self.args.max_object_distance)
            current_angle = base_angle + (i * angle_step)
            x = distance * cos(current_angle)
            y = distance * sin(current_angle)
            positions.append((x, y))
        shuffle(positions)
        return positions

    def object_faces_up(self, object_index):
        """
        Orient object to face toward the agent at episode start.
        """
        obj_pos, obj_orn = p.getBasePositionAndOrientation(object_index, physicsClientId=self.physicsClient)
        agent_pos, _ = p.getBasePositionAndOrientation(self.robot_index, physicsClientId=self.physicsClient)
        delta_x = agent_pos[0] - obj_pos[0]
        delta_y = agent_pos[1] - obj_pos[1]
        angle_to_agent = math.atan2(delta_y, delta_x)
        roll, pitch, _ = p.getEulerFromQuaternion(obj_orn, physicsClientId=self.physicsClient)
        new_orn = p.getQuaternionFromEuler([0, 0, angle_to_agent if not isnan(angle_to_agent) else 0])
        p.resetBasePositionAndOrientation(object_index, (obj_pos[0], obj_pos[1], object_upper_starting_pos), new_orn, physicsClientId=self.physicsClient)

    def get_object_positions(self):
        """
        Return the world positions of all objects in play.
        """
        object_positions = {}
        for object_index in self.objects_in_play.values():
            pos, spe, roll, pitch, yaw = self.get_pos_spe_rpy(object_index)
            object_positions[object_index] = pos
        return object_positions

    def get_local_position_of_object(self, object_id, agent_pos, agent_orn):
        """
        Get an object's position relative to the robot's frame.
        """
        inv_agent_pos, inv_agent_orn = p.invertTransform(agent_pos, agent_orn)
        obj_pos, obj_orn = p.getBasePositionAndOrientation(object_id)
        local_obj_pos, _ = p.multiplyTransforms(inv_agent_pos, inv_agent_orn, obj_pos, obj_orn)
        return local_obj_pos

    def get_object_angle(self, object_index):
        """
        Compute the yaw-angle of an object relative to the robot's heading.
        """
        object_pos, _ = p.getBasePositionAndOrientation(object_index, physicsClientId=self.physicsClient)
        agent_pos, agent_ori = p.getBasePositionAndOrientation(self.robot_index, physicsClientId=self.physicsClient)

        distance_vector = np.subtract(object_pos[:2], agent_pos[:2])
        distance = np.linalg.norm(distance_vector)
        normalized_distance_vector = distance_vector / distance

        rotation_matrix = p.getMatrixFromQuaternion(agent_ori, physicsClientId=self.physicsClient)
        forward_vector = np.array([rotation_matrix[0], rotation_matrix[3]])
        forward_vector /= np.linalg.norm(forward_vector)

        dot_product = np.dot(forward_vector, normalized_distance_vector)
        angle_radians = np.arccos(np.clip(dot_product, -1.0, 1.0))

        cross_product = np.cross(np.append(forward_vector, 0), np.append(normalized_distance_vector, 0))
        if cross_product[2] < 0:
            angle_radians = -angle_radians
        return angle_radians

    def touching_object(self, object_index):
        """
        Check all sensor links on robot for contact with a specific object.
        """
        touching = {}
        for link_name, sensor_index in self.sensors.items():
            touching_this = bool(p.getContactPoints(bodyA=self.robot_index, bodyB=object_index, linkIndexA=sensor_index, physicsClientId=self.physicsClient))
            touching[link_name] = 1 if touching_this else 0
        return touching

    def touching_any_object(self):
        """
        Check all robot sensors against all objects.
        """
        touching = {}
        for object_index in self.objects_in_play.values():
            touching[object_index] = self.touching_object(object_index)
        return touching

    # -----------------------------
    # Robot-related Functions
    # -----------------------------

    def get_pos_spe_rpy(self, index):
        """
        Return position, linear speed (forward), and orientation of a body.
        """
        pos, ors = p.getBasePositionAndOrientation(index, physicsClientId=self.physicsClient)
        roll, pitch, yaw = p.getEulerFromQuaternion(ors, physicsClientId=self.physicsClient)
        forward_dir = np.array([np.cos(yaw), np.sin(yaw)])
        (vx, vy, _), _ = p.getBaseVelocity(index, physicsClientId=self.physicsClient)
        velocity_vec = np.array([vx, vy])
        spe = float(np.dot(velocity_vec, forward_dir))
        return pos, spe, roll, pitch, yaw

    def get_robot_velocities(self):
        """
        Get linear and angular velocity of robot in its local frame.
        """
        linear_velocity, angular_velocity = p.getBaseVelocity(self.robot_index, physicsClientId=self.physicsClient)
        vx, vy, _ = linear_velocity
        _, _, wz = angular_velocity
        _, _, _, _, yaw = self.get_pos_spe_rpy(self.robot_index)
        local_vx = cos(yaw) * vx + sin(yaw) * vy
        return local_vx, wz

    def get_wheel_speeds(self):
        """
        Compute wheel speeds from robot velocities using differential drive.
        """
        linear_velocity, angular_velocity = self.get_robot_velocities()
        left_wheel = linear_velocity - (angular_velocity / self.args.angular_scaler) / 2
        right_wheel = linear_velocity + (angular_velocity / self.args.angular_scaler) / 2
        return left_wheel, right_wheel

    def set_pos(self, pos=(0, 0)):
        """
        Set robot world position, maintaining current yaw.
        """
        pos = (pos[0], pos[1], agent_upper_starting_pos)
        _, _, _, _, yaw = self.get_pos_spe_rpy(self.robot_index)
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.robot_index, pos, orn, physicsClientId=self.physicsClient)

    def set_yaw(self, yaw=0):
        """
        Set robot yaw orientation while keeping position fixed.
        """
        orn = p.getQuaternionFromEuler([0, 0, yaw], physicsClientId=self.physicsClient)
        pos, _, _, _, _ = self.get_pos_spe_rpy(self.robot_index)
        p.resetBasePositionAndOrientation(self.robot_index, pos, orn, physicsClientId=self.physicsClient)

    def set_wheel_speeds(self, left_wheel_speed=0, right_wheel_speed=0):
        """
        Apply wheel speeds and update base linear/angular velocity.
        """
        linear_velocity = (left_wheel_speed + right_wheel_speed) / 2
        _, _, _, _, yaw = self.get_pos_spe_rpy(self.robot_index)
        x = linear_velocity * cos(yaw)
        y = linear_velocity * sin(yaw)
        angular_velocity = (right_wheel_speed - left_wheel_speed) * self.args.angular_scaler

        p.resetBaseVelocity(
            self.robot_index,
            linearVelocity=[x, y, 0],
            angularVelocity=[0, 0, angular_velocity],
            physicsClientId=self.physicsClient
        )

        for index, name in self.wheels:
            speed = left_wheel_speed if name == 'left_wheel' else right_wheel_speed
            p.setJointMotorControl2(
                self.robot_index, index,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=-4 * speed,
                physicsClientId=self.physicsClient
            )

    def get_joint_angles(self):
        """
        Get current joint angles from robot.
        """
        return {
            key: p.getJointState(self.robot_index, index, physicsClientId=self.physicsClient)[0]
            for key, index in self.joint_indices.items()
        }

    def get_joint_speeds(self):
        """
        Get current joint angular velocities from robot.
        """
        return {
            key: p.getJointState(self.robot_index, index, physicsClientId=self.physicsClient)[1]
            for key, index in self.joint_indices.items()
        }

    def set_joint_angles(self, joint_angles=None):
        """
        Set joint angles directly (useful for resets).
        """
        if joint_angles is None:
            joint_angles = {key: None for key in self.joint_indices}
        for key, index in self.joint_indices.items():
            if joint_angles[key] is not None:
                p.resetJointState(self.robot_index, index, joint_angles[key], physicsClientId=self.physicsClient)

    def set_joint_target_velocities(self, joint_target_velocities=None):
        """
        Set target angular velocities for robot joints.
        """
        if joint_target_velocities is None:
            joint_target_velocities = {key: 0 for key in self.joint_indices}
        for key, index in self.joint_indices.items():
            if joint_target_velocities[key] is not None:
                p.setJointMotorControl2(
                    self.robot_index, index,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=joint_target_velocities[key],
                    force=self.args.force,
                    maxVelocity=self.args.max_joint_speed
                )

    def fix_joints(self, joint_target_velocities):
        """
        Clamp joint velocities to within physical limits.
        """
        joint_angles = self.get_joint_angles()
        joint_speeds = self.get_joint_speeds()

        for key in self.joint_indices:
            max_angle = getattr(self.args, f'max_joint_{key}_angle')
            min_angle = getattr(self.args, f'min_joint_{key}_angle')
            max_speed = self.args.max_joint_speed

            if joint_speeds[key] > max_speed:
                joint_target_velocities[key] = max_speed
            if joint_speeds[key] < -max_speed:
                joint_target_velocities[key] = -max_speed

            if joint_angles[key] < min_angle:
                diff = min_angle - joint_angles[key]
                new_speed = diff * max_speed
                if joint_target_velocities[key] < new_speed:
                    joint_target_velocities[key] = new_speed

            if joint_angles[key] > max_angle:
                diff = max_angle - joint_angles[key]
                new_speed = diff * max_speed
                if joint_target_velocities[key] > new_speed:
                    joint_target_velocities[key] = new_speed

        return joint_target_velocities
            
        
        
    # -----------------------------
    # Calculate Extrinsic Rewards
    # -----------------------------
    
    def rewards(self, verbose=False):
        """ 
        Find if the robot has performed any goals with any objects.
        """
        objects_goals = {}
        win = False
        reward = 0
        v_rx = cos(self.robot_start_yaw)
        v_ry = sin(self.robot_start_yaw)

        if verbose:
            for object_key, object_dict in self.objects_touch.items():
                for link_name, value in object_dict.items():
                    if value:
                        print(f'Touching {object_key} with {link_name}.')

        # Iterate over objects
        for i, ((color_index, shape_index, _), object_index) in enumerate(self.objects_in_play.items()):
            watched = been_near = topped = pushed = lefted = righted = excepted = False

            # Touch state
            objects_touch = self.objects_touch[object_index]
            objects_touch_body = {key: value for key, value in objects_touch.items() if 'body' in key}
            touching = any(objects_touch.values())
            touching_body = any(objects_touch_body.values())

            # Distance & movement
            object_pos, _ = p.getBasePositionAndOrientation(object_index, physicsClientId=self.physicsClient)
            agent_pos, agent_ori = p.getBasePositionAndOrientation(self.robot_index, physicsClientId=self.physicsClient)
            distance_vector = np.subtract(object_pos[:2], agent_pos[:2])
            distance = np.linalg.norm(distance_vector)

            (x_before, y_before, z_before) = self.objects_start[object_index]
            (x_after, y_after, z_after) = self.objects_end[object_index]
            delta_x = x_after - x_before
            delta_y = y_after - y_before
            global_movement_forward = delta_x * v_rx + delta_y * v_ry
            global_movement_left = delta_x * (-v_ry) + delta_y * v_rx

            object_local_pos_start = self.objects_local_pos_start[object_index]
            object_local_pos_end = self.objects_local_pos_end[object_index]
            local_movement_forward = object_local_pos_end[0] - object_local_pos_start[0]
            local_movement_left = object_local_pos_end[1] - object_local_pos_start[1]

            object_angle_start = self.objects_angle_start[object_index]
            object_angle_end = self.objects_angle_end[object_index]
            angle_change = object_angle_end - object_angle_start
            angle_change_degrees = degrees(angle_change)
            object_angle_start_degrees = degrees(object_angle_start)
            object_angle_end_degrees = degrees(object_angle_end)

            if verbose:
                print(f'Object: {color_map[color_index].name} {shape_map[shape_index].name}')
                print(f'Angle from agent to object:\t{round(object_angle_start_degrees, 2)} before,\t{round(object_angle_end_degrees, 2)} after,\t{round(angle_change_degrees, 2)} change')
                print(f'Movement forward:\t{round(global_movement_forward, 2)} global,\t{round(local_movement_forward, 2)} local')
                print(f'Movement left:\t\t{round(global_movement_left, 2)} global,\t{round(local_movement_left, 2)} local')
                print(f'Angle of movement: {(round(v_rx, 2), round(v_ry, 2))}')

            # Conditions for different behaviors
            good_watching_angle = abs(object_angle_end) <= self.args.pointing_at_object_for_watch
            watching = good_watching_angle and not touching and distance <= self.args.watch_distance

            good_being_near_angle = abs(object_angle_end) <= self.args.pointing_at_object_for_being_near
            being_near = good_being_near_angle and not touching and distance <= self.args.be_near_distance

            good_touch_top_angle = abs(object_angle_end) <= self.args.pointing_at_object_for_touch_top
            link_index = None
            for sensor_name, sensor_index in self.sensors.items():
                if sensor_name.startswith('hand_sensor_') and sensor_name.endswith('_stop'):
                    link_index = sensor_index
            hand_height = p.getLinkState(self.robot_index, linkIndex=link_index)[0][2]
            good_hand_height = hand_height >= self.args.touch_top_min_height
            topping = touching and not touching_body and good_hand_height and good_touch_top_angle

            good_push_distance = global_movement_forward >= self.args.global_push_amount
            pushing = touching and good_push_distance and good_watching_angle

            left_wheel_speed, right_wheel_speed = self.get_wheel_speeds()
            good_wheel_speed = max(abs(left_wheel_speed), abs(right_wheel_speed)) <= self.args.max_wheel_speed_for_left_right
            arm_speed = self.get_joint_speeds()[1]
            good_arm_speed = abs(arm_speed) >= self.args.min_arm_speed_for_left_right

            good_push_left_distance = global_movement_left >= self.args.global_left_right_amount
            good_push_right_distance = global_movement_left <= -self.args.global_left_right_amount
            good_left_right_angle = abs(object_angle_end) <= self.args.pointing_at_object_for_left_right

            lefting = touching and good_push_left_distance and good_left_right_angle and good_wheel_speed and good_arm_speed
            righting = touching and good_push_right_distance and good_left_right_angle and good_wheel_speed and good_arm_speed

            if verbose:
                print(f'\nTouching: {touching}. Touching body: {touching_body}.')
                print(f'Watching angle: {good_watching_angle}. Lefting angle: {good_left_right_angle}.')
                print(f'Good wheel speed: {good_wheel_speed}. Good arm speed: {good_arm_speed}.')
                print(f'Watching\t({watching}):\t\t{self.durations["watch"][object_index]} steps')
                print(f'Being Near\t({being_near}):\t\t{self.durations["be_near"][object_index]} steps')
                print(f'Topping\t\t({topping}):\t\t{self.durations["top"][object_index]} steps')
                print(f'Pushing\t\t({pushing}):\t\t{self.durations["push"][object_index]} steps')
                print(f'Lefting\t\t({lefting}):\t\t{self.durations["left"][object_index]} steps')
                print(f'Righting\t({righting}):\t\t{self.durations["right"][object_index]} steps\n')

            # Choose one motion type
            active_changes = []
            if pushing:
                active_changes.append(('pushing', global_movement_forward))
            if lefting:
                active_changes.append(('lefting', global_movement_left))
            if righting:
                active_changes.append(('righting', abs(global_movement_left)))
            if len(active_changes) > 1:
                active_changes.sort(key=lambda x: x[1], reverse=True)
                highest_change = active_changes[0][0]
                pushing = lefting = righting = False
                if highest_change == 'pushing':
                    pushing = True
                elif highest_change == 'lefting':
                    lefting = True
                elif highest_change == 'righting':
                    righting = True

            if being_near:
                watching = False
            if topping:
                pushing = lefting = righting = False

            if verbose:
                print('After consideration:')
                print(f'Watching:\t{watching}')
                print(f'Being Near:\t{being_near}')
                print(f'Topping:\t{topping}')
                print(f'Pushing:\t{pushing}')
                print(f'Lefting:\t{lefting}')
                print(f'Righting:\t{righting}\n')

            def update_duration(action_name, action_now, object_index, duration_threshold):
                if action_now:
                    self.durations[action_name][object_index] += 1
                else:
                    self.durations[action_name][object_index] = 0
                return self.durations[action_name][object_index] >= duration_threshold

            watched   = update_duration('watch', watching, object_index, self.args.watch_duration)
            been_near = update_duration('be_near', being_near, object_index, self.args.be_near_duration)
            topped    = update_duration('top', topping, object_index, self.args.top_duration)
            pushed    = update_duration('push', pushing, object_index, self.args.push_duration)
            lefted    = update_duration('left', lefting, object_index, self.args.left_right_duration)
            righted   = update_duration('right', righting, object_index, self.args.left_right_duration)

            key = (color_map[color_index], shape_map[shape_index])
            new_value = [watched, been_near, topped, pushed, lefted, righted,
                         watching, being_near, topping, pushing, lefting, righting]

            if key in objects_goals:
                objects_goals[key] = [old or new for old, new in zip(objects_goals[key], new_value)]
            else:
                objects_goals[key] = new_value

            if verbose:
                ings = sum([watching, being_near, topping, pushing, lefting, righting, excepted])
                print('Finished:')
                if ings > 1:
                    print('\t\tWARNING! MULTIPLE TASKS AT ONCE!')
                print(f'Watching:\t{watching}\tWatched:\t{watched}\t{self.durations["watch"][object_index]} steps')
                print(f'Being Near:\t{being_near}\tBeen Near:\t{been_near}\t{self.durations["be_near"][object_index]} steps')
                print(f'Topping:\t{topping}\tTopped:\t\t{topped}\t{self.durations["top"][object_index]} steps')
                print(f'Pushing:\t{pushing}\tPushed:\t\t{pushed}\t{self.durations["push"][object_index]} steps')
                print(f'Lefting:\t{lefting}\tLefted:\t\t{lefted}\t{self.durations["left"][object_index]} steps')
                print(f'Righting:\t{righting}\tRighted:\t{righted}\t{self.durations["right"][object_index]} steps\n')

        feedback_voice = empty_goal
        wrong_object = False
        task_performed = None

        for (color, shape), (watched, been_near, topped, pushed, lefted, righted, watching, being_near, topping, pushing, lefting, righting) in objects_goals.items():
            if sum([watched, been_near, topped, pushed, lefted, righted]) == 1:
                task_performed = 'watched' if watched else 'other'
                if color == self.real_goal.color and shape == self.real_goal.shape:
                    task_name = self.real_goal.task.name
                    if (
                        (task_name == 'WATCH'         and watched  and not (being_near or topping or pushing or lefting or righting)) or
                        (task_name == 'BE NEAR'       and been_near and not (watching or topping or pushing or lefting or righting)) or
                        (task_name == 'TOUCH THE TOP' and topped and not (watching or being_near or pushing or lefting or righting)) or
                        (task_name == 'PUSH FORWARD'  and pushed and not (watching or being_near or topping or lefting or righting)) or
                        (task_name == 'PUSH LEFT'     and lefted and not (watching or being_near or topping or pushing or righting)) or
                        (task_name == 'PUSH RIGHT'    and righted and not (watching or being_near or topping or pushing or lefting))
                    ):
                        win = True
                        reward = self.args.reward
                else:
                    wrong_object = True

            task_in_progress = None
            if sum([watching, being_near, topping, pushing, lefting, righting]) >= 1:
                if watching: task_in_progress = task_map[1]
                elif being_near: task_in_progress = task_map[2]
                elif topping: task_in_progress = task_map[3]
                elif pushing: task_in_progress = task_map[4]
                elif lefting: task_in_progress = task_map[5]
                elif righting: task_in_progress = task_map[6]
                feedback_voice = Goal(task_in_progress, color, shape, parenting=False)

                if not self.supposed_to_be_exception:
                    pass
                elif self.supposed_to_be_exception and feedback_voice.digits == self.goal.digits:
                    feedback_voice = empty_goal
                elif self.supposed_to_be_exception and feedback_voice.digits == self.real_goal.digits:
                    feedback_voice = self.goal

        if wrong_object:
            win = False
            reward = 0 if task_performed == 'watched' else self.args.wrong_object_punishment

        feedback_voice.make_texts()

        if verbose:
            print(f'feedback voice: \'{feedback_voice.human_text}\'')
            print('Total reward:', reward)
            print('Win:', win)

        return reward, win, feedback_voice

    
    # -----------------------------
    # Camera-related Functions
    # -----------------------------
    
    def birds_eye_view(self):
        """
        This takes a photo of the robot from a bird's eye view.
        """
        pos, _, _, _, yaw = self.get_pos_spe_rpy(self.robot_index)
        x, y = 4 * cos(-3 * pi / 4), 4 * sin(-3 * pi / 4)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[pos[0] + x, pos[1] + y, 10],
            cameraTargetPosition=[pos[0], pos[1], 2],
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.physicsClient
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=90,
            aspect=1,
            nearVal=0.01,
            farVal=20,
            physicsClientId=self.physicsClient
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=256,
            height=256,
            projectionMatrix=proj_matrix,
            viewMatrix=view_matrix,
            shadow=0,
            physicsClientId=self.physicsClient
        )
        if not isinstance(rgba, np.ndarray):
            rgba = np.array(rgba).reshape(32, 32, 4)
        rgb = rgba[:, :, :-1] / 255
        return rgb

    def photo_from_above(self):
        """
        This takes a photo of the robot from behind its right shoulder.
        """
        pos, spe, roll, pitch, yaw = self.get_pos_spe_rpy(self.robot_index)
        quat = p.getQuaternionFromEuler([roll, pitch, yaw])
        rot_matrix_flat = p.getMatrixFromQuaternion(quat)
        rot_matrix = np.array(rot_matrix_flat).reshape(3, 3)
        forward_vector = rot_matrix[:, 0]
        left_vector = rot_matrix[:, 1]
        up_vector = rot_matrix[:, 2]

        cam_eye_pos = [pos[0], pos[1], pos[2] + 5]
        cam_eye = np.array(cam_eye_pos) + forward_vector * -3.0 + left_vector * -2.0
        cam_target = np.array(pos) + forward_vector * 4.0

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_eye.tolist(),
            cameraTargetPosition=cam_target.tolist(),
            cameraUpVector=up_vector.tolist(),
            physicsClientId=self.physicsClient
        )
        proj_matrix = p.computeProjectionMatrix(
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            nearVal=near,
            farVal=25
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=256,
            height=256,
            projectionMatrix=proj_matrix,
            viewMatrix=view_matrix,
            shadow=0,
            physicsClientId=self.physicsClient
        )
        if not isinstance(rgba, np.ndarray):
            rgba = np.array(rgba).reshape(256, 256, 4)
        rgb = rgba[:, :, :-1] / 255
        return rgb

    def photo_for_agent(self):
        """
        This takes a photo from the robot's camera.
        """
        pos, spe, roll, pitch, yaw = self.get_pos_spe_rpy(self.robot_index)
        quat = p.getQuaternionFromEuler([roll, pitch, yaw])
        rot_matrix_flat = p.getMatrixFromQuaternion(quat)
        rot_matrix = np.array(rot_matrix_flat).reshape(3, 3)
        forward_vector = rot_matrix[:, 0]
        up_vector = rot_matrix[:, 2]

        cam_eye = np.array(pos) + forward_vector * 0.1
        cam_target = np.array(pos) + forward_vector * 2.0

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_eye.tolist(),
            cameraTargetPosition=cam_target.tolist(),
            cameraUpVector=up_vector.tolist(),
            physicsClientId=self.physicsClient
        )
        proj_matrix = p.computeProjectionMatrix(
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            nearVal=near,
            farVal=far
        )
        _, _, rgba, depth, _ = p.getCameraImage(
            width=self.args.image_size * 2,
            height=self.args.image_size * 2,
            projectionMatrix=proj_matrix,
            viewMatrix=view_matrix,
            shadow=0,
            physicsClientId=self.physicsClient
        )

        if not isinstance(rgba, np.ndarray):
            rgba = np.array(rgba).reshape(32, 32, 4)
            depth = np.array(depth).reshape(32, 32)

        rgb = rgba[:, :, :-1] / 255
        d = np.nan_to_num(np.expand_dims(depth, axis=-1), nan=1)

        if d.max() != d.min():
            d = (d - d.min()) / (d.max() - d.min())

        vision = np.concatenate([rgb, d], axis=-1)
        vision = resize(vision, (self.args.image_size, self.args.image_size, 4))
        return vision



# ============================
# SCRIPT ENTRY POINT
# ============================

    if __name__ == '__main__':
        from utils import args
        arena = Arena(GUI=True, args=args)
        while True:
            sleep(0.01)
            p.stepSimulation(physicsClientId=arena.physicsClient)
