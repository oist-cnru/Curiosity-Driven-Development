#%%
import math

try:
    from .part import Part  
except ImportError:
    from part import Part  
    
joint_1_height = .2

arm_mass = 2
arm_length = 1.75
arm_width = .2

hand_length = 1
hand_width = arm_width * 3
hand_height = 1.25

spoke_length = .3 
spoke_width = .01

outline_size = .005


parts = [
    
    Part(
        name = 'body',
        mass = 1000,
        size = (1, 1, 1),
        joint_origin = (0, 0, 1.05), 
        joint_axis = (0, 0, 1),
        joint_type = 'fixed',
        sensors = 1,
        inertia = [15, 0, 0, 15, 0, 15]),
    
    Part(
        name = 'camera_1', 
        mass = 0, 
        shape = 'cylinder',
        size = (.2, .05, .05), 
        joint_parent = 'body', 
        joint_origin = (.5, 0, 0), 
        joint_axis = (0, 0, 1),
        joint_rpy=(0, 1.5708, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),

    Part(
        name = 'camera_2', 
        mass = 0, 
        shape = 'cylinder',
        size = (.175, .05, .05), 
        joint_parent = 'body', 
        joint_origin = (.501, 0, 0), 
        joint_axis = (0, 0, 1),
        joint_rpy=(0, 1.5708, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'camera_3', 
        mass = 0, 
        shape = 'cylinder',
        size = (.1, .05, .05), 
        joint_parent = 'body', 
        joint_origin = (.502, 0, 0), 
        joint_axis = (0, 0, 1),
        joint_rpy=(0, 1.5708, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'camera_4', 
        mass = 0, 
        shape = 'cylinder',
        size = (.025, .05, .05), 
        joint_parent = 'body', 
        joint_origin = (.503, 0, 0), 
        joint_axis = (0, 0, 1),
        joint_rpy=(0, 1.5708, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'left_wheel', 
        mass = .1, 
        shape = 'cylinder',
        size = (.2, .1, .1), 
        joint_parent = 'body', 
        joint_origin = (0, .5, -.3), 
        joint_axis = (0, 0, 1),
        joint_rpy=(1.5708, 0, 0),
        joint_type = 'continuous',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'left_spoke_1',
        mass = 0,
        size = (spoke_length, spoke_width, .1),
        joint_parent = 'left_wheel', 
        joint_origin = (0, 0, -.01), 
        joint_axis = (0, 0, 1),
        joint_rpy=(0, 0, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'left_spoke_2',
        mass = 0,
        size = (spoke_width, spoke_length, .1),
        joint_parent = 'left_wheel', 
        joint_origin = (0, 0, -.01), 
        joint_axis = (0, 0, 1),
        joint_rpy=(0, 0, .524),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'left_spoke_3',
        mass = 0,
        size = (spoke_width, spoke_length, .1),
        joint_parent = 'left_wheel', 
        joint_origin = (0, 0, -.01), 
        joint_axis = (0, 0, 1),
        joint_rpy=(0, 0, -.524),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'right_wheel', 
        mass = .1, 
        shape = 'cylinder',
        size = (.2, .1, .1), 
        joint_parent = 'body', 
        joint_origin = (0, -.5, -.3), 
        joint_axis = (0, 0, 1),
        joint_rpy=(1.5708, 0, 0),
        joint_type = 'continuous',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'right_spoke_1',
        mass = 0,
        size = (spoke_length, spoke_width, .1),
        joint_parent = 'right_wheel', 
        joint_origin = (0, 0, .01), 
        joint_axis = (0, 0, 1),
        joint_rpy=(0, 0, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'right_spoke_2',
        mass = 0,
        size = (spoke_width, spoke_length, .1),
        joint_parent = 'right_wheel', 
        joint_origin = (0, 0, .01), 
        joint_axis = (0, 0, 1),
        joint_rpy=(0, 0, .524),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'right_spoke_3',
        mass = 0,
        size = (spoke_width, spoke_length, .1),
        joint_parent = 'right_wheel', 
        joint_origin = (0, 0, .01), 
        joint_axis = (0, 0, 1),
        joint_rpy=(0, 0, -.524),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'joint_1',
        mass = arm_mass,
        size = (arm_width, arm_width, joint_1_height),
        joint_parent = 'body', 
        joint_origin = (0, 0, .5 + joint_1_height / 2), 
        joint_axis = (0, 0, 1),
        joint_type = 'revolute',
        joint_limits = [-math.pi/4, math.pi/4, 999, 999],
        inertia = [.01, 0, 0, .01, 0, .01]),
    
    
    
    Part(
        name = 'joint_2',
        mass = arm_mass,
        size = (arm_width, arm_width, arm_width),
        joint_parent = 'joint_1', 
        joint_origin = (0, 0, joint_1_height / 2 + arm_width / 2), 
        joint_axis = (0, 1, 0),
        joint_type = 'revolute',
        joint_limits = [-math.pi/2, 0, 999, 999],
        inertia = [.01, 0, 0, .01, 0, .01]),
    
    Part(
        name = 'joint_2_outline_1',
        mass = 0,
        size = (arm_width, outline_size, outline_size),
        joint_parent = 'joint_2', 
        joint_origin = (0, arm_width/2 + outline_size/2, arm_width/2 + outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'joint_2_outline_2',
        mass = 0,
        size = (arm_width, outline_size, outline_size),
        joint_parent = 'joint_2', 
        joint_origin = (0, -arm_width/2 - outline_size/2, arm_width/2 + outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'joint_2_outline_3',
        mass = 0,
        size = (outline_size, outline_size, arm_width + outline_size),
        joint_parent = 'joint_2', 
        joint_origin = (-arm_width/2 - outline_size/2, arm_width/2 + outline_size/2, outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'joint_2_outline_4',
        mass = 0,
        size = (outline_size, outline_size, arm_width + outline_size),
        joint_parent = 'joint_2', 
        joint_origin = (-arm_width/2 - outline_size/2, -arm_width/2 - outline_size/2, outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'joint_2_outline_5',
        mass = 0,
        size = (arm_width + outline_size, outline_size, outline_size),
        joint_parent = 'joint_2', 
        joint_origin = (outline_size/2, arm_width/2 + outline_size/2, -arm_width/2 - outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'joint_2_outline_6',
        mass = 0,
        size = (arm_width + outline_size, outline_size, outline_size),
        joint_parent = 'joint_2', 
        joint_origin = (outline_size/2, -arm_width/2 - outline_size/2, -arm_width/2 - outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'joint_2_outline_7',
        mass = 0,
        size = (outline_size, arm_width + outline_size, outline_size),
        joint_parent = 'joint_2', 
        joint_origin = (-arm_width/2 - outline_size/2, outline_size/2, arm_width/2 + outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    
    
    Part(
        name = 'arm',
        mass = arm_mass,
        size = (arm_length, arm_width, arm_width),
        joint_parent = 'joint_2', 
        joint_origin = (arm_width / 2 + arm_length / 2, 0, 0), 
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        sensors = 1,
        sensor_sides = ['left', 'right', 'top', 'bottom'],
        inertia = [0.01, 0, 0, 1.25, 0, 1.25]),
    
    Part(
        name = 'arm_outline_1',
        mass = 0,
        size = (arm_length, outline_size, outline_size),
        joint_parent = 'arm', 
        joint_origin = (0, arm_width/2 + outline_size/2, arm_width/2 + outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'arm_outline_2',
        mass = 0,
        size = (arm_length, outline_size, outline_size),
        joint_parent = 'arm', 
        joint_origin = (0, -arm_width/2 - outline_size/2, arm_width/2 + outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'arm_outline_3',
        mass = 0,
        size = (arm_length, outline_size, outline_size),
        joint_parent = 'arm', 
        joint_origin = (0, arm_width/2 + outline_size/2, -arm_width/2 - outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'arm_outline_4',
        mass = 0,
        size = (arm_length, outline_size, outline_size),
        joint_parent = 'arm', 
        joint_origin = (0, -arm_width/2 - outline_size/2, -arm_width/2 - outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'arm_outline_5',
        mass = 0,
        size = (outline_size, outline_size, arm_width + outline_size),
        joint_parent = 'arm', 
        joint_origin = (arm_length/2 + outline_size/2 -outline_size, arm_width/2 + outline_size/2, 0),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'arm_outline_6',
        mass = 0,
        size = (outline_size, outline_size, arm_width + outline_size),
        joint_parent = 'arm', 
        joint_origin = (arm_length/2 + outline_size/2 -outline_size, -arm_width/2 - outline_size/2, 0),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'arm_outline_7',
        mass = 0,
        size = (outline_size, arm_width + outline_size, outline_size),
        joint_parent = 'arm', 
        joint_origin = (arm_length/2 + outline_size/2 -outline_size, 0, -arm_width/2 - outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    

    Part(
        name = 'hand',
        mass = arm_mass,
        size = (hand_length, hand_width, hand_height),
        joint_parent = 'arm', 
        joint_origin = (arm_length / 2 + hand_length / 2, 0, - hand_height / 2 + arm_width / 2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        sensors = 1,
        inertia = [.3, 0, 0, .4, 0, .2]),
    
    Part(
        name = 'hand_outline_1',
        mass = 0,
        size = (hand_length + 2 * outline_size, outline_size, outline_size),
        joint_parent = 'hand', 
        joint_origin = (0, hand_width/2 + outline_size/2, hand_height/2 + outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'hand_outline_2',
        mass = 0,
        size = (hand_length + 2 * outline_size, outline_size, outline_size),
        joint_parent = 'hand', 
        joint_origin = (0, -hand_width/2 - outline_size/2, hand_height/2 + outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'hand_outline_3',
        mass = 0,
        size = (hand_length + 2 * outline_size, outline_size, outline_size),
        joint_parent = 'hand', 
        joint_origin = (0, hand_width/2 + outline_size/2, -hand_height/2 - outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'hand_outline_4',
        mass = 0,
        size = (hand_length + 2 * outline_size, outline_size, outline_size),
        joint_parent = 'hand', 
        joint_origin = (0, -hand_width/2 - outline_size/2, -hand_height/2 - outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'hand_outline_5',
        mass = 0,
        size = (outline_size, hand_width, outline_size),
        joint_parent = 'hand', 
        joint_origin = (hand_length/2 + outline_size/2, 0, hand_height/2 + outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'hand_outline_6a',
        mass = 0,
        size = (outline_size, (hand_width - arm_width)/2, outline_size),
        joint_parent = 'hand', 
        joint_origin = (-hand_length/2 - outline_size/2, arm_width, hand_height/2 + outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'hand_outline_6b',
        mass = 0,
        size = (outline_size, (hand_width - arm_width)/2, outline_size),
        joint_parent = 'hand', 
        joint_origin = (-hand_length/2 - outline_size/2, -arm_width, hand_height/2 + outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'hand_outline_7',
        mass = 0,
        size = (outline_size, hand_width, outline_size),
        joint_parent = 'hand', 
        joint_origin = (hand_length/2 + outline_size/2, 0, -hand_height/2 - outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),

    Part(
        name = 'hand_outline_8',
        mass = 0,
        size = (outline_size, hand_width, outline_size),
        joint_parent = 'hand', 
        joint_origin = (-hand_length/2 - outline_size/2, 0, -hand_height/2 - outline_size/2),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'hand_outline_9',
        mass = 0,
        size = (outline_size, outline_size, hand_height),
        joint_parent = 'hand', 
        joint_origin = (hand_length/2 + outline_size/2, hand_width/2 + outline_size/2, 0),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'hand_outline_10',
        mass = 0,
        size = (outline_size, outline_size, hand_height),
        joint_parent = 'hand', 
        joint_origin = (-hand_length/2 - outline_size/2, hand_width/2 + outline_size/2, 0),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'hand_outline_11',
        mass = 0,
        size = (outline_size, outline_size, hand_height),
        joint_parent = 'hand', 
        joint_origin = (hand_length/2 + outline_size/2, -hand_width/2 - outline_size/2, 0),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = 'hand_outline_12',
        mass = 0,
        size = (outline_size, outline_size, hand_height),
        joint_parent = 'hand', 
        joint_origin = (-hand_length/2 - outline_size/2, -hand_width/2 - outline_size/2, 0),
        joint_axis = (0, 1, 0),
        joint_type = 'fixed',
        inertia = [0, 0, 0, 0, 0, 0]),
    
    ]