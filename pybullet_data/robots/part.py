#%%

import numpy as np 
from scipy.spatial.transform import Rotation as R



def compute_transformation(joint_origin, joint_rpy):
    """
    Compute a 4x4 homogeneous transformation matrix.
    
    Parameters:
      joint_origin: tuple or list of (x, y, z) translation.
      joint_rpy: tuple or list of (roll, pitch, yaw) in radians.
      
    Returns:
      A 4x4 numpy array representing the transformation matrix.
    """
    T = np.eye(4)
    # Compute the rotation matrix from the Euler angles.
    rotation = R.from_euler('xyz', joint_rpy, degrees=False)
    T[:3, :3] = rotation.as_matrix()
    # Set the translation part.
    T[:3, 3] = np.array(joint_origin)
    return T



class Part:
    def __init__(
        self, 
        name, 
        mass = 0, 
        shape = 'box',
        size = (1, 1, 1), 
        joint_parent = None, 
        joint_origin = (0, 0, 0), 
        joint_axis = (1, 0, 0),
        joint_type = 'fixed',
        sensors = 0, 
        sensor_width = .03, 
        sensor_angle = 0,
        sensor_sides = ['start', 'stop', 'top', 'bottom', 'left', 'right'],
        joint_rpy=(0, 0, 0),
        joint_limits = [0, 0, 0, 0],
        inertia = [.05, .05, .05, .05, .05, .05]):
                
        params = locals()
        for param in params:
            if param != 'self':
                setattr(self, param, params[param])
                
        self.sensor_positions = []
        self.sensor_dimensions = []
        self.sensor_angles = []
        
        self.shape_text = self.get_shape_text()
        self.sensor_text = ''  # Initialize as empty
        self.joint_text = self.get_joint_text()
        
    def get_text(self):
        return(self.shape_text + self.sensor_text + self.joint_text)
        
    def get_shape_text(self):
        if(self.shape == 'box'):
            shape_sizes = f'box size="{self.size[0]} {self.size[1]} {self.size[2]}"'
        if(self.shape == 'cylinder'):
            shape_sizes = f'cylinder radius="{self.size[0]}" length="{self.size[1]}"'          
        return(
f"""\n\n
    <!-- {self.name} -->
    <link name='{self.name}'>
        <inertial>
            <origin xyz='0 0 0' rpy='0 0 0'/>
            <mass value='{self.mass}'/>
            <inertia ixx='{self.inertia[0]}' ixy='{self.inertia[1]}' ixz='{self.inertia[2]}' iyy='{self.inertia[3]}' iyz='{self.inertia[4]}' izz='{self.inertia[5]}'/>
        </inertial>
        <visual>
            <origin xyz='0 0 0' rpy='0 0 0'/>
            <geometry>
                <{shape_sizes}/>
            </geometry>
        </visual>
        <collision>
            <origin xyz='0 0 0' rpy='0 0 0'/>
            <geometry>
                <{shape_sizes}/>
            </geometry>
        </collision>
    </link>""")
        
    def make_sensor(self, i, side, size, origin, minus, first_plus, second_plus, parts):
        sensor_origin = [
            o if (i + first_plus) % 3 == self.sensor_angle 
            else o - self.size[i]/2 if (i + second_plus) % 3 == self.sensor_angle and minus 
            else o + self.size[i]/2 if (i + second_plus) % 3 == self.sensor_angle and not minus 
            else o
            for i, o in enumerate(origin)]
        
        cumulative_transform = np.eye(4)
        parent_part = self
        while parent_part:
            # Create a 4x4 transformation matrix for the parent's rotation and translation
            T = compute_transformation(parent_part.joint_origin, parent_part.joint_rpy)
            cumulative_transform = T @ cumulative_transform
            parent_part = next((part for part in parts if part.name == parent_part.joint_parent), None)

        # Apply the cumulative transformation to the sensor's local position
        sensor_position_homogeneous = np.array(list(sensor_origin) + [1])
        transformed_position = cumulative_transform @ sensor_position_homogeneous
        transformed_position = transformed_position[:3]  # Use only x, y, z
        
        self.sensor_positions.append(transformed_position)
        self.sensor_dimensions.append(size)
        self.sensor_angles.append(self.joint_rpy)
                
        sensor = Part(
            name = f'{self.name}_sensor_{i}_{side}',
            mass = 0,
            size = size,
            joint_parent = f'{self.name}',
            joint_origin = sensor_origin,
            joint_axis = self.joint_axis,
            joint_type = 'fixed',
            inertia = [0, 0, 0, 0, 0, 0])
        
        return(sensor.get_text())
    
    def get_sensors_text(self, parts):
        if(self.sensors == 0):
            return('')
        text = ''
        if(self.sensors == 1):
            origin = (0, 0, 0)
        else:
            origin = (
                -self.size[0]/2 + self.size[0]/(2*self.sensors) if self.sensor_angle == 0 else 0, 
                -self.size[1]/2 + self.size[1]/(2*self.sensors) if self.sensor_angle == 1 else 0, 
                -self.size[2]/2 + self.size[2]/(2*self.sensors) if self.sensor_angle == 2 else 0)
            
        start_stop_size = [
            s if (i + 2) % 3 == self.sensor_angle 
            else s if (i + 1) % 3 == self.sensor_angle
            else self.sensor_width 
            for i, s in enumerate(self.size)]
        
        top_bottom_size = [
            s / self.sensors if (i + 0) % 3 == self.sensor_angle 
            else s if (i + 2) % 3 == self.sensor_angle
            else self.sensor_width 
            for i, s in enumerate(self.size)]
        
        left_right_size = [
            s / self.sensors if (i + 0) % 3 == self.sensor_angle 
            else s if (i + 1) % 3 == self.sensor_angle
            else self.sensor_width 
            for i, s in enumerate(self.size)]
        
        for i in range(self.sensors):
            if(i == 0 and 'start' in self.sensor_sides):
                text += self.make_sensor(i, 'start', start_stop_size, (0, 0, 0), False, 2, 0, parts)
            if(i == self.sensors - 1 and 'stop' in self.sensor_sides):
                text += self.make_sensor(i, 'stop', start_stop_size, (0, 0, 0), True, 2, 0, parts)
            if('top' in self.sensor_sides):
                text += self.make_sensor(i, 'top', top_bottom_size, origin, False, 2, 1, parts)
            if('bottom' in self.sensor_sides):
                text += self.make_sensor(i, 'bottom', top_bottom_size, origin, True, 2, 1, parts)
            if('left' in self.sensor_sides):
                text += self.make_sensor(i, 'left', left_right_size, origin, False, 0, 2, parts)
            if('right' in self.sensor_sides):
                text += self.make_sensor(i, 'right', left_right_size, origin, True, 0, 2, parts)
            origin = (
                    origin[0] + self.size[0]/(self.sensors) if self.sensor_angle == 0 else origin[0], 
                    origin[1] + self.size[1]/(self.sensors) if self.sensor_angle == 1 else origin[1],
                    origin[2] + self.size[2]/(self.sensors) if self.sensor_angle == 2 else origin[2])
        return(text)
        
    def get_joint_text(self):
        if self.joint_parent is None:
            return ''
        # Use self.joint_rpy here instead of fixed 0 0 0
        return f"""
    <!-- Joint: {self.joint_parent}, {self.name} -->
    <joint name='{self.joint_parent}_{self.name}_joint' type='{self.joint_type}'>
        <parent link='{self.joint_parent}'/>
        <child link='{self.name}'/>
        <origin xyz='{self.joint_origin[0]} {self.joint_origin[1]} {self.joint_origin[2]}'
                rpy='{self.joint_rpy[0]} {self.joint_rpy[1]} {self.joint_rpy[2]}'/>
        <axis xyz='{self.joint_axis[0]} {self.joint_axis[1]} {self.joint_axis[2]}'/>
        {'' if self.joint_type == 'fixed' else f'<limit lower="{self.joint_limits[0]}" upper="{self.joint_limits[1]}" effort="{self.joint_limits[2]}" velocity="{self.joint_limits[3]}"/>'}
    </joint>"""
    
    
    
if(__name__ == '__main__'):
    part = Part(
        name = 'body', 
        mass = 100, 
        size = (1, 1, 1),
        sensors = 1,
        sensor_sides = ['start', 'stop', 'top', 'left', 'right']),