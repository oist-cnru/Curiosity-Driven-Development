#%%

import os
import pickle
import pybullet as p
from time import sleep
import builtins
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, ConnectionPatch
import argparse, ast
from math import exp, log, pi
from random import choice, choices
import torch
import psutil
from itertools import product
import tkinter as tk
import numpy as np

# -------------------------------
# TORCH SETUP
# -------------------------------

device = torch.device('cpu')  # I recommend CPU explicitly.

# -------------------------------
# DIRECTORY CHECK
# -------------------------------

if os.getcwd().split('/')[-1] != 'communication': 
    os.chdir('communication')
print(f'\n\nWorking in: {os.getcwd()}\n\n')

# -------------------------------
# Miscellaneous
# -------------------------------

font = {
    'family': 'sans-serif',
    'size': 22
}
matplotlib.rc('font', **font)

def print(*args, **kwargs):
    """Override built-in print to auto-flush."""
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)

torch.set_printoptions(precision=3, sci_mode=False)

start_time = datetime.datetime.now()

def duration(start_time=start_time):
    """Return elapsed time since given start time (default: script start)."""
    delta = datetime.datetime.now() - start_time
    return delta

def print_duration(start_time, end_time, text=None, end_text=''):
    """Print the duration between two times with optional prefix text."""
    delta = end_time - start_time
    if text:
        print(f'{text}: {delta}{end_text}')
    else:
        print(f'{delta}{end_text}')

def estimate_total_duration(proportion_completed, start_time=start_time):
    """Estimate total time given progress percentage and elapsed time."""
    if proportion_completed == 0:
        return '?:??:??'
    so_far = datetime.datetime.now() - start_time
    estimated_total = so_far / proportion_completed
    estimated_total = estimated_total - datetime.timedelta(microseconds=estimated_total.microseconds)
    return estimated_total

def cpu_memory_usage():
    """Print memory usage of current Python process (in GB)."""
    process = psutil.Process(os.getpid())
    mem_usage_bytes = process.memory_info().rss
    mem_usage_gb = mem_usage_bytes / (1024 ** 3)
    print('memory use:', round(mem_usage_gb, 3), 'gigabytes')

#%%

# -------------------------------
# TASK
# -------------------------------

class Task:
    """Represent a task with a character and name."""
    def __init__(self, char, name):
        self.char = char
        self.name = name

    def __str__(self):
        return f'{self.char}, {self.name}'


task_map = {
    0:  Task('A', 'SILENCE'),
    1:  Task('B', 'WATCH'),
    2:  Task('C', 'BE NEAR'),
    3:  Task('D', 'TOUCH THE TOP'),
    4:  Task('E', 'PUSH FORWARD'),     
    5:  Task('F', 'PUSH LEFT'),   
    6:  Task('G', 'PUSH RIGHT')
}

max_len_taskname = max(len(task.name) for task in task_map.values())
task_name_list = [task.name for task in task_map.values()]

# -------------------------------
# COLOR
# -------------------------------

class Color:
    """Represent a color with a character, name, and RGBA tuple."""
    def __init__(self, char, name, rgba):
        self.char = char
        self.name = name
        self.rgba = rgba

    def __str__(self):
        return f'{self.char}, {self.name}'


color_map = {
    0: Color('H', 'RED',     (1, 0, 0, 1)),
    1: Color('I', 'GREEN',   (0, 1, 0, 1)),
    2: Color('J', 'BLUE',    (0, 0, 1, 1)),
    3: Color('K', 'CYAN',    (0, 1, 1, 1)),
    4: Color('L', 'MAGENTA', (1, 0, 1, 1)),
    5: Color('M', 'YELLOW',  (1, 1, 0, 1)),
}

max_len_color_name = max(len(c.name) for c in color_map.values())
color_name_list = [c.name for c in color_map.values()]

# -------------------------------
# SHAPE
# -------------------------------

class Shape:
    """Represent a shape using a character and shape file name."""
    def __init__(self, char, file_name):
        self.char = char
        self.file_name = file_name
        self.name = file_name.split('_')[-1][:-5]  # Extract name from filename

    def __str__(self):
        return f'{self.char}, {self.name}'


shape_files = [f.name for f in os.scandir('pybullet_data/shapes') if f.name.endswith('urdf')]
shape_files.sort()
shape_letter_file = [[f.split('_')[0], f] for f in shape_files]
shape_map = {i: Shape(letter, fname) for i, (letter, fname) in enumerate(shape_letter_file)}

max_len_shape_name = max(len(s.name) for s in shape_map.values())
shape_name_list = [s.name for s in shape_map.values()]

# -------------------------------
# DISPLAY ALL OPTIONS
# -------------------------------

if __name__ == '__main__':
    print('Tasks:')
    for key, value in task_map.items():
        print(f'\t{key} :\t{value}')
    print('Colors:')
    for key, value in color_map.items():
        print(f'\t{key} :\t{value}')
    print('Shapes:')
    for key, value in shape_map.items():
        print(f'\t{key} :\t{value}')



        
#%%

import torch

# -------------------------------
# GOAL
# -------------------------------

class Goal:
    """
    Combines a task, color, and shape into a single goal object.
    Used for interpreting command voices or specifying target behavior.

    - 'parenting' indicates if command voice is used (True) or agent is cooperating with another (False).
    """
    def __init__(self, task, color, shape, parenting):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        
        if self.task.name == 'SILENCE':
            self.color = self.task
            self.shape = self.task
        
        self.one_hots = torch.zeros((3, len(task_map) + len(color_map) + len(shape_map)))
        self.digits = ()
        self.make_texts()

    def make_texts(self):
        """Generate one-hot vector, index triplet, and text representations."""
        for i, char in enumerate([self.task.char, self.color.char, self.shape.char]):
            index = ord(char) - ord('A')
            self.one_hots[i, index] = 1
            
        task_index = ord(self.task.char) - ord('A')
        color_index = ord(self.color.char) - ord('A') - len(task_map)
        shape_index = ord(self.shape.char) - ord('A') - len(task_map) - len(color_map)
        self.digits = (task_index, color_index, shape_index)

        self.char_text = f'{self.task.char}{self.color.char}{self.shape.char}'
        self.human_text = f'{self.task.name} {self.color.name} {self.shape.name}'

    def human_friendly_text(self, command=True):
        """Return a more readable label for command/feedback voice."""
        return f'{"Command" if command else "Feedback"}: {self.human_text}'

# Create a default 'silent' goal
empty_goal = Goal(task_map[0], task_map[0], task_map[0], parenting=False)


# -------------------------------
# GOAL UTILITIES
# -------------------------------

def get_goal_from_one_hots(one_hots):
    """Construct a Goal object from a one-hot representation."""
    while one_hots.ndim > 2:
        one_hots = one_hots.squeeze(0)
        
    task_one_hot = one_hots[0, :len(task_map)]
    color_one_hot = one_hots[1, len(task_map):len(task_map) + len(color_map)]
    shape_one_hot = one_hots[2, len(task_map) + len(color_map):len(task_map) + len(color_map) + len(shape_map)]

    task = task_map[torch.argmax(task_one_hot).item()]
    color = color_map[torch.argmax(color_one_hot).item()]
    shape = shape_map[torch.argmax(shape_one_hot).item()]
    
    goal = Goal(task, color, shape, parenting=False)
    return empty_goal if task.name == 'SILENCE' else goal


def get_goal_from_digits(digits):
    """Construct a Goal object from a tuple of (task, color, shape) indices."""
    x, y, z = digits
    one_hots = torch.zeros((3, len(task_map) + len(color_map) + len(shape_map)))
    one_hots[0, x] = 1
    one_hots[1, len(task_map) + y] = 1
    one_hots[2, len(task_map) + len(color_map) + z] = 1
    return get_goal_from_one_hots(one_hots)

# -------------------------------
# REINFORCEMENT LEARNING WRAPPERS
# -------------------------------

class Obs:
    """
    A single observation consisting of:
    - vision
    - touch
    - proprioception
    - command voice
    - feedback voice
    """
    def __init__(self, vision, touch, prop, command_voice, feedback_voice):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})


class Action:
    """
    A single action consisting of:
    - wheels & joints
    - voice output
    """
    def __init__(self, wheels_joints, voice_out):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})


class To_Push:
    """
    Transition for replay buffer:
    (obs, action, reward, next_obs, done)
    """
    def __init__(self, obs, action, reward, next_obs, done):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})

    def push(self, memory):
        """Push the transition into the replay memory."""
        memory.push(
            self.obs.vision.to('cpu'),
            self.obs.touch.to('cpu'),
            self.obs.prop.to('cpu'),
            self.obs.command_voice.to('cpu'),
            self.obs.feedback_voice.to('cpu'),
            self.action.wheels_joints.to('cpu'), 
            self.action.voice_out.to('cpu'),
            self.reward, 
            self.next_obs.vision.to('cpu'),
            self.next_obs.touch.to('cpu'),
            self.next_obs.prop.to('cpu'),
            self.next_obs.command_voice.to('cpu'), 
            self.next_obs.feedback_voice.to('cpu'), 
            self.done
        )


class Inner_States:
    """
    Latent internal state:
    - zp: prior sample
    - zq: posterior sample
    - dkl: KL divergence
    """
    def __init__(self, zp, zq, dkl):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})


# -------------------------------
# CHAR ↔ INDEX MAPPINGS
# -------------------------------

used_chars = sorted(
    [t.char for t in task_map.values()] +
    [c.char for c in color_map.values()] +
    [s.char for s in shape_map.values()]
)

voice_map = {
    k: v for k, v in {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
        7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N',
        14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
        21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
    }.items() if v in used_chars
}

char_to_index = {v: k for k, v in voice_map.items()}


# -------------------------------
# TESTING EXAMPLES
# -------------------------------

if __name__ == '__main__':
    print('\n\nEmpty Goal:')
    example = empty_goal
    print(example.one_hots)
    print(example.char_text)
    print(example.human_text)
    print(get_goal_from_one_hots(example.one_hots).human_text)

    print('\n\nExample Goal:')
    example = Goal(task_map[1], color_map[2], shape_map[2], parenting=False)
    print(example.one_hots)
    print(example.char_text)
    print(example.human_text)
    print(get_goal_from_one_hots(example.one_hots).human_text)

    print('\n\nExample Goal:')
    example = Goal(task_map[4], color_map[3], shape_map[3], parenting=False)
    print(example.one_hots)
    print(example.char_text)
    print(example.human_text)
    print(get_goal_from_one_hots(example.one_hots).human_text)
    print('\n\n')





#%%


# -------------------------------
# GENERATING VALID TEST/TRAINING COMBINATIONS
# -------------------------------



# All possible combinations of (task, color, shape)
all_combos = list(product(task_map.keys(), color_map.keys(), shape_map.keys()))

def get_matrix_pattern(a_values, rows=5, cols=6):
    """
    Returns a matrix pattern excluding (r, c) pairs for additive cyclic offsets.
    """
    excluded = set()
    for a in a_values:
        for r in range(rows):
            c = (r + a) % cols
            excluded.add((r, c))
    return [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in excluded]

# Pattern lookups for training set 3: 6 tasks, 6 colors, 5 shapes (180 goals total, 60 in training)
pattern_lookup_3 = {
    1: set(get_matrix_pattern([0, 1, 2, 3])),
    2: set(get_matrix_pattern([1, 2, 3, 4])),
    3: set(get_matrix_pattern([2, 3, 4, 5])),
    4: set(get_matrix_pattern([3, 4, 5, 6])),
    5: set(get_matrix_pattern([-2, -1, 0, 1])),
    6: set(get_matrix_pattern([-1, 0, 1, 2]))
}



def get_training_combos(pattern_lookup):
    """
    Uses a task-specific pattern lookup to filter all_combos for training.
    Returns:
        A list of (task, color, shape) combinations used in training.
    """
    return [
        (a, c, s) for (a, c, s) in all_combos if
        a == 0 or
        (a == 1 and (s, c) in pattern_lookup[1]) or
        (a == 2 and (s, c) in pattern_lookup[2]) or
        (a == 3 and (s, c) in pattern_lookup[3]) or
        (a == 4 and (s, c) in pattern_lookup[4]) or
        (a == 5 and (s, c) in pattern_lookup[5]) or
        (a == 6 and (s, c) in pattern_lookup[6])
    ]



# ------------------------------------------------------------
# Training & Testing Splits
# ------------------------------------------------------------

# === Set 1: 4 tasks, 4 colors, 3 shapes (48 goals total, 16 in training) ===
# 16 for training, 32 for testing
training_combos_1 = [
    # SILENCE 
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), 
    (0, 1, 1), (0, 1, 2), (0, 2, 0), (0, 2, 1), 
    (0, 2, 2), (0, 3, 0), (0, 3, 1), (0, 3, 2), 
    # Sparse task combos
    (1, 2, 0), (1, 3, 0), (1, 0, 1), (1, 1, 2),
    (4, 0, 0), (4, 1, 1), (4, 3, 1), (4, 2, 2), 
    (5, 1, 0), (5, 2, 1), (5, 0, 2), (5, 3, 2), 
    (6, 1, 0), (6, 2, 0), (6, 3, 1), (6, 0, 2)
]
testing_combos_1 = [combo for combo in all_combos if combo not in training_combos_1]

# === Set 2: 5 tasks, 5 colors, 3 shapes (75 goals total, 25 in training) ===
# 25 for training, 50 for testing
training_combos_2 = [
    # SILENCE
    (0, 0, 0), (0, 0, 1), (0, 0, 2), 
    (0, 1, 0), (0, 1, 1), (0, 1, 2), 
    (0, 2, 0), (0, 2, 1), (0, 2, 2), 
    (0, 3, 0), (0, 3, 1), (0, 3, 2), 
    (0, 4, 0), (0, 4, 1), (0, 4, 2), 
    # Sparse task combos
    (1, 3, 0), (1, 0, 1), (1, 4, 1), (1, 0, 2), (1, 1, 2),
    (2, 0, 0), (2, 4, 0), (2, 1, 1), (2, 1, 2), (2, 2, 2),
    (4, 0, 0), (4, 1, 0), (4, 1, 1), (4, 2, 1), (4, 3, 2),
    (5, 2, 0), (5, 2, 1), (5, 3, 1), (5, 3, 2), (5, 4, 2),
    (6, 2, 0), (6, 3, 1), (6, 4, 1), (6, 0, 2), (6, 4, 2)
]
testing_combos_2 = [combo for combo in all_combos if combo not in training_combos_2]

# === Set 3: 6 tasks, 6 colors, 5 shapes (180 goals total, 60 in training) ===
training_combos_3 = get_training_combos(pattern_lookup_3)
testing_combos_3 = [combo for combo in all_combos if combo not in training_combos_3]



# === Set 4: only watch, for testing baseline model ===
training_combos_4 = [
    # SILENCE
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4),
    (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4),
    (0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 2, 4), 
    (0, 3, 0), (0, 3, 1), (0, 3, 2), (0, 3, 3), (0, 3, 4), 
    (0, 4, 0), (0, 4, 1), (0, 4, 2), (0, 4, 3), (0, 4, 4), 
    (0, 5, 0), (0, 5, 1), (0, 5, 2), (0, 5, 3), (0, 5, 4), 
    # Sparse task combos
    (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 2, 1), (1, 2, 2), (1, 3, 2), (1, 3, 3), (1, 4, 3), (1, 4, 4), (1, 5, 4),
]
testing_combos_4 = [combo for combo in all_combos if combo not in training_combos_4]

# === Set 5: only be near, for testing baseline model ===
training_combos_5 = [
    # SILENCE
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4),
    (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4),
    (0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 2, 4), 
    (0, 3, 0), (0, 3, 1), (0, 3, 2), (0, 3, 3), (0, 3, 4), 
    (0, 4, 0), (0, 4, 1), (0, 4, 2), (0, 4, 3), (0, 4, 4), 
    (0, 5, 0), (0, 5, 1), (0, 5, 2), (0, 5, 3), (0, 5, 4), 
    # Sparse task combos
    (2, 0, 0), (2, 1, 0), (2, 1, 1), (2, 2, 1), (2, 2, 2), (2, 3, 2), (2, 3, 3), (2, 4, 3), (2, 4, 4), (2, 5, 4),
]
testing_combos_5 = [combo for combo in all_combos if combo not in training_combos_5]



# === Set 6: only watch or be near, for testing baseline model ===
training_combos_6 = [
    # SILENCE
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4),
    (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4),
    (0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 2, 4), 
    (0, 3, 0), (0, 3, 1), (0, 3, 2), (0, 3, 3), (0, 3, 4), 
    (0, 4, 0), (0, 4, 1), (0, 4, 2), (0, 4, 3), (0, 4, 4), 
    (0, 5, 0), (0, 5, 1), (0, 5, 2), (0, 5, 3), (0, 5, 4), 
    # Sparse task combos
    (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 2, 1), (1, 2, 2), (1, 3, 2), (1, 3, 3), (1, 4, 3), (1, 4, 4), (1, 5, 4),
    (2, 2, 0), (2, 3, 0), (2, 3, 1), (2, 4, 1), (2, 4, 2), (2, 5, 2), (2, 5, 3), (2, 5, 3), (2, 5, 4), (2, 0, 4),
]
testing_combos_6 = [combo for combo in all_combos if combo not in training_combos_6]



# === Set 7: only push_forward, for testing baseline model ===
training_combos_7 = [
    # SILENCE
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4),
    (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4),
    (0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 2, 4), 
    (0, 3, 0), (0, 3, 1), (0, 3, 2), (0, 3, 3), (0, 3, 4), 
    (0, 4, 0), (0, 4, 1), (0, 4, 2), (0, 4, 3), (0, 4, 4), 
    (0, 5, 0), (0, 5, 1), (0, 5, 2), (0, 5, 3), (0, 5, 4), 
    # Sparse task combos
    (4, 0, 0), (4, 1, 0), (4, 1, 1), (4, 2, 1), (4, 2, 2), (4, 3, 2), (4, 3, 3), (4, 4, 3), (4, 4, 4), (4, 5, 4),
]
testing_combos_7 = [combo for combo in all_combos if combo not in training_combos_7]



# === Set 8: only watch or be near or push forward, for testing baseline model ===
training_combos_8 = [
    # SILENCE
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4),
    (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4),
    (0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 2, 4), 
    (0, 3, 0), (0, 3, 1), (0, 3, 2), (0, 3, 3), (0, 3, 4), 
    (0, 4, 0), (0, 4, 1), (0, 4, 2), (0, 4, 3), (0, 4, 4), 
    (0, 5, 0), (0, 5, 1), (0, 5, 2), (0, 5, 3), (0, 5, 4), 
    # Sparse task combos
    (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 2, 1), (1, 2, 2), (1, 3, 2), (1, 3, 3), (1, 4, 3), (1, 4, 4), (1, 5, 4),
    (2, 2, 0), (2, 3, 0), (2, 3, 1), (2, 4, 1), (2, 4, 2), (2, 5, 2), (2, 5, 3), (2, 5, 3), (2, 5, 4), (2, 0, 4),
    (4, 4, 0), (4, 5, 0), (4, 5, 1), (4, 0, 1), (4, 0, 2), (4, 1, 2), (4, 1, 3), (4, 2, 3), (4, 2, 4), (4, 3, 4),
]
testing_combos_8 = [combo for combo in all_combos if combo not in training_combos_8]



# Exceptions dictionary
exceptions_dict = {
    0 : (                           # None
        [],            
        []),

    1 : (
        [(1, 4, 0), (2, 1, 1)],     # Swap Watch Magenta Pillar with Be Near Green Pole
        [(2, 1, 1), (1, 4, 0)]),
}


def add_control_exceptions(ex_dict):
    """
    Adds control conditions: each odd-numbered exception key gets a paired even-numbered key
    where red == blue (no actual change, tests robustness to exception highlighting).
    """
    new_dict = ex_dict.copy()
    for k in list(ex_dict.keys()):
        if k % 2 == 1:
            red = ex_dict[k][0]
            new_dict[k + 1] = (red, red)
    return new_dict

exceptions_dict = add_control_exceptions(exceptions_dict)



# In __main__, view plots showing training and testing combinations.
if __name__ == '__main__':
    def plot_combined_training_grid(training_combos, exception_num, title='Training Set'):
        task_items = [(a, t) for a, t in task_map.items() if t.name != 'SILENCE']
        num_tasks = len(task_items)
        num_cols = 3
        num_rows = (num_tasks + num_cols - 1) // num_cols

        fig = plt.figure(figsize=(22, 12))
        fig.suptitle(title, fontsize=28)
        outer_grid = gridspec.GridSpec(num_rows, num_cols, wspace=0.5, hspace=0.5)

        # Map each (a, c, s) combo to its Axes so we can connect across subplots later
        ax_map = {}

        for i, (a, task) in enumerate(task_items):
            inner_grid = gridspec.GridSpecFromSubplotSpec(
                len(shape_map), len(color_map),
                subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0
            )

            for s in range(len(shape_map)):
                for c in range(len(color_map)):
                    ax = fig.add_subplot(inner_grid[s, c])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)

                    combo = (a, c, s)
                    ax_map[combo] = ax  # remember where this combo lives

                    # Base cell
                    if combo in training_combos:
                        ax.add_patch(patches.Rectangle((0, 0), 1, 1, color='gray', alpha=0.5))
                    else:
                        ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black'))

                    # Red exception (centered at (0.5, 0.5))
                    if combo in exceptions_dict[exception_num][0]:
                        ax.add_patch(patches.Rectangle((.25, .25), .5, .5, facecolor='red', alpha = .5, edgecolor='red'))

                    # Blue exception (also centered at (0.5, 0.5))
                    if combo in exceptions_dict[exception_num][1]:
                        ax.add_patch(patches.Rectangle((.375, .375), .25, .25, facecolor='blue', alpha = .3, edgecolor='blue'))

                    # Label
                    ax.text(0.5, 0.5, f'{color_map[c].name}\n{shape_map[s].name}',
                            va='center', ha='center', fontsize=9, wrap=True)

            # Task title in the middle column
            center_col = len(color_map) // 2
            title_ax = fig.add_subplot(inner_grid[0, center_col])
            title_ax.set_title(task.name, fontsize=14, pad=12)
            title_ax.axis('off')

        # Make sure layout is finalized before drawing connectors
        fig.canvas.draw()

        # Draw arrows: from each red combo to the blue combo at the same index
        for start_combo, end_combo in zip(exceptions_dict[exception_num][0], exceptions_dict[exception_num][1]):
            start_ax = ax_map.get(start_combo)
            end_ax   = ax_map.get(end_combo)
            if start_ax is None or end_ax is None:
                continue

            # Both colored squares are centered at (0.5, 0.5) in their own axes
            con = ConnectionPatch(
                xyA=(0.5, 0.5), 
                coordsA=start_ax.transData,   # start (red)
                xyB=(0.5, 0.5), 
                coordsB=end_ax.transData,     # end (blue)
                arrowstyle='-|>', 
                ls = '-',
                mutation_scale=25, 
                lw=1.8, 
                color='black',
                shrinkA=10, 
                shrinkB=10  # keep arrowheads off the colored squares
            )
            con.set_zorder(1000)
            con.set_clip_on(False)  # don't let axes clip the arrow
            fig.add_artist(con)

        plt.show()
        plt.close()
    
    plot_combined_training_grid(training_combos_1, title='Training Set 1 – 4 tasks', exception_num = 0)
    plot_combined_training_grid(training_combos_2, title='Training Set 2 – 5 tasks', exception_num = 0)
    plot_combined_training_grid(training_combos_3, title='Training Set 3 – All Tasks', exception_num = 0)
    #plot_combined_training_grid(training_combos_4, title='Training Set 4 – Only watch', exception_num = 0)
    #plot_combined_training_grid(training_combos_5, title='Training Set 5 – Only watch', exception_num = 0)
    #plot_combined_training_grid(training_combos_6, title='Training Set 6 – Only watch and be near', exception_num = 0)
    #plot_combined_training_grid(training_combos_7, title='Training Set 7 – Only push forward', exception_num = 0)
    #plot_combined_training_grid(training_combos_8, title='Training Set 8 – Watch, Be Near, Push Forward', exception_num = 0)

    
    #for key in exceptions_dict.keys():
    #    lot_combined_training_grid(training_combos_3, title=f'Training Set 3 - All Tasks - Exceptions {key}', exception_num = key)
            
        
        
#%%



#%%
"""
Functions for generating task-goal combinations with constraints on tasks, colors, and shapes.
"""


def valid_color_shape(task_num, other_shape_colors, allowed_colors, allowed_shapes, test_train_num=3, test=False):
    """
    Choose a valid (color, shape) pair for a given task, filtering based on:
    - test vs. train set (controlled by `test` flag)
    - allowed colors and shapes
    - exclusion of duplicates from other_shape_colors
    """
    training_combos = (
        training_combos_1 if test_train_num == 1 else
        training_combos_2 if test_train_num == 2 else
        training_combos_3 if test_train_num == 3 else
        training_combos_4 if test_train_num == 4 else 
        training_combos_5 if test_train_num == 5 else
        training_combos_6 if test_train_num == 6 else
        training_combos_7 if test_train_num == 7 else
        training_combos_8
    )
    testing_combos = [combo for combo in all_combos if combo not in training_combos]

    if test is None:
        these_combos = testing_combos + training_combos
    elif test:
        these_combos = testing_combos
    else:
        these_combos = training_combos

    these_combos = [combo for combo in these_combos if combo[0] == task_num]
    these_combos = [
        (color, shape) for _, color, shape in these_combos
        if color in allowed_colors and shape in allowed_shapes
    ]
    if test is not None:
        these_combos = [combo for combo in these_combos if combo not in other_shape_colors]

    color_num, shape_num = choice(these_combos)
    return color_num, shape_num


def make_objects_and_task(num_objects, allowed_tasks_and_weights, allowed_colors, allowed_shapes, test_train_num=3, test=False):
    """
    Generate a full goal specification:
    - Randomly selects a task based on weights
    - Selects valid color-shape combinations for two object lists
    - Returns: (Task, List1 of (Color, Shape), List2 of (Color, Shape))
    """
    tasks = [v for v, w in allowed_tasks_and_weights]
    weights = [w for v, w in allowed_tasks_and_weights]
    task_num = choices(tasks, weights=weights, k=1)[0]

    goal_object = valid_color_shape(task_num, [], allowed_colors, allowed_shapes, test_train_num, test=test)
    colors_shapes_1 = [goal_object]
    colors_shapes_2 = [goal_object]

    for _ in range(num_objects - 1):
        colors_shapes_1.append(valid_color_shape(task_num, colors_shapes_1 + colors_shapes_2, allowed_colors, allowed_shapes, test_train_num, test=test))
    for _ in range(num_objects - 1):
        colors_shapes_2.append(valid_color_shape(task_num, colors_shapes_1 + colors_shapes_2, allowed_colors, allowed_shapes, test_train_num, test=test))

    task = task_map[task_num]
    colors_shapes_1 = [(color_map[c], shape_map[s]) for c, s in colors_shapes_1]
    colors_shapes_2 = [(color_map[c], shape_map[s]) for c, s in colors_shapes_2]

    return task, colors_shapes_1, colors_shapes_2


# ---------------------------------------
# Demonstration of goal generation logic
# ---------------------------------------
if __name__ == '__main__':
    print('Train')
    for _ in range(1):
        task, colors_shapes_1, colors_shapes_2 = make_objects_and_task(
            num_objects=2,
            allowed_tasks_and_weights=[(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],
            allowed_colors=[0, 1, 2, 3, 4, 5],
            allowed_shapes=[0, 1, 2, 3, 4]
        )
        print(task.name, [(c.name, s.name) for c, s in colors_shapes_1], [(c.name, s.name) for c, s in colors_shapes_2])

    print('\nTest')
    for _ in range(1):
        task, colors_shapes_1, colors_shapes_2 = make_objects_and_task(
            num_objects=2,
            allowed_tasks_and_weights=[(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],
            allowed_colors=[0, 1, 2, 3, 4, 5],
            allowed_shapes=[0, 1, 2, 3, 4],
            test=True
        )
        print(task.name, [(c.name, s.name) for c, s in colors_shapes_1], [(c.name, s.name) for c, s in colors_shapes_2])
        
        
        
        
#%% 



# ---------------------------------------
# LIST OF ARGUMENTS
# ---------------------------------------



# Type for booleons in arguments.
def literal(arg_string): 
    return(ast.literal_eval(arg_string))


# Arguments to parse. 
parser = argparse.ArgumentParser()

    # Meta 
parser.add_argument('--arg_title',                      type=str,           default = 'default',
                    help='Title of argument-set containing all non-default arguments.') 
parser.add_argument('--arg_name',                       type=str,           default = 'default',
                    help='Title of argument-set for human-understanding.') 
parser.add_argument('--agents',                         type=int,           default = 36,
                    help='How many agents are trained in this job?')
parser.add_argument('--previous_agents',                type=int,           default = 0,
                    help='How many agents with this argument-set are trained in previous jobs?')
parser.add_argument('--init_seed',                      type=float,         default = 33333,   
                    help='Random seed. Due to versions of python packages, results may vary.')
parser.add_argument('--comp',                           type=str,           default = 'deigo',
                    help='Cluster name (deigo or saion).')
parser.add_argument('--device',                         type=str,           default = device,
                    help='Which device to use for Torch.')
parser.add_argument('--cpu',                            type=int,           default = 0,
                    help='Which cpu for affinity.')
parser.add_argument('--local',                          type=bool,          default = False,
                    help='Is this running on a local machine for testing?')
parser.add_argument('--show_duration',                  type=bool,          default = False,
                    help='Should durations be printed?')
parser.add_argument('--load_agents',                    type=literal,       default = False,
                    help='Are we loading agents?')      

    

    # Simulation details
parser.add_argument('--time_step',                      type=float,         default = .005,
                    help='Length of step in pybullet environment.')
parser.add_argument('--steps_per_step',                 type=int,           default = 20,
                    help='Agent-steps for each action.')
parser.add_argument('--numSolverIterations',            type=int,           default = 1,
                    help='Precision of steps in pybullet environment.')
parser.add_argument('--numSubSteps',                    type=int,           default = 1,
                    help='numSubSteps in pybullet environment.')
parser.add_argument('--force',                          type=float,         default = 30000,
                    help='Force for moving joints.') 
parser.add_argument('--gravity',                        type=float,         default = -9.8,
                    help='Force of gravity.') 



    # Which tasks/colors/shapes are allowed in this test_train_num?
parser.add_argument('--watch',                          type=literal,       default = True,
                    help='Allow watch task?')
parser.add_argument('--be_near',                        type=literal,       default = True,
                    help='Allow be_near task?')
parser.add_argument('--touch_top',                      type=literal,       default = True,
                    help='Allow touch_top task?')
parser.add_argument('--push_forward',                   type=literal,       default = True,
                    help='Allow push_forward task?')
parser.add_argument('--push_left',                      type=literal,       default = True,
                    help='Allow push_left task?')
parser.add_argument('--push_right',                     type=literal,       default = True,
                    help='Allow push_right task?')

parser.add_argument('--red',                            type=literal,       default = True,
                    help='Allow red color?')
parser.add_argument('--green',                          type=literal,       default = True,
                    help='Allow green color?')
parser.add_argument('--blue',                           type=literal,       default = True,
                    help='Allow blue color?')
parser.add_argument('--cyan',                           type=literal,       default = True,
                    help='Allow cyan color?')
parser.add_argument('--magenta',                        type=literal,       default = True,
                    help='Allow magenta color?')
parser.add_argument('--yellow',                         type=literal,       default = True,
                    help='Allow yellow color?')

parser.add_argument('--pillar',                         type=literal,       default = True,
                    help='Allow pillar shape?')
parser.add_argument('--pole',                           type=literal,       default = True,
                    help='Allow pole shape?')
parser.add_argument('--dumbbell',                       type=literal,       default = True,
                    help='Allow dumbbell shape?')
parser.add_argument('--cone',                           type=literal,       default = True,
                    help='Allow cone shape?')
parser.add_argument('--hourglass',                      type=literal,       default = True,
                    help='Allow hourglass shape?')



    # Agent details
parser.add_argument('--robot_name',                     type=str,           default = 'robot',
                    help='Name of the robot\'s urdf file.')  
parser.add_argument('--body_size',                      type=float,         default = 2,
                    help='How large is the agent\'s body?')  
parser.add_argument('--image_size',                     type=int,           default = 16, 
                    help='Dimensions of the images observed.')
parser.add_argument('--max_wheel_speed',                type=float,         default = 10,
                    help='Max wheel speed.')
parser.add_argument('--angular_scaler',                 type=float,         default = .4,
                    help='How to scale angular velocity vs linear velocity.')
parser.add_argument('--max_joint_speed',                type=float,         default = 8,
                    help='Max joint speed.')
parser.add_argument('--max_joint_1_angle',              type=float,         default = pi/6,
                    help='Max yaw angle.')
parser.add_argument('--min_joint_2_angle',              type=float,         default = -pi/2,
                    help='Min pitch angle.')
parser.add_argument('--max_joint_2_angle',              type=float,         default = 0,
                    help='Max pitch angle.')



    # Arena/Processor details
parser.add_argument('--processor',                      type=str,           default = 'all',
                help='List of processors. Agent trains on each processor based on epochs in epochs parameter.')
parser.add_argument('--min_object_distance',            type=float,         default = 4,
                    help='How far objects can start from the agent.')
parser.add_argument('--max_object_distance',            type=float,         default = 8,
                    help='How far objects can start from the agent.')
parser.add_argument('--min_object_angle',               type=float,         default = pi/2,
                    help='How far objects must be from one another.')
parser.add_argument('--object_size',                    type=float,         default = 2.5,
                    help='How large are objects?')          

parser.add_argument('--reward',                         type=float,         default = 10,
                    help='Extrinsic reward for choosing correct task, shape, and color.') 
parser.add_argument('--wrong_object_punishment',        type=float,         default = 0,
                    help='Negative reward for punishing doing anything to the wrong object (except watching).') 
parser.add_argument('--hidden_state_eta_feedback_voice_reduction_type',  type=str,         default = 'None',
                    help='How should interest in feedback_voice chance?') 
parser.add_argument('--reward_inflation_type',          type=str,           default = 'None',
                    help='How should reward increase?')   
parser.add_argument('--tanh_touch',                     type=literal,       default = True,
                    help='Do sensors measure contact with Tanh?')

parser.add_argument('--max_steps',                      type=int,           default = 30,     
                    help='How many steps the agent can make in one episode.')
parser.add_argument('--step_lim_punishment',            type=float,         default = 0,
                    help='Extrinsic punishment for taking max_steps steps.')
parser.add_argument('--step_cost',                      type=float,         default = .99,    
                    help='How much extrinsic rewards are reduced per step.')
parser.add_argument('--max_voice_len',                  type=int,           default = 3,
                    help='Maximum length of voice.')



    # Task details.
parser.add_argument('--watch_duration',                 type=int,           default = 6,
                    help='How long the agent must watch the object to achieve watching.')
parser.add_argument('--pointing_at_object_for_watch',   type=float,         default = pi/12,
                    help='How directly the agent must point to the object to achieve watching.')
parser.add_argument('--watch_distance',                 type=float,         default = 12,
                    help='How closely the agent must watch the object to achieve watching.')

parser.add_argument('--be_near_duration',               type=int,           default = 5,
                    help='How long the agent must be near the object to achieve be_near.')
parser.add_argument('--pointing_at_object_for_being_near',  type=float,     default = pi/6,
                    help='How directly the agent must point to the object to achieve be_near.')
parser.add_argument('--be_near_distance',               type=float,         default = 6,
                    help='How close the agent must be near the object to achieve be_near.')

parser.add_argument('--top_duration',                   type=int,           default = 3,   
                    help='How long the agent must touch the top of the object to achieve touch_top.')
parser.add_argument('--pointing_at_object_for_touch_top',  type=float,      default = pi/3,
                    help='How directly the agent must point to the object to achieve touch top.')
parser.add_argument('--touch_top_min_height',           type=float,         default = 3.75,
                    help='How elevated the agent\'s arm must be to touch the object from above.')

parser.add_argument('--push_duration',                  type=int,           default = 3,
                    help='How long the agent must push the object to achieve push_forward.')
parser.add_argument('--pointing_at_object_for_push',    type=float,         default = pi/12,
                    help='How directly the agent must point to the object to achieve push_forward.')
parser.add_argument('--global_push_amount',             type=float,         default = .1,
                    help='Needed distance of an object\'s movement for push_forward')
parser.add_argument('--local_push_limit',               type=float,         default = .3,
                    help='Prevent bogus pushing by requiring local stillness.')

parser.add_argument('--left_right_duration',            type=int,           default = 3,   
                    help='How long the agent must push the object to achieve push_left or push_right.')
parser.add_argument('--pointing_at_object_for_left_right', type=float,      default = pi/3,
                    help='How directly the agent must point to the object to achieve push_left or push_right.')
parser.add_argument('--global_left_right_amount',       type=float,         default = .2,
                    help='Needed distance of an object\'s movement for push_left or push_right.')
parser.add_argument('--local_left_right_amount',        type=float,         default = .25,
                    help='Prevent bogus pushing by requiring local movement.')
parser.add_argument('--max_wheel_speed_for_left_right', type=float,         default = 5,
                    help='How fast the agent\'s wheels may move for push_left or push_right.')
parser.add_argument('--min_arm_speed_for_left_right',   type=float,         default = .01,
                    help='How fast the agent\'s arm must move for push_left or push_right.')

parser.add_argument('--exceptions',                     type=literal,       default = 0,
                    help='Add exceptions to goals?')



    # Model architecture
parser.add_argument('--lstm',                           type=literal,       default = False,
                    help='Should we use the baseline model, instead of the PVRNN-style model?')   
parser.add_argument('--hidden_size',                    type=int,           default = 64,
                    help='Parameters in hidden layers.')   
parser.add_argument('--pvrnn_mtrnn_size',               type=int,           default = 256,
                    help='Parameters in hidden layers of PVRNN\'s mtrnn.')   

parser.add_argument('--wheels_joints_encode_size',      type=int,           default = 8,
                    help='Parameters in encoding wheels_joints.')   
parser.add_argument('--touch_encode_size',              type=int,           default = 20,
                    help='Parameters in encoding image.')  
parser.add_argument('--touch_state_size',               type=int,           default = 20,
                    help='Parameters in prior and posterior inner-states.')

parser.add_argument('--vision_encode_size',             type=int,           default = 128,
                    help='Parameters in encoding image.')   
parser.add_argument('--vision_state_size',              type=int,           default = 128,
                    help='Parameters in prior and posterior inner-states.')

parser.add_argument('--prop_encode_size',               type=int,           default = 4,
                    help='Parameters in encoding image.')  
parser.add_argument('--prop_state_size',                type=int,           default = 4,
                    help='Parameters in prior and posterior inner-states.')

parser.add_argument('--char_encode_size',               type=int,           default = 8,
                    help='Parameters in encoding.')   
parser.add_argument('--voice_encode_size',              type=int,           default = 256,
                    help='Parameters in encoding voice.')   
parser.add_argument('--voice_state_size',               type=int,           default = 256,
                    help='Parameters in prior and posterior inner-states.')

parser.add_argument('--dropout',                        type=float,         default = .001,
                    help='Dropout percentage.')
parser.add_argument('--divisions',                      type=int,           default = 2,
                    help='How many times should RBGD_Out double size to image-size?')
parser.add_argument('--half',                           type=literal,       default = True,
                    help='Should the models use float16 instead of float32?')   



    # Training
parser.add_argument('--epochs',                         type=int,           default = 60000,
                    help='List of processors. Agent trains on each processor based on epochs in epochs parameter.')
parser.add_argument('--test_train_num',                 type=int,           default = 3,
                    help='Which collects of tasks/colors/shapes are used?')
parser.add_argument('--capacity',                       type=int,           default = 256,
                    help='How many episodes can the memory buffer contain.')
parser.add_argument('--batch_size',                     type=int,           default = 32, 
                    help='How many episodes are sampled for each epoch.')       
parser.add_argument('--weight_decay',                   type=float,         default = .00001,
                    help='Weight decay for modules.')       
parser.add_argument('--lr',                             type=float,         default = .0003,
                    help='Learning rate.')
parser.add_argument('--critics',                        type=int,           default = 2,
                    help='How many critics?')  
parser.add_argument('--tau',                            type=float,         default = .1,
                    help='Rate at which target-critics approach critics.')      
parser.add_argument('--GAMMA',                          type=float,         default = .9,
                    help='How heavily critics consider the future.')
parser.add_argument('--d',                              type=int,           default = 2,
                    help='Delay for training actors.') 



    # Entropy
parser.add_argument('--normal_alpha',                   type=float,         default = 0,
                    help='Nonnegative value, how much to consider policy prior.') 
parser.add_argument('--alpha',                          type=literal,       default = 0,
                    help='Nonnegative value, how much to consider entropy. Set to None to use target_entropy.')        
parser.add_argument('--target_entropy',                 type=float,         default = 0,
                    help='Target for choosing alpha if alpha set to None. Recommended: negative size of action-space.')      
parser.add_argument('--alpha_text',                     type=literal,       default = 0,
                    help='Nonnegative value, how much to consider entropy regarding agent voice. Set to None to use target_entropy_text.')        
parser.add_argument('--target_entropy_text',            type=float,         default = 0,
                    help='Target for choosing alpha_text if alpha_text set to None. Recommended: negative size of voice_out-space.')     



    # Curiosity
parser.add_argument('--std_min',                        type=int,           default = exp(-20),
                    help='Minimum value for standard deviation.')
parser.add_argument('--std_max',                        type=int,           default = exp(2),
                    help='Maximum value for standard deviation.')
parser.add_argument('--curiosity',                      type=str,           default = 'none',
                    help='Which kind of curiosity: none, prediction_error, or hidden_state.')  
parser.add_argument('--dkl_max',                        type=float,         default = 1,
                    help='Maximum value for clamping Kullback-Liebler divergence for hidden_state curiosity.')   



    # Vision
parser.add_argument('--vision_scaler',                  type=float,         default = 5, 
                    help='How much to consider vision prediction in accuracy compared to voice and touch.')   
parser.add_argument('--beta_vision',                    type=float,         default = .03, #.3,
                    help='Relative importance of complexity for vision.')
parser.add_argument('--prediction_error_eta_vision',    type=float,         default = 0,
                    help='Nonnegative value, how much to consider prediction_error curiosity for vision.')    
parser.add_argument('--hidden_state_eta_vision',        type=float,         default = 0,
                    help='Nonnegative values, how much to consider hidden_state curiosity for vision.') 



    # Touch
parser.add_argument('--touch_scaler',                   type=float,         default = .3, 
                    help='How much to consider touch prediction in accuracy compared to vision and voice.')   
parser.add_argument('--beta_touch',                     type=float,         default = .1, #.1,
                    help='Relative importance of complexity for touch.')     
parser.add_argument('--prediction_error_eta_touch',     type=float,         default = 0,
                    help='Nonnegative value, how much to consider prediction_error curiosity for touch.')   
parser.add_argument('--hidden_state_eta_touch',         type=float,         default = 0,
                    help='Nonnegative values, how much to consider hidden_state curiosity for touch.') 



    # Proprioception
parser.add_argument('--prop_scaler',                    type=float,         default = .01, 
                    help='How much to consider proprioception prediction in accuracy compared to vision and voice.')   
parser.add_argument('--beta_prop',                      type=float,         default = 1, #.3,
                    help='Relative importance of complexity for proprioception.')     
parser.add_argument('--prediction_error_eta_prop',      type=float,         default = 0,
                    help='Nonnegative value, how much to consider prediction_error curiosity for proprioception.')   
parser.add_argument('--hidden_state_eta_prop',          type=float,         default = 0,
                    help='Nonnegative values, how much to consider hidden_state curiosity for proprioception.') 



    # Command Voice
parser.add_argument('--voice_scaler',                   type=float,         default = 3,
                    help='How much to consider voice prediction in accuracy compared to vision and touch.') 
parser.add_argument('--beta_command_voice',             type=float,         default = .1,
                    help='Relative importance of complexity for command voice.')
parser.add_argument('--prediction_error_eta_command_voice', type=float,     default = 0,
                    help='Nonnegative value, how much to consider prediction_error curiosity for voice.')    
parser.add_argument('--hidden_state_eta_command_voice', type=float,         default = 0,
                    help='Nonnegative values, how much to consider hidden_state curiosity for voice.') 



    # Feedback Voice  
parser.add_argument('--beta_feedback_voice',            type=float,         default = .03,
                    help='Relative importance of complexity for feedback voice.')
parser.add_argument('--prediction_error_eta_feedback_voice', type=float,    default = 0,
                    help='Nonnegative value, how much to consider prediction_error curiosity for voice.')     
parser.add_argument('--hidden_state_eta_feedback_voice', type=float,        default = 0,
                    help='Nonnegative values, how much to consider hidden_state curiosity for voice.') 



    # Saving data
parser.add_argument('--keep_data',                      type=int,           default = 500,
                    help='How many epochs should pass before keeping data.')
parser.add_argument('--temp',                           type=literal,       default = False,
                    help='Should this use data saved temporarily?')      
parser.add_argument('--agents_for_plotting',            type=int,           default = 9999,
                    help='How many agents should be used in plotting?')      

parser.add_argument('--epochs_per_gen_test',            type=int,           default = 50,
                    help='How many epochs should pass before trying generalization test.')

parser.add_argument('--save_agents',                    type=literal,       default = False,
                    help='Do you save agents?')
parser.add_argument('--epochs_per_agent_save',          type=int,           default = 10000,
                    help='How many epochs should pass before saving agent model.')
parser.add_argument('--agents_per_agent_save',          type=int,           default = 2,
                    help='How many epochs should pass before saving agent model.')

parser.add_argument('--save_behaviors',                 type=literal,       default = True,
                    help='How many agents to save episodes.')
parser.add_argument('--episodes_per_behavior_analysis', type=int,           default = 10,
                    help='How many agents to save episodes.')
parser.add_argument('--agents_per_behavior_analysis',   type=int,           default = 1,
                    help='How many agents to save episodes.')

parser.add_argument('--save_compositions',              type=literal,       default = False,
                    help='How many agents to save episodes.')
parser.add_argument('--epochs_per_composition_data',    type=int,           default = 2500,
                    help='How many epochs should pass before saving an episode.')
parser.add_argument('--agents_per_composition_data',    type=int,           default = 2,
                    help='How many agents to save episodes.')



    # Currently testing
parser.add_argument('--pb_vector',              type=literal,       default = False,
                    help='True if separating command-voice.')
parser.add_argument('--complex_pb_vector',      type=literal,       default = True,
                    help='True if separating command-voice.')
parser.add_argument('--command_pb_size',        type=int,           default = 16,
                    help='Size of command\'s PB vector.')

# Make arguments.
try:
    default_args = parser.parse_args([])
    try:    
        args = parser.parse_args()
    except: 
        args, _ = parser.parse_known_args()
except:
    import sys 
    sys.argv=[''] 
    del sys           
    default_args = parser.parse_args([])
    try:    
        args = parser.parse_args()
    except: 
        args, _ = parser.parse_known_args()
    
    
    
# Checking how many sensors the robot has.
def get_num_sensors(robot_name):
    urdf_path = 'pybullet_data/robots/{}.urdf'.format(args.robot_name)
    physicsClient = p.connect(p.DIRECT)
    default_orn = p.getQuaternionFromEuler([0, 0, 0], physicsClientId = physicsClient)
    robot_index = p.loadURDF(urdf_path, (0, 0, 0), default_orn, useFixedBase=False, globalScaling = 1, physicsClientId = physicsClient)
    sensors = []
    for link_index in range(p.getNumJoints(robot_index, physicsClientId = physicsClient)):
        joint_info = p.getJointInfo(robot_index, link_index, physicsClientId = physicsClient)
        link_name = joint_info[12].decode('utf-8')  
        if 'sensor' in link_name:
            sensors.append(link_name)
    p.disconnect(physicsClientId = physicsClient)
    num_sensors = len(sensors)
    return(num_sensors, sensors)



# Based on arguments, adjust other arguments.
def update_args(arg_set):
    if arg_set.comp == 'deigo':
        arg_set.half = False
        
    arg_set.min_joint_1_angle = -arg_set.max_joint_1_angle
    arg_set.wheels_joints_shape = 4
       
    num_sensors, sensors = get_num_sensors(args.robot_name)
    arg_set.touch_shape = num_sensors
    arg_set.sensor_names = sensors
    arg_set.joint_aspects = 4
    
    arg_set.steps_per_epoch = arg_set.max_steps
    arg_set.voice_shape = len(voice_map)
    arg_set.obs_encode_size = arg_set.vision_encode_size + arg_set.touch_encode_size + arg_set.voice_encode_size
    arg_set.h_w_wheels_joints_size = arg_set.pvrnn_mtrnn_size + arg_set.wheels_joints_encode_size
    arg_set.h_w_action_size = arg_set.pvrnn_mtrnn_size + arg_set.wheels_joints_encode_size + arg_set.voice_encode_size
    
    allowed_task_dict = {
        1 : arg_set.watch,
        2 : arg_set.be_near,
        3 : arg_set.touch_top,
        4 : arg_set.push_forward,
        5 : arg_set.push_left,
        6 : arg_set.push_right}
    arg_set.allowed_tasks = [key for key, value in allowed_task_dict.items() if value]
    
    allowed_color_dict = {
        0 : arg_set.red,
        1 : arg_set.green,
        2 : arg_set.blue,
        3 : arg_set.cyan,
        4 : arg_set.magenta,
        5 : arg_set.yellow}
    arg_set.allowed_colors = [key for key, value in allowed_color_dict.items() if value]
    
    allowed_shape_dict = {
        0 : arg_set.pillar,
        1 : arg_set.pole,
        2 : arg_set.dumbbell,
        3 : arg_set.cone,
        4 : arg_set.hourglass}
    arg_set.allowed_shapes = [key for key, value in allowed_shape_dict.items() if value]

    return(arg_set)

for arg_set in [default_args, args]:
    default_args = update_args(default_args) 
    args = update_args(args)
    
    

# ---------------------------------------
# MAKE A TITLE FOR ARGUMENTS, COMPARED TO DEFAULT ARGUMENTS
# ---------------------------------------

    
        
# Don't include these parameters in title.
args_not_in_title = [
    'arg_title', 'id', 'agents', 'previous_agents', 'keep_data', 'epochs_per_pred_list', 
    'episodes_in_pred_list', 'agents_per_pred_list', 'epochs_per_pos_list', 'episodes_in_pos_list', 'agents_per_pos_list',
    'watch', 'be_near', 'touch_top', 'push_forward', 'push_left', 'push_right',
    'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 
    'pillar', 'pole', 'dumbbell', 'cone', 'hourglass']

# Make a title for the arguments. 
def get_args_title(default_args, args):
    if args.arg_title[:3] == '___': 
        return(args.arg_title)
    name = '' 
    first = True
    arg_list = list(vars(default_args).keys())
    arg_list.insert(0, arg_list.pop(arg_list.index('arg_name')))
    for arg in arg_list:
        if arg in args_not_in_title: 
            pass 
        else: 
            default = getattr(default_args, arg)
            try:
                this_time = getattr(args, arg)
            except:
                this_time = 'NONE'
            if this_time == default: 
                pass
            elif arg == 'arg_name':
                name += '{} ('.format(this_time)
            else: 
                if first: 
                    first = False
                else: 
                    name += ', '
                name += '{}: {}'.format(arg, this_time)
    if name == '': 
        name = 'default' 
    else:           
        name += ')'
    if name.endswith(' ()'): 
        name = name[:-3]
    parts = name.split(',')
    name = '' 
    line = ''
    for i, part in enumerate(parts):
        if len(line) > 50 and len(part) > 2: 
            name += line + '\n' 
            line = ''
        line += part
        if i+1 != len(parts): 
            line += ','
    name += line
    return(name)

args.arg_title = get_args_title(default_args, args)

# Generate folders for saving agents and plots.
save_file = f'saved_{args.comp}'
os.makedirs(f'{save_file}', exist_ok=True)
os.makedirs(f'{save_file}/thesis_pics', exist_ok=True)
os.makedirs(f'{save_file}/thesis_pics/final', exist_ok=True)
folder = f'{save_file}/{args.arg_name}'

if args.arg_title[:3] != '___' and not args.arg_name in ['default', 'finishing_dictionaries', 'plotting', 'plotting_predictions', 'plotting_positions']:
    os.makedirs(f'{folder}', exist_ok=True)
    os.makedirs(f'{folder}/agents', exist_ok=True)
    with open(f'{folder}/agents/args.pickle', 'wb') as handle:
        pickle.dump(args, handle)
if default_args.alpha == 'None': 
    default_args.alpha = None
if args.alpha == 'None':         
    args.alpha = None

# Print information about arguments.
if args == default_args: 
    print('Using default arguments.')
else:
    for arg in vars(default_args):
        default = getattr(default_args, arg)
        try:
            this_time = getattr(args, arg)
        except:
            this_time = 'NONE'
        if this_time != default:
            print('{}:\n\tDefault:\t{}\n\tThis time:\t{}'.format(arg, default, this_time))
        elif arg == 'device':
            print('{}:\n\tDefault:\t{}\n\tThis time:\t{}'.format(arg, default, this_time))
            
            
            
# If we are not showing durations, remove influence of this function.
if not args.show_duration:
    def print_duration(start_time, end_time, text = None, end_text = ''):
        pass
     


#%% 



""" 
For GUIs.
"""



def wait_for_button_press(button_label='Continue'):
    """Open a blocking Tkinter window with a button to resume execution."""
    def on_button_click():
        nonlocal continue_simulation
        continue_simulation = True
        root.destroy()

    root = tk.Tk()
    root.title('Wait for Input')
    root.geometry('200x100')
    button = tk.Button(root, text=button_label, command=on_button_click)
    button.pack(expand=True)
    continue_simulation = False
    root.mainloop()
    
    

def adjust_action(action_tensor):
    """Open a Tkinter window to adjust each element of an action tensor via sliders."""
    root = tk.Tk()
    root.title('Adjust Actions')
    shape = action_tensor.shape
    flat_action = action_tensor.view(-1).detach().numpy()
    num_elements = flat_action.size
    scales = []
    value_labels = []
    original_values = flat_action.copy()

    def update_value_label(val, label):
        label.config(text=f'{float(val):.2f}')

    def confirm():
        root.quit()

    def reset_to_original():
        for i, scale in enumerate(scales):
            scale.set(original_values[i])

    def reset_to_zero():
        for scale in scales:
            scale.set(0.0)

    for i in range(num_elements):
        frame = tk.Frame(root, padx=5, pady=5)
        frame.pack(fill=tk.X)
        label = tk.Label(frame, text=f'Action[{i}]')
        label.pack(side=tk.LEFT)
        current_val_label = tk.Label(frame, width=5, anchor='e')
        current_val_label.pack(side=tk.RIGHT)
        scale = tk.Scale(
            frame, from_=-1.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, length=300,
            command=lambda val, lbl=current_val_label: update_value_label(val, lbl))
        scale.set(flat_action[i])
        scale.pack(side=tk.RIGHT, padx=10)
        current_val_label.config(text=f'{scale.get():.2f}')
        scales.append(scale)
        value_labels.append(current_val_label)

    btn_frame = tk.Frame(root, pady=10)
    btn_frame.pack()
    tk.Button(btn_frame, text='Reset to Original', command=reset_to_original).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text='Reset to Zero', command=reset_to_zero).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text='Confirm', command=confirm).pack(side=tk.LEFT, padx=5)

    root.mainloop()
    updated_values = [scale.get() for scale in scales]
    root.destroy()
    return torch.tensor(updated_values).view(shape)



def plot_number_bars(numbers):
    """Plot a bar chart of motor commands with red (neg) and blue (pos) bars."""
    numbers = [n for n in numbers if n is not None]
    fontsize = 7
    plt.figure(figsize=(1.5, 1.5))
    plt.bar(range(len(numbers)), numbers, color=['red' if x < 0 else 'blue' for x in numbers])
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel('Index', fontsize=fontsize)
    plt.ylabel('Value', fontsize=fontsize)
    plt.title('Bar Plot of Motor Commands', fontsize=fontsize)
    plt.ylim(-1, 1)

    xticks = ['left wheel', 'right wheel']
    i = 1
    while(len(xticks) < len(numbers)):
        xticks.append(f'joint {i}')
        i += 1

    plt.xticks(range(len(xticks)), xticks, rotation=45, ha='right', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()




#%%



""" 
For various calculations.
"""



def wheels_joints_to_string(wheels_joints):
    """Convert tensor of wheels and joints into readable string format."""
    while(len(wheels_joints.shape) > 1):
        wheels_joints = wheels_joints.squeeze(0)
    print(f'\n\nIN WHEEL_JOINTS_TO_STRING: {wheels_joints}\n\n')
    string = f'Left Wheel: {round(wheels_joints[0].item(), 2)}'
    string += f'Right Wheel: {round(wheels_joints[1].item(), 2)}'
    string += f'Joint 1: {round(wheels_joints[2].item(), 2)}'
    if len(wheels_joints) == 4:
        string += f'Joint 2: {round(wheels_joints[3].item(), 2)}'
    return string



def relative_to(this, min, max):
    """Convert a value in [-1, 1] to the range [min, max]."""
    this = min + ((this + 1) / 2) * (max - min)
    this = [min, max, this]
    this.sort()
    return this[1]



def opposite_relative_to(this, min, max):
    """Convert a value in [min, max] to [-1, 1]."""
    return ((this - min) / (max - min)) * 2 - 1


    
def calculate_dkl(p_mu, p_std, q_mu, q_std):
    """Calculate Kullback-Leibler divergence between two Gaussians.
    DKL(Q||P) = .5 * ( (p_mu - q_mu)**2 / p_std**2 + q_std**2 / p_std**2 - log(q_std**2 / p_std**2) - 1 )
    """
    p_std = p_std ** 2
    q_std = q_std ** 2
    term_1 = (p_mu - q_mu) ** 2 / p_std
    term_2 = q_std / p_std
    term_3 = torch.log(term_2)
    out = 0.5 * (term_1 + term_2 - term_3 - 1)
    out = torch.nan_to_num(out)
    return out



def rolling_average(lst, window_size=500):
    """Compute rolling average over list, handling None values."""
    print('Rolling...', end=' ')
    try:
        new_list = [0 if lst[0] is None else float(lst[0])]
        for i in range(1, len(lst)):
            if lst[i] is None:
                new_list.append(new_list[-1])
            else:
                start_index = max(0, i - window_size + 1)
                window = [x for x in lst[start_index:i + 1] if x is not None]
                new_value = sum(window) / len(window) if window else 0
                new_list.append(new_value)
        return new_list
    except Exception as e:
        print('\n\nRolling average failed.\n\n')



#%%



""" 
Loading files for plotting, etc.
"""



def load_dicts(args):
    """Load plot_dicts and min_max_dicts for a given experiment (saved runs)."""
    if os.getcwd().split('/')[-1] != save_file:
        os.chdir(save_file)

    plot_dicts = []
    min_max_dicts = []

    if isinstance(args, dict):
        complete_order = args['titles']
    else:
        complete_order = args.arg_title[3:-3].split('+')

    order = [o for o in complete_order if o not in ['empty_space', 'break']]

    for name in order:
        print(f'Loading dictionaries for {name}...')
        got_plot_dicts = False
        got_min_max_dicts = False
        while not got_plot_dicts:
            with open(name + '/plot_dict.pickle', 'rb') as handle:
                plot_dicts.append(pickle.load(handle))
                got_plot_dicts = True
        while not got_min_max_dicts:
            try:
                with open(name + '/min_max_dict.pickle', 'rb') as handle:
                    min_max_dicts.append(pickle.load(handle))
                    got_min_max_dicts = True
            except:
                print(f'Stuck trying to get {name}\'s min_max_dicts...')
                sleep(1)

    print('Loaded all dicts! Making min/max dict...')

    min_max_dict = {}
    for key in plot_dicts[0].keys():
        if key not in [
            'args', 'arg_title', 'arg_name', 'all_task_names',
            'composition_data', 'component_data', 'episode_dicts',
            'agent_lists', 'spot_names', 'steps', 'goal_task',
            'all_processor_names', 'behavior'
        ]:
            if key == 'hidden_state':
                min_maxes = []
                for layer in range(len(min_max_dicts[0][key])):
                    minimum = None
                    maximum = None
                    for mm_dict in min_max_dicts:
                        if minimum is None or minimum > mm_dict[key][layer][0]:
                            minimum = mm_dict[key][layer][0]
                        if maximum is None or maximum < mm_dict[key][layer][1]:
                            maximum = mm_dict[key][layer][1]
                    min_maxes.append((minimum, maximum))
                min_max_dict[key] = min_maxes
            else:
                minimum = None
                maximum = None
                for mm_dict in min_max_dicts:
                    if mm_dict[key] != (None, None):
                        if minimum is None or minimum > mm_dict[key][0]:
                            minimum = mm_dict[key][0]
                        if maximum is None or maximum < mm_dict[key][1]:
                            maximum = mm_dict[key][1]
                min_max_dict[key] = (minimum, maximum)

    print('Made min/max dict!')

    final_complete_order = []
    final_plot_dicts = []

    for arg_name in complete_order:
        if arg_name in ['break', 'empty_space']:
            final_complete_order.append(arg_name)
        else:
            for plot_dict in plot_dicts:
                if plot_dict['args'].arg_name == arg_name or plot_dict['args'].arg_name + '_old' == arg_name:
                    final_complete_order.append(arg_name)
                    final_plot_dicts.append(plot_dict)

    while final_complete_order and final_complete_order[0] in ['break', 'empty_space']:
        final_complete_order.pop(0)

    print('Done with Load Dicts!')
    return final_plot_dicts, min_max_dict, complete_order
# %%
