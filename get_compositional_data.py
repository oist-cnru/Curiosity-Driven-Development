#%% 
import os
import pickle
import gzip
from math import pi
import numpy as np
import tkinter as tk
import pybullet as p

from utils import (
    make_objects_and_task, task_map, color_map, shape_map,
    testing_combos_1, testing_combos_2, testing_combos_3
)
from itertools import product
from processor import Processor
from models import Actor
from agent import Agent
print("Starting.")

set_goal = None
hyper_parameters = 'name_here'
agent_num = '0001'
epochs = '030000'
saved_file = 'saved_deigo'

load_path = f'{saved_file}/{hyper_parameters}/agents/agent_{agent_num}_epoch_{epochs}.pkl.gz'
with gzip.open(load_path, "rb") as f:
    agent = pickle.load(f)
    
print(agent)

agent.start_physics(GUI = False)



agent.args.agents_per_composition_data = -1
agent.epochs = int(epochs)

agent.all_processors = {
    f'{task_map[task].name}_{color_map[color].name}_{shape_map[shape].name}':
    Processor(
        agent.args, agent.arena_1, agent.arena_2,
        tasks_and_weights=[(task, 1)],
        objects=2, colors=[color], shapes=[shape], parenting=True
    )
    for task, color, shape in product(
        agent.args.allowed_tasks,
        agent.args.allowed_colors,
        agent.args.allowed_shapes
    )
}

agent.all_processor_names = list(agent.all_processors.keys())

agent.get_composition_data(episodes_per_goal = 5)


        
# Save.
with open(f"saved_deigo/{hyper_parameters}/plot_dict_agent_{str(agent_num).zfill(3)}_epoch_{epochs}_composition.pickle", "wb") as handle:
    pickle.dump(agent.plot_dict, handle)


# %%

