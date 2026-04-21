#%%
import os
import pickle
import gzip
from math import pi

import tkinter as tk
import pybullet as p

from utils import make_objects_and_task
from processor import Processor
from models import Actor
from agent import Agent 


# -------------------------------
#  INITIAL SETUP
# -------------------------------

# If the user has no specific goal in mind, leave this as None
set_goal = None

# Change these to the agent you would like to test
hyper_parameters = 'name_here'
agent_num = '0001'
epochs = '060000'
saved_file = 'saved_deigo'

print('\n\nLoading default agent...', end = ' ')

load_path = f'{saved_file}/{hyper_parameters}/agents/agent_{agent_num}_epoch_{epochs}.pkl.gz'
with gzip.open(load_path, "rb") as f:
    agent = pickle.load(f) 

agent.start_physics(GUI = True)

episodes = 0
wins = 0

print('Ready to go!')



#%%
# -------------------------------
#  SWAP AGENT + PROPAGATE ARGS
# -------------------------------

hyper_parameters = 'name_here'
agent_num = '0001'
epochs = '060000'
saved_file = 'saved_deigo'

def change_agent(hyper_parameters, agent_num, epochs, saved_file = 'saved_deigo'):
    print('\n\nLoading new agent...', end = ' ')
    load_path = f'{saved_file}/{hyper_parameters}/agents/agent_{agent_num}_epoch_{epochs}.pkl.gz'

    load_path = f'{saved_file}/{hyper_parameters}/agents/args.pickle'
    with open(load_path, 'rb') as f:
        new_args = pickle.load(f)
        
    load_path = f'{saved_file}/{hyper_parameters}/agents/agent_{agent_num}_epoch_{epochs}.pkl.gz'
    with gzip.open(load_path, 'rb') as f:
        state_dict = pickle.load(f)()
    change_args(new_args)

    global episodes, wins
    episodes = 0
    wins = 0
    print('Ready to go!')


# Some arguments are only relevant in the arena or processor
def change_args(new_args):
    agent.args = new_args

    agent.arena_1.args = new_args
    agent.arena_1.change_physicsClient()

    agent.arena_2.args = new_args
    agent.arena_2.change_physicsClient()


change_agent(hyper_parameters, agent_num, epochs)



#%%
# -------------------------------
#  DEFINE SPECIFIC GOAL
# -------------------------------

set_goal = make_objects_and_task(
    num_objects = agent.processors['all'].objects,
    allowed_tasks_and_weights = agent.processors['all'].tasks_and_weights,
    allowed_colors = agent.processors['all'].colors,
    allowed_shapes = agent.processors['all'].shapes,
    test_train_num = agent.processors['all'].args.test_train_num,
    test = None
)

# Print details about the goal
print(set_goal)
print(set_goal[0].name)
print(set_goal[1][0][0].name)
print(set_goal[1][0][1].name)
print(set_goal[1][1][0].name)
print(set_goal[1][1][1].name)



#%%
# -------------------------------
#  RUN ONE EPISODE
# -------------------------------

#   Tasks:
#   0 - Free Play
#   1 - Watch
#   2 - Be Near
#   3 - Touch the Top
#   4 - Push
#   5 - Push Left
#   6 - Push Right

#   Colors:
#   0 - Red
#   1 - Green
#   2 - Blue
#   3 - Cyan
#   4 - Magenta
#   5 - Yellow

#   Shapes:
#   0 - Pillar
#   1 - Pole
#   2 - Dumbbell
#   3 - Cone
#   4 - Hourglass

agent.processors = {
    0: Processor(
        agent.args,
        agent.arena_1,
        agent.arena_2,
        tasks_and_weights = [(6, 1)],     
        objects = 2,
        colors = [0, 1, 2, 3, 4, 5],
        shapes = [0, 1, 2, 3, 4],
        parenting = True
    )
}

agent.processor_name = 0

episodes += 1

win = agent.save_episodes(
    test = False,                    # Use training or test objects?
    verbose = False,                # Print extra info?
    display = False,                # Full plotting of model internals
    video_display = True,           # Plot observation video
    sleep_time = 0.25,              # Delay per step
    waiting = False,                # Wait for user input per step?
    user_action = False,            # Manual control?
    dreaming = False,               # Run hallucinated?
    set_positions = None, #([0, 5], [0, -5]),  # Object positions
    set_goal = set_goal             # Specific goal
)

if win:
    wins += 1

# print(f'\tWIN RATE: {round(100 * (wins / episodes), 2)}% \t ({wins} wins out of {episodes} episodes)')
# %%
