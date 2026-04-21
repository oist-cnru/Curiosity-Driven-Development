#%%

import os
import gzip
import pickle
import torch
import random
import numpy as np
from multiprocessing import Process, Queue, set_start_method
from time import sleep
from math import floor

from utils import args, folder, duration, estimate_total_duration, print
from utils import (
    folder, wheels_joints_to_string, cpu_memory_usage, duration, print_duration,
    wait_for_button_press, task_map, color_map, shape_map, task_name_list, print,
    To_Push, empty_goal, rolling_average, Obs, Action, get_goal_from_one_hots,
    Goal, adjust_action, testing_combos_1, testing_combos_2, testing_combos_3, exceptions_dict
)
from buffer import RecurrentReplayBuffer
from agent import Agent
from agent_lstm import Agent as Agent_lstm

print('\nname:\n{}'.format(args.arg_name))
print('\nagents: {}. previous_agents: {}.'.format(args.agents, args.previous_agents))


def train(q, i):
    """Train one agent (i) and send progress updates to queue q."""
    seed = args.init_seed + i
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    if str(args.device) != 'cpu':
        num_gpus = torch.cuda.device_count()
        gpu_id = i % num_gpus
        args.device = torch.device(f'cuda:{gpu_id}')

    num_cores = os.cpu_count()
    cpu_id = i % num_cores
    args.cpu = cpu_id

    print(f'\nagent {i}: cpu {cpu_id}\n')

    if args.load_agents:
        print("LOADING", i)
        with gzip.open(folder + '/agents/agent_' + str(i).zfill(4) + '.pkl.gz', 'rb') as handle:
            agent = pickle.load(handle)
        agent.args = args
        agent.memory = RecurrentReplayBuffer(agent.args)
        agent.plot_dict = {
            'args': agent.args,
            'arg_title': agent.args.arg_title,
            'arg_name': agent.args.arg_name,
            'all_processor_names': agent.all_processor_names,
            'testing_combos': (
                testing_combos_1 if agent.args.test_train_num == 1
                else testing_combos_2 if agent.args.test_train_num == 2
                else testing_combos_3
            ),

            'division_epochs': [],
            'steps': [],
            'behavior': {},
            'composition_data': {},

            'accuracy_loss': [],
            'complexity_loss': [],
            'vision_loss': [],
            'touch_loss': [],
            'prop_loss': [],
            'command_voice_loss': [],
            'feedback_voice_loss': [],

            'actor_loss': [],
            'critics_loss': [[] for _ in range(agent.args.critics)],

            'alpha_loss': [],
            'alpha_text_loss': [],

            'reward': [],
            'gen_reward': [],
            'q': [],
            'extrinsic': [],

            'intrinsic_curiosity': [],
            'intrinsic_entropy': [],

            'vision_prediction_error_curiosity': [],
            'touch_prediction_error_curiosity': [],
            'prop_prediction_error_curiosity': [],
            'command_voice_prediction_error_curiosity': [],
            'feedback_voice_prediction_error_curiosity': [],
            'prediction_error_curiosity': [],

            'vision_hidden_state_curiosity': [],
            'touch_hidden_state_curiosity': [],
            'prop_hidden_state_curiosity': [],
            'command_voice_hidden_state_curiosity': [],
            'feedback_voice_hidden_state_curiosity': [],
            'hidden_state_curiosity': [],

            'wins_all': [],
            'gen_wins_all': []
        }

        # Add keys per task
        for t in task_map.values():
            agent.plot_dict[f'wins_{t.name}'] = []
            agent.plot_dict[f'gen_wins_{t.name}'] = []

        agent.plot_dict['wins_exception'] = []
        agent.start_physics()
    else:
        if args.lstm:
            agent = Agent_lstm(args=args, i=i)
        else:    
            agent = Agent(args=args, i=i)

    agent.training(q)


if __name__ == '__main__':
    """Main entry point for multi-agent training."""
    set_start_method('spawn')  # Required for multiprocessing
    queue = Queue()
    processes = []

    for worker_id in range(1 + args.previous_agents, 1 + args.agents + args.previous_agents):
        process = Process(target=train, args=(queue, worker_id))
        processes.append(process)
        process.start()

    # Progress tracking
    progress_dict      = {i: '0'   for i in range(1 + args.previous_agents, 1 + args.agents + args.previous_agents)}
    prev_progress_dict = {i: None for i in range(1 + args.previous_agents, 1 + args.agents + args.previous_agents)}

    while any(process.is_alive() for process in processes) or not queue.empty():
        while not queue.empty():
            worker_id, progress_percentage = queue.get()
            progress_dict[worker_id] = progress_percentage

        # If there's been any progress update, print the new state.
        if any(progress_dict[k] != prev_progress_dict[k] for k in progress_dict):
            prev_progress_dict = progress_dict.copy()

            values = list(progress_dict.values())
            values.sort()
            so_far = duration()
            lowest = float(values[0])
            estimated_total = estimate_total_duration(lowest)
            to_do = '?:??:??' if estimated_total == '?:??:??' else estimated_total - so_far

            values_display = []
            hundreds = 0
            for value in values:
                val_str = str(floor(100 * float(value))).ljust(3, ' ')
                if val_str == '100':
                    hundreds += 1
                else:
                    values_display.append(val_str)

            bar = ' '.join(values_display)
            if hundreds > 0:
                bar += ' ##' + ' 100' * hundreds
            if hundreds == 0:
                bar += ' ##'
            bar = f'{so_far} ({to_do} left):\t' + bar.rstrip() + '.'

            print(bar)

        sleep(15)

    for process in processes:
        process.join()

    print('\nDuration: {}. Done!'.format(duration()))
