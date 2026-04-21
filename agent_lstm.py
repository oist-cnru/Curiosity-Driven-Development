#%%

# ============================
# IMPORTS
# ============================

# Standard libraries
import os
import sys
import gzip
import zipfile
import pickle
from time import sleep
from math import log
from copy import deepcopy
from itertools import accumulate, product
from collections.abc import Mapping, Container

# Third-party libraries
import psutil
import numpy as np
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim

# Local modules
from utils import (
    folder, wheels_joints_to_string, cpu_memory_usage, duration, print_duration,
    wait_for_button_press, task_map, color_map, shape_map, task_name_list, print,
    To_Push, empty_goal, rolling_average, Obs, Action, get_goal_from_one_hots,
    Goal, adjust_action, testing_combos_1, testing_combos_2, testing_combos_3, exceptions_dict
)
from utils_submodule import model_start
from arena import Arena, get_physics
from processor import Processor
from buffer import RecurrentReplayBuffer
from pvrnn import PVRNN
from models_lstm import Actor, Critic
from plotting_step import plot_step
from plotting_for_video import plot_video_step


# ============================
# SYSTEM MONITORING FUNCTIONS
# ============================

def print_cpu_usage(string, num):
    """
    Prints the CPU affinity and usage statistics for the current process.
    """
    pid = os.getpid()
    process = psutil.Process(pid)
    cpu_affinity = process.cpu_affinity()
    print(f'{string}: {num} CPU affinity: {cpu_affinity}')
    current_cpu = psutil.cpu_percent(interval=1, percpu=True)
    print(f'{string}: {num} Current CPU usage per core: {current_cpu}')


def sizeof_fmt(num, suffix='B'):
    """
    Convert a byte size into a human-readable string (KB, MB, GB, etc.).
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if abs(num) < 1024.0:
            return f'{num:.2f} {unit}{suffix}'
        num /= 1024.0
    return f'{num:.2f} P{suffix}'


# ============================
# TASK WEIGHT UTILITIES (Unused)
# ============================

def get_uniform_weight(first_weight, num_weights):
    remaining_sum = 100 - first_weight
    uniform_weight = remaining_sum / (num_weights - 1)
    return uniform_weight


def make_tasks_and_weights(first_weight):
    u = get_uniform_weight(first_weight, 6)
    return [(0, first_weight)] + [(v, u) for v in [1, 2, 3, 4, 5]]


fwpulr_tasks_and_weights = make_tasks_and_weights(50)



# ============================
# AGENT CLASS
# ============================

class Agent:
    """
    RL Agent containing actor, critic, memory, and task-specific processors.

    Attributes:
        args: Namespace of configuration arguments.
        agent_num: Identifier for the agent instance.
        actor, critics: Policy and value networks.
        memory: Replay buffer for training.
        plot_dict: Data logger for training and evaluation metrics.
    """

    def __init__(self, args, i=-1, GUI=False):

        self.args = args
        self.agent_num = i
        self.agent_name = f'{self.args.arg_name}_{self.agent_num}'

        # Tracking overall progress
        self.total_steps = 0
        self.total_episodes = 0
        self.total_epochs = 0
        
        self.steps = 0
        self.episodes = 0
        self.epochs = 0

        # Reward inflation config
        self.reward_inflation = 0
        if self.args.reward_inflation_type == 'None':
            self.reward_inflation = 1

        self.hidden_state_eta_feedback_voice_reduction = 1

        # Device info
        if self.args.device.type == 'cuda':
            print(
                f'\nIN AGENT: {i} DEVICE: {self.args.device} '
                f'({torch.cuda.current_device()} out of {[j for j in range(torch.cuda.device_count())]}, '
                f'{torch.cuda.get_device_name(torch.cuda.current_device())})\n'
            )
        else:
            print(f'\nIN AGENT: {i} DEVICE: {self.args.device}\n')

        self.start_physics(GUI)

        # ----------------------------------------
        # Initialize processors
        # ----------------------------------------
        self.processors = {
            'all': Processor(
                self.args,
                self.arena_1,
                self.arena_2,
                tasks_and_weights=[(t, 1) for t in self.args.allowed_tasks],
                objects=2,
                colors=[c for c in self.args.allowed_colors],
                shapes=[s for s in self.args.allowed_shapes],
                parenting=True,
                full_name='All Tasks'
            )
        }

        self.all_processors = {
            f'{task_map[task].name}_{color_map[color].name}_{shape_map[shape].name}':
            Processor(
                self.args,
                self.arena_1,
                self.arena_2,
                tasks_and_weights=[(task, 1)],
                objects=2,
                colors=[color],
                shapes=[shape],
                parenting=True
            )
            for task, color, shape in product(
                [t for t in self.args.allowed_tasks],
                [c for c in self.args.allowed_colors],
                [s for s in self.args.allowed_shapes]
            )
        }

        self.all_processor_names = list(self.all_processors.keys())

        # ----------------------------------------
        # Entropy temperature α
        # ----------------------------------------
        self.target_entropy = self.args.target_entropy
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)

        self.alpha_opt = optim.Adam(
            params=[self.log_alpha],
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

        if self.args.half:
            self.log_alpha = self.log_alpha.to(dtype=torch.float16)

        # Text entropy α (if used)
        self.target_entropy_text = self.args.target_entropy_text
        self.alpha_text = 1
        self.log_alpha_text = torch.tensor([0.0], requires_grad=True)

        self.alpha_text_opt = optim.Adam(
            params=[self.log_alpha_text],
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

        if self.args.half:
            self.log_alpha_text = self.log_alpha_text.to(dtype=torch.float16)

        # ----------------------------------------
        # Models
        # ----------------------------------------

        # Actor
        self.actor = Actor(self.args)
        self.actor_opt = optim.Adam(
            self.actor.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

        # Critics
        self.critics = []
        self.critic_targets = []
        self.critic_opts = []

        for _ in range(self.args.critics):
            self.critics.append(Critic(self.args))
            self.critic_targets.append(Critic(self.args))
            self.critic_targets[-1].load_state_dict(self.critics[-1].state_dict())
            self.critic_opts.append(
                optim.Adam(
                    self.critics[-1].parameters(), 
                    lr=self.args.lr, 
                    weight_decay = self.args.weight_decay))

        self.memory = RecurrentReplayBuffer(self.args)

        # ----------------------------------------
        # Plotting dictionary
        # ----------------------------------------
        self.plot_dict = {
            'args': self.args,
            'arg_title': self.args.arg_title,
            'arg_name': self.args.arg_name,
            'all_processor_names': self.all_processor_names,
            'testing_combos': (
                testing_combos_1 if self.args.test_train_num == 1
                else testing_combos_2 if self.args.test_train_num == 2
                else testing_combos_3
            ),

            'division_epochs': [],
            'steps': [],
            'behavior': {},
            'composition_data': {},

            'actor_loss': [],
            'critics_loss': [[] for _ in range(self.args.critics)],

            'alpha_loss': [],
            'alpha_text_loss': [],

            'reward': [],
            'gen_reward': [],
            'q': [],
            'extrinsic': [],

            'intrinsic_entropy': [],

            'wins_all': [],
            'gen_wins_all': []
        }

        # Add keys per task
        for t in task_map.values():
            self.plot_dict[f'wins_{t.name}'] = []
            self.plot_dict[f'gen_wins_{t.name}'] = []

        self.plot_dict['wins_exception'] = []



    def start_physics(self, GUI=False):
        """
        Initialize simulation arenas and counters.
        """

        self.arena_1 = Arena(GUI=GUI, args=self.args)
        self.arena_2 = Arena(GUI=False, args=self.args)

        self.processor_name = self.args.processor
        
        
        
    def regular_checks(self, force=False, swapping=False, sleep_time=None):
        """
        Save agent parameters, collect compositions, test generalization.
        """
        if self.args.save_agents:
            if (
                (self.agent_num <= self.args.agents_per_agent_save and self.epochs % self.args.epochs_per_agent_save == 0) or
                (self.agent_num <= self.args.agents_per_agent_save and force)
            ):
                self.save_agent()

        if self.epochs % self.args.epochs_per_gen_test == 0 or force:
            self.gen_test(sleep_time=sleep_time)
            
        
        
    def training(self, q=None, sleep_time=None):
        """
        Begin training and track progression.
        """
        self.regular_checks(sleep_time=sleep_time)

        while True:
            self.training_episode(sleep_time=sleep_time)

            # Early check via GUI ping
            if self.check_ping():
                self.save_dicts()

            percent_done = str(self.epochs / self.args.epochs)
            if q is not None:
                q.put((self.agent_num, percent_done))

            if self.epochs >= self.args.epochs:
                linestyle = self.processors[self.processor_name].linestyle
                full_name = self.processors[self.processor_name].full_name
                self.plot_dict['division_epochs'].append((self.total_epochs, linestyle, full_name))
                break

            self.regular_checks(sleep_time=sleep_time)

        self.regular_checks(force=True)
        self.save_dicts(final=True)
        
        

    def check_ping(self):
        """
        Check for an alert file and remove it.
        Used for GUI-based early analysis triggers.
        """
        file_path = os.path.join(folder, self.agent_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            return True
        return False
        
        
        
    def save_dicts(self, final=False):
        """
        Save dictionaries of plotting data and compute min/max statistics.
        """
        self.plot_dict['accumulated_reward'] = list(accumulate(self.plot_dict['reward']))
        self.plot_dict['accumulated_gen_reward'] = list(accumulate(self.plot_dict['gen_reward']))

        for task_name in task_name_list + ['all']:
            self.plot_dict['rolled_wins_' + task_name] = rolling_average(
                self.plot_dict['wins_' + task_name], window_size=500)
            self.plot_dict['rolled_gen_wins_' + task_name] = rolling_average(
                self.plot_dict['gen_wins_' + task_name], window_size=500)

        self.plot_dict['rolled_wins_exception'] = rolling_average(
            self.plot_dict['wins_exception'], window_size=500)

        # Generate min/max statistics
        self.min_max_dict = {key: [] for key in self.plot_dict.keys()}

        for key in self.min_max_dict.keys():
            if key in [
                'args', 'arg_title', 'arg_name',
                'all_processor_names', 
                'episode_dicts', 'agent_lists', 'spot_names',
                'steps', 'behavior'
            ]:
                continue

            l = deepcopy(self.plot_dict[key])
            l = [_ for _ in l if _ is not None]
            if l:
                minimum = min(l)
                maximum = max(l)
            else:
                minimum = maximum = None
            self.min_max_dict[key] = (minimum, maximum)

        file_end = str(self.agent_num).zfill(3)
        if not final:
            file_end = f'temp_{file_end}'

        with open(f'{folder}/plot_dict_{file_end}.pickle', 'wb') as handle:
            pickle.dump(self.plot_dict, handle)

        with open(f'{folder}/min_max_dict_{file_end}.pickle', 'wb') as handle:
            pickle.dump(self.min_max_dict, handle)
                
    
    
    def get_agent_obs(self, agent_1=True):
        """
        Get observations from the current processor for the specified agent.
        """
        parenting = self.processor.parenting
        if parenting and not agent_1:
            return None

        obs = self.processor.obs(agent_1)
        obs.command_voice = obs.command_voice.one_hots.unsqueeze(0).unsqueeze(0)
        obs.feedback_voice = obs.feedback_voice.one_hots.unsqueeze(0).unsqueeze(0)
        return obs
                
                

    def step_in_episode(
        self,
        prev_action_1, obs_1, actor_h_1, critic_h_1,
        prev_action_2, obs_2, actor_h_2, critic_h_2,
        verbose=False, sleep_time=None, user_action=False
    ):
        """
        Perform one step in an episode for both agents. Collect actions, rewards, and transition data.
        """
                
        with torch.no_grad():
            self.eval()
            parenting = self.processor.parenting
                              
            def agent_step(agent_1=True):
                """
                Inner function to compute a step for one agent.
                """
                if parenting and not agent_1:
                    return (
                        None, None, actor_h_2, critic_h_2, None
                    )

                prev_action = prev_action_1 if agent_1 else prev_action_2
                partner_prev_voice_out = (
                    prev_action_2.voice_out if agent_1 and not parenting
                    else torch.zeros((1, 1, self.args.max_voice_len, self.args.voice_shape))
                    if agent_1 else prev_action_1.voice_out
                )

                obs = obs_1 if agent_1 else obs_2
                actor_h = actor_h_1 if agent_1 else actor_h_2
                critic_h = critic_h_1 if agent_1 else critic_h_2


                if isinstance(obs.command_voice, Goal):
                    obs.command_voice = obs.command_voice.one_hots
                if isinstance(obs.feedback_voice, Goal):
                    obs.feedback_voice = obs.feedback_voice.one_hots

                action, _, _, actor_h = self.actor(obs, prev_action, actor_h, parenting)

                if user_action:
                    user_wheels_joints = adjust_action(action.wheels_joints)
                    action.wheels_joints = user_wheels_joints

                values = []
                for i in range(self.args.critics):
                    value, critic_h[i] = self.critics[i](obs, action, critic_h[i])
                    values.append(round(value.item(), 3))

                return (
                    obs, action, actor_h, critic_h, values,
                )
            
            # Run agent steps
            obs_1, action_1, actor_h_1, critic_h_1, values_1 = agent_step()
            obs_2, action_2, actor_h_2, critic_h_2, values_2 = agent_step(agent_1=False)

            # Run the environment step
            reward, done, win = self.processor.step(
                action_1.wheels_joints[0, 0].clone(),
                None if action_2 is None else action_2.wheels_joints[0, 0].clone(),
                sleep_time=sleep_time,
                verbose=verbose
            )

            # Adjust reward if using inflation
            reward *= self.reward_inflation

            def next_agent_step(agent_1=True):
                """
                Get next observation and transition for the agent.
                """
                if parenting and not agent_1:
                    return None, None

                next_obs = self.processor.obs(agent_1)
                obs = obs_1 if agent_1 else obs_2
                action = action_1 if agent_1 else action_2

                next_obs.command_voice = next_obs.command_voice.one_hots.unsqueeze(0).unsqueeze(0)
                next_obs.feedback_voice = next_obs.feedback_voice.one_hots.unsqueeze(0).unsqueeze(0)

                to_push = To_Push(obs, action, reward, next_obs, done)
                return next_obs, to_push

            # Get next states and transitions
            next_obs_1, to_push_1 = next_agent_step()
            next_obs_2, to_push_2 = next_agent_step(agent_1=False)

        torch.cuda.empty_cache()
        
        # Return all outputs
        return (
            action_1, values_1, actor_h_1, critic_h_1,
            action_2, values_2, actor_h_2, critic_h_2,
            reward, done, win, to_push_1, to_push_2
        )
            
           
           
    def start_episode(self):
        """
        Begin an episode, initializing states and actions.
        """
        done = False
        complete_reward = 0
        steps = 0

        def start_agent(agent_1=True):
            to_push_list = []
            prev_action = Action(
                torch.zeros((1, 1, self.args.wheels_joints_shape)),
                torch.zeros((1, 1, self.args.max_voice_len, self.args.voice_shape))
            )
            actor_h = torch.zeros((1, 1, self.args.pvrnn_mtrnn_size))
            critic_h = [torch.zeros((1, 1, self.args.pvrnn_mtrnn_size))] * len(self.critics)
            return to_push_list, prev_action, actor_h, critic_h

        return done, complete_reward, steps, start_agent(), start_agent(agent_1=False)
           
           
    
    def training_episode(self, sleep_time=None):
        """
        Run a full training episode: simulate, step through actions,
        record transitions, and update statistics.
        """
        # Initialize agents
        done, complete_reward, steps, \
            (to_push_list_1, prev_action_1, actor_h_1, critic_h_1), \
            (to_push_list_2, prev_action_2, actor_h_2, critic_h_2) = self.start_episode()

        start_time = duration()
        self.episodes += 1
        self.total_episodes += 1

        # Initialize behavior tracking
        self.plot_dict['behavior'][self.episodes] = []

        # Select processor
        self.processor = self.processors[self.processor_name]
        self.processor.begin()

        for step in range(self.args.max_steps):
            self.steps += 1
            self.total_steps += 1

            if not done:
                steps += 1

                # Get observations and run one step
                obs_1 = self.get_agent_obs()
                obs_2 = self.get_agent_obs(agent_1=False)

                prev_action_1, values_1, actor_h_1, critic_h_1, \
                    prev_action_2, values_2, actor_h_2, critic_h_2, \
                    reward, done, win, to_push_1, to_push_2 = self.step_in_episode(
                        prev_action_1, obs_1, actor_h_1, critic_h_1, 
                        prev_action_2, obs_2, actor_h_2, critic_h_2, 
                        sleep_time=sleep_time
                    )

                # Behavior tracking
                if self.args.save_behaviors:
                    if self.args.agents_per_behavior_analysis == -1 or self.agent_num <= self.args.agents_per_behavior_analysis:
                        if self.episodes == 0 or self.episodes % self.args.episodes_per_behavior_analysis == 0:
                            self.plot_dict['behavior'][self.episodes].append(
                                get_goal_from_one_hots(to_push_1.next_obs.feedback_voice)
                            )

                # Push transitions
                to_push_list_1.append(to_push_1)
                to_push_list_2.append(to_push_2)
                complete_reward += reward

            # Epoch-level training
            if self.steps % self.args.steps_per_epoch == 0:
                self.epoch(self.args.batch_size)

        # Finish episode
        self.processor.done()
        self.plot_dict['steps'].append(steps)
        self.plot_dict['reward'].append(complete_reward)

        # Task-specific win tracking
        goal_task = self.processor.goal.task.name
        exception_list_a, exception_list_b = exceptions_dict[self.args.exceptions]
        if self.processor.goal.digits in exception_list_a:
            goal_task = 'exception'

        self.plot_dict['wins_all'].append(win)
        for task_name in task_name_list + ['exception']:
            if task_name == goal_task:
                self.plot_dict['wins_' + task_name].append(win)
            else:
                self.plot_dict['wins_' + task_name].append(None)

        # Push transitions to memory
        for to_push in to_push_list_1:
            to_push.push(self.memory)

        for to_push in to_push_list_2:
            if to_push is not None:
                to_push.push(self.memory)

        # Progress percentage
        percent_done = self.epochs / self.args.epochs

        # Curiosity tracking adjustment (optional)
        if self.args.hidden_state_eta_feedback_voice_reduction_type == 'linear':
            self.hidden_state_eta_feedback_voice_reduction = 1 - percent_done
        if self.args.hidden_state_eta_feedback_voice_reduction_type.startswith('exp'):
            exp = float(self.args.hidden_state_eta_feedback_voice_reduction_type.split('_')[-1])
            self.hidden_state_eta_feedback_voice_reduction = 1 - (percent_done ** exp)
        if self.args.hidden_state_eta_feedback_voice_reduction_type.startswith('sigmoid'):
            k = float(self.args.hidden_state_eta_feedback_voice_reduction_type.split('_')[-1])
            self.hidden_state_eta_feedback_voice_reduction = 1 - (
                1 / (1 + np.exp(-k * (self.epochs - self.args.epochs / 2)))
            )

        # Reward inflation adjustment (optional)
        if self.args.reward_inflation_type == 'linear':
            self.reward_inflation = percent_done
        if self.args.reward_inflation_type.startswith('exp'):
            exp = float(self.args.reward_inflation_type.split('_')[-1])
            self.reward_inflation = percent_done ** exp
        if self.args.reward_inflation_type.startswith('sigmoid'):
            k = float(self.args.reward_inflation_type.split('_')[-1])
            self.reward_inflation = (
                1 / (1 + np.exp(-k * (self.epochs - self.args.epochs / 2)))
            )

        end_time = duration()
        print_duration(start_time, end_time, '\nTraining episode', '\n')

        return step
        
        
        
    def gen_test(self, sleep_time=None):
        """
        Episode for testing generalization.
        This is the same as a training episode, but without pushing transitions
        and with different result tracking.
        """
        done, complete_reward, steps, \
            (to_push_list_1, prev_action_1, actor_h_1, critic_h_1), \
            (to_push_list_2, prev_action_2, actor_h_2, critic_h_2) = self.start_episode()

        try:
            self.processor = self.processors[self.processor_name]
            self.processor.begin(test=True)

            for step in range(self.args.max_steps):
                if not done:
                    obs_1 = self.get_agent_obs()
                    obs_2 = self.get_agent_obs(agent_1=False)

                    prev_action_1, values_1, actor_h_1, critic_h_1, \
                        prev_action_2, values_2, actor_h_2, critic_h_2, \
                        reward, done, win, to_push_1, to_push_2 = self.step_in_episode(
                            prev_action_1, obs_1, actor_h_1, critic_h_1, 
                            prev_action_2, obs_2, actor_h_2, critic_h_2, 
                            sleep_time=sleep_time
                        )

                    complete_reward += reward

            self.processor.done()
            goal_task = self.processor.goal.task.name

            self.plot_dict['gen_wins_all'].append(win)
            for task_name in task_name_list:
                if task_name == goal_task:
                    self.plot_dict['gen_wins_' + task_name].append(win)
                else:
                    self.plot_dict['gen_wins_' + task_name].append(None)

        except Exception:
            complete_reward = 0
            win = False
            self.plot_dict['gen_wins_all'].append(None)
            for task_name in task_name_list:
                self.plot_dict['gen_wins_' + task_name].append(win)

        self.plot_dict['gen_reward'].append(complete_reward)
        return win
    
    
    
    def save_episodes(
        self, test=False, verbose=False, display=True,
        video_display=True, sleep_time=None, waiting=False,
        user_action=False, dreaming=False,
        set_positions=None, set_goal=None
    ):
        """
        Run a single episode and save detailed visual/logging data for debugging or demos.
        Supports real-time display and mental planning ('dreaming').
        """
        with torch.no_grad():
            self.processor = self.processors[self.processor_name]
            self.processor.begin(test=test, set_positions=set_positions, set_goal=set_goal)
            parenting = self.processor.parenting

            # Initialize episode dictionary
            common_keys = [
                'obs', 'action', 'birds_eye', 'reward', 'critic_predictions'
            ]
            episode_dict = {f'{key}_{agent_id}': [] for agent_id in [0, 1] for key in common_keys}
            episode_dict['reward'] = []
            episode_dict['processor'] = self.processor
            self.processor.goal.make_texts()
            episode_dict['goal'] = self.processor.goal

            # Episode state
            done, complete_reward, steps, \
                (to_push_list_1, prev_action_1, actor_h_1, critic_h_1), \
                (to_push_list_2, prev_action_2, actor_h_2, critic_h_2) = self.start_episode()

            # Save real observations per step
            def save_step(step, obs, agent_1=True):
                agent_num = 1 if agent_1 else 2
                birds_eye = self.processor.arena_1.photo_from_above() if agent_1 else self.processor.arena_2.photo_from_above()
                obs.command_voice = (
                    obs.command_voice if parenting else
                    prev_action_2.voice_out if agent_1 else prev_action_1.voice_out
                )
                if not isinstance(obs.command_voice, Goal):
                    obs.command_voice = get_goal_from_one_hots(obs.command_voice)
                if not isinstance(obs.feedback_voice, Goal):
                    obs.feedback_voice = get_goal_from_one_hots(obs.feedback_voice)

                episode_dict[f'obs_{agent_num}'].append(obs)
                #episode_dict[f'birds_eye_{agent_num}'].append(birds_eye[:, :, 0:3])

            def display_step(step, agent_1=True, done=False, stopping=False):
                return

            def video_display_step(step, agent_1=True, done=False, stopping=False):
                return

            # Episode loop
            for step in range(self.args.max_steps + 1):
                # Get real or imagined obs
                obs_1 = self.get_agent_obs()
                obs_2 = self.get_agent_obs(agent_1=False)

                save_step(step, obs_1, agent_1=True)
                if not parenting:
                    save_step(step, obs_2, agent_1=False)

                display_step(step)
                video_display_step(step)

                # Act and collect predictions
                prev_action_1, values_1, actor_h_1, critic_h_1, \
                    prev_action_2, values_2, actor_h_2, critic_h_2, \
                    reward, done, win, to_push_1, to_push_2 = self.step_in_episode(
                        prev_action_1, obs_1, actor_h_1, critic_h_1, 
                        prev_action_2, obs_2, actor_h_2, critic_h_2, 
                        sleep_time=sleep_time
                    )

                episode_dict['reward'].append(str(round(reward, 3)))

                def update_episode_dict(index, prev_action, values, reward):
                    episode_dict[f'action_{index}'].append(prev_action)
                    episode_dict[f'critic_predictions_{index}'].append(values)
                    episode_dict[f'reward_{index}'].append(str(round(reward, 3)))

                update_episode_dict(1, prev_action_1, values_1, reward)
                if not parenting:
                    update_episode_dict(2, prev_action_2, values_2, reward)

                if done:
                    obs_1 = self.get_agent_obs()
                    obs_2 = self.get_agent_obs(agent_1=False)
                    save_step(step, obs_1, agent_1=True)
                    if not parenting:
                        save_step(step, obs_2, agent_1=False)
                    display_step(step + 1, done=True)
                    video_display_step(step + 1, done=True)
                    self.processor.done()
                    break

            return win
        
        
        
    def get_batch(self, memory, batch_size, random_sample=True):
        """
        Retrieve and preprocess a batch from a memory buffer.
        """
        batch = memory.sample(batch_size, random_sample=random_sample)
        if batch is False:
            return False

        (vision, touch, prop, command_voice, feedback_voice,
         wheels_joints, voice_out, reward, done, mask) = batch

        vision = torch.from_numpy(vision).to(self.args.device)
        touch = torch.from_numpy(touch).to(self.args.device)
        prop = torch.from_numpy(prop).to(self.args.device)
        command_voice = torch.from_numpy(command_voice).to(self.args.device)
        feedback_voice = torch.from_numpy(feedback_voice).to(self.args.device)

        wheels_joints = torch.from_numpy(wheels_joints)
        voice_out = torch.from_numpy(voice_out)
        reward = torch.from_numpy(reward).to(self.args.device)
        done = torch.from_numpy(done).to(self.args.device)
        mask = torch.from_numpy(mask)

        wheels_joints = torch.cat(
            [torch.zeros(wheels_joints[:, 0].unsqueeze(1).shape), wheels_joints], dim=1
        ).to(self.args.device)

        voice_out = torch.cat(
            [torch.zeros(voice_out[:, 0].unsqueeze(1).shape), voice_out], dim=1
        ).to(self.args.device)

        all_mask = torch.cat(
            [torch.ones(mask.shape[0], 1, 1), mask], dim=1
        ).to(self.args.device)

        mask = mask.to(self.args.device)

        episodes = reward.shape[0]
        steps = reward.shape[1]

        if self.args.half:
            vision = vision.to(dtype=torch.float16)
            touch = touch.to(dtype=torch.float16)
            prop = prop.to(dtype=torch.float16)
            command_voice = command_voice.to(dtype=torch.float16)
            feedback_voice = feedback_voice.to(dtype=torch.float16)
            wheels_joints = wheels_joints.to(dtype=torch.float16)
            voice_out = voice_out.to(dtype=torch.float16)
            reward = reward.to(dtype=torch.float16)
            done = done.to(dtype=torch.float16)
            mask = mask.to(dtype=torch.float16)
            all_mask = all_mask.to(dtype=torch.float16)

        return (
            vision, touch, prop, command_voice, feedback_voice,
            wheels_joints, voice_out, reward, done, mask,
            all_mask, episodes, steps
        )
        
    
    
    # One epoch, training the actor, critics, and alpha values.
    def epoch(self, batch_size):
        start_time = duration()
        self.epochs += 1
        self.total_epochs += 1
        self.train()
        parenting = self.processor.parenting
                   
        # Collect information.               
        batch = self.get_batch(self.memory, batch_size)
        if batch == False:
            return(False)
        vision, touch, prop, command_voice, feedback_voice, wheels_joints, voice_out, reward, done, mask, all_mask, episodes, steps = batch
        extrinsic = torch.mean(reward).item()
        obs = Obs(vision, touch, prop, command_voice, feedback_voice)
        non_last_obs = Obs(vision[:,:-1], touch[:,:-1], prop[:,:-1], command_voice[:,:-1], feedback_voice[:,:-1])
        actions = Action(wheels_joints, voice_out)
        non_last_action = Action(wheels_joints[:,:-1], voice_out[:,:-1])
        non_blank_action = Action(wheels_joints[:,1:], voice_out[:,1:])

        starting_hidden_state = torch.zeros(1, episodes, self.args.pvrnn_mtrnn_size).detach()
        

                
        # Train critics
        with torch.no_grad():
            new_action, log_pis_next, log_pis_next_text, _ = \
                self.actor(obs, actions, starting_hidden_state.detach(), parenting)
            log_pis_next = log_pis_next[:,1:]
            log_pis_next_text = log_pis_next_text[:,1:]
            Q_target_nexts = []
            for i in range(self.args.critics):
                Q_target_next, _ = self.critic_targets[i](obs, new_action, starting_hidden_state.detach())
                Q_target_nexts.append(Q_target_next)                
                        
            Q_target_nexts_stacked = torch.stack(Q_target_nexts, dim=0)
            Q_target_next, _ = torch.min(Q_target_nexts_stacked, dim=0)
            Q_target_next = Q_target_next[:,1:]
            if self.args.alpha == None:      
                alpha = self.alpha 
            else:                            
                alpha = self.args.alpha
            if self.args.alpha_text == None: 
                alpha_text = self.alpha_text 
            else:                            
                alpha_text = self.args.alpha_text
            Q_targets = reward + (self.args.GAMMA * (1 - done) * (Q_target_next - (alpha * log_pis_next) - (alpha_text * log_pis_next_text)))
        
        critic_losses = []
        Qs = []
        for i in range(self.args.critics):
            Q, _ = self.critics[i](non_last_obs, non_blank_action, starting_hidden_state.detach())
            critic_loss = 0.5*F.mse_loss(Q*mask, Q_targets*mask)
            critic_losses.append(critic_loss)
            Qs.append(Q[0,0].item())
            self.critic_opts[i].zero_grad()
            critic_loss.backward()
            self.critic_opts[i].step()
        
            self.soft_update(self.critics[i], self.critic_targets[i], self.args.tau)
        
        torch.cuda.empty_cache()
                                    
            
        
        # Train actor
        if self.epochs % self.args.d == 0:
            if self.args.alpha == None:      
                alpha = self.alpha 
            else:                            
                alpha = self.args.alpha
            if self.args.alpha_text == None: 
                alpha_text = self.alpha_text 
            else:                            
                alpha_text = self.args.alpha_text
            new_action, log_pis, log_pis_text, _ = self.actor(non_last_obs, non_last_action, starting_hidden_state.detach(), parenting)
            
            loc = torch.zeros(self.args.wheels_joints_shape, dtype=torch.float64).to(self.args.device).float()
            n = self.args.wheels_joints_shape
            scale_tril = torch.eye(n)        
            policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
            policy_prior_log_prrgbd = self.args.normal_alpha * policy_prior.log_prob(new_action.wheels_joints).unsqueeze(-1)
            intrinsic_entropy = torch.mean((alpha * log_pis - policy_prior_log_prrgbd)*mask).item()
                
            Qs = []
            for i in range(self.args.critics):
                Q, _ = self.critics[i](non_last_obs, new_action, starting_hidden_state.detach())
                Qs.append(Q)
            Qs_stacked = torch.stack(Qs, dim=0)
            Q, _ = torch.min(Qs_stacked, dim=0)
            Q = Q.mean(-1).unsqueeze(-1)
            
            actor_loss = ((alpha * log_pis - policy_prior_log_prrgbd) + (alpha_text * log_pis_text) - Q)*mask
            actor_loss = actor_loss.mean() / mask.mean()
            
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
        else:
            Q = None
            intrinsic_entropy = None
            intrinsic_imitation = None
            actor_loss = None
        
            
            
        # Train alpha values.
        if self.args.alpha == None:
            _, log_pis, _, _ = self.actor(non_last_obs, non_last_action, starting_hidden_state.detach(), parenting)
            alpha_loss = -(self.log_alpha.to(self.args.device) * (log_pis + self.target_entropy))*mask
            alpha_loss = alpha_loss.mean() / mask.mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = torch.exp(self.log_alpha.to(dtype=torch.float32)).to(self.args.device)
            torch.cuda.empty_cache()
        else:
            alpha_loss = None
            
        if self.args.alpha_text == None:
            _, _, log_pis_text, _ = self.actor(non_last_obs, non_last_action, starting_hidden_state.detach(), parenting)
            alpha_text_loss = -(self.log_alpha_text.to(self.args.device) * (log_pis_text + self.target_entropy_text))*mask
            alpha_text_loss = alpha_text_loss.mean() / mask.mean()
            self.alpha_text_opt.zero_grad()
            alpha_text_loss.backward()
            self.alpha_text_opt.step()
            self.alpha_text = torch.exp(self.log_alpha_text.to(dtype=torch.float32)).to(self.args.device)
            torch.cuda.empty_cache()
        else:
            alpha_text_loss = None
                                
                                
                                
        # Save information.
        if alpha_loss is not None:              alpha_loss = alpha_loss.item()
        if alpha_text_loss is not None:         alpha_text_loss = alpha_text_loss.item()
        if actor_loss is not None:              actor_loss = actor_loss.item()
        if Q is not None:                       Q = -Q.mean().item()
        for i in range(self.args.critics):
            if critic_losses[i] is not None: 
                critic_losses[i] = critic_losses[i].item()
                critic_losses[i] = log(critic_losses[i]) if critic_losses[i] > 0 else critic_losses[i]

        if self.epochs == 1 or self.epochs >= self.args.epochs or self.epochs % self.args.keep_data == 0:                                                             
            self.plot_dict["alpha_loss"].append(alpha_loss)
            self.plot_dict["alpha_text_loss"].append(alpha_text_loss)
            self.plot_dict["actor_loss"].append(actor_loss)
            for layer, f in enumerate(critic_losses):
                self.plot_dict["critics_loss"][layer].append(f)    
            self.plot_dict["critics_loss"].append(critic_losses)
            self.plot_dict["extrinsic"].append(extrinsic)
            self.plot_dict["q"].append(Q)
            self.plot_dict["intrinsic_entropy"].append(intrinsic_entropy)
            
        end_time = duration()
        print_duration(start_time, end_time, "\nEpoch", "\n")
        
    
    
    def soft_update(self, local_model, target_model, tau):
        """
        Soft update target network parameters toward local network parameters.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    
        
    def sizeof_plot_dict(self):
        """
        Print approximate deep sizes of plot_dict contents and selected agent attributes.
        """
        print('\n\n\n')
        print('-' * 50)
        print('Key'.ljust(30), 'Size')
        print('-' * 50)

        # Deep plot_dict
        for key, value in self.plot_dict.items():
            size_estimate = len(pickle.dumps(value))
            if size_estimate > 100000:
                print(f'{key}:\t{sizeof_fmt(size_estimate)}')

        print('-' * 50)

        # Selected attributes
        keys_to_check = [
            'arena_1', 'arena_2',
            'log_alpha', 'alpha_opt',
            'actor', 'actor_opt',
            'memory'
        ]

        for attr_name in keys_to_check:
            attr = getattr(self, attr_name, None)
            if attr is not None:
                try:
                    size_estimate = len(pickle.dumps(attr))
                    if size_estimate > 100000:
                        print(f'{attr_name}:\t{sizeof_fmt(size_estimate)}')
                except Exception as e:
                    print(f'{attr_name}:\t(Unserializable: {e})')
            else:
                print(f'{attr_name}: (None)')

        print('-' * 50)
        print(f'{"TOTAL".ljust(30)}\t{sizeof_fmt(len(pickle.dumps(self)))}')
        print('-' * 50)
        print('\n\n\n')
        
        
    
    def save_agent(self):
        if not self.args.local:
            self.sizeof_plot_dict()
            plot_dict_backup = self.plot_dict
            memory_backup = self.memory
            self.plot_dict = self.empty_plot_dict
            self.memory = RecurrentReplayBuffer(self.args)
            save_path = f"{folder}/agents/agent_{str(self.agent_num).zfill(4)}_epoch_{str(self.epochs).zfill(6)}.pkl.gz"
            with gzip.open(save_path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.plot_dict = plot_dict_backup
            self.memory = memory_backup
                
    def load_agent(self, load_path):
        with gzip.open(load_path, "rb") as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)
                
    def state_dict(self):
        to_return = [self.actor.state_dict()]
        for i in range(self.args.critics):
            to_return.append(self.critics[i].state_dict())
            to_return.append(self.critic_targets[i].state_dict())
        return(to_return)

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict = state_dict[0])
        for i in range(self.args.critics):
            self.critics[i].load_state_dict(state_dict = state_dict[1+2*i])
            self.critic_targets[i].load_state_dict(state_dict = state_dict[2+2*i])
        self.memory = RecurrentReplayBuffer(self.args)

    def eval(self):
        self.actor.eval()
        for i in range(self.args.critics):
            self.critics[i].eval()
            self.critic_targets[i].eval()

    def train(self):
        self.actor.train()
        for i in range(self.args.critics):
            self.critics[i].train()
            self.critic_targets[i].train()


# ============================
# SCRIPT ENTRY POINT
# ============================

if __name__ == '__main__':
    agent = Agent(args=args)
    agent.save_episodes()
# %%