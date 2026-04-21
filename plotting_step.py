#%%

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import imageio

from utils import print, args, duration, load_dicts, wheels_joints_to_string, get_goal_from_one_hots, plot_number_bars
from pybullet_data.robots.robot_maker import robot_dict


"""
This file plots a comprehensive visual step-by-step overview of agent perception.
Includes:
    - Raw and predicted observations
    - Predicted actions
    - DKL divergence
    - Goal and feedback comparison
"""

def human_friendly_text(goal):
    """Format goal as human-readable text with its char-text in parentheses."""
    return f'{goal.human_text} ({goal.char_text})'


def plot_step(step, episode_dict, agent_1=True, last_step=False, saving=True, dreaming=False, args=args):
    """
    Plot all data for a given step of the agent.
    Can plot for either agent_1 or agent_2.
    """
    sensor_plotter, sensor_values = robot_dict[args.robot_name]
    agent_num = 1 if agent_1 else 2

    obs = episode_dict[f'obs_{agent_num}'][step]
    vision = obs.vision[0, :, :, :-1]
    touch = obs.touch.tolist()[0]
    command_voice = obs.command_voice
    feedback_voice = obs.feedback_voice

    if step != 0:
        prior = episode_dict[f'prior_predictions_{agent_num}'][step - 1]
        posterior = episode_dict[f'posterior_predictions_{agent_num}'][step - 1]
        action = episode_dict[f'action_{agent_num}'][step - 1]

        prior_vision = prior.vision[0, 0, :, :, :-1]
        posterior_vision = posterior.vision[0, 0, :, :, :-1]

        prior_touch = prior.touch.tolist()[0][0]
        posterior_touch = posterior.touch.tolist()[0][0]

        prior_command_voice = prior.command_voice
        posterior_command_voice = posterior.command_voice

        prior_feedback_voice = prior.feedback_voice
        posterior_feedback_voice = posterior.feedback_voice

    data = []

    data.append(['Goal', [human_friendly_text(episode_dict['goal'])], .1])
    if step != 0:
        data.append(['Acheived Goal', [human_friendly_text(feedback_voice)], .1])

    data.append(['Bird\'s Eye View', [episode_dict[f'birds_eye_{agent_num}'][step], 'image'], 1])

    if step != 0:
        real_label = 'Real (not seen in dream; \nagent sees posterior)' if dreaming else ''
        data.append(['', [real_label], ['Prior'], ['Posterior'], .1])

    # If this is the first step, the agent hasn't predicted anything yet
    if step == 0:
        data.extend([
            [f'Vision ({agent_num})', [vision, 'image'], 1],
            [f'Touch ({agent_num})', [touch, 'touch'], 1],
            [f'Command voice ({agent_num})', [human_friendly_text(command_voice)], 1],
            [f'Feedback voice ({agent_num})', [human_friendly_text(feedback_voice)], 1],
        ])
    else:
        data.extend([
            [f'Vision ({agent_num})', [vision, 'image'], [prior_vision, 'image'], [posterior_vision, 'image'], 1],
            [f'Touch ({agent_num})', [touch, 'touch'], [prior_touch, 'touch'], [posterior_touch, 'touch'], 1],
            [f'Command voice ({agent_num})', [human_friendly_text(command_voice)],
             [human_friendly_text(prior_command_voice)], [human_friendly_text(posterior_command_voice)], .3],
            [f'Feedback voice ({agent_num})', [human_friendly_text(feedback_voice)],
             [human_friendly_text(prior_feedback_voice)], [human_friendly_text(posterior_feedback_voice)], .3],
            [f'Wheels, Joints ({agent_num})', [action.wheels_joints, 'bar_plot'], .5],
            [f'Voice Out ({agent_num})', [human_friendly_text(get_goal_from_one_hots(action.voice_out))], .3],
            [f'Vision DKL ({agent_num})', [episode_dict[f'vision_dkl_{agent_num}'][:step], 'line_plot'], .5],
            [f'Touch DKL ({agent_num})', [episode_dict[f'touch_dkl_{agent_num}'][:step], 'line_plot'], .5],
            [f'Command voice DKL ({agent_num})', [episode_dict[f'command_voice_dkl_{agent_num}'][:step], 'line_plot'], .5],
            [f'Feedback voice DKL ({agent_num})', [episode_dict[f'feedback_voice_dkl_{agent_num}'][:step], 'line_plot'], .5],
        ])

    max_cols = max(len(row) for row in data)

    def plot_text(ax, value):
        ax.text(0.1, 0.5, f'{value}', fontsize=12, va='center', transform=ax.transAxes)
        ax.axis('off')

    def plot_image(ax, image):
        ax.imshow(image, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_touch(ax, touch_data):
        touch_image = sensor_plotter(touch_data)[80:-70, 10:-10]
        ax.imshow(touch_image)
        ax.axis('off')

    def plot_bar_plot(ax, values):
        numbers = values.flatten().tolist()
        ax.bar(range(len(numbers)), numbers, color=['red' if x < 0 else 'blue' for x in numbers])
        ax.axhline(0, color='black', lw=1)
        ax.set_ylim(-1, 1)
        xticks = ['left wheel', 'right wheel'] + [f'joint {i+1}' for i in range(len(numbers)-2)]
        ax.set_xticks(range(len(numbers)))
        ax.set_xticklabels(xticks, rotation=0, ha='right', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)

    def plot_line_plot(ax, values):
        ax.plot(values)
        ax.set_ylabel('DKL')
        ax.set_xlabel('Time')

    def plot_sublist(fig, gs, sublist, row):
        ax = fig.add_subplot(gs[row, 0])
        plot_text(ax, sublist[0] + (':' if sublist[0] else ''))

        for col, item in enumerate(sublist[1:-1]):
            ax = fig.add_subplot(gs[row, col + 1])
            if isinstance(item[0], str):
                plot_text(ax, item[0])
            elif item[-1] == 'image':
                plot_image(ax, item[0])
            elif item[-1] == 'touch':
                plot_touch(ax, item[0])
            elif item[-1] == 'bar_plot':
                ax = fig.add_subplot(gs[row, 1:])
                plot_bar_plot(ax, item[0])
            elif item[-1] == 'line_plot':
                ax = fig.add_subplot(gs[row, 1:])
                plot_line_plot(ax, item[0])

    # Generate full plot
    fig = plt.figure(figsize=(20, 25))
    height_ratios = [row[-1] for row in data]
    gs = gridspec.GridSpec(len(data), max_cols, figure=fig, height_ratios=height_ratios)

    for row_idx, sublist in enumerate(data):
        plot_sublist(fig, gs, sublist, row_idx)

    if saving:
        plt.savefig(f'Step {step} Agent {agent_num}.png')
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    print('name:\n{}\n'.format(args.arg_name))
    plot_dicts, min_max_dict, complete_order = load_dicts(args)
    plot_episodes(complete_order, plot_dicts)
    print(f'\nDuration: {duration()}. Done!')

#%%
