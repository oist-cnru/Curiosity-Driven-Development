#%%

import os
import re
import time
import imageio
import threading
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.table import Table

from utils import print, args, duration, load_dicts, wheels_joints_to_string, plot_number_bars, empty_goal
from pybullet_data.robots.robot_maker import robot_dict


"""
This script creates concise visualizations of agent observations and curiosity metrics.
Includes:
    - Vision, touch
    - Command and feedback voices
    - Feedback prediction
    - Curiosity DKL values (vision, touch, proprioception, feedback voice)
"""

def plot_video_step(step, episode_dict, agent_1=True, last_step=False, saving=True, dreaming=False, args=args):
    """Plot a visual summary of a single step in the agent's episode."""

    sensor_plotter, sensor_values = robot_dict[args.robot_name]
    agent_num = 1 if agent_1 else 2

    # --- Observations ---
    obs = episode_dict[f'obs_{agent_num}'][step]
    vision = obs.vision[0, :, :, :-1]
    touch = obs.touch.tolist()[0]
    
    # --- Voice (human readable) ---
    command_voice = obs.command_voice.human_friendly_text()
    feedback_voice = obs.feedback_voice.human_friendly_text(command=False)

    command_task = obs.command_voice.task.name
    command_color = obs.command_voice.color.name
    command_shape = obs.command_voice.shape.name

    feedback_task = obs.feedback_voice.task.name
    feedback_color = obs.feedback_voice.color.name
    feedback_shape = obs.feedback_voice.shape.name

    if step != 0:
        posterior = episode_dict[f'posterior_predictions_{agent_num}'][step - 1]
        posterior_feedback_voice = posterior.feedback_voice
    else:
        posterior_feedback_voice = empty_goal

    predicted_feedback_task = posterior_feedback_voice.task.name
    predicted_feedback_color = posterior_feedback_voice.color.name
    predicted_feedback_shape = posterior_feedback_voice.shape.name

    # --- DKL Curiosity values ---
    #visual_curiosity = [0] + [c * args.hidden_state_eta_vision for c in episode_dict[f'vision_is_{agent_num}'.dkl.sum().item()][:step]]
    #touch_curiosity = [0] + [c * args.hidden_state_eta_touch for c in episode_dict[f'touch_is_{agent_num}'.dkl.sum().item()][:step]]
    #prop_curiosity = [0] + [c * args.hidden_state_eta_prop for c in episode_dict[f'prop_is_{agent_num}'.dkl.sum().item()][:step]]
    #feedback_voice_curiosity = [0] + [c * args.hidden_state_eta_feedback_voice for c in episode_dict[f'feedback_voice_is_{agent_num}'.dkl.sum().item()][:step]]
    
    # --- Latent values ---
    latent_state_num = 3
    voice_posterior = []
    hq = []
    for i in range(latent_state_num):
        voice_posterior += [[0] + [c.zp[0,i].item() for c in episode_dict[f'command_voice_is_{agent_num}'][:step]]]
        hq += [[0] + [hq[0,i].item() for hq in episode_dict[f'hq_{agent_num}'][:step]]]
    all_values = [v for sublist in voice_posterior for v in sublist] + \
                [v for sublist in hq for v in sublist]
    min_val = min(all_values)
    max_val = max(all_values)

    # --- Figure setup ---
    dpi = 100
    fig = plt.figure(figsize=(4, 8), dpi=dpi, facecolor='none')
    fig.patch.set_alpha(0)

    main_ax = fig.add_axes([0, 0, 1, 1])
    main_ax.set_axis_off()
    main_ax.patch.set_alpha(0)
    
    # Step label
    main_ax.text(
        0.94, 0.98, f'Step {step}',
        fontsize=15,
        transform=main_ax.transAxes,
        zorder=3,
        ha='right',
        va='center',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=1.0, linewidth=2)
    )

    # --- Vision ---
    vision_ax = fig.add_axes([0.05, 0.27, 0.9, 0.9])
    vision_ax.imshow(vision)
    vision_ax.set_xticks([])
    vision_ax.set_yticks([])
    vision_ax.patch.set_alpha(0)
    for spine in vision_ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(3)
    
    # --- Touch ---
    touch_ax = fig.add_axes([0.05, -0.14, 0.9, 0.9])
    touch_image = sensor_plotter(touch)[80:-70, 80:-60]
    touch_ax.imshow(touch_image)
    touch_ax.patch.set_alpha(0)
    touch_ax.set_xticks([])
    touch_ax.set_yticks([])
    for spine in touch_ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(3)

    # --- Command and Feedback Table ---
    table_ax = fig.add_axes([0.05, -0.18, 0.9, 0.25])
    table_ax.set_axis_off()
    fontsize = 12
    table_ax.text(0, 0.8, f'Command:\n{command_task} {command_color} {command_shape}.',
                  ha='left', va='center', fontsize=fontsize)
    table_ax.text(0, 0.45, f'Feedback:\n{feedback_task} {feedback_color} {feedback_shape}.',
                  ha='left', va='center', fontsize=fontsize)
    table_ax.text(0, 0.1, f'Predicted Feedback:\n{predicted_feedback_task} {predicted_feedback_color} {predicted_feedback_shape}.',
                  ha='left', va='center', fontsize=fontsize)
            
    # --- Curiosity Plots ---
    #all_curiosities = visual_curiosity + touch_curiosity + prop_curiosity + feedback_voice_curiosity
    #if not all_curiosities:
    #    all_curiosities = [0]
    #min_curi = min(all_curiosities) * 0.9
    #max_curi = max(all_curiosities) * 1.1

    plot_height = 0.07
    base_bottom = -0.30
    #curiosity_titles = ['Vision Curiosity', 'Touch Curiosity', 'Proprioception Curiosity', 'Feedback Voice Curiosity']
    #curiosity_data = [visual_curiosity, touch_curiosity, prop_curiosity, feedback_voice_curiosity]
    latent_titles = ['voice_posterior', 'hq']
    latent_data = [voice_posterior, hq]

    """for idx, (title, data) in enumerate(zip(curiosity_titles, curiosity_data)):
        bottom_pos = base_bottom - idx * (plot_height + 0.03)
        ax = fig.add_axes([0.1, bottom_pos, 0.8, plot_height])
        if len(data) > 1:
            ax.plot(data, color='black', linewidth=2)
        elif len(data) == 1:
            ax.plot([0], data, marker='o', markersize=6, color='black')
        ax.set_xlim([0, step])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(title, fontsize=10)
        ax.patch.set_alpha(0)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1)"""
            
    for idx, (title, data) in enumerate(zip(latent_titles, latent_data)):
        bottom_pos = base_bottom - idx * (plot_height + 0.03)
        ax = fig.add_axes([0.1, bottom_pos, 0.8, plot_height])
        if len(data) > 1:
            for i in range(latent_state_num):
                ax.plot(data[i], linewidth=2)
        elif len(data) == 1:
            for i in range(latent_state_num):
                ax.plot([0], data, marker='o', markersize=6, color='black')
        ax.set_xlim([0, step])
        ax.set_ylim([min_val, max_val])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(title, fontsize=10)
        ax.patch.set_alpha(0)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1)
    

    
    # --- Save or Show ---
    if saving:
        os.makedirs('saved_deigo/thesis_pics/video_pics', exist_ok=True)
        filename = f'saved_deigo/thesis_pics/video_pics/Goal {command_voice} Step {step}.png'
        plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()

    plt.close()
            
    

if __name__ == '__main__':
    print(f'name:\n{args.arg_name}\n')
    plot_dicts, min_max_dict, complete_order = load_dicts(args)
    plot_episodes(complete_order, plot_dicts)
    print(f'\nDuration: {duration()}. Done!')
    
# %%