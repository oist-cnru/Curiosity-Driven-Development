#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats
import random

from utils import args, duration, load_dicts, task_map, color_map, shape_map

# This file makes a plot showing agent behavior from beginning of training to end.
# I.e., which actions is the agent performing?
# It seems as though behaviors aren't being properly saved in training episodes.

print('name:\n{}'.format(args.arg_name))

try:
    os.chdir(f'saved_{args.comp}')
except:
    pass

try:
    os.mkdir('thesis_pics/behavior')
except:
    pass


def dict_to_ordered_list(d):
    """
    Convert dictionary to a list of values, ordered by sorted keys.
    """
    return [d[key] for key in sorted(d.keys())]


# Define task, color, and shape sets
tasks = dict_to_ordered_list(task_map)
colors = dict_to_ordered_list(color_map)
shapes = dict_to_ordered_list(shape_map)


def behaviors_to_data(behaviors, start_epoch, finish_epoch):
    """
    Count occurrences of behavior patterns over a given epoch range.

    Returns:
        A dict mapping char combinations to raw counts.
    """
    all_strings = ['AAA'] + [''.join((t.char, c.char, s.char)) for (t, c, s) in product(tasks, colors, shapes)]
    range_keys = range(start_epoch, finish_epoch)
    string_counts = {string: 0 for string in all_strings}
    total_count = 0

    for i in range_keys:
        for feedback_voice in behaviors[i]:
            if feedback_voice.char_text[0] != 'A':
                string_counts[feedback_voice.char_text] += 1
                total_count += 1

    return string_counts


def create_ranges(start, end, step):
    """
    Split the full training range into smaller epoch ranges.
    
    Returns:
        A list of (start, end) tuples.
    """
    ranges = []
    for i in range(start, end, step):
        ranges.append((i, min(i + step, end + 1)))
    return ranges


def plot_behaviors(plot_dict):
    """
    Create heatmaps of agent behavior across training time and tasks.
    """
    all_behaviors = plot_dict['behavior'][0]
    epoch_ranges = create_ranges(1, 20000, 2000)

    # Compute behavior data across epochs
    data = {
        epoch: behaviors_to_data(all_behaviors, start, end)
        for epoch, (start, end) in enumerate(epoch_ranges)
    }

    # Get maximum value for colormap normalization
    vmax = max(
        percentage
        for epoch_data in data.values()
        for percentage in epoch_data.values()
    )

    fig, axes = plt.subplots(
        len(epoch_ranges),
        len(tasks),
        figsize=(3 * len(tasks), 3 * len(epoch_ranges))
    )

    for row, (start, end) in enumerate(epoch_ranges):
        percentages = data[row]

        for col, task in enumerate(tasks):
            heatmap = np.zeros((len(shapes), len(colors)))

            for i, shape in enumerate(shapes):
                for j, color in enumerate(colors):
                    key = 'AAA' if task.name == 'SILENCE' else task.char + color.char + shape.char
                    heatmap[i, j] = percentages.get(key, 0)

            ax = axes[row, col]
            ax.imshow(heatmap, vmin=0, vmax=100, cmap='gray_r')

            if col == 0:
                ax.set_ylabel(f'Epoch {start}-{end}', fontsize=12)

            if task.name == 'SILENCE':
                ax.set_title('NO ACTION', pad=10)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            else:
                ax.set_title(task.name, pad=10)
                ax.set_xticks(np.arange(len(colors)))
                ax.set_yticks(np.arange(len(shapes)))
                ax.set_xticklabels([c.name for c in colors], rotation=90, fontsize=10)
                ax.set_yticklabels([s.name for s in shapes], rotation=45, fontsize=10)

    fig.suptitle('Behavior Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'thesis_pics/behavior/{plot_dict["args"].arg_name}.png')


# Load dicts and run plotting
plot_dicts, min_max_dict, complete_order = load_dicts(args)

for plot_dict in plot_dicts:
    plot_behaviors(plot_dict)

print(f'\nDuration: {duration()}. Done!')
#%%
