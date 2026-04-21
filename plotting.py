#%%
import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.lines as mlines
from itertools import accumulate
from math import log
from statistics import mode
from collections import Counter
from scipy import interpolate
from scipy.interpolate import interp1d

from utils import args, duration, load_dicts, print, rolling_average

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Prevents kernel crash with pyplot

print('name:\n{}\n'.format(args.arg_name))

dpi = 50

custom_ls = (0, (3, 5, 1, 5))


def to_percent(y, position):
    """Format y-axis tick values as percentages."""
    return f'{y:.0f}%'


def use_letter_path(xs, ys, letter, here, color, label):
    """Draw a line with a letter marker (e.g., $A$) on the plot."""
    x_dense = np.linspace(min(xs), max(xs), 30)
    y_dense = np.interp(x_dense, xs, ys)
    handle, = here.plot(
        x_dense, y_dense, marker=f'${letter}$', linestyle='None',
        markersize=12, color=color, label=label
    )
    return handle


def clean_and_interpolate_nan(arr, xs, non_nan_mask):
    """Fill in NaNs by linear interpolation."""
    arr = np.array(arr, dtype=np.float64)
    if np.isnan(arr[0]):
        arr[0] = 0.0
    if np.isnan(arr[-1]):
        arr[-1] = arr[~np.isnan(arr)][-1]
    nan_mask = np.isnan(arr)
    interpolator = interpolate.interp1d(xs[~nan_mask], arr[~nan_mask], bounds_error=False, fill_value='extrapolate')
    arr[nan_mask] = interpolator(xs[nan_mask])
    return arr


def get_quantiles(plot_dict, name, levels=[99, 0], adjust_xs=None):
    """Compute confidence intervals for a given data entry in the plot_dict."""
    if 0 not in levels:
        levels.append(0)
    max_len = mode([len(agent) for agent in plot_dict[name]])
    lists = np.array([[np.nan if x in [None, 'not_used'] else x for x in agent]
                      for agent in plot_dict[name] if len(agent) == max_len], dtype=float)

    valid_indexes = [i for i in plot_dict['args'].agents_for_plotting if 0 <= i < len(lists)]
    lists = lists[valid_indexes]
    non_nan_mask = ~np.isnan(lists).all(axis=0)
    xs = np.arange(lists.shape[1])
    if adjust_xs is not None:
        xs = xs * adjust_xs

    quantile_dict = {'xs': xs}
    levels_array = np.array(levels)
    lower_levels = ((100 - levels_array) / 2) / 100
    upper_levels = (levels_array + (100 - levels_array) / 2) / 100
    all_levels = np.concatenate([lower_levels, upper_levels])
    quantiles = np.nanquantile(lists, all_levels, axis=0)

    for i, level in enumerate(all_levels):
        quantile_dict[level] = quantiles[i]

    for key, values in quantile_dict.items():
        if key != 'xs':
            if np.any(np.isnan(values)):
                quantile_dict[key] = clean_and_interpolate_nan(values, xs, non_nan_mask)

    return quantile_dict


def get_list_quantiles(list_of_lists, plot_dict, levels=[99, 0]):
    """Compute confidence intervals for a list of lists (e.g., multiple critics)."""
    if 0 not in levels:
        levels.append(0)

    def largest_mode(values):
        """Find the largest mode in a list."""
        count = Counter(values)
        max_count = max(count.values())
        modes = [key for key, value in count.items() if value == max_count]
        return max(modes)

    quantile_dicts = []
    for layer in range(len(list_of_lists[0])):
        l = [l[layer] for l in list_of_lists]
        lengths = [len(sublist) for sublist in l]
        max_len = largest_mode(lengths)
        lists = np.array([l[i] for i in range(len(l)) if len(l[i]) == max_len], dtype=float)
        xs = list(range(len(lists[0])))
        lists = lists[:, xs]
        quantile_dict = {'xs': [x * plot_dict['args'].keep_data for x in xs]}
        for level in levels:
            lower_level = ((100 - level) / 2) / 100
            upper_level = (level + (100 - level) / 2) / 100
            quantile_dict[lower_level] = np.nanquantile(lists, lower_level, axis=0)
            quantile_dict[upper_level] = np.nanquantile(lists, upper_level, axis=0)
        quantile_dicts.append(quantile_dict)

    return quantile_dicts


def combine_quantile_dicts(train_dict, test_dict, train_weight=1, test_weight=2):
    """Blend two quantile dictionaries with weighted averaging."""
    total_weight = train_weight + test_weight
    combined_dict = {}

    xs_train = train_dict['xs']
    xs_test = test_dict['xs'] * 50  # Rescale test xs to align with training

    combined_dict['xs'] = xs_train

    for key in train_dict:
        if key == 'xs':
            continue
        train_vals = np.array(train_dict[key])
        test_vals = np.array(test_dict[key])
        test_interp_func = interp1d(xs_test, test_vals, bounds_error=False, fill_value='extrapolate')
        test_vals_interp = test_interp_func(xs_train)
        combined_vals = (train_weight * train_vals + test_weight * test_vals_interp) / total_weight
        combined_dict[key] = combined_vals

    return combined_dict


def get_logs(quantile_dict):
    """Convert quantile values to log-space (if needed)."""
    log_quantile_dict = {'xs': quantile_dict['xs']}
    for key in quantile_dict:
        if key != 'xs':
            log_quantile_dict[key] = np.log(quantile_dict[key])
    return log_quantile_dict


def pair_list(nums):
    """Pair quantile levels symmetrically, e.g. (.025, .975)."""
    nums.sort()
    pairs = []
    single = None
    while nums:
        if nums[-1] == .5:
            single = nums.pop()
        else:
            largest = nums.pop()
            for i in range(len(nums)):
                if nums[i] + largest == 1:
                    pairs.append((nums.pop(i), largest))
                    break
    pairs.sort(key=lambda x: abs(x[0] - x[1]), reverse=True)
    result = pairs
    if single is not None:
        result.append((single,))
    return result


def awesome_plot(here, quantile_dict, color, label, min_max=None,
                 line_transparency=0.9, fill_transparency=0.1, linestyle='solid', letter=None):
    """Draw line and fill confidence intervals for a quantile dict."""
    xs = quantile_dict['xs']
    keys = list(quantile_dict.keys())[1:]
    pairs = pair_list(keys)
    if letter is None:
        for i, pair in enumerate(pairs):
            if pair == (.5,):
                handle, = here.plot(xs, quantile_dict[.5], color=color, alpha=line_transparency,
                                    label=label, linestyle=linestyle)
                print(f'{label} : {quantile_dict[.5][-1]}')
            else:
                start, stop = pair
                transparency = line_transparency * (.05 * (i + 1))
                here.fill_between(xs, quantile_dict[start], quantile_dict[stop],
                                  color=color, alpha=fill_transparency, linewidth=0)
                here.plot(xs, quantile_dict[start], color=color, alpha=transparency,
                          label=label, linestyle=linestyle)
                here.plot(xs, quantile_dict[stop], color=color, alpha=transparency,
                          label=label, linestyle=linestyle)
    else:
        handle = use_letter_path(xs, quantile_dict[.5], letter, here, color, label)

    if min_max is not None and min_max[0] != min_max[1]:
        here.set_ylim([min_max[0], min_max[1]])
    return handle


def many_min_max(min_max_list):
    """Combine multiple min/max values into one range."""
    mins = [min_val for min_val, _ in min_max_list if min_val is not None]
    maxs = [max_val for _, max_val in min_max_list if max_val is not None]
    return min(mins), max(maxs)




def plots(plot_dicts, min_max_dict):
    """Create all plots and save them to disk."""
    
    # If there are more than 1 but less than 16 agents, we may plot them all together in one enourmous plot.
    too_many_plot_dicts = len(plot_dicts) <= 1 or len(plot_dicts) > 16
    levels = [99]  # Using just a 99% confidence interval.

    # If few agents: make a conjoined plot with all of them.
    if not too_many_plot_dicts:
        fig, axs = plt.subplots(36, len(plot_dicts), figsize=(20 * len(plot_dicts), 305))                

    # Iterate over agents.
    for i, plot_dict in enumerate(plot_dicts):
        row_num = 0
        if not too_many_plot_dicts:
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1

        args = plot_dict['args']
        args.robot_name = 'robot'
        print(f'\nStarting {plot_dict["arg_name"]}.')
        epochs = plot_dict['division_epochs'][0]

        def divide_arenas(xs, here):
            """(Optional) Add vertical lines to separate training phases."""
            if isinstance(xs, dict):
                xs = xs['xs']
            length_xs = len(xs)
            mapped_values = []

            # Uncomment to show vertical divisions:
            # for epoch, linestyle, full_name in epochs:
            #     position = (epoch / epochs[-1][0]) * (length_xs - 1)
            #     index = round(position)
            #     x_val = xs[index]
            #     here.axvline(x=x_val, color=(0, 0, 0, .2), linestyle=linestyle)
            #     here.text(x_val, here.get_ylim()[1] * 0.95, full_name,
            #         rotation=90, verticalalignment='top', fontsize=10, color='black')

        # -------------------------
        # Rolling Win-Rate Plots
        # -------------------------
        try:
            os.mkdir('thesis_pics/rolling_win_rate')
        except:
            pass

        task_name_list = [k[5:] for k in plot_dict if k.startswith('wins_') and k[5:] != 'SILENCE']

        if plot_dict['args'].exceptions == 0:
            task_name_list = task_name_list[:-1]

        fig2, ax2 = plt.subplots(len(task_name_list), 3, figsize=(30, 30))
        fig2.suptitle(plot_dict['arg_title'])
        fig2_row_num = 0

        for task_name in task_name_list:
            if task_name == 'exception':
                plot_dict['rolled_wins_' + task_name] = [
                    rolling_average(wins, window_size=3000)
                    for wins in plot_dict['wins_' + task_name]
                ]

            win_dict = get_quantiles(plot_dict, f'rolled_wins_{task_name}', levels=levels)
            if task_name == 'exception':
                gen_win_dict = None
                combine_win_dict = None
            else:
                gen_win_dict = get_quantiles(plot_dict, f'rolled_gen_wins_{task_name}', levels=levels)
                combine_win_dict = combine_quantile_dicts(win_dict, gen_win_dict)

            for key, value in win_dict.items():
                if key != 'xs':
                    win_dict[key] *= 100
                    if combine_win_dict is not None:
                        combine_win_dict[key] *= 100
                    if task_name != 'exception':
                        gen_win_dict[key] *= 100
                else:
                    if task_name != 'exception':
                        gen_win_dict[key] *= args.epochs_per_gen_test

            def plot_rolling_average_wins(here, which_dict=None):
                """Plot rolling average win rates."""
                this_win_dict = (
                    combine_win_dict if which_dict == 'all'
                    else gen_win_dict if which_dict == 'gen'
                    else win_dict
                )
                if this_win_dict is None:
                    return
                awesome_plot(here, this_win_dict, 'black', 'WinRate', (0, 100))
                here.set_ylabel(
                    'Rolling-Average Gen-Win-Rate' if which_dict == 'gen'
                    else 'Rolling-Average Win-Rate'
                )
                here.yaxis.set_major_formatter(FuncFormatter(to_percent))
                here.yaxis.set_minor_locator(MultipleLocator(5))
                here.set_xlabel('Epochs')
                here.set_title(
                    plot_dict['arg_title'] +
                    ('\nRolling-Average Gen-Win-Rate ({})'.format(task_name) if which_dict == 'gen'
                     else '\nRolling-Average Win-Rate ({})'.format(task_name))
                )
                divide_arenas(this_win_dict, here)

            if not too_many_plot_dicts:
                plot_rolling_average_wins(ax)
                ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
                row_num += 1
                plot_rolling_average_wins(ax, which_dict='gen')
                ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
                row_num += 1

            plot_rolling_average_wins(ax2[fig2_row_num, 0])
            ax2[fig2_row_num, 0].set_title(f'Rolling-Average Win-Rate ({task_name})')
            plot_rolling_average_wins(ax2[fig2_row_num, 1], which_dict='gen')
            ax2[fig2_row_num, 1].set_title(f'Rolling-Average Gen-Win-Rate ({task_name})')
            plot_rolling_average_wins(ax2[fig2_row_num, 2], which_dict='all')
            ax2[fig2_row_num, 2].set_title(f'Rolling-Average All-Win-Rate ({task_name})')

            fig2_row_num += 1
            print(f'\tFinished win-rates ({task_name}).')

        fig2.savefig(f'thesis_pics/rolling_win_rate/win_rates_{plot_dict["arg_name"]}.png',
                     bbox_inches='tight', dpi=dpi)
        plt.close(fig2)
                
                
        # -------------------------
        # Cumulative Reward
        # -------------------------
        try:
            os.mkdir('thesis_pics/cumulative_reward')
        except:
            pass

        rew_dict = get_quantiles(plot_dict, 'accumulated_reward', levels=levels, adjust_xs=None)
        gen_rew_dict = get_quantiles(plot_dict, 'accumulated_gen_reward', levels=levels,
                                     adjust_xs=plot_dict['args'].epochs_per_gen_test)

        def plot_cumulative_reward(here, gen=False, min_max=None):
            """Plot cumulative rewards (with optional generalization test)."""
            awesome_plot(here, gen_rew_dict if gen else rew_dict, 'black', 'Reward', min_max)
            here.axhline(y=0, color='black', linestyle='--', alpha=0.2)
            here.set_ylabel('Cumulative Reward')
            here.set_xlabel('Epochs')
            here.set_title(
                plot_dict['arg_title'] + '\nCumulative reward' +
                ('\nin Generalization-Tests' if gen else '') +
                ('' if min_max is None else ', shared min/max')
            )
            divide_arenas(gen_rew_dict if gen else rew_dict, here)

        if not too_many_plot_dicts:
            plot_cumulative_reward(ax)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1
            plot_cumulative_reward(ax, min_max=min_max_dict['accumulated_reward'])
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1
            plot_cumulative_reward(ax, gen=True)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1
            plot_cumulative_reward(ax, gen=True, min_max=min_max_dict['accumulated_gen_reward'])
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1

        fig2, ax2 = plt.subplots(4, 1, figsize=(20, 60))
        fig2.suptitle(plot_dict['arg_title'])
        plot_cumulative_reward(ax2[0])
        ax2[0].set_title('Cumulative reward')
        plot_cumulative_reward(ax2[1], min_max=min_max_dict['accumulated_reward'])
        ax2[1].set_title('Cumulative reward, shared min/max')
        plot_cumulative_reward(ax2[2], gen=True)
        ax2[2].set_title('Cumulative reward\nin Generalization-Tests')
        plot_cumulative_reward(ax2[3], gen=True, min_max=min_max_dict['accumulated_gen_reward'])
        ax2[3].set_title('Cumulative reward\nin Generalization-Tests, shared min/max')
        fig2.savefig(f'thesis_pics/cumulative_reward/cumulative_reward_{plot_dict["arg_name"]}.png',
                     bbox_inches='tight', dpi=dpi)
        plt.close(fig2)

        print(f'\tFinished cumulative reward.')

        # -------------------------
        # Forward Losses
        # -------------------------
        try:
            os.mkdir('thesis_pics/forward_losses')
        except:
            pass

        vision_dict = get_quantiles(plot_dict, 'vision_loss', levels=levels, adjust_xs=plot_dict['args'].keep_data)
        touch_dict = get_quantiles(plot_dict, 'touch_loss', levels=levels, adjust_xs=plot_dict['args'].keep_data)
        command_voice_dict = get_quantiles(plot_dict, 'command_voice_loss', levels=levels, adjust_xs=plot_dict['args'].keep_data)
        feedback_voice_dict = get_quantiles(plot_dict, 'feedback_voice_loss', levels=levels, adjust_xs=plot_dict['args'].keep_data)
        accuracy_dict = get_quantiles(plot_dict, 'accuracy_loss', levels=levels, adjust_xs=plot_dict['args'].keep_data)
        comp_dict = get_quantiles(plot_dict, 'complexity_loss', levels=levels, adjust_xs=plot_dict['args'].keep_data)

        all_values = np.concatenate([v for k, v in comp_dict.items() if k != 'xs'])
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        forward_losses_min_max = (min_val, max_val)

        log_vision_dict = get_logs(vision_dict)
        log_touch_dict = get_logs(touch_dict)
        log_command_voice_dict = get_logs(command_voice_dict)
        log_feedback_voice_dict = get_logs(feedback_voice_dict)
        log_accuracy_dict = get_logs(accuracy_dict)
        log_comp_dict = get_logs(comp_dict)

        if forward_losses_min_max[0] == 0:
            log_forward_losses_min_max = (0.01, forward_losses_min_max[1])
        else:
            log_forward_losses_min_max = forward_losses_min_max

        log_forward_losses_min_max = (
            log(log_forward_losses_min_max[0]),
            log(log_forward_losses_min_max[1])
        )

        def plot_forward_losses(here, log=False, min_max=None):
            """Plot (optionally log-scale) forward model losses."""
            handles = []
            # Only plotting complexity for now.
            # handles.append(awesome_plot(here, log_vision_dict if log else vision_dict, 'blue', 'Vision-Loss', min_max))
            # handles.append(awesome_plot(here, log_touch_dict if log else touch_dict, 'orange', 'Touch-Loss', min_max))
            # handles.append(awesome_plot(here, log_command_voice_dict if log else command_voice_dict, 'red', 'Command Voice-Loss', min_max))
            # handles.append(awesome_plot(here, log_feedback_voice_dict if log else feedback_voice_dict, 'red', 'Feedback Voice-Loss', min_max))
            # handles.append(awesome_plot(here, log_accuracy_dict if log else accuracy_dict, 'purple', 'Accuracy', min_max))
            handles.append(awesome_plot(here, log_comp_dict if log else comp_dict, 'green', 'Complexity', min_max))
            here.set_ylabel('Loss')
            here.set_xlabel('Epochs')
            here.legend(handles=handles)
            here.set_title(
                plot_dict['arg_title'] + '\n' +
                ('log ' if log else '') + 'Forward Losses' +
                (', shared min/max' if min_max else '')
            )
            divide_arenas(accuracy_dict, here)

        if not too_many_plot_dicts:
            plot_forward_losses(ax)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1
            plot_forward_losses(ax, min_max=forward_losses_min_max)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1
            plot_forward_losses(ax, log=True)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1
            plot_forward_losses(ax, log=True, min_max=log_forward_losses_min_max)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1

        fig2, ax2 = plt.subplots(4, 1, figsize=(20, 60))
        fig2.suptitle(plot_dict['arg_title'])
        plot_forward_losses(ax2[0])
        ax2[0].set_title('Forward Losses')
        plot_forward_losses(ax2[1], min_max=forward_losses_min_max)
        ax2[1].set_title('Forward Losses, shared min/max')
        plot_forward_losses(ax2[2], log=True)
        ax2[2].set_title('log Forward Losses')
        plot_forward_losses(ax2[3], log=True, min_max=log_forward_losses_min_max)
        ax2[3].set_title('log Forward Losses, shared min/max')
        fig2.savefig(f'thesis_pics/forward_losses/forward_losses_{plot_dict["arg_name"]}.png',
                     bbox_inches='tight', dpi=dpi)
        plt.close(fig2)

        print(f'\tFinished forward losses.')

        # -------------------------
        # Other Losses
        # -------------------------
        try:
            os.mkdir('thesis_pics/other_losses')
        except:
            pass

        alpha_dict = get_quantiles(plot_dict, 'alpha_loss', levels=levels, adjust_xs=plot_dict['args'].keep_data)
        actor_dict = get_quantiles(plot_dict, 'actor_loss', levels=levels, adjust_xs=plot_dict['args'].keep_data)
        crit_dicts = get_list_quantiles(plot_dict['critics_loss'], plot_dict, levels=levels)
        crit_min_max = many_min_max([c for c in min_max_dict['critics_loss']])

        def plot_other_losses(here, min_max=False):
            """Plot actor, critic, and alpha losses on separate axes."""
            handles = []
            handles.append(awesome_plot(here, actor_dict, 'red', 'Actor',
                                        min_max_dict['actor_loss'] if min_max else None))
            here.set_ylabel('Actor Loss')

            ax2 = here.twinx()
            handles.append(awesome_plot(ax2, crit_dicts[0], 'blue', 'log Critic 1',
                                        crit_min_max if min_max else None))
            ax2.set_ylabel('log Critic Losses')

            ax3 = here.twinx()
            ax3.spines['right'].set_position(('axes', 1.08))
            handles.append(awesome_plot(ax3, alpha_dict, 'black', 'Alpha',
                                        min_max_dict['alpha_loss'] if min_max else None))
            ax3.set_ylabel('Alpha Loss')

            here.set_xlabel('Epochs')
            here.legend(handles=handles)
            here.set_title(plot_dict['arg_title'] + '\nActor, Critic Losses' +
                           (', shared min/max' if min_max else ''))
            divide_arenas(actor_dict, here)

        if not too_many_plot_dicts:
            plot_other_losses(ax)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1
            plot_other_losses(ax, min_max=True)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1

        fig2, ax2 = plt.subplots(2, 1, figsize=(20, 30))
        fig2.suptitle(plot_dict['arg_title'])
        plot_other_losses(ax2[0])
        ax2[0].set_title('Actor, Critic Losses')
        plot_other_losses(ax2[1], min_max=True)
        ax2[1].set_title('Actor, Critic Losses, shared min/max')
        fig2.savefig(f'thesis_pics/other_losses/actor_critic_losses_{plot_dict["arg_name"]}.png',
                     bbox_inches='tight', dpi=dpi)
        plt.close(fig2)

        print(f'\tFinished other losses.')

        
            
        # -------------------------
        # Extrinsic and Intrinsic Reward
        # -------------------------
        try:
            os.mkdir('thesis_pics/reward')
        except:
            pass

        lowest, highest = None, None
        for level in levels:
            lower = ((100 - level) / 2) / 100
            upper = (level + (100 - level) / 2) / 100
            if lowest is None or lower < lowest:
                lowest = lower
            if highest is None or upper > highest:
                highest = upper
        keys = (lowest, highest)

        ext_dict = get_quantiles(plot_dict, 'extrinsic', levels=levels, adjust_xs=plot_dict['args'].keep_data)
        ent_dict = get_quantiles(plot_dict, 'intrinsic_entropy', levels=levels, adjust_xs=plot_dict['args'].keep_data)
        cur_dict = get_quantiles(plot_dict, 'intrinsic_curiosity', levels=levels, adjust_xs=plot_dict['args'].keep_data)
        reward_min_max = many_min_max([
            min_max_dict['extrinsic'],
            min_max_dict['intrinsic_entropy'],
            min_max_dict['intrinsic_curiosity']
        ])

        def plot_extrinsic_and_intrinsic_reward(here, min_max=False):
            """Plot extrinsic, entropy, and curiosity reward (separate axes)."""
            handles = []
            handles.append(awesome_plot(here, ext_dict, 'red', 'Extrinsic',
                                        min_max_dict['extrinsic'] if min_max else None))
            here.set_ylabel('Extrinsic')
            here.set_xlabel('Epochs')

            if (ent_dict[keys[0]] != ent_dict[keys[1]]).any():
                ax2 = here.twinx()
                handles.append(awesome_plot(ax2, ent_dict, 'black', 'Entropy',
                                            min_max_dict['intrinsic_entropy'] if min_max else None))
                ax2.set_ylabel('Entropy')

            if (cur_dict[keys[0]] != cur_dict[keys[1]]).any():
                ax3 = here.twinx()
                ax3.spines['right'].set_position(('axes', 1.08))
                handles.append(awesome_plot(ax3, cur_dict, 'green', 'Curiosity',
                                            min_max_dict['intrinsic_curiosity'] if min_max else None))
                ax3.set_ylabel('Curiosity')

            here.legend(handles=handles)
            here.set_title(
                plot_dict['arg_title'] + '\nExtrinsic and Intrinsic reward' +
                (', shared min/max' if min_max else '')
            )
            divide_arenas(ext_dict, here)

        def plot_extrinsic_and_intrinsic_reward_shared_dim(here, min_max=False):
            """Plot all reward types on same axis."""
            handles = []
            handles.append(awesome_plot(here, ext_dict, 'red', 'Extrinsic',
                                        reward_min_max if min_max else None))
            here.set_ylabel('Reward')
            here.set_xlabel('Epochs')

            if (ent_dict[keys[0]] != ent_dict[keys[1]]).any():
                handles.append(awesome_plot(here, ent_dict, 'black', 'Entropy',
                                            reward_min_max if min_max else None))
            if (cur_dict[keys[0]] != cur_dict[keys[1]]).any():
                handles.append(awesome_plot(here, cur_dict, 'green', 'Curiosity',
                                            reward_min_max if min_max else None))

            here.legend(handles=handles)
            here.set_title(
                plot_dict['arg_title'] + '\nExtrinsic and Intrinsic reward, shared dims' +
                (', shared min/max' if min_max else '')
            )
            divide_arenas(ext_dict, here)

        if not too_many_plot_dicts:
            plot_extrinsic_and_intrinsic_reward(ax)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1
            plot_extrinsic_and_intrinsic_reward(ax, min_max=True)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1
            plot_extrinsic_and_intrinsic_reward_shared_dim(ax)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1
            plot_extrinsic_and_intrinsic_reward_shared_dim(ax, min_max=True)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1

        fig2, ax2 = plt.subplots(4, 1, figsize=(20, 60))
        fig2.suptitle(plot_dict['arg_title'])
        plot_extrinsic_and_intrinsic_reward(ax2[0])
        ax2[0].set_title('Extrinsic and Intrinsic reward')
        plot_extrinsic_and_intrinsic_reward(ax2[1], min_max=True)
        ax2[1].set_title('Extrinsic and Intrinsic reward, shared min/max')
        plot_extrinsic_and_intrinsic_reward_shared_dim(ax2[2])
        ax2[2].set_title('Extrinsic and Intrinsic reward, shared dim')
        plot_extrinsic_and_intrinsic_reward_shared_dim(ax2[3], min_max=True)
        ax2[3].set_title('Extrinsic and Intrinsic reward, shared dim, shared min/max')
        fig2.savefig(f'thesis_pics/reward/reward_{plot_dict["arg_name"]}.png', bbox_inches='tight', dpi=dpi)
        plt.close(fig2)

        print(f'\tFinished extrinsic and intrinsic reward.')

        # -------------------------
        # Curiosities
        # -------------------------
        try:
            os.mkdir('thesis_pics/curiosities')
        except:
            pass

        vision_prediction_error_dict = get_quantiles(
            plot_dict, 'vision_prediction_error_curiosity', levels=[],
            adjust_xs=plot_dict['args'].keep_data)
        touch_prediction_error_dict = get_quantiles(
            plot_dict, 'touch_prediction_error_curiosity', levels=[],
            adjust_xs=plot_dict['args'].keep_data)
        command_voice_prediction_error_dict = get_quantiles(
            plot_dict, 'command_voice_prediction_error_curiosity', levels=[],
            adjust_xs=plot_dict['args'].keep_data)
        feedback_voice_prediction_error_dict = get_quantiles(
            plot_dict, 'feedback_voice_prediction_error_curiosity', levels=[],
             adjust_xs=plot_dict['args'].keep_data)
        prediction_error_dict = get_quantiles(
            plot_dict, 'prediction_error_curiosity', levels=[],
            adjust_xs=plot_dict['args'].keep_data)

        vision_hidden_state_dict = get_quantiles(
            plot_dict, 'vision_hidden_state_curiosity', levels=[],
            adjust_xs=plot_dict['args'].keep_data)
        touch_hidden_state_dict = get_quantiles(
            plot_dict, 'touch_hidden_state_curiosity', levels=[],
            adjust_xs=plot_dict['args'].keep_data)
        command_voice_hidden_state_dict = get_quantiles(
            plot_dict, 'command_voice_hidden_state_curiosity', levels=[],
            adjust_xs=plot_dict['args'].keep_data)
        feedback_voice_hidden_state_dict = get_quantiles(
            plot_dict, 'feedback_voice_hidden_state_curiosity', levels=[],
            adjust_xs=plot_dict['args'].keep_data)
        hidden_state_dict = get_quantiles(
            plot_dict, 'hidden_state_curiosity', levels=[],
            adjust_xs=plot_dict['args'].keep_data)

        curiosity_min_max = many_min_max([
            min_max_dict['vision_prediction_error_curiosity'],
            min_max_dict['touch_prediction_error_curiosity'],
            min_max_dict['command_voice_prediction_error_curiosity'],
            min_max_dict['prediction_error_curiosity'],
            min_max_dict['vision_hidden_state_curiosity'],
            min_max_dict['touch_hidden_state_curiosity'],
            min_max_dict['command_voice_hidden_state_curiosity'],
            min_max_dict['hidden_state_curiosity'],
            min_max_dict['feedback_voice_prediction_error_curiosity'],
            min_max_dict['feedback_voice_hidden_state_curiosity']
        ])

        log_vision_prediction_error_dict = get_logs(vision_prediction_error_dict)
        log_touch_prediction_error_dict = get_logs(touch_prediction_error_dict)
        log_command_voice_prediction_error_dict = get_logs(command_voice_prediction_error_dict)
        log_feedback_voice_prediction_error_dict = get_logs(feedback_voice_prediction_error_dict)
        log_prediction_error_dict = get_logs(prediction_error_dict)

        log_vision_hidden_state_dict = get_logs(vision_hidden_state_dict)
        log_touch_hidden_state_dict = get_logs(touch_hidden_state_dict)
        log_command_voice_hidden_state_dict = get_logs(command_voice_hidden_state_dict)
        log_feedback_voice_hidden_state_dict = get_logs(feedback_voice_hidden_state_dict)
        log_hidden_state_dict = get_logs(hidden_state_dict)
        
        if curiosity_min_max[0] == 0:
            log_curiosity_min_max = (0.01, curiosity_min_max[1])
        else:
            log_curiosity_min_max = curiosity_min_max

        log_curiosity_min_max = (
            log(log_curiosity_min_max[0]),
            log(log_curiosity_min_max[1])
        )

        def plot_prediction_error_curiosities(here, log=False, min_max=False):
            """Plot prediction error-based curiosity signals."""
            handles = []
            this_min_max = (
                log_curiosity_min_max if (log and min_max)
                else curiosity_min_max if min_max
                else None
            )
            handles.append(awesome_plot(
                here,
                log_prediction_error_dict if log else prediction_error_dict,
                'green',
                'Total ' + ('log ' if log else '') + 'Prediction Error',
                min_max=this_min_max,
                linestyle='solid'
            ))
            handles.append(awesome_plot(
                here,
                log_vision_prediction_error_dict if log else vision_prediction_error_dict,
                'green',
                'Vision',
                min_max=this_min_max,
                linestyle='dotted'
            ))
            handles.append(awesome_plot(
                here,
                log_touch_prediction_error_dict if log else touch_prediction_error_dict,
                'green',
                'Touch',
                min_max=this_min_max,
                linestyle=custom_ls
            ))
            handles.append(awesome_plot(
                here,
                log_command_voice_prediction_error_dict if log else command_voice_prediction_error_dict,
                'green',
                'Command voice',
                min_max=this_min_max,
                letter='C'
            ))
            handles.append(awesome_plot(
                here,
                log_feedback_voice_prediction_error_dict if log else feedback_voice_prediction_error_dict,
                'green',
                'Feedback voice',
                min_max=this_min_max,
                letter='R'
            ))
            here.set_ylabel('Prediction Error Curiosity')
            here.set_xlabel('Epochs')
            here.legend(handles=handles)
            here.set_title(
                plot_dict['arg_title'] + '\n' +
                ('log ' if log else '') + 'Possible Prediction Error Curiosities' +
                (', shared min/max' if min_max else '')
            )
            divide_arenas(prediction_error_dict, here)

        def plot_hidden_state_curiosities(here, log=False, min_max=False):
            """Plot hidden-state-based curiosity signals."""
            handles = []
            this_min_max = (
                log_curiosity_min_max if (log and min_max)
                else curiosity_min_max if min_max
                else None
            )
            handles.append(awesome_plot(
                here,
                log_hidden_state_dict if log else hidden_state_dict,
                'red',
                'Total Hidden State',
                min_max=this_min_max,
                linestyle='solid'
            ))
            handles.append(awesome_plot(
                here,
                log_vision_hidden_state_dict if log else vision_hidden_state_dict,
                'red',
                'Vision',
                min_max=this_min_max,
                linestyle='dotted'
            ))
            handles.append(awesome_plot(
                here,
                log_touch_hidden_state_dict if log else touch_hidden_state_dict,
                'red',
                'Touch',
                min_max=this_min_max,
                linestyle=custom_ls
            ))
            handles.append(awesome_plot(
                here,
                log_command_voice_hidden_state_dict if log else command_voice_hidden_state_dict,
                'red',
                'Command voice',
                min_max=this_min_max,
                letter='C'
            ))
            handles.append(awesome_plot(
                here,
                log_feedback_voice_hidden_state_dict if log else feedback_voice_hidden_state_dict,
                'red',
                'Feedback voice',
                min_max=this_min_max,
                letter='R'
            ))
            here.set_ylabel('Hidden State Curiosity')
            here.set_xlabel('Epochs')
            here.legend(handles=handles)
            here.set_title(
                plot_dict['arg_title'] + '\n' +
                ('log ' if log else '') + 'Possible Hidden State Curiosities' +
                (', shared min/max' if min_max else '')
            )
            divide_arenas(prediction_error_dict, here)

        if not too_many_plot_dicts:
            plot_prediction_error_curiosities(ax)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1

            plot_hidden_state_curiosities(ax)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1

            plot_prediction_error_curiosities(ax, log=True)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1

            plot_hidden_state_curiosities(ax, log=True)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1

            plot_prediction_error_curiosities(ax, min_max=True)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1

            plot_hidden_state_curiosities(ax, min_max=True)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1

            plot_prediction_error_curiosities(ax, log=True, min_max=True)
            ax = axs[row_num, i] if len(plot_dicts) > 1 else axs[row_num]
            row_num += 1

            plot_hidden_state_curiosities(ax, log=True, min_max=True)

        fig2, ax2 = plt.subplots(4, 2, figsize=(40, 60))
        fig2.suptitle(plot_dict['arg_title'])

        plot_prediction_error_curiosities(ax2[0, 0])
        ax2[0, 0].set_title('Possible Prediction Error Curiosities')

        plot_hidden_state_curiosities(ax2[1, 0])
        ax2[1, 0].set_title('Possible Hidden State Curiosities')

        plot_prediction_error_curiosities(ax2[0, 1], log=True)
        ax2[0, 1].set_title('Possible log Prediction Error Curiosities')

        plot_hidden_state_curiosities(ax2[1, 1], log=True)
        ax2[1, 1].set_title('Possible log Hidden State Curiosities')

        plot_prediction_error_curiosities(ax2[2, 0], min_max=True)
        ax2[2, 0].set_title('Prediction Error Curiosities, shared min/max')

        plot_hidden_state_curiosities(ax2[3, 0], min_max=True)
        ax2[3, 0].set_title('Hidden State Curiosities, shared min/max')

        plot_prediction_error_curiosities(ax2[2, 1], log=True, min_max=True)
        ax2[2, 1].set_title('log Prediction Error Curiosities, shared min/max')

        plot_hidden_state_curiosities(ax2[3, 1], log=True, min_max=True)
        ax2[3, 1].set_title('log Hidden State Curiosities, shared min/max')

        fig2.savefig(
            f'thesis_pics/curiosities/curiosities_{plot_dict["arg_name"]}.png',
            bbox_inches='tight',
            dpi=dpi
        )
        plt.close(fig2)

        print(f'\tFinished curiosities.')
        print(f'Finished {plot_dict["arg_name"]}.')
        print(f'Duration: {duration()}')

    # Save large full-plot if many agents
    if not too_many_plot_dicts:
        print('\nSaving full plot...')
        fig.tight_layout(pad=1.0)
        plt.savefig('thesis_pics/plot.png', bbox_inches='tight')
        plt.close(fig)
    
    
    
# Plot
plot_dicts, min_max_dict, complete_order = load_dicts(args)
for i, plot_dict in enumerate(plot_dicts):
    new_plot_dict = deepcopy(plot_dict)
    new_plot_dict['args'].agents_for_plotting = [i for i in range(new_plot_dict['args'].agents_for_plotting)]
    plot_dicts[i] = new_plot_dict

plots(plot_dicts, min_max_dict)



# Plot for individual agents
#for i in range(10):
#    new_plot_dicts = []
#    for plot_dict in plot_dicts:
#        new_plot_dict = deepcopy(plot_dict)
#        new_plot_dict['arg_name'] = new_plot_dict['arg_name'] + f'_{i}'
#        new_plot_dict['args'].agents_for_plotting = [i]
#        new_plot_dicts.append(new_plot_dict)
#    plots(new_plot_dicts, min_max_dict)
    
print(f'\nDuration: {duration()}. Done!')
# %%