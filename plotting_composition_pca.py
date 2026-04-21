#%%
import os
import pickle
import random
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from utils import task_map, color_map, shape_map, duration, print

# ============================
# CONFIGURATION
# ============================
hyper_parameters = 'name_here'
agent_num_str = '0001'
epochs_str = '060000'
saved_file = 'saved_deigo'

# ============================
# LOAD SAVED PLOT DICT
# ============================
load_path = f'{saved_file}/{hyper_parameters}/plot_dict_agent_{agent_num_str}_epoch_{epochs_str}_composition.pickle'
with open(load_path, 'rb') as f:
    raw_plot_dict = pickle.load(f)

plot_dict = raw_plot_dict.copy()
plot_dict['composition_data'] = [raw_plot_dict['composition_data']]

the_epoch = int(epochs_str)
these_epochs = [the_epoch]

# ============================
# PLOTTING CONSTANTS
# ============================
dpi = 400
letter_size = 120
letter_w_size = 200
shape_size = 0.4
test_size = 120
color_size = 120
fontsize = 10
max_agent_num = 0

task_mapping_color = {
    'WATCH': '#FF0000', 'BE NEAR': '#00FF00', 'TOUCH THE TOP': '#0000FF',
    'PUSH FORWARD': '#00DDDD', 'PUSH LEFT': '#FF00FF', 'PUSH RIGHT': '#DDDD00'}

task_mapping_letter = {
    'WATCH': 'W', 'BE NEAR': 'N', 'TOUCH THE TOP': 'T',
    'PUSH FORWARD': 'F', 'PUSH LEFT': 'L', 'PUSH RIGHT': 'R'}

color_mapping_color = {
    'RED': '#FF0000', 'GREEN': '#00FF00', 'BLUE': '#0000FF',
    'CYAN': '#00DDDD', 'MAGENTA': '#FF00FF', 'YELLOW': '#DDDD00'}

shape_mapping_marker = {
    'PILLAR':    mpimg.imread('pybullet_data/shapes/pillar.png'),
    'POLE':      mpimg.imread('pybullet_data/shapes/pole.png'),
    'DUMBBELL':  mpimg.imread('pybullet_data/shapes/dumbbell.png'),
    'CONE':      mpimg.imread('pybullet_data/shapes/cone.png'),
    'HOURGLASS': mpimg.imread('pybullet_data/shapes/hourglass.png')}

def colorize_marker_image(marker_img, hex_color, alpha=0.3):
    hex_color = hex_color.lstrip('#')
    target_rgb = np.array([int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)])
    colorized = marker_img.copy()
    mask = colorized[..., 3] > 0
    colorized[mask, 0] = target_rgb[0]
    colorized[mask, 1] = target_rgb[1]
    colorized[mask, 2] = target_rgb[2]
    colorized[mask, 3] = alpha
    return colorized

shape_mapping_colored_marker = {}
for shape_name, shape_marker in shape_mapping_marker.items():
    shape_mapping_colored_marker[shape_name] = {}
    shape_mapping_colored_marker[shape_name]['BLACK'] = colorize_marker_image(shape_marker, '#000000')
    for color_name, color_color in color_mapping_color.items():
        shape_mapping_colored_marker[shape_name][color_name] = colorize_marker_image(shape_marker, color_color)

# ============================
# PIPELINE FUNCTIONS
# ============================

meta_data_dict = {}

def get_all_data(plot_dict, component):
    args = plot_dict['args']
    print(f'Getting {args.arg_name}\'s {component} data...')
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        if values_for_composition == {} or agent_num > max_agent_num:
            break
        meta_data_dict[(args.arg_name, agent_num, component)] = {}
        for epochs, comp_dict in values_for_composition.items():
            all_mask = comp_dict['all_mask'].astype(bool)
            max_episode_len = all_mask.shape[1]
            all_mask = all_mask.reshape(-1, all_mask.shape[-1]).squeeze()
            def process_component(key):
                data = comp_dict[key]
                if key == 'b':
                    data = data.transpose(0, 2, 1)
                data = data.reshape(-1, data.shape[-1])
                data = data[all_mask]
                return data
            meta_data_dict[(args.arg_name, agent_num, component)][epochs] = {
                'labels': process_component('labels'),
                'component': process_component(component)}


meta_reducer_dict = {}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_reducer(data_dict, reducer_type):
    reducer_dict = {}
    for classes in [('task', 'color'), ('task', 'shape'), ('color', 'shape')]:
        print(f'\t\t\tClasses {classes}...')
        set_seed(42)
        scaler = StandardScaler()
        labels = data_dict['labels']
        data_scaled = scaler.fit_transform(data_dict['component'])
        if reducer_type == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            reducer.fit(data_scaled)
        if reducer_type == 'lda':
            i = 0
            these_labels = np.zeros_like(labels)
            if 'task' in classes:
                these_labels[:, i] = labels[:, 0]; i += 1
            if 'color' in classes:
                these_labels[:, i] = labels[:, 1]; i += 1
            if 'shape' in classes:
                these_labels[:, i] = labels[:, 2]
            class_labels = [f'{a}_{b}' for a, b in zip(these_labels[:, 0], these_labels[:, 1])]
            reducer = LDA(n_components=2)
            reducer.fit(data_scaled, class_labels)
        reducer_dict[classes] = {'scaler': scaler, 'reducer': reducer}
    return reducer_dict

def make_all_reducers(plot_dict, component, reducer_type, these_epochs):
    args = plot_dict['args']
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        if values_for_composition == {} or agent_num > max_agent_num:
            break
        meta_reducer_dict[(args.arg_name, agent_num, component, reducer_type)] = {}
        for epochs in these_epochs:
            data_dict = meta_data_dict[(args.arg_name, agent_num, component)][epochs]
            meta_reducer_dict[(args.arg_name, agent_num, component, reducer_type)][epochs] = make_reducer(data_dict, reducer_type)


meta_reduced_data_dict = {}

def use_reducer(data_dict, reducer_dict):
    reduced_data_dict = {}
    labels = data_dict['labels']
    reduced_data_dict['labels'] = labels
    reduced_data_dict['tasks'] = labels[:, 0]
    reduced_data_dict['colors'] = labels[:, 1]
    reduced_data_dict['shapes'] = labels[:, 2]
    reduced_data_dict['unique_tasks'] = np.unique(reduced_data_dict['tasks'])
    reduced_data_dict['unique_colors'] = np.unique(reduced_data_dict['colors'])
    reduced_data_dict['unique_shapes'] = np.unique(reduced_data_dict['shapes'])
    for classes in [('task', 'color'), ('task', 'shape'), ('color', 'shape')]:
        data_scaled = reducer_dict[classes]['scaler'].transform(data_dict['component'])
        reduced_data_dict[classes] = reducer_dict[classes]['reducer'].transform(data_scaled)
    return reduced_data_dict

def make_all_reduced_data(plot_dict, component, reducer_type):
    args = plot_dict['args']
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        if values_for_composition == {} or agent_num > max_agent_num:
            break
        meta_reduced_data_dict[(args.arg_name, agent_num, component, reducer_type)] = {}
        for data_epochs in meta_data_dict[(args.arg_name, agent_num, component)].keys():
            data_dict = meta_data_dict[(args.arg_name, agent_num, component)][data_epochs]
            reducer_dict = meta_reducer_dict[(args.arg_name, agent_num, component, reducer_type)][data_epochs]
            meta_reduced_data_dict[(args.arg_name, agent_num, component, reducer_type)][data_epochs, data_epochs] = use_reducer(data_dict, reducer_dict)


meta_aligned_data_dict = {}

def align_data(reduced_data_dict_1, reduced_data_dict_2):
    aligned_data_dict = {}
    labels = reduced_data_dict_1['labels']
    aligned_data_dict['labels'] = labels
    aligned_data_dict['tasks'] = labels[:, 0]
    aligned_data_dict['colors'] = labels[:, 1]
    aligned_data_dict['shapes'] = labels[:, 2]
    aligned_data_dict['unique_tasks'] = np.unique(aligned_data_dict['tasks'])
    aligned_data_dict['unique_colors'] = np.unique(aligned_data_dict['colors'])
    aligned_data_dict['unique_shapes'] = np.unique(aligned_data_dict['shapes'])
    for classes in [('task', 'color'), ('task', 'shape'), ('color', 'shape')]:
        aligned_data, _, _ = procrustes(reduced_data_dict_1[classes], reduced_data_dict_2[classes])
        aligned_data_dict[classes] = aligned_data
    return aligned_data_dict

def make_all_aligned_data(plot_dict, component, reducer_type):
    args = plot_dict['args']
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        if values_for_composition == {} or agent_num > max_agent_num:
            break
        meta_aligned_data_dict[(args.arg_name, agent_num, component, reducer_type)] = {}
        for data_epochs, reducer_epochs in meta_reduced_data_dict[(args.arg_name, agent_num, component, reducer_type)].keys():
            reduced_data_dict = meta_reduced_data_dict[(args.arg_name, agent_num, component, reducer_type)][data_epochs, reducer_epochs]
            # Single epoch: align with itself
            meta_aligned_data_dict[(args.arg_name, agent_num, component, reducer_type)][data_epochs, reducer_epochs] = align_data(reduced_data_dict, reduced_data_dict)


def plot_by_attribute(ax, aligned_data, classes, title):
    print(f'\t\t\t\t{title}...')
    coords = aligned_data[classes]
    tasks  = aligned_data['tasks']
    colors = aligned_data['colors']
    shapes = aligned_data['shapes']

    grouped_points = defaultdict(list)
    grouped_letters = {}
    for task, color, shape, (x, y) in zip(tasks, colors, shapes, coords):
        t = task  if 'task'  in classes else None
        c = color if 'color' in classes else None
        s = shape if 'shape' in classes else None
        key = (t, c, s)
        grouped_points[key].append((x, y))
        if key not in grouped_letters and t is not None:
            grouped_letters[key] = task_mapping_letter[task_map[t].name]

    xs_all, ys_all = [], []
    for (task, color, shape), pts in grouped_points.items():
        xs, ys = zip(*pts)
        x, y = sum(xs)/len(xs), sum(ys)/len(ys)
        xs_all.append(x); ys_all.append(y)

        if 'color' in classes:
            color_name = color_map[color].name
            color_val  = color_mapping_color[color_name]
            text_color_val = color_val if 'task' in classes else 'black'
        else:
            text_color_val = 'black'

        if 'shape' in classes:
            shape_name = shape_map[shape].name
            colored_marker = (shape_mapping_colored_marker[shape_name][color_name]
                              if 'color' in classes
                              else shape_mapping_colored_marker[shape_name]['BLACK'])
            imagebox = OffsetImage(colored_marker, zoom=shape_size)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, alpha=0.4, zorder=1)
            ax.add_artist(ab)

        if 'task' in classes:
            letter = grouped_letters[(task, color, shape)]
            ax.scatter(x, y, color=text_color_val,
                       marker=f'${letter}$', alpha=1,
                       s=letter_w_size if letter == 'W' else letter_size,
                       edgecolor='none', zorder=2)

    if xs_all:
        pad = 0.1
        xr, yr = max(xs_all)-min(xs_all), max(ys_all)-min(ys_all)
        ax.set_xlim([min(xs_all)-xr*pad, max(xs_all)+xr*pad])
        ax.set_ylim([min(ys_all)-yr*pad, max(ys_all)+yr*pad])

    ax.set_title(title)
    ax.set_xlabel('Component 1')
    if title == 'Task and Color':
        ax.set_ylabel('Component 2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(False)


def plot_single_epoch(plot_dict, component, reducer_type, epoch):
    args = plot_dict['args']
    agent_num = 0
    aligned_data = meta_aligned_data_dict[(args.arg_name, agent_num, component, reducer_type)][epoch, epoch]

    fig, axes = plt.subplots(
        1, 4, figsize=(19, 6), dpi=dpi,
        sharex=False, sharey=False, constrained_layout=True,
        gridspec_kw={'width_ratios': [1, 1, 1, 0.4]})

    plt.suptitle(
        f'Tests for Compositionality with {args.arg_name}\n'
        f'Agent {agent_num_str} • epoch {epoch} • {reducer_type} with {component}.', fontsize=16)

    plot_by_attribute(axes[0], aligned_data, ('task', 'color'),  'Task and Color')
    plot_by_attribute(axes[1], aligned_data, ('task', 'shape'),  'Task and Shape')
    plot_by_attribute(axes[2], aligned_data, ('color', 'shape'), 'Color and Shape')

    # Legend panel
    ax = axes[3]
    y = 10
    for task_name, letter in task_mapping_letter.items():
        ax.scatter(.1, y, color='black', marker=f'${letter}$', alpha=0.4,
                   s=letter_w_size if letter == 'W' else letter_size, edgecolor='none')
        ax.text(.3, y, s=task_name, horizontalalignment='left',
                verticalalignment='center', fontsize=fontsize)
        y -= .5
    for color_name, color in color_mapping_color.items():
        ax.scatter(.1, y, facecolors=color, edgecolors='none', s=color_size, alpha=0.4)
        ax.text(.3, y, s=color_name, horizontalalignment='left',
                verticalalignment='center', fontsize=fontsize)
        y -= .5
    for shape_name in shape_mapping_marker.keys():
        colored_marker = shape_mapping_colored_marker[shape_name]['BLACK']
        imagebox = OffsetImage(colored_marker, zoom=shape_size)
        ab = AnnotationBbox(imagebox, (.1, y), frameon=False, alpha=0.4, zorder=1)
        ax.add_artist(ab)
        ax.text(.3, y, s=shape_name, horizontalalignment='left',
                verticalalignment='center', fontsize=fontsize)
        y -= .5
    ax.scatter(.1, y, facecolors='none', edgecolors='#000000', linewidths=0.5,
               marker='o', alpha=.5, s=test_size, zorder=0, linestyle='dotted')
    ax.text(.3, y, s='TEST', horizontalalignment='left',
            verticalalignment='center', fontsize=fontsize)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_xlim([0, .8]); ax.set_ylim([1, 10.5])

    outdir = f'thesis_pics/composition/{args.arg_name}/agent_{agent_num_str}/separate {component}/{reducer_type}'
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f'{outdir}/data_{str(epoch).zfill(6)}.001.png', bbox_inches='tight')
    plt.close()
    print(f'Saved {component} / {reducer_type} plot for epoch {epoch}.')


# ============================
# RUN
# ============================
for component in ['command_voice_zq']:
    get_all_data(plot_dict, component)
    for reducer_type in ['pca']:
        make_all_reducers(plot_dict, component, reducer_type, these_epochs)
        make_all_reduced_data(plot_dict, component, reducer_type)
        make_all_aligned_data(plot_dict, component, reducer_type)
        plot_single_epoch(plot_dict, component, reducer_type, the_epoch)

print(f'\nDuration: {duration()}. Done!')
# %%