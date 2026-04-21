#%%
import os
import pickle
import random
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import torch
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from utils import task_map, color_map, shape_map, duration, print

# ============================
# CONFIGURATION
# ============================
hyper_parameters = 'name_here'
agent_num_str    = '0001'
epochs_str       = '045000'
saved_file       = 'saved_deigo'

# ── HCA knobs ─────────────────────────────────────────────────────────────────
K              = 999        # number of clusters to cut
LINKAGE_METHOD = 'ward'     # 'ward' | 'average' | 'complete' | 'single'

# Which dendrograms to produce.  Comment out any you don't want.
DENDROGRAMS_TO_PLOT = [
    'all_180',       # all Task×Color×Shape centroids  (180 leaves)
    'task_x_color',  # avg over shape                  (36 leaves)
    'task_x_shape',  # avg over color                  (30 leaves)
    'color_x_shape', # avg over task                   (30 leaves)
    'task_only',     # avg over color+shape            (6 leaves)
    'color_only',    # avg over task+shape             (6 leaves)
    'shape_only',    # avg over task+color             (5 leaves)
]

# ── Visual constants (your style) ─────────────────────────────────────────────
dpi            = 400
shape_size     = 1.2
letter_size    = 550
letter_w_size  = 600
color_dot_size = 500
fontsize       = 20
step_y_scaler  = 0.20   # vertical gap between stacked glyphs (fraction of y_max)

max_agent_num  = 0

# ── Priority orderings ────────────────────────────────────────────────────────
task_priority  = {'WATCH': 0, 'BE NEAR': 1, 'TOUCH THE TOP': 2,
                  'PUSH FORWARD': 3, 'PUSH LEFT': 4, 'PUSH RIGHT': 5}
color_priority = {'RED': 0, 'GREEN': 1, 'BLUE': 2,
                  'CYAN': 3, 'MAGENTA': 4, 'YELLOW': 5}
shape_priority = {'PILLAR': 0, 'POLE': 1, 'DUMBBELL': 2, 'CONE': 3, 'HOURGLASS': 4}

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
    hex_color  = hex_color.lstrip('#')
    target_rgb = np.array([int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)])
    colorized  = marker_img.copy()
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
# LOAD SAVED PLOT DICT
# ============================
load_path = (f'{saved_file}/{hyper_parameters}/'
             f'plot_dict_agent_{agent_num_str}_epoch_{epochs_str}_composition.pickle')
with open(load_path, 'rb') as f:
    raw_plot_dict = pickle.load(f)

plot_dict = raw_plot_dict.copy()
plot_dict['composition_data'] = [raw_plot_dict['composition_data']]

the_epoch    = int(epochs_str)
these_epochs = [the_epoch]


# ============================
# PIPELINE  (my original: all timesteps → centroids → linkage)
# ============================
meta_data_dict = {}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_all_data(plot_dict, component):
    """Flatten ALL valid (non-padded) timesteps across every episode."""
    args = plot_dict['args']
    print(f'Getting {args.arg_name}\'s {component} data...')
    for agent_num, values_for_composition in enumerate(plot_dict['composition_data']):
        if values_for_composition == {} or agent_num > max_agent_num:
            break
        meta_data_dict[(args.arg_name, agent_num, component)] = {}
        for epochs, comp_dict in values_for_composition.items():
            all_mask = comp_dict['all_mask'].astype(bool)

            def process_component(key):
                data = comp_dict[key]
                episodes, steps, D = data.shape
                mask_flat = all_mask.reshape(episodes * steps, -1).squeeze(-1)
                data_flat = data.reshape(episodes * steps, D)
                return data_flat[mask_flat]

            labels_data = comp_dict['labels']
            episodes, steps, _ = labels_data.shape
            mask_flat     = all_mask.reshape(episodes * steps, -1).squeeze(-1).astype(bool)
            labels_flat   = labels_data.reshape(episodes * steps, 3)
            labels_masked = labels_flat[mask_flat]

            meta_data_dict[(args.arg_name, agent_num, component)][epochs] = {
                'labels':    labels_masked,
                'component': process_component(component)}


def _centroid_matrix(data_scaled, labels, key_fn):
    """
    Group rows of data_scaled by key_fn(label_row) and average within each group.
    Returns (centroid_matrix, ordered_keys).
    """
    groups = defaultdict(list)
    for row, lbl in zip(data_scaled, labels):
        groups[key_fn(lbl)].append(row)
    ordered_keys = sorted(groups.keys())
    centroids    = np.array([np.mean(groups[k], axis=0) for k in ordered_keys])
    return centroids, ordered_keys


def build_centroid_sets(data_dict):
    """
    Compute one centroid matrix per requested dendrogram type.
    Scaling is fit on the full timestep pool before centroid averaging,
    matching the approach from the original script.

    Returns dict: dg_name -> {'X': ndarray, 'keys': list, 'label_fn': callable}
    label_fn(key) -> (letter_or_None, hex_color, shape_name_or_None)
    """
    labels    = data_dict['labels']
    component = data_dict['component']

    set_seed(42)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(component)

    result = {}

    if 'all_180' in DENDROGRAMS_TO_PLOT:
        centroids, keys = _centroid_matrix(
            X_scaled, labels, lambda l: (int(l[0]), int(l[1]), int(l[2])))
        def label_fn_180(key):
            t, c, s    = key
            letter     = task_mapping_letter[task_map[t].name]
            hex_color  = color_mapping_color[color_map[c].name]
            shape_name = shape_map[s].name
            return letter, hex_color, shape_name
        result['all_180'] = {'X': centroids, 'keys': keys, 'label_fn': label_fn_180}

    if 'task_x_color' in DENDROGRAMS_TO_PLOT:
        centroids, keys = _centroid_matrix(
            X_scaled, labels, lambda l: (int(l[0]), int(l[1])))
        def label_fn_tc(key):
            t, c      = key
            letter    = task_mapping_letter[task_map[t].name]
            hex_color = color_mapping_color[color_map[c].name]
            return letter, hex_color, None
        result['task_x_color'] = {'X': centroids, 'keys': keys, 'label_fn': label_fn_tc}

    if 'task_x_shape' in DENDROGRAMS_TO_PLOT:
        centroids, keys = _centroid_matrix(
            X_scaled, labels, lambda l: (int(l[0]), int(l[2])))
        def label_fn_ts(key):
            t, s       = key
            letter     = task_mapping_letter[task_map[t].name]
            shape_name = shape_map[s].name
            return letter, '#000000', shape_name
        result['task_x_shape'] = {'X': centroids, 'keys': keys, 'label_fn': label_fn_ts}

    if 'color_x_shape' in DENDROGRAMS_TO_PLOT:
        centroids, keys = _centroid_matrix(
            X_scaled, labels, lambda l: (int(l[1]), int(l[2])))
        def label_fn_cs(key):
            c, s       = key
            hex_color  = color_mapping_color[color_map[c].name]
            shape_name = shape_map[s].name
            return None, hex_color, shape_name
        result['color_x_shape'] = {'X': centroids, 'keys': keys, 'label_fn': label_fn_cs}

    if 'task_only' in DENDROGRAMS_TO_PLOT:
        centroids, keys = _centroid_matrix(
            X_scaled, labels, lambda l: (int(l[0]),))
        def label_fn_t(key):
            (t,)   = key
            letter = task_mapping_letter[task_map[t].name]
            return letter, '#000000', None
        result['task_only'] = {'X': centroids, 'keys': keys, 'label_fn': label_fn_t}

    if 'color_only' in DENDROGRAMS_TO_PLOT:
        centroids, keys = _centroid_matrix(
            X_scaled, labels, lambda l: (int(l[1]),))
        def label_fn_c(key):
            (c,)      = key
            hex_color = color_mapping_color[color_map[c].name]
            return None, hex_color, None
        result['color_only'] = {'X': centroids, 'keys': keys, 'label_fn': label_fn_c}

    if 'shape_only' in DENDROGRAMS_TO_PLOT:
        centroids, keys = _centroid_matrix(
            X_scaled, labels, lambda l: (int(l[2]),))
        def label_fn_s(key):
            (s,)       = key
            shape_name = shape_map[s].name
            return None, '#000000', shape_name
        result['shape_only'] = {'X': centroids, 'keys': keys, 'label_fn': label_fn_s}

    return result


def _human_title(name):
    return {
        'all_180':       'All Task × Color × Shape (180 centroids)',
        'task_x_color':  'Task × Color — averaged over shape (36 centroids)',
        'task_x_shape':  'Task × Shape — averaged over color (30 centroids)',
        'color_x_shape': 'Color × Shape — averaged over task (30 centroids)',
        'task_only':     'Task only — averaged over color + shape (6 centroids)',
        'color_only':    'Color only — averaged over task + shape (6 centroids)',
        'shape_only':    'Shape only — averaged over task + color (5 centroids)',
    }[name]


# ============================
# PLOT  (your vertical style)
# ============================

def plot_dendrogram(centroid_info, dg_name, plot_dict, component, epoch,
                    max_labels_per_cluster=None):
    args   = plot_dict['args']
    X      = centroid_info['X']
    keys   = centroid_info['keys']
    lbl_fn = centroid_info['label_fn']

    Z           = linkage(X, method=LINKAGE_METHOD)
    cluster_ids = fcluster(Z, t=K, criterion='maxclust')

    # Leaf order from the full un-truncated dendrogram
    d_full     = dendrogram(Z, no_plot=True, no_labels=True)
    leaf_order = d_full['leaves']

    # Build cluster membership in left-to-right tree order
    cluster_to_indices = {}
    cluster_order      = []
    seen               = set()
    for leaf_idx in leaf_order:
        cid = cluster_ids[leaf_idx]
        cluster_to_indices.setdefault(cid, []).append(leaf_idx)
        if cid not in seen:
            seen.add(cid)
            cluster_order.append(cid)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 16), dpi=dpi, constrained_layout=True,
        gridspec_kw={'width_ratios': [4, 1]})
    ax_tree   = axes[0]
    ax_legend = axes[1]

    plt.suptitle(
        f'Hierarchical Clustering (HCA)\n'
        f'{args.arg_name} | Agent {agent_num_str} | Epoch {epoch} | {component}',
        fontsize=16)

    # Draw truncated tree (K pseudo-nodes)
    dendrogram(Z, ax=ax_tree, no_labels=True,
               truncate_mode='lastp', p=K, color_threshold=None)

    ax_tree.set_title(f'Hierarchical Clustering (showing {K} clusters)\n'
                      f'{_human_title(dg_name)}')
    ax_tree.set_ylabel('Distance')
    ax_tree.set_xticks([])

    # ── Glyph positions ───────────────────────────────────────────────────────
    x_min, x_max  = ax_tree.get_xlim()
    cluster_width = (x_max - x_min) / len(cluster_order)
    y_max  = ax_tree.get_ylim()[1]
    base_y = -0.06 * y_max
    step_y = -step_y_scaler * y_max

    max_cluster_size_seen = 0

    for col, cid in enumerate(cluster_order):
        x_pos   = x_min + (col + 0.5) * cluster_width
        members = cluster_to_indices[cid]

        # Sort by task → color → shape priority
        members = sorted(members, key=lambda idx: (
            task_priority.get(task_map[keys[idx][0]].name,   999) if len(keys[idx]) > 0 and keys[idx][0] != -1 else 999,
            color_priority.get(color_map[keys[idx][1]].name, 999) if len(keys[idx]) > 1 and keys[idx][1] != -1 else 999,
            shape_priority.get(shape_map[keys[idx][2]].name, 999) if len(keys[idx]) > 2 and keys[idx][2] != -1 else 999,
        ))

        max_cluster_size_seen = max(max_cluster_size_seen, len(members))

        if max_labels_per_cluster is not None and len(members) > max_labels_per_cluster:
            members_to_plot = members[:max_labels_per_cluster]
            truncated = True
        else:
            members_to_plot = members
            truncated = False

        for j, idx in enumerate(members_to_plot):
            key        = keys[idx]
            letter, hex_color, shape_name = lbl_fn(key)
            color_val  = hex_color if hex_color else '#777777'

            y = base_y + j * step_y

            if shape_name is not None:
                icon_color  = color_map[key[1]].name if len(key) > 1 else 'BLACK'
                # For color_x_shape and shape_only, use the color from the key if present
                if hex_color and hex_color != '#000000':
                    # find the color name matching this hex
                    icon_color = next(
                        (cn for cn, cv in color_mapping_color.items() if cv == hex_color),
                        'BLACK')
                colored_img = shape_mapping_colored_marker[shape_name][icon_color]
                imagebox    = OffsetImage(colored_img, zoom=shape_size)
                ab          = AnnotationBbox(imagebox, (x_pos, y), frameon=False, zorder=1)
                ax_tree.add_artist(ab)

            if letter is not None:
                ax_tree.scatter(x_pos, y, color=color_val,
                                marker=f'${letter}$',
                                s=letter_w_size if letter == 'W' else letter_size,
                                edgecolor='none', zorder=2)

            if letter is None and shape_name is None:
                # color-only dot
                ax_tree.scatter(x_pos, y, color=color_val,
                                marker='o', s=color_dot_size,
                                edgecolor='none', zorder=2)

        if truncated:
            y_trunc = base_y + len(members_to_plot) * step_y
            ax_tree.text(x_pos, y_trunc, '...', ha='center', va='top', fontsize=10)

    ax_tree.set_ylim(bottom=base_y + (max_cluster_size_seen + 2) * step_y)

    # ── Legend ────────────────────────────────────────────────────────────────
    y = 10
    for t_name, letter in task_mapping_letter.items():
        ax_legend.scatter(.1, y, color='black', marker=f'${letter}$', alpha=0.4,
                          s=letter_w_size if letter == 'W' else letter_size,
                          edgecolor='none')
        ax_legend.text(.3, y, t_name, ha='left', va='center', fontsize=fontsize)
        y -= .5

    for c_name, c_color in color_mapping_color.items():
        ax_legend.scatter(.1, y, facecolors=c_color, edgecolors='none',
                          s=color_dot_size, alpha=0.4)
        ax_legend.text(.3, y, c_name, ha='left', va='center', fontsize=fontsize)
        y -= .5

    for s_name in shape_mapping_marker:
        colored_img = shape_mapping_colored_marker[s_name]['BLACK']
        imagebox    = OffsetImage(colored_img, zoom=shape_size)
        ab          = AnnotationBbox(imagebox, (.1, y), frameon=False, alpha=0.4, zorder=1)
        ax_legend.add_artist(ab)
        ax_legend.text(.3, y, s_name, ha='left', va='center', fontsize=fontsize)
        y -= .5

    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    ax_legend.set_xlim([0, .8])
    ax_legend.set_ylim([1, 10.5])

    # ── Save ──────────────────────────────────────────────────────────────────
    save_dir = (f'saved_deigo/thesis_pics/composition/{args.arg_name}/'
                f'agent_{agent_num_str}/{component}/hca')
    os.makedirs(save_dir, exist_ok=True)
    fname = (f'{save_dir}/{component}_{K}_clusters_{dg_name}_method_{LINKAGE_METHOD}_{str(epoch).zfill(6)}.png')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')


# ============================
# RUN
# ============================

def run_hca(plot_dict, component, epoch):
    args      = plot_dict['args']
    agent_num = 0
    data_dict = meta_data_dict[(args.arg_name, agent_num, component)][epoch]
    centroid_sets = build_centroid_sets(data_dict)
    for dg_name, centroid_info in centroid_sets.items():
        plot_dendrogram(centroid_info, dg_name, plot_dict, component, epoch)


for component in ['command_voice_zq']:
    get_all_data(plot_dict, component)
    run_hca(plot_dict, component, the_epoch)

print(f'\nDuration: {duration()}. Done!')
# %%