import os
import pickle
import re
import sys

from utils import duration, args, print, save_file

print('name:\n{}'.format(args.arg_name))

# Adjust file patterns based on 'temp' variable
if args.temp:
    plot_dict_pattern = re.compile(r'^plot_dict_temp_\d{3}\.pickle$')
    min_max_dict_pattern = re.compile(r'^min_max_dict_temp_\d{3}\.pickle$')
else:
    plot_dict_pattern = re.compile(r'^plot_dict_\d{3}\.pickle$')
    min_max_dict_pattern = re.compile(r'^min_max_dict_\d{3}\.pickle$')

os.chdir(save_file)
complete_order = args.arg_title[3:-3].split('+')
folders = [o for o in complete_order if o not in ['empty_space', 'break']]


def truncate_lists_in_dict(d):
    """Ensure inner lists are truncated to the shortest length across entries."""
    for key, value in d.items():
        if isinstance(value, list) and all(isinstance(item, list) for item in value):
            if value:
                lengths = [len(lst) for lst in value if lst]
                if lengths:
                    min_length = min(lengths)
                    d[key] = [lst[:min_length] for lst in value]
                else:
                    d[key] = [[] for _ in value]
    return d


# Iterate over folders with robots' saved dictionaries of data.
for folder in folders:
    plot_dict = {}
    min_max_dict = {}

    files = os.listdir(folder)
    files.sort()

    print('{} files in folder {}.'.format(len(files), folder))

    # Only process files full of information regarding plotting.
    filtered_files = [
        file for file in files
        if plot_dict_pattern.match(file) or min_max_dict_pattern.match(file)
    ]

    if len(filtered_files) == 0:
        print(f'No matching files in {folder}, skipping.')
        continue

    # Process each file and populate the combined dictionaries.
    for file in filtered_files:
        print(file)
        if file.startswith('plot'):
            d = plot_dict
        elif file.startswith('min'):
            d = min_max_dict
        else:
            continue

        with open(os.path.join(folder, file), 'rb') as handle:
            saved_d = pickle.load(handle)

        for key in saved_d.keys():
            if key not in d:
                d[key] = []
            if key in ['args', 'arg_title', 'arg_name', 'all_processor_names']:
                d[key] = saved_d[key]
            else:
                d[key].append(saved_d[key])

    # Merge 'episode_dicts' and 'agent_lists' dictionaries
    episode_dicts = {}
    for d_item in plot_dict.get('episode_dicts', []):
        episode_dicts.update(d_item)
    plot_dict['episode_dicts'] = episode_dicts

    agent_lists = {}
    for d_item in plot_dict.get('agent_lists', []):
        agent_lists.update(d_item)
    plot_dict['agent_lists'] = agent_lists

    # Compute min and max for 'min_max_dict'.
    for key in min_max_dict.keys():
        if key not in [
            'args', 'arg_title', 'arg_name', 'all_processor_names',
            'composition_data', 'episode_dicts', 'agent_lists',
            'spot_names', 'steps', 'goal_task', 'behavior'
        ]:
            minimum = None
            maximum = None
            for min_max in min_max_dict[key]:
                if any(item in [None, 'not_used'] for item in min_max):
                    pass
                if minimum is None or minimum > min_max[0]:
                    minimum = min_max[0]
                if maximum is None or maximum < min_max[1]:
                    maximum = min_max[1]
            min_max_dict[key] = (minimum, maximum)

    # Truncate lists in dictionaries.
    plot_dict = truncate_lists_in_dict(plot_dict)
    min_max_dict = truncate_lists_in_dict(min_max_dict)

    # Write the new output files (overwrite if they already exist).
    with open(os.path.join(folder, 'min_max_dict.pickle'), 'wb') as handle:
        pickle.dump(min_max_dict, handle)

    with open(os.path.join(folder, 'plot_dict.pickle'), 'wb') as handle:
        pickle.dump(plot_dict, handle)

    # Remove only the files we processed
    for file in filtered_files:
        os.remove(os.path.join(folder, file))

    print(f'Finished processing and saving for folder {folder}.')

print('\nDuration: {}. Done!\n'.format(duration()))
