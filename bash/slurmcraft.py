#%%
from copy import deepcopy
import argparse, json
from math import pi
from itertools import product
parser = argparse.ArgumentParser()
parser.add_argument('--comp',         type=str,  default = 'deigo')
parser.add_argument('--agents',       type=int,  default = 10)
parser.add_argument('--arg_list',     type=str,  default = [])
try:    args = parser.parse_args()
except: args, _ = parser.parse_known_args()

# This file works with communication.sh to implement many different parameters.

if type(args.arg_list) != list: 
    args.arg_list = json.loads(args.arg_list)
combined = '___{}___'.format('+'.join(args.arg_list))    

import os 
try:    os.chdir('communication/bash')
except: pass



# Given sets of arguments, return all possible combinations.
def expand_args(name, args):
    combos = [{}]
    complex = False
    for key, value in args.items():
        if type(value) != list:
            for combo in combos:
                combo[key] = value
        else: 
            complex = True
            if value[0] == 'num_min_max': 
                num, min_val, max_val = value[1]
                num = int(num)
                min_val = float(min_val)
                max_val = float(max_val)
                value = [min_val + i*((max_val - min_val) / (num - 1)) for i in range(num)]
            new_combos = []
            for v in value:
                temp_combos = deepcopy(combos)
                for combo in temp_combos: 
                    combo[key] = v 
                    new_combos.append(combo)   
            combos = new_combos  
    if complex and name[-1] != '_': 
        name += '_'
    return(name, combos)

def convert_list(input_list):
    converted = ['\\[' + ','.join(map(str, sub_list)) + '\\]' for sub_list in input_list]
    return(converted)



slurm_dict = {'d' : {}}



# Add these arguments to slurm_dict.
def add_this(name, args):
    keys, values = [], []
    for key, value in slurm_dict.items(): keys.append(key) ; values.append(value)
    for key, value in zip(keys, values):  
        if key == 'd': 
            key = ''
        between = '' if key == '' or len(name) == 1 else '_'
        new_key = key + between + name 
        new_value = deepcopy(value)
        for arg_name, arg in args.items():
            if type(arg) != list: 
                new_value[arg_name] = arg
            elif type(arg[0]) != list: 
                new_value[arg_name] = arg
            else:
                for condition in arg:
                    for if_arg_name, if_arg in condition[0].items():
                        if if_arg_name in value and value[if_arg_name] == if_arg:
                            new_value[arg_name] = condition[1]
        slurm_dict[new_key] = new_value

# Agents with entropy
add_this('entropy',   {
    'alpha' : 'None', 
    'normal_alpha' : .05,
    'target_entropy' : -1.5})    

vision_eta = .03     
touch_eta = 1       
prop_eta = 1      
feedback_eta = .3   

# Curiosity of senses only
add_this('sensory_motor_curiosity',   {                                           
    'curiosity' : 'hidden_state',
    'hidden_state_eta_vision' : vision_eta,
    'hidden_state_eta_touch' : touch_eta,
    'hidden_state_eta_prop' : prop_eta})

# Agents with curiosity (all)
add_this('all_curiosity',   {
    'curiosity' : 'hidden_state',
    'hidden_state_eta_vision' : vision_eta,
    'hidden_state_eta_touch' : touch_eta, 
    'hidden_state_eta_prop' : prop_eta,     
    'hidden_state_eta_feedback_voice' : feedback_eta})

add_this('lstm', {
    'lstm' : 'True',
    'pvrnn_mtrnn_size' : 384})

add_this('three_tasks', {
    'watch' : True,
    'be_near' : True,
    'touch_top' : False,
    'push_forward' : True,
    'push_left' : False,
    'push_right' : False
})


add_this('exceptions', {
    'exceptions' : 1,
})

add_this('pseudoexceptions', {
    'exceptions' : 2,
})

add_this('smallest_vocab',   { 
    'touch_top' : False,
    'be_near' : False,
    'magenta' : False,
    'yellow' : False,
    'cone' : False,
    'hourglass' : False,
    'test_train_num' : 1
})

add_this('smaller_vocab',   {
    'touch_top' : False,
    'yellow' : False,
    'cone' : False,
    'hourglass' : False,
    'test_train_num' : 2
})

add_this('long', {
    'epochs' : 120000,
})



new_slurm_dict = {}
for key, value in slurm_dict.items():
    key, combos = expand_args(key, value)
    if len(combos) == 1: 
        new_slurm_dict[key] = combos[0] 
    else:
        for i, combo in enumerate(combos): new_slurm_dict[key + str(i+1)] = combo
        
slurm_dict = new_slurm_dict

def get_args(name):
    s = '' 
    for key, value in slurm_dict[name].items(): s += '--{} {} '.format(key, value)
    return(s)

def all_like_this(this): 
    if this in ['break', 'empty_space']: 
        result = [this]
    elif this[-1] != '_':                
        result = [this]
    else: 
        result = [key for key in slurm_dict.keys() if key.startswith(this) and key[len(this):].isdigit()]
    return(json.dumps(result))

            

max_cpus = args.agents if args.agents < 30 else 30
 
if __name__ == '__main__' and args.arg_list == []:
    print('ALL POSSIBLE HYPERPARAMETERS:')
    for key, value in slurm_dict.items(): 
        print(key, ':', value)
    print(len(slurm_dict.items()))
    interesting = []
    if len(interesting) != 0:
        print('\n\n\nTHESE HYPERPARAMETERS:')
        for this in interesting:
            print('{} : {}'.format(this,slurm_dict[this]))



if __name__ == '__main__' and args.arg_list != []:
    
    if args.comp == 'deigo':
        nv = ''
        module = 'module load singularity'
        partition = \
'''
#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 96:00:00
#SBATCH --mem=50G'''

    if args.comp == 'saion':
        nv = ' --nv'
        module = 'module load singularity cuda'
        partition = \
'''
#!/bin/bash -l
#SBATCH --partition=taniu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 48:00:00
#SBATCH --mem=490G
#SBATCH --gres=gpu:4'''
    for name in args.arg_list:
        if name in ['break', 'empty_space']: 
            pass 
        else:
            with open('main_{}.slurm'.format(name), 'w') as f:
                f.write(
f'''
{partition}
#SBATCH --ntasks={max_cpus}
{module}
singularity exec{nv} communication.sif python communication/main.py --comp {args.comp} --arg_name {name} {get_args(name)} --agents $agents_per_job --previous_agents $previous_agents
'''[2:])
            


    with open('finish_dicts.slurm', 'w') as f:
        f.write(
f'''
{partition}
{module}
singularity exec{nv} communication.sif python communication/finish_dicts.py --comp {args.comp} --arg_title {combined} --arg_name finishing_dictionaries
'''[2:])
        
    with open('plotting.slurm', 'w') as f:
        f.write(
f'''
{partition}
{module}
singularity exec{nv} communication.sif python communication/plotting.py --comp {args.comp} --arg_title {combined} --arg_name plotting
'''[2:])
        
    with open('plotting_composition.slurm', 'w') as f:
        f.write(
f'''
{partition}
{module}
singularity exec{nv} communication.sif python communication/plotting_composition.py --comp {args.comp} --arg_title {combined} --arg_name plotting_composition
'''[2:])
        
    with open('plotting_episodes.slurm', 'w') as f:
        f.write(
f'''
{partition}
{module}
singularity exec{nv} communication.sif python communication/plotting_episodes.py --comp {args.comp} --arg_title {combined} --arg_name plotting_episodes
'''[2:])
        
    with open('plotting_p_values.slurm', 'w') as f:
        f.write(
f'''
{partition}
{module}
singularity exec{nv} communication.sif python communication/plotting_p_val.py --comp {args.comp} --arg_title {combined} --arg_name plotting_p_values
'''[2:])
        
    with open('plotting_behavior.slurm', 'w') as f:
        f.write(
f'''
{partition}
{module}
singularity exec{nv} communication.sif python communication/plotting_behavior.py --comp {args.comp} --arg_title {combined} --arg_name plotting_behavior
'''[2:])
        
    with open('combine_plots.slurm', 'w') as f:
        f.write(
f'''
{partition}
{module}
singularity exec{nv} communication.sif python communication/combine_plots.py --comp {args.comp} --arg_title {combined} --arg_name combining_plots
'''[2:])
# %%

