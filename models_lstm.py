#%%

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary as torch_summary

from utils import default_args, print, duration, Action, Obs
from utils_submodule import init_weights, episodes_steps, var, sample, model_start, model_end
from mtrnn import MTRNN
from submodules import Vision_IN, Touch_IN, Prop_IN, Voice_IN, Voice_OUT, Obs_OUT, Wheels_Joints_IN



class Actor(nn.Module):
    """
    Actor (policy network): generates motor commands and, optionally, the robot's voice.
    """

    def __init__(self, args=default_args):
        super(Actor, self).__init__()

        self.args = args
        
        self.vision_in = Vision_IN(self.args)
        self.touch_in = Touch_IN(self.args)
        self.prop_in = Prop_IN(self.args)
        self.command_voice_in = Voice_IN(self.args)
        self.feedback_voice_in = Voice_IN(self.args)
        self.wheels_joints_in = Wheels_Joints_IN(self.args)
        self.prev_voice_in = Voice_IN(self.args)
        
        self.mtrnn = MTRNN(
            self.args.vision_state_size + 
                        self.args.touch_state_size +
                        self.args.prop_state_size + 
                        self.args.voice_state_size * 3 + 
                        self.args.wheels_joints_encode_size,
            hidden_size = self.args.pvrnn_mtrnn_size,
            time_constant = 1,
            args = self.args)
        
        self.lin = nn.Sequential(
            nn.Linear(self.args.pvrnn_mtrnn_size, self.args.hidden_size),
            nn.PReLU(),
            nn.Linear(self.args.hidden_size, self.args.hidden_size),
            nn.PReLU()
        )

        self.voice_out = Voice_OUT(actor=True, args=self.args)

        self.mu = nn.Linear(self.args.hidden_size, self.args.wheels_joints_shape)
        self.std = nn.Sequential(
            nn.Linear(self.args.hidden_size, self.args.wheels_joints_shape),
            nn.Softplus()
        )

        self.apply(init_weights)
        self.to(self.args.device)
        if self.args.half:
            self.half()

    def forward(self, obs, prev_action, hidden_state, parenting=True):
        """
        Returns motor commands and optionally voice output from the actor.
        hidden_state should be a tuple (h0, c0)
        """
        obs_and_prev_action_encoded = torch.cat([
            self.vision_in(obs.vision), 
            self.touch_in(obs.touch), 
            self.prop_in(obs.prop),
            self.command_voice_in(obs.command_voice),
            self.feedback_voice_in(obs.feedback_voice),
            self.wheels_joints_in(prev_action.wheels_joints),
            self.prev_voice_in(prev_action.voice_out)], dim = -1)
                
        batch_size = obs_and_prev_action_encoded.size(0)

        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, 1, self.args.pvrnn_mtrnn_size)
            
        new_hidden_state = self.mtrnn(obs_and_prev_action_encoded, hidden_state)
        
        x = self.lin(new_hidden_state)
        
        mu, std = var(x, self.mu, self.std, self.args)
        sampled = sample(mu, std, self.args.device)
        if self.args.half:
            sampled = sampled.half()
        wheels_joints = torch.tanh(sampled)

        log_prob = Normal(mu, std).log_prob(sampled) - torch.log(1 - wheels_joints.pow(2) + 1e-6)
        log_prob = log_prob.mean(-1).unsqueeze(-1)

        encoded_wheels_joints = self.wheels_joints_in(wheels_joints)
        concatenated = torch.cat([new_hidden_state, encoded_wheels_joints], dim=-1)
        voice_out, voice_log_prob = self.voice_out(concatenated)

        if parenting:
            voice_out = torch.zeros_like(voice_out)
            voice_log_prob = torch.zeros_like(voice_log_prob)
            if self.args.half:
                voice_out = voice_out.half()
                voice_log_prob = voice_log_prob.half()
                
        return Action(wheels_joints, voice_out), log_prob, voice_log_prob, new_hidden_state
    
    
    
if __name__ == '__main__':
    
    from utils import args
    episodes = args.batch_size
    steps = args.max_steps
    
    actor = Actor(args = args)
    
    print('\n\nActor')
    print(actor)
    print()
    
    example_obs = Obs(
        vision = torch.zeros((episodes, steps, args.image_size, args.image_size, 4)), 
        touch = torch.zeros((episodes, steps, args.touch_shape)), 
        prop = torch.zeros((episodes, steps, args.joint_aspects)), 
        command_voice = torch.zeros((episodes, steps, args.max_voice_len, args.voice_shape)),
        feedback_voice = torch.zeros((episodes, steps, args.max_voice_len, args.voice_shape)))
    
    example_prev_action = Action(
        wheels_joints = torch.zeros((episodes, steps, args.wheels_joints_shape)), 
        voice_out = torch.zeros((episodes, steps, args.max_voice_len, args.voice_shape)))
    
    example_hidden_state = torch.zeros((1, episodes, args.pvrnn_mtrnn_size), device=args.device)
    
    actor(example_obs, example_prev_action, example_hidden_state)
        
    total = sum(p.numel() for p in actor.parameters())
    trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    print(f"Total: {total:,}")
    print(f"Trainable: {trainable:,}")

    


#%%

class Critic(nn.Module):
    """
    Critic (Q-network): predicts expected value given an action and forward state.
    """

    def __init__(self, args=default_args):
        super(Critic, self).__init__()

        self.args = args
        
        self.vision_in = Vision_IN(self.args)
        self.touch_in = Touch_IN(self.args)
        self.prop_in = Prop_IN(self.args)
        self.command_voice_in = Voice_IN(self.args)
        self.feedback_voice_in = Voice_IN(self.args)
        self.wheels_joints_in = Wheels_Joints_IN(self.args)
        self.prev_voice_in = Voice_IN(self.args)
        
        self.mtrnn = MTRNN(
            self.args.vision_state_size + 
                        self.args.touch_state_size +
                        self.args.prop_state_size + 
                        self.args.voice_state_size * 3 + 
                        self.args.wheels_joints_encode_size,
            hidden_size = self.args.pvrnn_mtrnn_size,
            time_constant = 1,
            args = self.args)

        self.value = nn.Sequential(
            nn.Linear(self.args.pvrnn_mtrnn_size, self.args.hidden_size),
            nn.PReLU(),
            nn.Linear(self.args.hidden_size, self.args.hidden_size),
            nn.PReLU(),
            nn.Linear(self.args.hidden_size, 1)
        )

        self.apply(init_weights)
        self.to(self.args.device)
        if self.args.half:
            self.half()

    def forward(self, obs, action, hidden_state):
        """
        Returns predicted Q-value for given action and hidden state.
        hidden_state should be a tuple (h0, c0)
        """
        
        obs_and_prev_action_encoded = torch.cat([
            self.vision_in(obs.vision), 
            self.touch_in(obs.touch), 
            self.prop_in(obs.prop),
            self.command_voice_in(obs.command_voice),
            self.feedback_voice_in(obs.feedback_voice),
            self.wheels_joints_in(action.wheels_joints),
            self.prev_voice_in(action.voice_out)], dim = -1)
        
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, 1, self.args.pvrnn_mtrnn_size)

        new_hidden_state = self.mtrnn(obs_and_prev_action_encoded, hidden_state)
        value = self.value(new_hidden_state)

        return value, new_hidden_state



if __name__ == "__main__":
    """
    Test-time execution of Actor and Critic models with summary and profiling.
    Issue with device type.
    """
    args = default_args
    episodes = args.batch_size
    steps = args.max_steps

    critic = Critic(args)

    print("\n\nCRITIC\n")
    print(critic)
    
    example_obs = Obs(
        vision = torch.zeros((episodes, steps, args.image_size, args.image_size, 4)), 
        touch = torch.zeros((episodes, steps, args.touch_shape)), 
        prop = torch.zeros((episodes, steps, args.joint_aspects)), 
        command_voice = torch.zeros((episodes, steps, args.max_voice_len, args.voice_shape)),
        feedback_voice = torch.zeros((episodes, steps, args.max_voice_len, args.voice_shape)))
    
    example_action = Action(
        wheels_joints = torch.zeros((episodes, steps, args.wheels_joints_shape)), 
        voice_out = torch.zeros((episodes, steps, args.max_voice_len, args.voice_shape)))
    
    example_hidden_state = torch.zeros((1, episodes, args.pvrnn_mtrnn_size), device=args.device)
    
    critic(example_obs, example_action, example_hidden_state)
    
    total = sum(p.numel() for p in critic.parameters())
    trainable = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"Total: {total:,}")
    print(f"Trainable: {trainable:,}")

# %%
