#%%
import numpy as np
from math import log

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary as torch_summary

from utils import print
from utils_submodule import model_start, model_end, init_weights, pad_zeros, var, sample
from mtrnn import MTRNN



# Module for encoding RGBD images.
class Vision_IN(nn.Module):

    def __init__(self, args):
        super(Vision_IN, self).__init__()

        self.args = args
        image_dims = 4

        vision_size = (1, image_dims, self.args.image_size, self.args.image_size)
        example = torch.zeros(vision_size)

        self.a = nn.Sequential()
        example = self.a(example)
        vision_latent_size = example.flatten(1).shape[1]

        self.b = nn.Sequential(
            nn.Linear(vision_latent_size, self.args.vision_encode_size),
            nn.PReLU(),
            nn.Dropout(self.args.dropout)
        )

        self.apply(init_weights)
        self.to(self.args.device)
        if self.args.half:
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)

    def forward(self, vision):
        start_time, episodes, steps, [vision] = model_start([(vision, 'cnn')], self.args.device, self.args.half)
        vision = (vision * 2) - 1
        a = self.a(vision).flatten(1)
        encoding = self.b(a)
        [encoding] = model_end(start_time, episodes, steps, [(encoding, 'lin')], '\tVision_IN')
        return encoding



if __name__ == '__main__':
    from utils import args
    episodes = args.batch_size ; steps = args.max_steps

    vision_in = Vision_IN(args = args)

    print('\n\n')
    print(vision_in)
    print()
    with profile(activities = [ProfilerActivity.CPU], record_shapes = True) as prof:
        with record_function('model_inference'):
            print(torch_summary(vision_in, (episodes, steps, args.image_size, args.image_size, 4)))
    print(prof.key_averages().table(sort_by = 'cpu_time_total', row_limit = 100))



#%%



# Module for decoding RGBD images.
class Vision_OUT(nn.Module):

    def __init__(self, args):
        super(Vision_OUT, self).__init__()

        self.args = args
        self.out_features_channels = self.args.hidden_size

        self.a = nn.Sequential(
            nn.Linear(
                self.args.h_w_wheels_joints_size,
                self.out_features_channels * (self.args.image_size // self.args.divisions) ** 2
            )
        )

        self.b = nn.Sequential(
            nn.BatchNorm2d(self.out_features_channels),
            nn.PReLU(),
            nn.Dropout(self.args.dropout),

            nn.Conv2d(
                in_channels = self.out_features_channels,
                out_channels = 4 * (1 if self.args.divisions == 1 else 2 ** self.args.divisions),
                kernel_size = 3,
                padding = 1,
                padding_mode = 'reflect'
            ),
            nn.PixelShuffle(self.args.divisions),
            nn.Tanh()
        )

        self.apply(init_weights)
        self.to(self.args.device)
        if self.args.half:
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)

    def forward(self, h_w_wheels_joints):
        start_time, episodes, steps, [h_w_wheels_joints] = model_start([(h_w_wheels_joints, 'lin')], self.args.device, self.args.half)

        a = self.a(h_w_wheels_joints)
        a = a.reshape(
            episodes * steps,
            self.out_features_channels,
            self.args.image_size // self.args.divisions,
            self.args.image_size // self.args.divisions
        )
        vision = self.b(a)
        vision = (vision + 1) / 2

        [vision] = model_end(start_time, episodes, steps, [(vision, 'cnn')], '\tVision_OUT')
        return vision



if __name__ == '__main__':
    from utils import args
    episodes = args.batch_size ; steps = args.max_steps

    vision_out = Vision_OUT(args = args)

    print('\n\n')
    print(vision_out)
    print()
    with profile(activities = [ProfilerActivity.CPU], record_shapes = True) as prof:
        with record_function('model_inference'):
            print(torch_summary(vision_out, (episodes, steps, args.h_w_wheels_joints_size)))
    print(prof.key_averages().table(sort_by = 'cpu_time_total', row_limit = 100))



#%%


# Module for encoding tactile sensations.
class Touch_IN(nn.Module):

    def __init__(self, args):
        super(Touch_IN, self).__init__()

        self.args = args

        self.a = nn.Sequential(
            nn.Linear(self.args.touch_shape, self.args.touch_encode_size),
            nn.BatchNorm1d(self.args.touch_encode_size),
            nn.PReLU()
        )

        self.apply(init_weights)
        self.to(self.args.device)
        if self.args.half:
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)

    def forward(self, touch):
        start_time, episodes, steps, [touch] = model_start([(touch, 'lin')], self.args.device, self.args.half)
        encoding = self.a(touch)
        [encoding] = model_end(start_time, episodes, steps, [(encoding, 'lin')], '\tTouch_IN')
        return encoding



if __name__ == '__main__':
    from utils import args
    episodes = args.batch_size ; steps = args.max_steps

    touch_in = Touch_IN(args = args)

    print('\n\n')
    print(touch_in)
    print()
    with profile(activities = [ProfilerActivity.CPU], record_shapes = True) as prof:
        with record_function('model_inference'):
            print(torch_summary(touch_in, (episodes, steps, args.touch_shape)))
    print(prof.key_averages().table(sort_by = 'cpu_time_total', row_limit = 100))



#%%
# Module for decoding tactile sensations.
class Touch_OUT(nn.Module):

    def __init__(self, args):
        super(Touch_OUT, self).__init__()

        self.args = args

        self.a = nn.Sequential(
            nn.Linear(
                in_features = self.args.h_w_wheels_joints_size,
                out_features = self.args.touch_shape
            ),
            nn.Tanh()
        )

        self.apply(init_weights)
        self.to(self.args.device)
        if self.args.half:
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)

    def forward(self, h_w_wheels_joints):
        start_time, episodes, steps, [h_w_wheels_joints] = model_start([(h_w_wheels_joints, 'lin')], self.args.device, self.args.half)
        touch = self.a(h_w_wheels_joints)
        touch = (touch + 1) / 2
        [touch] = model_end(start_time, episodes, steps, [(touch, 'lin')], '\tTouch_OUT')
        return touch



if __name__ == '__main__':
    from utils import args
    episodes = args.batch_size ; steps = args.max_steps

    touch_out = Touch_OUT(args = args)

    print('\n\n')
    print(touch_out)
    print()
    with profile(activities = [ProfilerActivity.CPU], record_shapes = True) as prof:
        with record_function('model_inference'):
            print(torch_summary(touch_out, (episodes, steps, args.h_w_wheels_joints_size)))
    print(prof.key_averages().table(sort_by = 'cpu_time_total', row_limit = 100))



#%%


# Module for encoding proprioception.
class Prop_IN(nn.Module):

    def __init__(self, args):
        super(Prop_IN, self).__init__()

        self.args = args

        self.a = nn.Sequential(
            nn.Linear(self.args.joint_aspects, self.args.prop_encode_size),
            nn.BatchNorm1d(self.args.prop_encode_size),
            nn.PReLU()
        )

        self.apply(init_weights)
        self.to(self.args.device)
        if self.args.half:
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)

    def forward(self, prop):
        start_time, episodes, steps, [prop] = model_start([(prop, 'lin')], self.args.device, self.args.half)
        encoding = self.a(prop)
        [encoding] = model_end(start_time, episodes, steps, [(encoding, 'lin')], '\tProp_IN')
        return encoding



if __name__ == '__main__':
    from utils import args
    episodes = args.batch_size ; steps = args.max_steps

    prop_in = Prop_IN(args = args)

    print('\n\n')
    print(prop_in)
    print()
    with profile(activities = [ProfilerActivity.CPU], record_shapes = True) as prof:
        with record_function('model_inference'):
            print(torch_summary(prop_in, (episodes, steps, args.joint_aspects)))
    print(prof.key_averages().table(sort_by = 'cpu_time_total', row_limit = 100))



#%%



# Module for decoding proprioception.
class Prop_OUT(nn.Module):

    def __init__(self, args):
        super(Prop_OUT, self).__init__()

        self.args = args

        self.a = nn.Sequential(
            nn.Linear(
                in_features = self.args.h_w_wheels_joints_size,
                out_features = self.args.joint_aspects
            ),
            nn.Tanh()
        )

        self.apply(init_weights)
        self.to(self.args.device)
        if self.args.half:
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)

    def forward(self, h_w_wheels_joints):
        start_time, episodes, steps, [h_w_wheels_joints] = model_start([(h_w_wheels_joints, 'lin')], self.args.device, self.args.half)
        prop = self.a(h_w_wheels_joints)
        prop = (prop + 1) / 2
        [prop] = model_end(start_time, episodes, steps, [(prop, 'lin')], '\tProp_OUT')
        return prop



if __name__ == '__main__':
    from utils import args
    episodes = args.batch_size ; steps = args.max_steps

    prop_out = Prop_OUT(args = args)

    print('\n\n')
    print(prop_out)
    print()
    with profile(activities = [ProfilerActivity.CPU], record_shapes = True) as prof:
        with record_function('model_inference'):
            print(torch_summary(prop_out, (episodes, steps, args.h_w_wheels_joints_size)))
    print(prof.key_averages().table(sort_by = 'cpu_time_total', row_limit = 100))




#%%



# Module for encoding voices (same model for command and feedback).
class Voice_IN(nn.Module):

    def __init__(self, args):
        super(Voice_IN, self).__init__()

        self.args = args

        self.a = nn.Sequential(
            nn.Embedding(
                num_embeddings = self.args.voice_shape,
                embedding_dim = self.args.char_encode_size
            ),
            nn.PReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(
                in_features = self.args.char_encode_size,
                out_features = self.args.hidden_size
            )
        )

        self.ab = nn.Sequential(
            nn.PReLU()
        )

        self.b = nn.GRU(
            input_size = self.args.hidden_size,
            hidden_size = self.args.hidden_size,
            batch_first = True
        )

        self.c = nn.Sequential()  # I tested BatchNorm1d here. It was not helpful.

        self.cb = nn.Sequential(
            nn.PReLU(),
            nn.Linear(
                in_features = self.args.hidden_size,
                out_features = self.args.voice_encode_size
            )
        )

        self.apply(init_weights)
        self.to(self.args.device)
        if self.args.half:
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)

    def forward(self, voice, return_a_b = False):
        start_time, episodes, steps, [voice] = model_start([(voice, 'voice')], self.args.device, self.args.half)

        self.to('cpu')
        voice = pad_zeros(voice, self.args.max_voice_len)
        voice = torch.argmax(voice, dim = -1).int()
        
        a = self.a(voice)
        a = a.permute(0, 2, 1)
        a = self.ab(a)
        a = a.permute(0, 2, 1)
        _, b = self.b(a)
        
        b = b.reshape(episodes, steps, self.args.hidden_size)
        b = b.permute(0, 2, 1)
        c = self.c(b)
        c = c.permute(0, 2, 1)

        encoding = self.cb(c)
        
        [encoding] = model_end(start_time, episodes, steps, [(encoding, 'lin')], '\tVoice_IN')
        if return_a_b:
            a = a.reshape(episodes, steps, a.shape[-2], a.shape[-1])
            return encoding, a, b
        else:
            return encoding


if __name__ == '__main__':
    from utils import args
    episodes = args.batch_size ; steps = args.max_steps

    voice_in = Voice_IN(args = args)

    print('\n\n')
    print(voice_in)
    print()
    with profile(activities = [ProfilerActivity.CPU], record_shapes = True) as prof:
        with record_function('model_inference'):
            print(torch_summary(voice_in, (episodes, steps, args.max_voice_len, args.voice_shape)))
    print(prof.key_averages().table(sort_by = 'cpu_time_total', row_limit = 100))



#%%



# Module for decoding voices (two separate modules for command and feedback).
class Voice_OUT(nn.Module):

    def __init__(self, args, actor = False):
        super(Voice_OUT, self).__init__()

        self.args = args
        self.actor = actor

        self.a = nn.Sequential(
            nn.Linear(
                in_features = self.args.h_w_wheels_joints_size,
                out_features = self.args.hidden_size * self.args.max_voice_len
            )
        )

        self.ab = nn.Sequential(
            nn.BatchNorm1d(self.args.hidden_size * self.args.max_voice_len)
        )

        self.ac = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(self.args.dropout)
        )

        self.b = nn.GRU(
            input_size = self.args.hidden_size,
            hidden_size = self.args.hidden_size,
            batch_first = True
        )

        self.mu = nn.Sequential(
            nn.Linear(
                in_features = self.args.hidden_size,
                out_features = self.args.voice_shape
            )
        )

        self.std = nn.Sequential(
            nn.Linear(
                in_features = self.args.hidden_size,
                out_features = self.args.voice_shape
            ),
            nn.Softmax(dim = -1)
        )

        self.apply(init_weights)
        self.to(self.args.device)
        if self.args.half:
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)

    def forward(self, h_w_wheels_joints):
        start_time, episodes, steps, [h_w_wheels_joints] = model_start([(h_w_wheels_joints, 'lin')], self.args.device, self.args.half)

        h_w_wheels_joints = h_w_wheels_joints.reshape(episodes * steps, self.args.h_w_wheels_joints_size)
        a = self.a(h_w_wheels_joints)
        a = self.ab(a)
        a = self.ac(a)
        a = a.reshape(episodes * steps, self.args.max_voice_len, self.args.hidden_size)

        b, _ = self.b(a)

        if self.actor:
            mu, std = var(b, self.mu, self.std, self.args)
            voice = sample(mu, std, self.args.device)
            voice_out = torch.tanh(voice)
            log_prob = Normal(mu, std).log_prob(voice) - torch.log(1 - voice_out.pow(2) + 1e-6)
            log_prob = torch.mean(log_prob, -1).unsqueeze(-1)
            log_prob = log_prob.mean(-2)
            voice_out = voice_out.reshape(episodes, steps, self.args.max_voice_len, self.args.voice_shape)
            log_prob = log_prob.reshape(episodes, steps, 1)
            return voice_out, log_prob
        else:
            voice_pred = self.mu(b)
            voice_pred = voice_pred.reshape(episodes, steps, self.args.max_voice_len, self.args.voice_shape)
            return voice_pred



if __name__ == '__main__':
    from utils import args
    episodes = args.batch_size ; steps = args.max_steps

    voice_out = Voice_OUT(actor = False, args = args)

    print('\n\n')
    print(voice_out)
    print()
    with profile(activities = [ProfilerActivity.CPU], record_shapes = True) as prof:
        with record_function('model_inference'):
            print(torch_summary(voice_out, (episodes, steps, args.h_w_wheels_joints_size)))
    print(prof.key_averages().table(sort_by = 'cpu_time_total', row_limit = 100))



#%%



# Module for decoding all parts of the sensory observation.
class Obs_OUT(nn.Module):

    def __init__(self, args):
        super(Obs_OUT, self).__init__()

        self.args = args

        self.vision_out = Vision_OUT(self.args)
        self.touch_out = Touch_OUT(self.args)
        self.prop_out = Prop_OUT(self.args)
        self.command_voice_out = Voice_OUT(actor = False, args = self.args)
        self.feedback_voice_out = Voice_OUT(actor = False, args = self.args)

        self.apply(init_weights)
        self.to(self.args.device)
        if self.args.half:
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)

    def forward(self, h_w_wheels_joints):
        vision_pred = self.vision_out(h_w_wheels_joints)
        touch_pred = self.touch_out(h_w_wheels_joints)
        prop_pred = self.prop_out(h_w_wheels_joints)
        command_voice_pred = self.command_voice_out(h_w_wheels_joints)
        feedback_voice_pred = self.feedback_voice_out(h_w_wheels_joints)
        return vision_pred, touch_pred, prop_pred, command_voice_pred, feedback_voice_pred



if __name__ == '__main__':
    from utils import args
    episodes = args.batch_size ; steps = args.max_steps

    obs_out = Obs_OUT(args = args)

    print('\n\n')
    print(obs_out)
    print()
    with profile(activities = [ProfilerActivity.CPU], record_shapes = True) as prof:
        with record_function('model_inference'):
            print(torch_summary(obs_out, (episodes, steps, args.h_w_wheels_joints_size)))
    print(prof.key_averages().table(sort_by = 'cpu_time_total', row_limit = 100))



#%%



# Module for encoding motor commands.
class Wheels_Joints_IN(nn.Module):

    def __init__(self, args):
        super(Wheels_Joints_IN, self).__init__()

        self.args = args

        self.a = nn.Sequential(
            nn.Linear(
                in_features = self.args.wheels_joints_shape,
                out_features = self.args.wheels_joints_encode_size
            ),
            nn.PReLU()
        )

        self.apply(init_weights)
        self.to(self.args.device)
        if self.args.half:
            self = self.half()
            torch.nn.utils.clip_grad_norm_(self.parameters(), .1)

    def forward(self, wheels_joints):
        start_time, episodes, steps, [wheels_joints] = model_start([(wheels_joints, 'lin')], self.args.device, self.args.half)
        encoded = self.a(wheels_joints)
        [encoded] = model_end(start_time, episodes, steps, [(encoded, 'lin')], '\tWheels_joints_IN')
        return encoded



if __name__ == '__main__':
    from utils import args
    episodes = args.batch_size ; steps = args.max_steps

    wheels_joints_in = Wheels_Joints_IN(args = args)

    print('\n\n')
    print(wheels_joints_in)
    print()
    with profile(activities = [ProfilerActivity.CPU], record_shapes = True) as prof:
        with record_function('model_inference'):
            print(torch_summary(wheels_joints_in, (episodes, steps, args.wheels_joints_shape)))
    print(prof.key_averages().table(sort_by = 'cpu_time_total', row_limit = 100))
