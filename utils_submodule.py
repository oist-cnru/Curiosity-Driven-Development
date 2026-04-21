#%%
import torch
from torch.distributions import Normal
from torchvision.transforms import Resize
from math import log, sqrt
from torch import nn

from utils import duration, print, print_duration


# -------------------------------
#  INITIALIZATION
# -------------------------------

def init_weights(m):
    """Initialize weights of a neural network layer using Xavier normal and zero bias."""
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)
    except:
        pass


def episodes_steps(this):
    """Return the number of episodes and steps in a given batch tensor."""
    return this.shape[0], this.shape[1]


# -------------------------------
#  PREPROCESSING
# -------------------------------

def pad_zeros(value, length):
    """Pad a batch of sequences with zeros along the sequence (second-to-last) dimension."""
    rows_to_add = length - value.size(-2)
    if rows_to_add == 0:
        return value

    padding_shape = list(value.shape)
    padding_shape[-2] = rows_to_add

    device = value.device if value.get_device() != -1 else "cpu"
    padding = torch.zeros(padding_shape).to(device)
    padding[..., 0] = 1  # First element marks padding

    value = torch.cat([value, padding], dim=-2)
    return value


# -------------------------------
#  DISTRIBUTION HELPERS
# -------------------------------

def var(x, mu_func, std_func, args):
    """Compute mean and clamped standard deviation for a distribution."""
    mu = mu_func(x)
    std = torch.clamp(std_func(x), min=args.std_min, max=args.std_max)
    return mu, std


def sample(mu, std, device):
    """Sample from a normal distribution using reparameterization trick."""
    e = Normal(0, 1).sample(std.shape).to(device)
    return mu + e * std


# -------------------------------
#  CNN OVER RNN-LIKE INPUT
# -------------------------------

def rnn_cnn(do_this, to_this):
    """Apply a CNN to a batched sequence by flattening episodes and steps."""
    episodes, steps = episodes_steps(to_this)
    x = to_this.view(episodes * steps, to_this.shape[2], to_this.shape[3], to_this.shape[4])
    x = do_this(x)
    x = x.view(episodes, steps, x.shape[1], x.shape[2], x.shape[3])
    return x


# -------------------------------
#  MODEL PIPELINE ENTRY
# -------------------------------

def model_start(model_input_list, device="cpu", half=False, recurrent=False):
    """
    Prepares input tensors for model execution.
    
    Applies device/precision conversion, reshaping depending on layer type:
    - 'lin' for linear layers
    - 'cnn' for convolutional layers
    - 'voice' for sequence embeddings
    """
    start_time = duration()
    new_model_inputs = []

    for model_input, layer_type in model_input_list:
        model_input = model_input.to(device)
        if half:
            model_input = model_input.to(dtype=torch.float16)

        if layer_type == "lin":
            if len(model_input.shape) == 2:
                model_input = model_input.unsqueeze(1)
            episodes, steps = episodes_steps(model_input)
            if not recurrent:
                model_input = model_input.reshape(episodes * steps, model_input.shape[2])

        if layer_type == "cnn":
            if len(model_input.shape) == 4:
                model_input = model_input.unsqueeze(1)
            episodes, steps = episodes_steps(model_input)
            model_input = model_input.reshape(
                episodes * steps,
                model_input.shape[2],
                model_input.shape[3],
                model_input.shape[4]
            ).permute(0, -1, 1, 2)

        if layer_type == "voice":
            if len(model_input.shape) == 2:
                model_input = model_input.unsqueeze(0)
            if len(model_input.shape) == 3:
                model_input = model_input.unsqueeze(1)
            episodes, steps = episodes_steps(model_input)
            model_input = model_input.reshape(episodes * steps, model_input.shape[2], model_input.shape[3])

        new_model_inputs.append(model_input)

    return start_time, episodes, steps, new_model_inputs


# -------------------------------
#  MODEL PIPELINE EXIT
# -------------------------------

def model_end(start_time, episodes, steps, model_output_list, duration_text=None):
    """
    Restores model outputs to their original shape depending on layer type.
    
    Optionally prints execution time.
    """
    new_model_outputs = []

    for model_output, layer_type in model_output_list:
        if layer_type == "lin":
            model_output = model_output.reshape(episodes, steps, model_output.shape[-1])

        if layer_type == "cnn":
            model_output = model_output.permute(0, 2, 3, 1)
            model_output = model_output.reshape(episodes, steps, model_output.shape[1], model_output.shape[2], model_output.shape[3])

        if layer_type == "voice":
            model_output = model_output.reshape(episodes, steps, model_output.shape[1], model_output.shape[2])

        new_model_outputs.append(model_output)

    print_duration(start_time, duration(), duration_text)
    return new_model_outputs
