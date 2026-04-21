#%%

import torch 
from torch import nn 
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary as torch_summary

from utils_submodule import episodes_steps, init_weights


# In a multi-layer PVRNN, MTRNN can be used to make a "short term" and "long term" memory.


class MTRNNCell(nn.Module):
    """
    A single cell in a Multiple Timescale Recurrent Neural Network (MTRNN), based on GRU.
    """

    def __init__(self, input_size, hidden_size, time_constant, args):
        super(MTRNNCell, self).__init__()

        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_constant = time_constant
        self.new = 1 / time_constant
        self.old = 1 - self.new

        self.r_x = nn.Linear(input_size, hidden_size)
        self.r_h = nn.Linear(hidden_size, hidden_size)

        self.z_x = nn.Linear(input_size, hidden_size)
        self.z_h = nn.Linear(hidden_size, hidden_size)

        self.n_x = nn.Linear(input_size, hidden_size)
        self.n_h = nn.Linear(hidden_size, hidden_size)

        self.apply(init_weights)
        self.to(self.args.device)
        if self.args.half:
            self.half()

    def forward(self, x, h):
        """
        Compute the next hidden state based on input x and previous hidden state h.
        """
        r = torch.sigmoid(self.r_x(x) + self.r_h(h))
        z = torch.sigmoid(self.z_x(x) + self.z_h(h))
        new_h = torch.tanh(self.n_x(x) + r * self.n_h(h))
        new_h = new_h * (1 - z) + h * z
        new_h = new_h * self.new + h * self.old

        if len(new_h.shape) == 2:
            new_h = new_h.unsqueeze(1)
        return new_h


#%%

class MTRNN(nn.Module):
    """
    A full MTRNN, iterating over time steps and applying a single MTRNNCell.
    """

    def __init__(self, input_size, hidden_size, time_constant, args):
        super(MTRNN, self).__init__()

        self.args = args
        self.hidden_size = hidden_size
        self.mtrnn_cell = MTRNNCell(input_size, hidden_size, time_constant, args)

        self.apply(init_weights)
        self.to(self.args.device)
        if self.args.half:
            self.half()

    def forward(self, x, h=None):
        """
        Process input x over time using MTRNNCell. Optional hidden state h.
        """
        if h is None:
            h = torch.zeros((x.shape[0], 1, self.hidden_size), device=self.args.device)

        if self.args.half:
            x = x.half()
            h = h.half()

        episodes, steps = episodes_steps(x)
        outputs = []

        for step in range(steps):
            h = self.mtrnn_cell(x[:, step], h[:, 0])
            outputs.append(h)

        outputs = torch.cat(outputs, dim=1)
        return outputs


#%%

if __name__ == "__main__":
    """
    Test and profile MTRNNCell and MTRNN model execution with synthetic input.
    """
    from utils import args
    episodes = args.batch_size
    steps = args.max_steps

    print("\n\nMTRNNCell Summary\n")
    cell = MTRNNCell(input_size=16, hidden_size=32, time_constant=1, args=args)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("cell_forward"):
            print(torch_summary(cell, ((episodes, 1, 16), (episodes, 1, 32))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

    print("\n\nMTRNN Summary\n")
    mtrnn = MTRNN(input_size=16, hidden_size=32, time_constant=1, args=args)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("mtrnn_forward"):
            print(torch_summary(mtrnn, ((episodes, steps, 16), (episodes, steps, 32))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

#%%
