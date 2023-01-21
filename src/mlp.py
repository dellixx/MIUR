import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class GELU(nn.Module): 
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x, 3))))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x



class MLPMixer(nn.Module):

    def __init__(self, dim, depth, token_dim, channel_dim, num_patch):
        super().__init__()

        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(
                dim, num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x = x.squeeze()
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        return x


class MIUR(nn.Module):

    def __init__(self, dim, depth, token_dim, channel_dim, num_patch):
        super().__init__()

        self.mlp = MLPMixer(dim, depth, token_dim, channel_dim, num_patch)
        self.num_patch = num_patch

    def forward(self, x):
        bz, x_width, hidden_size = x.shape

        temp_tensor = x.data.new(bz, self.num_patch, hidden_size).fill_(0)
        temp_tensor[:, :x_width, :] = x
        x = temp_tensor

        x = self.mlp(x)
        x = x[:, :x_width, :]
        return x



