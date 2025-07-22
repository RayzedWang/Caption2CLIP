import  torch
from torch import nn, Tensor
from typing import Callable, List, Any, Tuple, Dict
import math
import torch.nn.functional as F

class DCShareAdapter(nn.Module):

    def __init__(self, hidden_dim, num_heads):
        super(DCShareAdapter, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.l1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.l2 = nn.Linear(hidden_dim//2, hidden_dim)

        # Add multi-head attention
        self.multihead_attention1 = nn.MultiheadAttention(hidden_dim//2, num_heads)
        self.gate1 = nn.Parameter(torch.tensor(0.6), requires_grad=True)

        self.init_weights()
        
    def init_weights(self):
        """Initializes weights with zeros."""
        nn.init.xavier_uniform_(self.l1.weight)
        self.l1.bias.data.zero_()
        self.l2.weight.data.zero_()
        self.l2.bias.data.zero_()

    def forward(self, x):
        xinit = x
        x = self.l1(x)
        x2 = x
        attn_output, _ = self.multihead_attention1(x, x, x)
        x = F.gelu(x)
        alpha = torch.sigmoid(self.gate1)
        attn = alpha * attn_output + (1 - alpha) * x2
        x = self.l2(attn)

        return x + xinit

class MMadapter(nn.Module):

    def __init__(self, share_adapter, hidden_size, neck_dim=128):

        super(MMadapter, self).__init__()
        self.img_proj_down = nn.Linear(hidden_size, neck_dim)
        self.img_proj_up = nn.Linear(neck_dim, hidden_size)
        self.BiShareAdapterxx = share_adapter
        if share_adapter is None:
            self.multihead_attention = nn.MultiheadAttention(neck_dim, 8)
            self.shared=False
        else:
            self.multihead_attention = share_adapter
            self.shared=True
        self.gate1 = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.img_proj_down.weight)
        self.img_proj_down.bias.data.zero_()
        self.img_proj_up.weight.data.zero_()
        self.img_proj_up.bias.data.zero_()

    def forward(self, x):
        x_init = x
        x = self.img_proj_down(x)
        x = F.gelu(x)
        xmid = x
        if self.shared==False:
            x, _ = self.multihead_attention(x, x, x)
        else:
            x = self.multihead_attention(x)
        alpha = torch.sigmoid(self.gate1)
        x = alpha * xmid + (1 - alpha) * x
        x = self.img_proj_up(x)
        x = x_init + x

        return x
    
