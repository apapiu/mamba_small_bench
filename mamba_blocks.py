#!pip install causal-conv1d==1.0.2 mamba-ssm

import torch
import torch.nn as nn

class MambaBlock(nn.Module):
    def __init__(self, embed_dim, dropout_level=0):
        super().__init__()

        self.mamba =  Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_level)

    def forward(self, x):
        x = self.norm(self.mamba(x) + x)
        return self.dropout(x)


class MambaTower(nn.Module):
    def __init__(self, embed_dim, n_layers, seq_len=None, global_pool=False):
        super().__init__()
        self.blocks = nn.Sequential(*[MambaBlock(embed_dim) for _ in range(n_layers)])
        self.global_pool = global_pool #for classification or other supervised learning.

    def forward(self, x):
        #for input (bs, n, d) it returns either (bs, n, d) or (bs, d) is global_pool
        out = self.blocks(x) if not self.global_pool else torch.mean(self.blocks(x),1)
        return out
