# Mamba Small Benchmarks:

Exploring the the [Mamba codebase](https://github.com/state-spaces/mamba) on small example datasets (CIFAR-10, Shakespeare character-level, etc.).


Shere the paper below: 

    Mamba: Linear-Time Sequence Modeling with Selective State Spaces
    Albert Gu*, Tri Dao*
    Paper: https://arxiv.org/abs/2312.00752

**Note**: I am not by any means an expert at any of this. Currently this is just a first pass at getting something up and running. There most likely are ways to improve both the architecture and the speed of the mamba code.  

#### TLDR of first impressions:

- **CIFAR-10 Classification**: The Mamba-based model slightly outperforms the Transformer ViT-like model (85% vs. 84% accuracy) on CIFAR-10 for models with similar # of params. However, the Mamba model is about 2x slower to train despite faster learning in terms of iterations.

- **Shakespeare Character-Level Model**: Mamba shows quicker convergence and a slightly better validation loss (1.463 (lower than the example in nano-gpt which gets 1.4697)). However, it's more prone to overfitting, particularly in configurations without dropout.

## Stacking Mamba Layers:
The Mamba architecture is a sequence-to-sequence model based on a state space model architecture. Based on my basic understanding of the original paper and the GitHub repository, the code below is a reasonable (although likely not optimal) way to utilize the Mamba architecture. The concept is simple: stack several Mamba layers with normalization and optionally dropout. There's no need to add positional encoding or masking.


It's also worth noting that one can incorporate the Mamba layer into other architectures, for example, replacing self-attention or the FFN in a transformer with Mamba (see Mamba Architecture: Interleaving Blocks on [page 31](https://arxiv.org/pdf/2312.00752.pdf).



```python
import torch.nn as nn
from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    def __init__(self, embed_dim, dropout_level=0):
        super().__init__()

        self.mamba = Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
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
```

## Cifar-10 Classification:

We'll use the MambaTower class above as the backbone of a vision model on the patchified version of cifar-10 images. 

### Setup:

We compare the model above with a Transformer ViT-like model based on the same patches. 
Both models have the following config: 

- embed_dim = 256
- 6 layers
- the Transformer model has an FFN dim of 2*embed_dim (512) to maintain similar # of parameters between the two models.
- patch size of 4 (so 64 patches of dimension 48) and various basic augmentation techniques (see the code).

Here's the code for the setup - it's fairly straightforward (To get a ViT like model I replace the MambaTower with a stack of Transformer Encoders):

```python
class ImgClassifier(nn.Module):
    def __init__(self, patch_size=4, img_size=32, n_channels=3, embed_dim=256, n_layers=6, dropout=0):
        super().__init__()

        seq_len = int((img_size/patch_size)*((img_size/patch_size)))
        patch_dim = n_channels*patch_size*patch_size

        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                                   p1=self.patch_size, p2=self.patch_size)

        self.func = nn.Sequential(self.rearrange,
                                  nn.LayerNorm(patch_dim),
                                  nn.Linear(patch_dim, embed_dim),
                                  nn.LayerNorm(embed_dim),
                                  MambaTower(embed_dim, n_layers, seq_len=seq_len, global_pool=True),
                                  nn.Linear(embed_dim, 10))

    def forward(self, x):
        return self.func(x)
```


### Results:
The two models perform comparably, with the Mamba-based model having a slight edge (85% accuracy vs. 84% accuracy on the CIFAR-10 test set). While the Mamba model learns "faster" in terms of iterations, it's about twice as slow to train (note that I am using the simple Mamba class - their LLM [example](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py) looks more optimized but harder to read).

Either way 85% accuracy on cifar-10 straight out of the box with no convolutions is not bad at all - so I was pretty impressed. 

<img width="1173" alt="image" src="https://github.com/apapiu/mamba_small_bench/assets/13619417/1e315867-5c05-4782-9eeb-06513a73557c">

https://api.wandb.ai/links/apapiu/00tsl03a



## Shakespeare Char Level Language Model:

The paper has quite a few examples showcasing that mamba is better or equal to the best transformers recepie out there. Still I wanted to try it out on a small dataset so decided to try it out on the shakespeare dataset. I use the split and data setup found in the [nano-gpt)(https://github.com/karpathy/nanoGPT/tree/master/data/shakespeare_char) repo.

Model setup: Embed dimension is 256 with 256 context window and transformer has a ffn dim of 512. Both models have roughly 2 million parameters. The code is again very simple:

```python
class GPMamba(nn.Module):
    def __init__(self, embed_dim, seq_len, n_layers, dropout):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.tower = MambaTower(embed_dim, n_layers, seq_len=seq_len, global_pool=False)
        self.out_proj = nn.Sequential(nn.LayerNorm(embed_dim),
                                      nn.Linear(embed_dim, vocab_size))

    def forward(self, x):
        x = self.tower(self.embed(x))
        return self.out_proj(x)
```

Results: The mamba model does seems to converge faster (altough it's also more prone to severe overfitting see below). Mamba got a val loss of 1.463 (lower than the example in [nano-gpt](https://github.com/karpathy/nanoGPT/tree/master#:~:text=validation%20loss%20is-,1.4697,-.%20Based%20on%20the) which gets 1.4697). 

<img width="1136" alt="image" src="https://github.com/apapiu/mamba_small_bench/assets/13619417/3fda926d-3c25-4baa-8639-cb7e174658a5">

### Overfitting:
It looks like the mamba model is more likely to overfit and completely memorize the training data - especially without dropout. See below for a model with embed_dim = 512 and no dropout. Will need to explore this more.. also this is likely not an issue when training on larger datasets.

<img width="530" alt="image" src="https://github.com/apapiu/mamba_small_bench/assets/13619417/1e4266d6-066b-4ac1-8566-7347b881b1b8">

### Future ideas:

- Explore scaling in terms of epoch time vs. sequence length on mamba vs. transformer 
- Use it for autoregressice pixel generation
- Use it in a diffusion like model.


