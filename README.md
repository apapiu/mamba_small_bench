# Mamba Small Benchmarks:

Trying out the [Mamba codebase](https://github.com/state-spaces/mamba) on small examples (cifar-10, shakespeare char level etc).

Shere [here] for the paper.

**Note**: I am note by any means an expert at any of this so don't take this at face value - there most likely are ways to improve the mamba code - this is just a first pass at getting something up and running,

## Stacking Mamba Layers:
The Mamba architecture is a sequence to sequence architecture. Based on my basic understanding of the original paper and the Github the code below is a reasonable (altough likely not optimal) way to use the Mamba architeccutre. The idea is simple: stack a few mamba layers with normalization and optionally dropout. No need to add positional encoding or masking.

Also note that one can use the mamba layer as part of other architectures as well for example replacing self-attention or the FFN in a transformer with Mamba (link paper).


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

We'll use the MambaTower class above as the backbone of a vision model on the patchified version of cifar-10 images. Here's the code for that - it's fairly straightforward:

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


### Setup:

We'll compare the model above with a transformer ViT-like model based on the same patches. 
Both models have embed_dim of 256, 6 layers, and the transformer model has ffn dim of 2*embed_dim (512) to keep the parameters in the two model similar. To get a ViT like model we'll simply replace the MambaTower with a Transformer Decoder Tower. 

We'll use patch size of 4 and various basic augmentation technizues (see the code).

### Results:
We see below that the two models are fairly comporable with the mamba based model having a bit of an edge (85% accuracy vs. 84% accuracy on the cifar-10 test set). While the mamba model learns "faster" in terms of iterations it's about twice as slow to train (note that I am using the simple mamba class - 

Either way 85% accuracy on cifar-10 straight out of the box with no convolutions is not bad at all - so I was pretty impressed. 

<img width="1173" alt="image" src="https://github.com/apapiu/mamba_small_bench/assets/13619417/1e315867-5c05-4782-9eeb-06513a73557c">

https://api.wandb.ai/links/apapiu/00tsl03a




## Shakespeare Char Level Language Model:

The paper has quite a few examples showcasing that mamba is better or equal to the best transformers recepie out there. Still I wanted to try it out on a small dataset so decided to try it out on the shakespeare dataset. I use the split and setup just like in the [nano-gpt)(https://github.com/karpathy/nanoGPT/tree/master/data/shakespeare_char) repo.

Setup: Embed dimension is 256 with 256 context window and transformer has a ffn dim of 512. Both models have roughly 2 million parameters.

Results: The mamba model does seems to converge faster (altough it's also more prone to severe overfitting). Mamba got a val loss of 1.463 (lower than the example in [nano-gpt](https://github.com/karpathy/nanoGPT/tree/master#:~:text=validation%20loss%20is-,1.4697,-.%20Based%20on%20the)). 

<img width="1136" alt="image" src="https://github.com/apapiu/mamba_small_bench/assets/13619417/3fda926d-3c25-4baa-8639-cb7e174658a5">

### Overfitting:
It looks like the mamba model was more likely to overfit and completely memorize the training data - especially without dropout. See below for a model with embed_dim = 512 and no dropout.

<img width="530" alt="image" src="https://github.com/apapiu/mamba_small_bench/assets/13619417/1e4266d6-066b-4ac1-8566-7347b881b1b8">




