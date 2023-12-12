
#!pip install torchview torchmetrics einops wandb causal-conv1d==1.0.2 mamba-ssm

# !git clone https://github.com/apapiu/generic_transformer.git
# import sys
# sys.path.append('generic_transformer/')
from trainer import Trainer
#from transformer_blocks import EncoderBlock, Tower, MLPSepConv, MLP 


from mamba_ssm import Mamba
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import wandb
from einops import rearrange
from einops.layers.torch import Rearrange
import torchmetrics
import torchvision
import torchvision.transforms as transforms
from mamba_blocks import MambaTower, MambaBlock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ImgClassifier(nn.Module):
    def __init__(self, patch_size=4, img_size=32, n_channels=3, embed_dim=256, n_layers=6, dropout=0):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.n_channels = 3
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        seq_len = int((self.img_size/self.patch_size)*((self.img_size/self.patch_size)))
        patch_dim = self.n_channels*self.patch_size*self.patch_size

        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                                   p1=self.patch_size, p2=self.patch_size)

        self.func = nn.Sequential(self.rearrange,
                                  nn.LayerNorm(patch_dim),
                                  nn.Linear(patch_dim, embed_dim),
                                  nn.LayerNorm(embed_dim),
                                  MambaTower(embed_dim, n_layers, seq_len=seq_len, global_pool=True, dropout=dropout),
                                  nn.Linear(embed_dim, 10))

    def forward(self, x):

        return self.func(x)


transform = transforms.Compose([
    transforms.ToTensor()
])

train_aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5)
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_aug_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=4, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False, num_workers=4, pin_memory=True)

#os.environ["WANDB_API_KEY"]='your_wandb_key'
!wandb login

block = 'mamba'
n_layers = 6
patch_size = 4
img_size = 32
embed_dim = 256
dropout = 0.1
n_layers = 6
n_channels = 3

weight_decay = 1e-5
lr = 0.0003
T_max = 30000
n_epochs = 100

config = {k: v for k, v in locals().items() if k in ['block', 'n_layers', 'patch_size', 'img_size', 'embed_dim',
                                                     'dropout', 'n_layer',
                                                     'lr', 'T_max', 'weight_decay']}

model = ImgClassifier(patch_size, img_size, n_channels, embed_dim, n_layers, dropout)
model = model.to(device)
loss = nn.CrossEntropyLoss()
val_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
trainer = Trainer(model, loss, lr=lr, T_max=T_max, weight_decay=weight_decay, wandb_log=True)

wandb.init(
    project="cifar10_classification",
    config = config)

#note this function does not work well with the mamba block:
trainer.plot_architecture(train_loader, depth=6)

wandb.save('model_graph_new.png')
print(f'Num params {sum(p.numel() for p in model.parameters())}')
trainer.train_loop(train_loader, test_loader, n_epochs=n_epochs, val_metric=val_metric)
wandb.finish()
