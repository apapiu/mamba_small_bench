from mamba_blocks import MambaTower, MambaBlock
import torch.nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from itertools import cycle
import numpy as np
from tqdm import tqdm
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR

from lightning.pytorch.callbacks import ModelCheckpoint


class MambaGPT(nn.Module):
    def __init__(self, embed_dim, seq_len, n_layers, dropout):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.tower = MambaTower(embed_dim, n_layers, seq_len=seq_len, global_pool=False)
        self.out_proj = nn.Sequential(nn.LayerNorm(embed_dim),
                                      nn.Linear(embed_dim, vocab_size))

    def forward(self, x):
        x = self.tower(self.embed(x))
        return self.out_proj(x)

# class GPTmodel(nn.Module):
#     def __init__(self, embed_dim, seq_len, n_layers, dropout):
#         super().__init__()
#         self.embed = nn.Embedding(vocab_size, embed_dim)

#         self.tower = Tower(embed_dim, seq_len, n_layers, use_pos_embeddings=True,
#                            dropout=dropout,
#                            n_heads=4, n_class=1, mlp_multiplier=2,
#                            is_causal=True, global_pool=False,
#                            block_class=EncoderBlock, mlp_class=MLP)

#         self.out_proj = nn.Sequential(nn.LayerNorm(embed_dim),
#                                       nn.Linear(embed_dim, vocab_size))

#     def forward(self, x):
#         x = self.tower(self.embed(x))

#         return self.out_proj(x)


class SequenceGenerator(IterableDataset):
    def __init__(self, token_ids, seq_length, batch_size):
        self.token_ids = torch.tensor(token_ids)
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.n_tokens = len(token_ids)
        self.indices = torch.arange(0, self.n_tokens - seq_length)

    def __iter__(self):
        self.indices = self.indices[torch.randperm(len(self.indices))]
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            X_batch = self.token_ids[batch_indices[:, None] + torch.arange(self.seq_length)]
            y_batch = self.token_ids[batch_indices[:, None] + torch.arange(1, self.seq_length + 1)]
            yield X_batch, y_batch
          
class TrainerClass(L.LightningModule):
    def __init__(self, vocab_size, embed_dim, seq_length, n_heads, attention_layers, dropout, mlp_multiplier, lr, epsilon, max_steps):
        super(Transformer, self).__init__()

        #self.model = GPTmodel(embed_dim, seq_len=seq_length, n_layers=attention_layers, dropout=dropout)
        self.model = MambaGPT(embed_dim, seq_len=seq_length, n_layers=attention_layers, dropout=dropout)

        self.max_steps = max_steps
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=epsilon, weight_decay=1e-5)

        self.batch_val_losses = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred.view(-1, y_pred.size(-1)), y.view(-1))

        if self.global_step % save_every_n_iterations == 0 and self.global_step>0:
            print('saving_model')
            checkpoint_path = f"model_checkpoint_{self.global_step}.pth"
            torch.save(self.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)

        wandb.log({"train_loss": loss}, step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
        self.batch_val_losses.append(loss.item())
        return loss

    def on_validation_epoch_end(self):

        val_loss = np.array(self.batch_val_losses).mean()
        self.batch_val_losses = []

        wandb.log({"val_loss": val_loss}, step=self.global_step)
        wandb.log({"learning_rate": self.optimizer.param_groups[0]['lr']}, step=self.global_step)

        query = """A fox was in the forest and"""
        example = encode(query)
        gen = generate_text(example, model, nchar=196, k=5, one_char_at_a_time=False,  end_on_zero=False)
        #text_table.add_data(self.global_step, gen)

    def configure_optimizers(self):

        scheduler = {
            'scheduler': CosineAnnealingLR(self.optimizer, T_max=self.max_steps, eta_min=model.optimizer.param_groups[0]['lr']/10),
            'interval': 'step',
            'frequency': 1,
            'strict': True,
        }
        return {'optimizer': self.optimizer, 'lr_scheduler': scheduler}

