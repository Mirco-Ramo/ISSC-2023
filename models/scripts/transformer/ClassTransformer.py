import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.scripts.defaults import *
from models.scripts.transformer.Transformer import Transformer


class MLP(nn.Module):
    def __init__(
            self,
            input_dim=HID_DIM,
            dropout=ENC_DROPOUT,
            n_tokens=DECODER_OUTPUT_LENGTH,
            hidden_units=MLP_HIDDEN_UNITS,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_units, n_tokens),
            nn.Softmax(dim=1)
        )

    def forward(self, enc_src):
        return self.layers(enc_src)


class ClassTransformer(Transformer):

    def __init__(self,
                 name,
                 vocab,
                 vector_size=VECTOR_SIZE,
                 device=DEVICE,
                 bos_idx=BOS_IDX,
                 eos_idx=EOS_IDX,
                 hid_dim=HID_DIM,
                 enc_heads=ENC_HEADS,
                 n_tokens=DECODER_OUTPUT_LENGTH,
                 enc_layers=ENC_LAYERS,
                 enc_pf_dim=ENC_PF_DIM,
                 enc_dropout=ENC_DROPOUT,
                 src_pad_idx=SRC_PAD_IDX,
                 enc_max_length=ENCODER_INPUT_LENGTH,
                 mlp_hidden_units=MLP_HIDDEN_UNITS,
                 encoder_name='new',
                 **kwargs):
        super().__init__(name=name,
                         vocab=vocab,
                         vector_size=vector_size,
                         device=device,
                         bos_idx=bos_idx,
                         eos_idx=eos_idx,
                         hid_dim=hid_dim,
                         enc_heads=enc_heads,
                         n_tokens=n_tokens,
                         enc_layers=enc_layers,
                         enc_pf_dim=enc_pf_dim,
                         enc_dropout=enc_dropout,
                         src_pad_idx=src_pad_idx,
                         enc_max_length=enc_max_length,
                         encoder_name=encoder_name,
                         decoder_name=None,
                         **kwargs)

        self.mlp_hidden_units = mlp_hidden_units
        self.n_tokens = n_tokens
        self.mlp = MLP(self.hid_dim, self.enc_dropout, self.n_tokens, mlp_hidden_units)
        self.class_embed = nn.Embedding(1, self.vector_size)
        self.device = device

    def forward(self, src, trg):
        src2 = torch.zeros([src.size(0)]).int().to(self.device)
        learnable_token_class = self.class_embed(src2).to(self.device)
        src = (torch.cat([learnable_token_class.unsqueeze(1), src], dim=1)).to(self.device)
        dev = src.device
        src_mask = self.make_src_mask(src).to(dev)
        enc_src, enc_attn = self.encoder(src, src_mask)
        return self.mlp(enc_src[:, 0, :]), enc_attn

    def save_hyperparameters_to_json(self, params=None):
        hp = {'enc_heads': self.enc_heads, 'enc_layers': self.enc_layers,
              'enc_dropout': self.enc_dropout, 'vector_size': self.vector_size,
              'bos_idx': self.bos_idx, 'eos_idx': self.eos_idx, 'n_tokens': self.n_tokens,
              'hid_dim': self.hid_dim,
              'src_pad_idx': self.src_pad_idx, 'enc_pf_dim': self.enc_pf_dim,
              'n_features': 2, 'enc_max_length': self.enc_max_length,
              'vocab': {self.vocab.itos[i]: i for i in range(self.n_tokens)}, 'mlp_hidden_units': self.mlp_hidden_units}
        with open(os.path.join("models", "hyperparameters", self.name + ".json"), "w+") as hpf:
            json.dump(hp, hpf)

    def trace_and_export(self, src, trg, version):
        # TODO
        super().trace_and_export(src, trg, version)

    def make_trg_mask(self, trg):
        raise

    def train_f(self, iterator, optimizer, criterion, clip, scheduler=None):
        self.train()

        epoch_loss = 0

        for i, batch in enumerate(iterator):
            src = batch[0].to(self.device)
            trg = batch[1].to(self.device)

            optimizer.zero_grad()

            output, enc_attn = self.forward(src=src, trg=)
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg.contiguous().view(-1)

            loss = criterion(output, trg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()

        if scheduler:
            scheduler.step()

        return epoch_loss / len(iterator)

    def evaluate_f(self, iterator, criterion=None):
        self.eval()
        if not criterion:
            if self.criterion:
                criterion = self.criterion
            else:
                criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
        epoch_loss = 0
        epoch_loss = 0
        correct = 0

        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src = batch[0].to(self.device)
                trg = batch[1].to(self.device)

                output, enc_attn = self.forward(src=src, trg=)
                output = output.contiguous().view(-1, output.shape[-1])
                trg = trg.contiguous().view(-1)

                loss = criterion(output, trg)
                _, pred = torch.max(output, -1)
                correct += torch.tensor(pred == trg).sum().item()
                epoch_loss += loss.item()

        print("Accuracy is:", correct/len(iterator))
        return epoch_loss / len(iterator)

    def predict(self, src, max_length=None):
        raise

    def evaluate_Levenshtein_accuracy(self, t_set):
        raise

    def evaluate_CER(self, t_set):
        raise

    def evaluate_postfix_accuracy(self, t_set):
        raise
