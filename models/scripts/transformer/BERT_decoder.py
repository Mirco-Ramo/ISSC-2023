import json
import os

import torch.nn as nn
from transformers import BertConfig, BertModel

from models.scripts.defaults import *
from models.scripts.transformer.PreEncoders import Conv1DTransformer
from models.scripts.transformer.utils import load_json_hypeparameters
from models.scripts.utils import load_checkpoint


class BertDecoder(nn.Module):
    def __init__(
            self,
            output_dim=VOCABULARY_SIZE,
            hid_dim=HID_DIM,
            n_heads=DEC_HEADS,
            max_length=DECODER_OUTPUT_LENGTH,
            **kvargs
    ):
        super().__init__()
        self.output_dim = output_dim

        configuration = BertConfig.from_pretrained("bert-base-cased")
        # configuration.cross_attention_hidden_size = hid_dim
        # configuration.num_attention_heads = n_heads
        configuration.is_decoder = True
        configuration.add_cross_attention = True
        configuration.ignore_mismatched_sizes = True

        self.bert = BertModel.from_pretrained("bert-base-cased", ignore_mismatched_sizes=True, config=configuration)
        self.max_length = max_length
        self.enc_to_dec_proj = nn.Linear(VECTOR_SIZE, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, self.output_dim)

    def forward(self, trg, enc_src, trg_mask=None, src_mask=None):
        src_mask = src_mask.reshape(-1, enc_src.shape[1])
        trg_mask = trg_mask[:, -1, -1, :]
        enc_src = self.enc_to_dec_proj(enc_src)
        bert_out = self.bert(input_ids=trg,
                             encoder_hidden_states=enc_src,
                             encoder_attention_mask=src_mask,
                             attention_mask=trg_mask,
                             output_attentions=True)
        bert_output = bert_out['last_hidden_state']
        dec_attentions = bert_out['attentions']
        cross_attentions = bert_out['cross_attentions']
        out = self.out(bert_output)
        return out, cross_attentions, dec_attentions


class BertTransformer(Conv1DTransformer):
    def __init__(self,
                 name,
                 vocab,
                 vector_size=VECTOR_SIZE,
                 src_pad_idx=SRC_PAD_IDX,
                 n_features=N_FEATURES,
                 device=DEVICE,
                 enc_heads=ENC_HEADS,
                 n_tokens=VOCABULARY_SIZE,
                 enc_layers=ENC_LAYERS,
                 enc_pf_dim=ENC_PF_DIM,
                 enc_dropout=ENC_DROPOUT,
                 enc_max_length=ENCODER_INPUT_LENGTH,
                 trainable_alpha_scaling=TRAINABLE_ALPHA_SCALING,
                 encoder_name='new',
                 decoder_name='new',
                 hid_dim=HID_DIM,
                 dec_heads=DEC_HEADS,
                 dec_max_length=DECODER_OUTPUT_LENGTH,
                 **kvargs):
        super().__init__(name,
                         vocab,
                         vector_size=vector_size,
                         src_pad_idx=src_pad_idx,
                         n_features=n_features,
                         device=device,
                         hid_dim=hid_dim,
                         enc_heads=enc_heads,
                         n_tokens=n_tokens,
                         enc_layers=enc_layers,
                         enc_pf_dim=enc_pf_dim,
                         enc_dropout=enc_dropout,
                         enc_max_length=enc_max_length,
                         trainable_alpha_scaling=trainable_alpha_scaling,
                         encoder_name=encoder_name,
                         decoder_name=None,
                         **kvargs)

        if decoder_name != 'new' and decoder_name is not None:
            try:
                print("WARNING: transferred decoder hyperparameters will overwrite current ones for shape consistency")
                version = decoder_name
                hp = load_json_hypeparameters(version)
                try:
                    hp.pop('vocab')
                except:
                    pass
                tmp_model = BertTransformer(version, vocab, **hp, device=self.device, decoder_name="new")
                tmp_model, *_ = load_checkpoint(
                    checkpoint_path=os.path.join("models", "check_points", f'best_model_{version}.pt'),
                    model=tmp_model, strict=False, device=self.device)
                self.decoder = tmp_model.decoder
                self.n_tokens = self.decoder.output_dim
            except:
                print("Specified decoder does not exist")
                raise
        else:
            decoder = BertDecoder(output_dim=n_tokens, hid_dim=hid_dim, n_heads=dec_heads,
                                  max_length=dec_max_length)
            self.decoder = decoder
            self.n_tokens = n_tokens
            self.hid_dim = hid_dim
            self.dec_heads = dec_heads
            self.dec_max_length = dec_max_length

    def save_hyperparameters_to_json(self, params=None):
        hp = {'enc_heads': self.enc_heads, 'enc_layers': self.enc_layers,
              'enc_dropout': self.enc_dropout, 'bos_idx': self.bos_idx, 'eos_idx': self.eos_idx,
              'n_tokens': self.n_tokens, 'hid_dim': self.hid_dim, 'trg_pad_idx': self.trg_pad_idx,
              'vector_size': VECTOR_SIZE, 'src_pad_idx': self.src_pad_idx, 'enc_pf_dim': self.enc_pf_dim,
              'n_features': self.n_features, 'enc_max_length': self.enc_max_length,
              'dec_max_length': self.dec_max_length, 'dec_heads': self.dec_heads,
              'vocab': {self.vocab.itos[i]: i for i in range(self.n_tokens)} if not type(
                  self.vocab).__name__ == "Tokenizer" else None}
        if params:
            hp = {**hp, **params}
        with open(os.path.join("models", "hyperparameters", self.name + ".json"), "w+") as hpf:
            json.dump(hp, hpf)
