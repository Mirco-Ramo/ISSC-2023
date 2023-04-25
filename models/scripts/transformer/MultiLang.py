from models.scripts.transformer.PreEncoders import Conv1DTransformer
import torch.nn as nn
from models.scripts.defaults import *


class ConditionableTransformer(Conv1DTransformer):
    def __init__(self, name, vocab, lang_base=10, **kwargs):
        super().__init__(name, vocab, **kwargs)
        self.enc_to_dec_proj = nn.Linear(self.encoder.hid_dim, self.hid_dim)
        self.lang_base = lang_base

    def forward(self, src, trg):
        src = self.preencoder(src)
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        scaled_src = self.alpha_scaling(src)
        pos_src = self.pos_encoding(scaled_src)
        enc_src, enc_attn = self.encoder(pos_src, src_mask)
        enc_src = self.enc_to_dec_proj(enc_src)
        return self.decoder(trg, enc_src, trg_mask, src_mask), enc_attn

    def predict(self, src, max_length=None, bos = None):
        self.eval()
        with torch.no_grad():
            src = self.preencoder(src) + src if self.preencoder is not None else src
        if max_length is None:
            max_length = self.decoder.max_length
        src_mask = self.make_src_mask(src)

        with torch.no_grad():
            scaled_src = self.alpha_scaling(src)
            pos_src = self.pos_encoding(scaled_src)
            enc_src, enc_self_attention = self.encoder(pos_src, src_mask)
            enc_src = self.enc_to_dec_proj(enc_src)

        # Start with a beginning of sequence token
        trg_indexes = [bos]

        for i in range(max_length):
            trg = torch.tensor(trg_indexes, dtype=torch.int64).unsqueeze(0).to(self.device)
            trg_mask = self.make_trg_mask(trg)

            with torch.no_grad():
                output, cross_attention, dec_self_attention = self.decoder(trg, enc_src, trg_mask, src_mask)

            pred_token = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token)

            if pred_token == self.eos_idx:
                break

        trg_indexes = trg_indexes[1:]  # Exclude '<bos>'

        if type(self.vocab).__name__ == "Tokenizer":
            trg_tokens = self.vocab.decode(trg_indexes)
        else:
            trg_tokens = "".join([self.vocab.itos[i] for i in trg_indexes])

        return trg_tokens, (cross_attention, dec_self_attention, enc_self_attention), trg_indexes
