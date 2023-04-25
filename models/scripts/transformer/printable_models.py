from abc import ABC

import torch
from models.scripts.transformer.Transformer import Transformer, MultiHeadAttentionLayer
from models.scripts.transformer.PreEncoders import Conv1DTransformer


class ExplainableTransformer(Conv1DTransformer):
    def __init__(
            self,
            name,
            vocab,
            printable_modules=None,
            printable_vectors=None,
            **kwargs
    ):
        super().__init__(name, vocab, **kwargs)
        if printable_modules:
            for layer in self.encoder.layers:
                layer.self_attention.__class__ = PrintableMultiHeadAttentionLayer
                layer.self_attention.store = {}
            for layer in printable_modules:
                la = self.summary_to_layer(layer)
                la.store = printable_vectors

    def summary_to_layer(self, printable_modules):
        if 'encoder' in printable_modules['MODULE']:
            layer = self.encoder
        elif 'decoder' in printable_modules['MODULE']:
            layer = self.decoder
        else:
            raise "MODULE wrong or not existing"

        layer = layer.layers[printable_modules['LAYER']]

        if 'self-attention' in printable_modules['ATTENTION_TYPE']:
            layer = layer.self_attention
        elif 'cross-attention' in printable_modules['ATTENTION_TYPE']:
            layer = layer.cross_attention
        else:
            raise "ATTENTION_TYPE wrong or not existing"

        return layer

    def predict_with_alternatives(self, src, max_length=None, num_alternatives=1):
        self.eval()
        if max_length is None:
            max_length = self.decoder.max_length
        src_mask = self.make_src_mask(src)

        with torch.no_grad():
            src = self.preencoder(src) + src if self.preencoder is not None else src
            scaled_src = self.alpha_scaling(src)
            pos_src = self.pos_encoding(scaled_src)
            enc_src, enc_self_attention = self.encoder(pos_src, src_mask)

        # Start with a beginning of sequence token
        trg_indexes = [self.bos_idx]
        top_k_indexes = []

        for i in range(max_length):
            trg = torch.tensor(trg_indexes, dtype=torch.int64).unsqueeze(0).to(self.device)
            trg_mask = self.make_trg_mask(trg)

            with torch.no_grad():
                output, cross_attention, dec_self_attention = self.decoder(trg, enc_src, trg_mask, src_mask)

            pred_tokens = torch.topk(output[:, -1],
                                     k=num_alternatives,
                                     largest=True,
                                     sorted=True)
            best_k = pred_tokens[1].tolist()[0]
            print(best_k)
            top_k_indexes.append(best_k)
            trg_indexes.append(best_k[0])

            if best_k[0] == self.eos_idx:
                break

        trg_indexes = trg_indexes[1:]  # Exclude '<bos>'

        if type(self.vocab).__name__ == "Tokenizer":
            trg_tokens = self.vocab.decode(trg_indexes)
        else:
            trg_tokens = "".join([self.vocab.itos[i] for i in trg_indexes])

        return trg_tokens, (cross_attention, dec_self_attention, enc_self_attention), trg_indexes, top_k_indexes


class PrintableMultiHeadAttentionLayer(MultiHeadAttentionLayer):
    def __init__(
            self,
            hid_dim,
            n_heads,
            dropout,
    ):
        super().__init__(hid_dim=hid_dim, n_heads=n_heads, dropout=dropout)
        self.store = {}

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        if 'print_q' in self.store:
            torch.save(Q, self.store['print_q'])
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 3, 1)
        if 'print_k' in self.store:
            torch.save(K, self.store['print_k'])
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        if 'print_v' in self.store:
            torch.save(V, self.store['print_v'])

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]
        energy = torch.matmul(Q, K) / self.scale.to(query.device)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)

        return x, attention
