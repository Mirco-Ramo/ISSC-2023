import json
import os
import time
import torch.nn as nn
from IPython.display import FileLinks
import coremltools as ct
from torchmetrics import CharErrorRate, WordErrorRate
from tqdm.notebook import tqdm

from models.scripts.transformer.utils import tensor_to_word, display_attention, load_json_hypeparameters
from models.scripts.defaults import *

from models.scripts.utils import load_checkpoint, log_epoch, epoch_time, plot_training, \
    Levenshtein_Normalized_distance


# Transformer Implementation

class Transformer(nn.Module):
    def __init__(
            self,
            name: str,
            vocab,
            vector_size: int = VECTOR_SIZE,
            src_pad_idx: int = SRC_PAD_IDX,
            n_features: int = N_FEATURES,
            device=DEVICE,
            hid_dim: int = HID_DIM,
            enc_heads: int = ENC_HEADS,
            dec_heads: int = DEC_HEADS,
            n_tokens: int = VOCABULARY_SIZE,
            enc_layers: int = ENC_LAYERS,
            dec_layers: int = DEC_LAYERS,
            enc_pf_dim: int = ENC_PF_DIM,
            dec_pf_dim: int = DEC_PF_DIM * 3,
            enc_dropout: float = ENC_DROPOUT,
            dec_dropout: float = DEC_DROPOUT,
            enc_max_length: int = ENCODER_INPUT_LENGTH,
            dec_max_length: int = DECODER_OUTPUT_LENGTH,
            trainable_alpha_scaling: bool = TRAINABLE_ALPHA_SCALING,
            encoder_name: str = 'new',
            decoder_name: str = 'new',
            bos: str = '<bos>',
            eos: str = '<eos>',
            pad: str = '<pad>',
            **kwargs
    ):
        super().__init__()
        self.name = name
        self.vocab = vocab
        self.device = device
        self.bos_idx = self.vocab.token_to_id(bos) if type(self.vocab).__name__ == "Tokenizer" else vocab[
            bos]
        self.eos_idx = self.vocab.token_to_id(eos) if type(self.vocab).__name__ == "Tokenizer" else vocab[
            eos]
        self.trg_pad_idx = self.vocab.token_to_id(pad) if type(self.vocab).__name__ == "Tokenizer" else vocab[
            pad]
        self.src_pad_idx = src_pad_idx
        self.log_path = os.path.join("models", "logs", f'{self.name}.log')
        self.bm_path = os.path.join("models", "check_points", f'best_model_{self.name}.pt')
        self.criterion = None
        self.alpha_scaling = AlphaScaling(hid_dim, enc_dropout, device, trainable=trainable_alpha_scaling)

        if encoder_name != 'new' and encoder_name is not None:
            try:
                print("WARNING: transferred encoder hyperparameters will overwrite current ones for shape consistency")
                version = encoder_name
                hp = load_json_hypeparameters(version)
                try:
                    hp.pop('vocab')
                except:
                    pass
                tmp_model = Transformer(version, vocab, **hp, device=self.device, encoder_name="new")
                tmp_model, *_ = load_checkpoint(
                    checkpoint_path=os.path.join("models", "check_points", f'best_model_{version}.pt'),
                    model=tmp_model, strict=False, device=self.device)
                self.encoder = tmp_model.encoder
                self.pos_encoding = tmp_model.pos_encoding if hasattr(tmp_model,
                                                                      'pos_encoding') else tmp_model.encoder.pos_encoding
                self.enc_layers = self.encoder.n_layers
                self.enc_heads = self.encoder.n_heads
                self.enc_pf_dim = self.encoder.pf_dim
                self.enc_dropout = hp['enc_dropout']
                self.enc_max_length = self.encoder.max_length
                self.vector_size = self.encoder.vector_size
                self.n_features = self.vector_size // 64


            except:
                print("Specified encoder does not exist")
                raise
        elif encoder_name == 'new':
            encoder = Encoder(hid_dim=hid_dim,
                              vector_size=vector_size,
                              n_layers=enc_layers,
                              n_heads=enc_heads,
                              pf_dim=enc_pf_dim,
                              dropout=enc_dropout,
                              max_length=enc_max_length,
                              device=self.device)

            self.encoder = encoder
            self.enc_layers = enc_layers
            self.enc_heads = enc_heads
            self.enc_pf_dim = enc_pf_dim
            self.enc_dropout = enc_dropout
            self.enc_max_length = enc_max_length
            self.vector_size = vector_size
            self.n_features = n_features
            self.pos_encoding = PositionalEncoding(enc_max_length, vector_size)

        else:
            self.encoder = None

        if decoder_name != 'new' and decoder_name is not None:
            try:
                print("WARNING: transferred decoder hyperparameters will overwrite current ones for shape consistency")
                version = decoder_name
                hp = load_json_hypeparameters(version)
                try:
                    hp.pop('vocab')
                except:
                    pass
                tmp_model = Transformer(version, vocab, **hp, device=self.device, decoder_name="new")
                tmp_model, *_ = load_checkpoint(
                    checkpoint_path=os.path.join("models", "check_points", f'best_model_{version}.pt'),
                    model=tmp_model, strict=False, device=self.device)
                self.decoder = tmp_model.decoder
                self.hid_dim = self.decoder.hid_dim
                self.n_tokens = self.decoder.output_dim
                self.dec_layers = self.decoder.n_layers
                self.dec_heads = self.decoder.n_heads
                self.dec_pf_dim = self.decoder.pf_dim
                self.dec_dropout = hp['dec_dropout']
                self.dec_max_length = self.decoder.max_length
            except:
                print("Specified decoder does not exist")
                raise
        elif decoder_name == 'new':
            decoder = Decoder(output_dim=n_tokens,
                              hid_dim=hid_dim,
                              n_layers=dec_layers,
                              n_heads=dec_heads,
                              pf_dim=dec_pf_dim,
                              dropout=dec_dropout,
                              max_length=dec_max_length,
                              device=device)
            self.decoder = decoder
            self.n_tokens = n_tokens
            self.dec_layers = dec_layers
            self.dec_heads = dec_heads
            self.dec_pf_dim = dec_pf_dim
            self.dec_dropout = dec_dropout
            self.dec_max_length = dec_max_length
            self.hid_dim = hid_dim

        else:
            self.decoder = None
            self.hid_dim = self.encoder.hid_dim

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        scaled_src = self.alpha_scaling(src)
        pos_src = self.pos_encoding(scaled_src)
        enc_src, enc_attn = self.encoder(pos_src, src_mask)
        return self.decoder(trg, enc_src, trg_mask, src_mask), enc_attn

    def count_parameters(self):
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {count:,} trainable parameters.')

    def save_hyperparameters_to_json(self, params=None):
        hp = {'enc_heads': self.enc_heads, 'dec_heads': self.dec_heads, 'enc_layers': self.enc_layers,
              'dec_layers': self.dec_layers, 'enc_dropout': self.enc_dropout, 'dec_dropout': self.dec_dropout,
              'bos_idx': self.bos_idx, 'eos_idx': self.eos_idx, 'n_tokens': self.n_tokens,
              'hid_dim': self.hid_dim, 'trg_pad_idx': self.trg_pad_idx, 'vector_size': VECTOR_SIZE,
              'src_pad_idx': self.src_pad_idx, 'enc_pf_dim': self.enc_pf_dim, 'dec_pf_dim': self.dec_pf_dim,
              'n_features': self.n_features, 'dec_max_length': self.dec_max_length,
              'enc_max_length': self.enc_max_length,
              'vocab': {self.vocab.itos[i]: i for i in range(self.n_tokens)} if not type(
                  self.vocab).__name__ == "Tokenizer" else None}
        if params:
            hp = {**hp, **params}
        with open(os.path.join("models", "hyperparameters", self.name + ".json"), "w+") as hpf:
            json.dump(hp, hpf)

    def load_best_version(self):
        state_dict = torch.load(self.bm_path, map_location=torch.device(self.device))['state_dict']
        if 'encoder.pos_embedding.weight' in state_dict:
            state_dict['encoder.pos_embedding.weight'
            .replace('encoder.pos_embedding.weight', 'pos_encoding.pos_embedding.weight')] = \
                state_dict.pop('encoder.pos_embedding.weight')
        if 'decoder.pos_embedding.weight' in state_dict:
            state_dict['decoder.pos_embedding.weight'
            .replace('decoder.pos_embedding.weight', 'decoder.pos_embedding.pos_embedding.weight')] = \
                state_dict.pop('decoder.pos_embedding.weight')

        self.load_state_dict(state_dict)

    def load_checkpoint(self, strict=True, optimizer=None, scheduler=None):
        """Load model and optimizer from a checkpoint"""

        checkpoint = torch.load(self.bm_path, map_location=torch.device(self.device))
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])

        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])

        self.load_state_dict(checkpoint['state_dict'], strict=strict)
        return optimizer, scheduler

    def plot_training(self):
        plot_training(self.log_path)

    def trace_and_export(self, src, trg, version):

        # Export Encoder to CoreML (direct route)
        output_dir = "models/exports"
        src = src.to('cpu')
        trg = trg.to('cpu')
        src_mask = self.make_src_mask(src).int().to('cpu')
        print(src.dtype, src_mask.dtype)
        print(src.shape, src_mask.shape)
        model_input = (src, src_mask)

        with torch.no_grad():
            self.encoder.eval().to('cpu')
            traced_model = torch.jit.trace(self.encoder.to('cpu'), model_input)

        ml_model = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="src", shape=model_input[0].shape, dtype=float),
                ct.TensorType(name="src_mask", shape=model_input[1].shape, dtype=bool)
            ],
            #     minimum_deployment_target=ct.target.iOS15,
        )

        # rename output description and save
        spec = ml_model.get_spec()
        output_desc = list(map(lambda x: '%s' % x.name, ml_model.output_description._fd_spec))
        ct.utils.rename_feature(spec, output_desc[0], 'src_enc')
        ct.utils.rename_feature(spec, output_desc[1], 'self_attentions')
        ml_model = ct.models.MLModel(spec)
        ml_model.save(os.path.join(output_dir, f'StrokeSequenceEncoder_', version))

        # Export Decoder to CoreML (direct route)
        trg_mask = self.make_trg_mask(trg).to('cpu')
        enc_src, _ = self.encoder(src, src_mask)
        model_input = (trg.int(), enc_src.to('cpu'), trg_mask, src_mask)

        with torch.no_grad():
            self.decoder.eval().to('cpu')
            traced_model = torch.jit.trace(self.decoder, model_input)

        ml_model = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="trg", shape=model_input[0].shape, dtype=int),
                ct.TensorType(name="enc_src", shape=model_input[1].shape, dtype=float),
                ct.TensorType(name="trg_mask", shape=model_input[2].shape, dtype=bool),
                ct.TensorType(name="src_mask", shape=model_input[3].shape, dtype=bool)
            ]
        )
        # rename output description and save
        spec = ml_model.get_spec()
        output_desc = list(map(lambda x: '%s' % x.name, ml_model.output_description._fd_spec))
        ct.utils.rename_feature(spec, output_desc[0], 'out_dec')
        ct.utils.rename_feature(spec, output_desc[1], 'cross_attentions')
        ct.utils.rename_feature(spec, output_desc[2], 'self_attentions')
        ml_model = ct.models.MLModel(spec)
        ml_model.save(os.path.join(output_dir, f'StrokeSequenceDecoder_', version))

        # List files
        FileLinks(os.path.join(output_dir))


    def make_src_mask(self, src):
        pad_tensor = torch.zeros(src.shape[1:], device=src.device) + self.src_pad_idx
        src_mask = torch.logical_not(torch.all(torch.eq(pad_tensor, src), dim=2)).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = torch.tensor(trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def train_f(self, iterator, optimizer, criterion, clip, scheduler=None):

        self.train()

        epoch_loss = 0
        for i, batch in enumerate(iterator):
            src = batch[0].to(self.device)
            trg = batch[1].to(self.device)

            optimizer.zero_grad()
            dec, enc_attn = self.forward(src, trg[:, :-1])  # Remove last (eos or pad)
            output = dec[0]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)  # Remove first (bos)

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

        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src = batch[0].to(self.device)
                trg = batch[1].to(self.device)

                dec, enc_attn = self.forward(src, trg[:, :-1])  # Remove last (eos or pad)
                output = dec[0]
                output_dim = output.shape[-1]

                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)  # Remove first (bos)

                loss = criterion(output, trg)

                epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def train_loop(self,
                   train_set,
                   valid_set,
                   test_set=None,
                   resume=False,
                   optimizer=None,
                   scheduler=None,
                   n_epochs=8000,
                   criterion=nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX),
                   clip=1
                   ):
        best_valid_loss = float('inf')
        self.criterion = criterion
        if resume:
            if os.path.exists(self.bm_path):
                print(f"Loaded previous model '{self.bm_path}'\n")
                optimizer, scheduler = self.load_checkpoint(optimizer=optimizer, scheduler=scheduler)
            else:
                print("Cannot resume training: no weights file found")

        else:
            if not optimizer:
                optimizer = torch.optim.Adam(self.parameters(), lr=0.003)

        print(f"Training started using device: {self.device}")

        for epoch in range(n_epochs):

            start_time = time.time()

            train_loss = self.train_f(train_set, optimizer, criterion, clip, scheduler)
            valid_loss = self.evaluate_f(valid_set, criterion)
            if test_set:
                test_loss = self.evaluate_f(test_set, criterion)

            log_epoch(self.log_path, epoch, train_loss, valid_loss, test_loss=test_loss if test_set else None)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # Save only best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                checkpoint = {
                    'vocab': self.vocab,
                    'state_dict': self.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                torch.save(checkpoint, self.bm_path)

            print(f'Epoch: {epoch + 1:02}/{n_epochs} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f}')
            if test_set:
                print(f'\t Test Loss: {test_loss:.3f}')

    def predict(self, src, max_length=None):
        self.eval()
        if max_length is None:
            max_length = self.decoder.max_length
        src_mask = self.make_src_mask(src)

        with torch.no_grad():
            scaled_src = self.alpha_scaling(src)
            pos_src = self.pos_encoding(scaled_src)
            enc_src, enc_self_attention = self.encoder(pos_src, src_mask)

        # Start with a beginning of sequence token
        trg_indexes = [self.bos_idx]

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

    def evaluate_Levenshtein_accuracy(self, t_set):
        loss = 0
        count = 0
        for b_x, b_y in tqdm(t_set, leave=False):
            b_x = b_x.to(self.device)
            b_y = b_y.to(self.device)
            assert len(b_x) == len(b_y), "Mismatch in test dimensions"
            for x_i, y_i in zip(b_x, b_y):
                prediction, attention, _ = self.predict(src=x_i.unsqueeze(0))
                if type(self.vocab).__name__ == "Tokenizer":
                    prediction = prediction.replace(" ", '').replace("Ġ", " ")
                    gt = self.vocab.decode(y_i.tolist()).replace(" ", '').replace("Ġ", " ")
                else:
                    gt = tensor_to_word(y_i, self.vocab)
                    gt = ''.join(gt).strip('<bos>').strip('<pad>').strip('<eos>')
                    prediction = ''.join(prediction).strip('<bos>').strip('<pad>').strip('<eos>')
                dis = Levenshtein_Normalized_distance(gt, prediction)
                count += 1
                loss = loss + ((dis - loss) / count)
        return 1 - loss

    def evaluate_CER(self, t_set):
        loss = 0
        count = 0
        metric = CharErrorRate()
        for b_x, b_y in tqdm(t_set, leave=False):
            b_x = b_x.to(self.device)
            b_y = b_y.to(self.device)
            assert len(b_x) == len(b_y), "Mismatch in test dimensions"
            for x_i, y_i in zip(b_x, b_y):
                prediction, attention, _ = self.predict(x_i.unsqueeze(0))
                if type(self.vocab).__name__ == "Tokenizer":
                    prediction = prediction.replace(" ", '').replace("Ġ", " ")
                    gt = self.vocab.decode(y_i.tolist()).replace(" ", '').replace("Ġ", " ")
                else:
                    gt = tensor_to_word(y_i, self.vocab)
                    gt = ''.join(gt).strip('<bos>').strip('<pad>').strip('<eos>')
                    prediction = ''.join(prediction).strip('<bos>').strip('<pad>').strip('<eos>')
                dis = metric(gt, prediction)
                count += 1
                loss = loss + ((dis - loss) / count)
        return loss

    def evaluate_WER(self, t_set):
        loss = 0
        count = 0
        metric = WordErrorRate()
        for b_x, b_y in tqdm(t_set, leave=False):
            b_x = b_x.to(self.device)
            b_y = b_y.to(self.device)
            assert len(b_x) == len(b_y), "Mismatch in test dimensions"
            for x_i, y_i in zip(b_x, b_y):
                prediction, attention, _ = self.predict(x_i.unsqueeze(0))
                if type(self.vocab).__name__ == "Tokenizer":
                    prediction = prediction.replace(" ", '').replace("Ġ", " ")
                    gt = self.vocab.decode(y_i.tolist()).replace(" ", '').replace("Ġ", " ")
                else:
                    gt = tensor_to_word(y_i, self.vocab)
                    gt = ''.join(gt).strip('<bos>').strip('<pad>').strip('<eos>')
                    prediction = ''.join(prediction).strip('<bos>').strip('<pad>').strip('<eos>')
                dis = metric(gt, prediction)

                count += 1
                loss = loss + ((dis - loss) / count)
        return loss

    def evaluate_multiple(self, t_set, metrics: list):
        vals = dict.fromkeys(metrics)
        for k in vals.keys():
            vals[k] = 0
        count = 0
        for b_x, b_y in tqdm(t_set):  # (256,256)
            batch = [(b_x, b_y)]
            count += 1
            if "Lev_acc" in vals:
                vals["Lev_acc"] += (self.evaluate_Levenshtein_accuracy(batch) - vals["Lev_acc"]) / count
            if "CER" in vals:
                vals["CER"] += (self.evaluate_CER(batch) - vals["CER"]) / count
            if "WER" in vals:
                vals["WER"] += (self.evaluate_WER(batch) - vals["WER"]) / count
        return vals

    def display_cross_attention(self, raw_input, pred, attention):
        raw_input = raw_input.cpu()
        attention = attention.cpu()
        pad_tensor = torch.zeros(raw_input.shape[-1], device='cpu') + self.src_pad_idx
        to_cut = 0
        for i, row in enumerate(raw_input):
            if torch.all(row.eq(pad_tensor)):
                to_cut = i
                break

        # attention = attention[:, :, :, :]

        display_attention(input=raw_input[:to_cut],
                          output=pred,
                          attention_mass=attention,
                          graphical_input=True,
                          graphical_output=False
                          )

    def display_encoder_self_attention(self, raw_input, enc_src, attention):
        raw_input = raw_input.cpu()
        enc_src = enc_src.cpu()
        attention = attention.cpu()
        pad_tensor = torch.zeros(raw_input.shape[-1], device='cpu') + self.src_pad_idx
        to_cut = 0
        for i, row in enumerate(raw_input):
            if torch.all(row.eq(pad_tensor)):
                to_cut = i
                break

        attention = attention[:, :, :to_cut, :to_cut]
        display_attention(input=raw_input[:to_cut],
                          output=enc_src[:to_cut],
                          attention_mass=attention,
                          graphical_input=True,
                          graphical_output=True
                          )

    def display_decoder_self_attention(self, trg, pred, attention):
        display_attention(input=trg,
                          output=pred,
                          attention_mass=attention[:, :, :len(trg), :len(pred)],
                          graphical_input=False,
                          graphical_output=False
                          )


# Decoder Implementation
class Decoder(nn.Module):
    def __init__(
            self,
            output_dim,
            hid_dim,
            n_layers,
            n_heads,
            pf_dim,
            dropout,
            max_length,
            device
    ):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.max_length = max_length
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = PositionalEncoding(max_length, hid_dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(
                hid_dim,
                n_heads,
                pf_dim,
                dropout,
                device) for _ in range(n_layers)]
        )
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = AlphaScaling(hid_dim, dropout, device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.pos_embedding(self.scale(self.tok_embedding(trg)))
        cross_attentions = []
        self_attentions = []
        for layer in self.layers:
            trg, cross_attention, self_attention = layer(trg, enc_src, trg_mask, src_mask)
            cross_attentions.append(cross_attention)
            self_attentions.append(self_attention)
        output = self.fc_out(trg)
        return output, torch.cat(cross_attentions), torch.cat(self_attentions)


# DecoderLayer Implementation

class DecoderLayer(nn.Module):
    def __init__(
            self,
            hid_dim,
            n_heads,
            pf_dim,
            dropout,
            device
    ):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, self_attention = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        _trg, cross_attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        return trg, cross_attention, self_attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


# Encoder Implementation

class Encoder(nn.Module):
    """Encoder is a stack of n_layers EncoderLayer"""

    def __init__(
            self,
            vector_size,
            hid_dim,
            n_layers,
            n_heads,
            pf_dim,
            dropout,
            max_length,
            device
    ):
        super().__init__()
        self.device = device
        self.vector_size = vector_size
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.max_length = max_length
        self.layers = nn.ModuleList(
            [EncoderLayer(
                hid_dim,
                n_heads,
                pf_dim,
                dropout,
                device) for _ in range(n_layers)]
        )

    def forward(self, src, src_mask):
        self_attentions = []
        for layer in self.layers:
            src, self_attention = layer(src, src_mask)
            self_attentions.append(self_attention)
        return src, torch.cat(self_attentions)


# EncoderLayer Implementation

class EncoderLayer(nn.Module):
    def __init__(
            self,
            hid_dim,
            n_heads,
            pf_dim,
            dropout,
            device
    ):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, src, src_mask):
        _src, self_attention = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src, self_attention


# positional encoding implementation
class PositionalEncoding(nn.Module):
    def __init__(self, max_length, vector_size):
        super().__init__()
        self.max_length = max_length
        self.vector_size = vector_size
        self.pos_embedding = nn.Embedding(max_length, vector_size)

    def forward(self, src):
        src_len = src.shape[1]
        batch_size = src.shape[0]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(src.device)
        return src + self.pos_embedding(pos)


class AlphaScaling(nn.Module):
    def __init__(self, hid_dim=HID_DIM, dropout=ENC_DROPOUT, device=DEVICE, trainable=False):
        super(AlphaScaling, self).__init__()
        if trainable:
            self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([hid_dim])))
        else:
            self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        return self.dropout((src * self.scale.to(src.device)))


# MultiHeadAttentionLayer Implementation

class MultiHeadAttentionLayer(nn.Module):
    def __init__(
            self,
            hid_dim,
            n_heads,
            dropout,
    ):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

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
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 3, 1)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

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
