import os
from abc import ABC, abstractmethod

import torch.nn as nn
import coremltools as ct
from IPython.display import FileLinks
from models.scripts.defaults import *
from models.scripts.transformer.Transformer import Transformer, PositionalEncoding, AlphaScaling
from models.scripts.transformer.utils import load_json_hypeparameters
from models.scripts.utils import load_checkpoint


def get_masks(src: torch.Tensor, pad_idx=SRC_PAD_IDX, bos_idx=BOS_IDX, eos_idx=EOS_IDX):
    if pad_idx:
        # pad_tensor = (torch.zeros(src.shape[1:]) + pad_idx).to(src.device)
        pad_tensor = (torch.ones(src.shape[1:]) + pad_idx - 1).to(src.device)
        pad_mask = ~torch.eq(pad_tensor, src)
        summed = torch.sum(pad_mask, 2) / VECTOR_SIZE
        pad_mask = summed.bool().unsqueeze(2)
    # torch.logical_not(torch.all(torch.eq(pad_tensor, src), dim=2)).unsqueeze(2)

    if bos_idx:
        # pad_tensor = (torch.zeros(src.shape[1:]) + pad_idx).to(src.device)
        bos_tensor = (torch.ones(src.shape[1:]) + bos_idx - 1).to(src.device)
        bos_mask = ~torch.eq(bos_tensor, src)
        summed = torch.sum(bos_mask, 2) / VECTOR_SIZE
        bos_mask = summed.bool().unsqueeze(2)

    if eos_idx:
        # pad_tensor = (torch.zeros(src.shape[1:]) + pad_idx).to(src.device)
        eos_tensor = (torch.ones(src.shape[1:]) + eos_idx - 1).to(src.device)
        eos_mask = ~torch.eq(eos_tensor, src)
        summed = torch.sum(eos_mask, 2) / VECTOR_SIZE
        eos_mask = summed.bool().unsqueeze(2)

    #     # create masks for all (pad, bos, eos)
    all_mask = torch.ones(src.shape[:-1]).unsqueeze(2).to(src.device)

    if pad_idx:
        all_mask = torch.logical_and(all_mask, pad_mask)

    if bos_idx:
        all_mask = torch.logical_and(all_mask, bos_mask)

    if eos_idx:
        all_mask = torch.logical_and(all_mask, eos_mask)

    mask = all_mask.expand_as(src).float().to(src.device)
    not_mask = (~all_mask).expand_as(src).float().to(src.device)
    a = (src * mask + src * not_mask).to(src.device)
    #     assert torch.all(torch.eq(a - src, (torch.zeros(src.shape)).to(src.device))), "error masking"

    return mask, not_mask


class PEEncoder(nn.Module):
    def __init__(self, preennc, scale, pos, enc):
        super(PEEncoder, self).__init__()
        self.preenc = preennc
        self.alpha_scal = scale
        self.pos_enc = pos
        self.enc = enc

    def forward(self, src, src_mask):
        # scaled_src = src
        src = self.preenc(src) + src
        scaled_src = self.alpha_scal(src)
        pos_src = self.pos_enc(scaled_src)
        return self.enc(pos_src, src_mask)


class ElementwisePreencoderTransformer(Transformer, ABC):
    def __init__(self,
                 name,
                 vocab,
                 **kwargs):
        super().__init__(name=name,
                         vocab=vocab,
                         **kwargs)

        self.preencoder = None

    @abstractmethod
    def define_preencoder(self, **kvargs):
        pass

    def forward(self, src, trg):
        src = self.preencoder(src) + src if self.preencoder else src  # skip connection
        return super().forward(src, trg)

    def predict(self, src, max_length=None):
        self.eval()
        with torch.no_grad():
            src = self.preencoder(src) + src if self.preencoder is not None else src
        return super().predict(src, max_length)

    def trace_and_export(self, src, trg, version):
        # Export Encoder to CoreML (direct route)
        output_dir = "models/exports"
        src = src.to('cpu')
        trg = trg.to('cpu')
        src_mask = self.make_src_mask(src).int().to('cpu')
        print(src.dtype, src_mask.dtype)
        print(src.shape, src_mask.shape)
        model_input = (src, src_mask)

        my_PE = PEEncoder(self.preencoder, self.alpha_scaling, self.pos_encoding, self.encoder)

        with torch.no_grad():
            my_PE.eval()
            self.requires_grad_(False)
            traced_model = torch.jit.trace(my_PE, model_input)

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
        ml_model.save(os.path.join(output_dir, f'StrokeSequenceEncoder_{version}'))

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
        ml_model.save(os.path.join(output_dir, f'StrokeSequenceDecoder_{version}'))

        # List files
        FileLinks(os.path.join(output_dir))


class EntanglingPreencoderTransformer(Transformer, ABC):
    def __init__(self,
                 name,
                 vocab,
                 **kwargs):
        super().__init__(name=name,
                         vocab=vocab,
                         **kwargs)

        self.preencoder = None
        self.pos_encoding = PositionalEncoding(self.enc_max_length, self.vector_size)
        self.alpha_scaling = AlphaScaling(self.hid_dim, self.enc_dropout, self.device)

    @abstractmethod
    def define_preencoder(self, **kvargs):
        pass

    def forward(self, src, trg):

        src_mask, _ = get_masks(src, pad_idx=self.src_pad_idx, bos_idx=self.bos_idx, eos_idx=self.eos_idx)
        trg_mask = self.make_trg_mask(trg)

        src = src * src_mask
        src = self.pos_encoding(src)

        src = self.preencoder(src) + src
        src = self.alpha_scaling(src)

        new_mask = torch.ones([src.shape[0], 1, 1, src.shape[1]], device=src.device)

        enc_src, enc_attn = self.encoder(src, src_mask=new_mask)
        return self.decoder(trg, enc_src, trg_mask, new_mask), enc_attn

    def predict(self, src, max_length=None):
        self.eval()
        if max_length is None:
            max_length = self.decoder.max_length
        src_mask, _ = get_masks(src, pad_idx=self.src_pad_idx, bos_idx=self.bos_idx, eos_idx=self.eos_idx)
        with torch.no_grad():
            src = src * src_mask
            src = self.pos_encoding(src)
            src = self.preencoder(src) + src if self.preencoder is not None else src
            src = self.alpha_scaling(src)

            new_mask = torch.ones([src.shape[0], 1, 1, src.shape[1]], device=src.device)

            enc_src, enc_self_attention = self.encoder(src, src_mask=new_mask)

        # Start with a beginning of sequence token
        trg_indexes = [self.bos_idx]

        for i in range(max_length):
            trg = torch.tensor(trg_indexes, dtype=torch.int64).unsqueeze(0).to(self.device)
            trg_mask = self.make_trg_mask(trg)

            with torch.no_grad():
                output, cross_attention, dec_self_attention = self.decoder(trg, enc_src, trg_mask, new_mask)

            pred_token = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token)

            if pred_token == self.eos_idx:
                break

        trg_indexes = trg_indexes[1:]  # Exclude '<bos>'

        if type(self.vocab).__name__ == "Tokenizer":
            trg_tokens = self.vocab.decode(trg_indexes)
        else:
            trg_tokens = [self.vocab.itos[i] for i in trg_indexes]

        return trg_tokens, (cross_attention, dec_self_attention, enc_self_attention), trg_indexes


#######################################################################################################################
#######################################################################################################################
################################################### CONV1D PREENCODER #################################################
#######################################################################################################################
#######################################################################################################################

class Conv1DPreencoder(nn.Module):
    def __init__(
            self,
            input_dims=(ENCODER_INPUT_LENGTH, VECTOR_SIZE // N_FEATURES, N_FEATURES),
            n_conv_layers=N_CONV_LAYERS,
            pad_idx=SRC_PAD_IDX,
            bos_idx=BOS_IDX,
            eos_idx=EOS_IDX,
            device=DEVICE
    ):
        super().__init__()
        self.input_dims = input_dims
        self.n_conv_layers = n_conv_layers
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.device = device

        self.layers = self._build_layers()

    def _build_layers(self):
        layers = nn.Sequential().to(self.device)

        seq_len, vec_size, channels = self.input_dims
        total_dims = seq_len * vec_size * channels
        for i in range(self.n_conv_layers):
            layers.append(nn.Conv1d(in_channels=channels, out_channels=channels * 2,
                                    kernel_size=3,
                                    padding='same',
                                    bias=False))
            layers.append(nn.MaxPool1d(kernel_size=2))
            vec_size //= 2
            channels *= 2
            assert seq_len * vec_size * channels == total_dims, "Dimension error in conv building"
        return layers

    def forward(self, src: torch.Tensor):
        batch_size = src.shape[0]
        mask, not_mask = get_masks(src, pad_idx=self.pad_idx, bos_idx=self.bos_idx, eos_idx=self.eos_idx)
        conv_src = src * mask
        # (batch_size * n_samples, vector_size//n_features, n_features), dim[1] is convoluted over
        conv_src = torch.reshape(conv_src,
                                 (batch_size * self.input_dims[0], self.input_dims[1], self.input_dims[2])).float()
        conv_src = conv_src.permute(0, 2, 1)  # torch requires channels before convolved dimensions
        # (batch_size*n_samples, n_features, vector_size//n_features)

        conv_src = conv_src.to(self.device)
        conv_res = self.layers(conv_src)
        conv_res = conv_res.permute(0, 2, 1)  # (batch_size * n_samples, vector_size//channels, channels)
        conv_res = torch.reshape(conv_res, (batch_size, self.input_dims[0], self.input_dims[1] * self.input_dims[2]))

        # (batch_size, n_samples, vector_size//n_features, n_features)
        conv_res = (conv_res * mask).to(self.device)

        assert src.shape == conv_res.shape

        return conv_res


class Conv1DTransformer(ElementwisePreencoderTransformer):

    def __init__(self,
                 name,
                 vocab,
                 conv_layer_name='new',
                 n_conv_layers=N_CONV_LAYERS,
                 **kwargs):
        super().__init__(name=name,
                         vocab=vocab,
                         **kwargs)
        self.n_conv_layers = None
        self.define_preencoder(vocab=vocab, conv_layer_name=conv_layer_name, n_conv_layers=n_conv_layers)

    def define_preencoder(self, **kvargs):
        vocab = kvargs['vocab']
        conv_layer_name = kvargs['conv_layer_name']
        n_conv_layers = kvargs['n_conv_layers']
        if conv_layer_name != 'new' and conv_layer_name is not None:
            try:
                print(
                    "WARNING: transferred conv_layer hyperparameters will overwrite current ones for shape consistency")
                version = conv_layer_name
                hp = load_json_hypeparameters(version)
                if "vocab" in hp:
                    hp.pop("vocab")
                self.n_conv_layers = hp['n_conv_layers']
                tmp_model = Conv1DTransformer(version, vocab, **hp, device=self.device, conv_layer_name="new")
                tmp_model, *_ = load_checkpoint(
                    checkpoint_path=os.path.join("models", "check_points", f'best_model_{version}.pt'),
                    model=tmp_model, strict=False, device=self.device)
                self.preencoder = tmp_model.preencoder
            except:
                print("Specified conv_layer version does not exist")
                raise
        elif conv_layer_name == 'new':
            self.n_conv_layers = n_conv_layers
            self.preencoder = Conv1DPreencoder(
                input_dims=(self.enc_max_length, self.vector_size // self.n_features, self.n_features),
                n_conv_layers=self.n_conv_layers,
                pad_idx=self.src_pad_idx,
                eos_idx=self.eos_idx,
                bos_idx=self.bos_idx,
                device=self.device
            )
        else:  # None
            print("Conv pre-encoding disabled")
            self.n_conv_layers = 0
            self.preencoder = None

    def save_hyperparameters_to_json(self, params=None):
        hp = {'n_conv_layers': self.n_conv_layers}
        super().save_hyperparameters_to_json(params=hp)


#######################################################################################################################
#######################################################################################################################
####################################### ENTANGLING CONV PREENCODER ####################################################
#######################################################################################################################
#######################################################################################################################

class Conv2DPreencoder(nn.Module):

    def __init__(
            self,
            input_dims=(ENCODER_INPUT_LENGTH, VECTOR_SIZE // N_FEATURES, N_FEATURES),
            n_conv_layers=N_CONV_LAYERS,
            pad_idx=SRC_PAD_IDX,
            bos_idx=BOS_IDX,
            eos_idx=EOS_IDX,
            device=DEVICE
    ):
        super().__init__()
        self.input_dims = input_dims
        self.n_conv_layers = n_conv_layers
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.device = device

        self.layers = self._build_layers()

    def _build_layers(self):
        layers = nn.Sequential().to(self.device)

        seq_len, vec_size, channels = self.input_dims
        total_dims = seq_len * vec_size * channels
        for i in range(self.n_conv_layers):
            layers.append(nn.Conv2d(in_channels=channels, out_channels=channels * 2,
                                    kernel_size=3,
                                    padding='same',
                                    bias=False))
            layers.append(nn.MaxPool2d(kernel_size=(1, 2)))
            vec_size //= 2
            channels *= 2
            assert seq_len * vec_size * channels == total_dims, "Dimension error in conv building"
        return layers

    def forward(self, src: torch.Tensor):
        batch_size = src.shape[0]
        # (batch_size, n_samples, vector_size//n_features, n_features), dim[1,2]are convoluted over
        new_shape = [batch_size] + [dim for dim in self.input_dims]
        conv_src = torch.reshape(src, new_shape).float()
        conv_src = conv_src.permute((0, 3, 1, 2))  # torch requires channels before convolved dimensions
        # (batch_size, n_features, n_samples, vector_size//n_features)
        conv_src = conv_src.to(self.device)
        conv_res = self.layers(conv_src)
        conv_res = conv_res.permute(0, 2, 3, 1)  # (batch_size, n_samples, vector_size//channels, channels)
        conv_res = torch.reshape(conv_res, (batch_size, self.input_dims[0], self.input_dims[1] * self.input_dims[2]))
        # (batch_size, n_samples, vector_size//n_features, n_features)

        assert src.shape == conv_res.shape

        return conv_res


class Conv2DTransformer(EntanglingPreencoderTransformer):

    def __init__(self,
                 name,
                 vocab,
                 conv_layer_name='new',
                 n_conv_layers=N_CONV_LAYERS,
                 **kwargs):
        super().__init__(name=name,
                         vocab=vocab,
                         **kwargs)
        self.n_conv_layers = None
        self.define_preencoder(vocab=vocab, conv_layer_name=conv_layer_name, n_conv_layers=n_conv_layers)

    def define_preencoder(self, **kvargs):
        vocab = kvargs['vocab']
        conv_layer_name = kvargs['conv_layer_name']
        n_conv_layers = kvargs['n_conv_layers']
        if conv_layer_name != 'new' and conv_layer_name is not None:
            try:
                print(
                    "WARNING: transferred conv_layer hyperparameters will overwrite current ones for shape consistency")
                version = conv_layer_name
                hp = load_json_hypeparameters(version)
                if "vocab" in hp:
                    hp.pop("vocab")
                self.n_conv_layers = hp['n_conv_layers']
                tmp_model = Conv2DTransformer(version, vocab, **hp, device=self.device, conv_layer_name="new")
                tmp_model, *_ = load_checkpoint(
                    checkpoint_path=os.path.join("models", "check_points", f'best_model_{version}.pt'),
                    model=tmp_model, strict=False, device=self.device)
                self.preencoder = tmp_model.preencoder
            except:
                print("Specified conv_layer version does not exist")
                raise
        elif conv_layer_name == 'new':
            self.n_conv_layers = n_conv_layers
            self.preencoder = Conv2DPreencoder(
                input_dims=(self.enc_max_length, self.vector_size // self.n_features, self.n_features),
                n_conv_layers=self.n_conv_layers,
                pad_idx=self.src_pad_idx,
                eos_idx=self.eos_idx,
                bos_idx=self.bos_idx,
                device=self.device
            )
        else:  # None
            print("Conv pre-encoding disabled")
            self.n_conv_layers = 0
            self.preencoder = None

    def save_hyperparameters_to_json(self, params=None):
        hp = {'n_conv_layers': self.n_conv_layers}
        super().save_hyperparameters_to_json(params=hp)


#####################################################################################################################
#####################################################################################################################
################################################ MLP MIXER ##########################################################
#####################################################################################################################
#####################################################################################################################


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int = HID_DIM,
            dropout: float = ENC_DROPOUT,
            output_dim: int = DECODER_OUTPUT_LENGTH,
            hidden_units: int = MLP_HIDDEN_UNITS,
            non_linearity=nn.ReLU,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_units, bias=False),
            nn.Dropout(p=dropout),
            non_linearity(),
            nn.Linear(hidden_units, output_dim, bias=False),
        )

    def forward(self, src):
        return self.layers(src)


class MLPMixer(nn.Module):
    def __init__(
            self,
            seq_len,
            n_channels,
            mlp_hidden_units=MLP_HIDDEN_UNITS
    ):
        super().__init__()

        self.seq_len = seq_len
        self.n_channels = n_channels
        self.non_linearity = nn.GELU
        self.channel_mlp = MLP(input_dim=self.seq_len,
                               output_dim=self.seq_len,
                               non_linearity=self.non_linearity,
                               hidden_units=mlp_hidden_units)
        self.token_mlp = MLP(input_dim=self.n_channels,
                             output_dim=self.n_channels,
                             non_linearity=self.non_linearity,
                             hidden_units=mlp_hidden_units)
        self.channel_layer_norm = nn.LayerNorm(self.seq_len)
        self.token_layer_norm = nn.LayerNorm(self.n_channels)

    def forward(self, src):
        in_shapes = src.shape
        mask, not_mask = get_masks(src)
        src = src * mask
        src = src.reshape([-1, self.seq_len, self.n_channels])
        assert len(src.shape) == 3, "Bad source shape, tensors must be bi-dimensional (excluding batch size)"
        assert src.shape[1] == self.seq_len and src.shape[
            2] == self.n_channels, "Tensor must have seq_len x n_channels shape"
        # apply layer norm and transpose
        transposed_src = torch.permute(self.token_layer_norm(src), (0, 2, 1))

        # apply channel mlp and transpose
        channel_processed_tensor = torch.permute(self.channel_mlp(transposed_src), (0, 2, 1))

        # skip connection
        channel_processed_tensor += src

        # apply layer norm 2, token mlp and skip connection
        token_processed_tensor = torch.permute(
            self.channel_layer_norm(
                torch.permute(channel_processed_tensor, (0, 2, 1))),
            (0, 2, 1))

        out = channel_processed_tensor + self.token_mlp(token_processed_tensor)
        out = torch.reshape(out, in_shapes)

        return out * mask


class ElemwiseMLPMixerTransformer(ElementwisePreencoderTransformer):

    def __init__(self,
                 name,
                 vocab,
                 mlp_name='new',
                 mlp_hidden_units=8,
                 **kwargs):
        super().__init__(name=name,
                         vocab=vocab,
                         **kwargs)
        self.mlp_hidden_units = None
        self.define_preencoder(vocab=vocab, mlp_name=mlp_name, mlp_hidden_units=mlp_hidden_units)

    def define_preencoder(self, **kvargs):
        vocab = kvargs['vocab']
        mlp_name = kvargs['mlp_name']
        mlp_hidden_units = kvargs['mlp_hidden_units']
        if mlp_name != 'new' and mlp_name is not None and mlp_hidden_units > 0:
            try:
                print(
                    "WARNING: transferred conv_layer hyperparameters will overwrite current ones for shape consistency")
                version = mlp_name
                hp = load_json_hypeparameters(version)
                if "vocab" in hp:
                    hp.pop("vocab")
                self.mlp_hidden_units = hp['mlp_hidden_units']
                tmp_model = ElemwiseMLPMixerTransformer(version, vocab, **hp, device=self.device, mlp_name="new")
                tmp_model, *_ = load_checkpoint(
                    checkpoint_path=os.path.join("models", "check_points", f'best_model_{version}.pt'),
                    model=tmp_model, strict=False, device=self.device)
                self.preencoder = tmp_model.preencoder
            except:
                print("Specified conv_layer version does not exist")
                raise
        elif mlp_name == 'new':
            self.mlp_hidden_units = mlp_hidden_units
            self.preencoder = MLPMixer(
                seq_len=self.vector_size // self.n_features,
                n_channels=self.n_features,
                mlp_hidden_units=mlp_hidden_units
            )
        else:  # None
            print("MLP mixer pre-encoding disabled")
            self.mlp_hidden_units = 0
            self.preencoder = None


    def save_hyperparameters_to_json(self, params=None):
        hp = {'mlp_hidden_units': self.mlp_hidden_units}
        super().save_hyperparameters_to_json(hp)


class EntanglingMLPMixerTransformer(EntanglingPreencoderTransformer):

    def __init__(self,
                 name,
                 vocab,
                 mlp_name='new',
                 mlp_hidden_units=MLP_HIDDEN_UNITS,
                 **kwargs):
        super().__init__(name=name,
                         vocab=vocab,
                         **kwargs)
        self.mlp_hidden_units = None
        self.define_preencoder(vocab=vocab, mlp_name=mlp_name, mlp_hidden_units=mlp_hidden_units)

    def define_preencoder(self, **kvargs):
        vocab = kvargs['vocab']
        mlp_name = kvargs['mlp_name']
        mlp_hidden_units = kvargs['mlp_hidden_units']
        if mlp_name != 'new' and mlp_name is not None and mlp_hidden_units > 0:
            try:
                print(
                    "WARNING: transferred conv_layer hyperparameters will overwrite current ones for shape consistency")
                version = mlp_name
                hp = load_json_hypeparameters(version)
                if "vocab" in hp:
                    hp.pop("vocab")
                self.mlp_hidden_units = hp['mlp_hidden_units']
                tmp_model = EntanglingMLPMixerTransformer(version, vocab, **hp, device=self.device, mlp_name="new")
                tmp_model, *_ = load_checkpoint(
                    checkpoint_path=os.path.join("models", "check_points", f'best_model_{version}.pt'),
                    model=tmp_model, strict=False, device=self.device)
                self.preencoder = tmp_model.preencoder
            except:
                print("Specified conv_layer version does not exist")
                raise
        elif mlp_name == 'new':
            self.mlp_hidden_units = mlp_hidden_units
            self.preencoder = MLPMixer(
                seq_len=self.enc_max_length,
                n_channels=self.vector_size,
                mlp_hidden_units=mlp_hidden_units
            )
        else:  # None
            print("MLP mixer pre-encoding disabled")
            self.mlp_hidden_units = 0
            self.preencoder = None

    def save_hyperparameters_to_json(self, params=None):
        hp = {'mlp_hidden_units': self.mlp_hidden_units}
        super().save_hyperparameters_to_json(hp)
