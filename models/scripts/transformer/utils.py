import itertools
import multiprocessing
import os
import random
import json

import torch
from tqdm import tqdm
from collections import Counter, defaultdict
from itertools import repeat

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.nn.utils.rnn import pad_sequence

from models.scripts.utils import chunker
from models.scripts.defaults import *

EXPR_MODES = ["all", "digits", "alphabets"]
GRANULARITIES = ["touch", "stroke", "glyph"]

mp_dict = multiprocessing.Manager().dict()


def seed_all(seed):
    """Seed everything"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def preprocess_dataset(d_set, vocab, fname, total_len=0):
    """Create x and y tensors for a dataset, `d_set` given its `vocab`"""
    cache_dir = os.path.join("cache", "preprocessed_datasets")
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    path = os.path.join(cache_dir, fname)
    print(path)
    if os.path.exists(path):
        print('Using cached processed dataset')
        return torch.load(path, map_location='cpu')

    tensors = []
    with tqdm(total=total_len) as pbar:
        for (x, y) in d_set:
            if type(vocab).__name__ == "Tokenizer":
                tensors.append((torch.tensor(x), torch.tensor(vocab.encode(y).ids)))
            else:
                tensors.append((torch.tensor(x), torch.stack([torch.tensor(vocab.stoi[i]) for i in y])))
            pbar.update(1)

    cache_processed_data(tensors, path)
    return tensors


def preprocess_with_lang(d_set, vocab, fname, total_len=0, bos=2):
    """Create x and y tensors for a dataset, `d_set` given its `vocab`, add the language code on top of x"""
    cache_dir = os.path.join("cache", "preprocessed_datasets")
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    path = os.path.join(cache_dir, fname)
    print(path)
    if os.path.exists(path):
        print('Using cached processed dataset')
        return torch.load(path, map_location='cpu')

    tensors = []
    with tqdm(total=total_len) as pbar:
        for (x, y) in d_set:
            if type(vocab).__name__ == "Tokenizer":
                new_x, new_y = pad_collate_fn([(torch.tensor(x), torch.tensor(vocab.encode(y).ids))], bos_idx=bos)
            else:
                new_x, new_y = pad_collate_fn(
                    [(torch.tensor(x), torch.stack([torch.tensor(vocab.stoi[i]) for i in y]))], bos_idx=bos
                )
            tensors.append((new_x.squeeze(0), new_y.squeeze(0)))
            pbar.update(1)

    cache_processed_data(tensors, path)
    return tensors


def preprocess_with_lang_or_unk(d_set, vocab, fname, total_len=0, language=Languages.UNK, split=UNK_SPLIT):
    """Create x and y tensors for a dataset, `d_set` given its `vocab`, add the language code on top of x"""
    cache_dir = os.path.join("cache", "preprocessed_datasets")
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    path = os.path.join(cache_dir, fname)
    print(path)
    if os.path.exists(path):
        print('Using cached processed dataset')
        return torch.load(path, map_location='cpu')

    tensors = []
    with tqdm(total=total_len) as pbar:
        for index, (x, y) in enumerate(d_set):
            lang = language if random.random() > UNK_SPLIT else Languages.UNK
            x = torch.cat([torch.zeros([1, VECTOR_SIZE]) + lang.value, torch.tensor(x)], dim=0)
            assert x.shape == (ENCODER_INPUT_LENGTH + 1, VECTOR_SIZE)
            if type(vocab).__name__ == "Tokenizer":
                tensors.append((x, torch.tensor(vocab.encode(y).ids)))
            else:
                tensors.append((x, torch.stack([torch.tensor(vocab.stoi[i]) for i in y])))
            pbar.update(1)
            if index >= total_len:
                break

    cache_processed_data(tensors, path)
    return tensors


def cache_processed_data(data, path):
    cur_dir = str(os.path.split(path)[0])
    if not os.path.exists(cur_dir):
        os.mkdir(cur_dir)
    torch.save(data, path)


def pad_collate_fn(batch, bos_idx=BOS_IDX, trg_pad_idx=TRG_PAD_IDX, eos_idx=EOS_IDX, max_len=DECODER_OUTPUT_LENGTH):
    """Pad and collate batches"""

    xx, yy = [], []
    padding_tensor = torch.tensor([trg_pad_idx])

    for (x, y) in batch:
        xx.append(x)
        diff = (max_len - (y.shape[0])) - 2
        y_ = torch.cat([torch.tensor([bos_idx]),
                        y, torch.tensor([eos_idx]), torch.tensor(list(repeat(padding_tensor, diff)))], dim=0)

        # Because equally-padded tensors have float
        yy.append(torch.tensor(y_, dtype=torch.int64))

    # Pad sequence to even size
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=trg_pad_idx)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=trg_pad_idx)

    return xx_pad, yy_pad


def pad_bert_collate_fn(batch, trg_pad_idx=TRG_PAD_IDX, max_len=DECODER_OUTPUT_LENGTH):
    """Pad and collate batches"""

    xx, yy = [], []
    padding_tensor = torch.tensor([trg_pad_idx])

    for (x, y) in batch:
        xx.append(x)
        diff = (max_len - (y.shape[0])) - 2
        y_ = torch.cat([y, torch.tensor(list(repeat(padding_tensor, diff)))], dim=0)

        # Because equally-padded tensors have float
        yy.append(torch.tensor(y_, dtype=torch.int64))

    # Pad sequence to even size
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=trg_pad_idx)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=trg_pad_idx)

    return xx_pad, yy_pad


def build_vocab(words):
    """Build word vocabulary"""

    counter = Counter()
    for char in words:
        counter.update(char)

    return DatasetVocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


def tensor_to_word(tensor: torch.Tensor, vocab):
    """Converts a tensor to its word representation"""

    if type(vocab).__name__ == "ByteLevelBPETokenizer":
        trg = vocab.decode(tensor.tolist())

    else:  # `torchtext.vocab.Vocab`
        itos = vocab.itos
        trg = "".join([itos[tensor[i]] for i in range(tensor.shape[0])])

    return trg


def load_json_hypeparameters(version: str):
    with open(os.path.join("models", "hyperparameters", version + ".json"), "r+") as hpf:
        hp = json.load(hpf)
    return hp


def display_attention(input, output, attention_mass, graphical_input=False, graphical_output=False, scale_factor=1,
                      exclude_paddings=True):
    n_layers = attention_mass.shape[0]
    n_heads = attention_mass.shape[1]

    for j in range(n_layers):
        for i in range(n_heads):
            # Per matplotlib documentation, create new figure instance
            f = plt.figure(figsize=(30, 25))
            f.tight_layout(pad=2)
            grid = gridspec.GridSpec(2, 2, wspace=0.01, hspace=0.01, width_ratios=(1, len(input)),
                                     height_ratios=(1, len(output)))
            ax = plt.Subplot(f, grid[3])
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%", pad=0.25)

            layer_attention_head = attention_mass[j][i]

            if exclude_paddings:
                pad_attention = []
                for l in range(layer_attention_head.shape[0]):
                    # Cut off zeros from the attention vector
                    pad_attention.append(
                        layer_attention_head[l][layer_attention_head[l] != 0])
                layer_attention_head = torch.stack(
                    [k for k in pad_sequence(pad_attention, batch_first=True)])

            _attention = layer_attention_head.cpu().detach().numpy()
            pos = ax.matshow(_attention)

            # add the colorbar using the figure's method,
            # telling which mappable we're talking about and
            # which axes object it should be near
            # plt.colorbar(pos, cax=cax)

            # ax.tick_params(labelsize=10)
            if not graphical_input:
                ax.set_xticklabels([""] + input, rotation=90)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            else:
                ax.tick_params(bottom=False, top=False)
                ax.axes.get_xaxis().set_ticks([])
                inner = gridspec.GridSpecFromSubplotSpec(1, len(input) - 1, subplot_spec=grid[1], wspace=0, hspace=0)
                for k in range(len(input) - 1):
                    ax_in = plt.Subplot(f, inner[k])
                    f.add_subplot(ax_in)
                    subplot_glyph_(ax_in, input[k + 1], 2)

            if not graphical_output:
                ax.set_yticklabels([""] + output)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            else:
                ax.tick_params(left=False, right=False)
                ax.axes.get_yaxis().set_ticks([])
                inner = gridspec.GridSpecFromSubplotSpec(len(output) - 1, 1, subplot_spec=grid[2], wspace=0, hspace=0)
                for k in range(len(output) - 1):
                    ax_in = plt.Subplot(f, inner[k])
                    f.add_subplot(ax_in)
                    subplot_glyph_(ax_in, output[k + 1], 2)
            f.add_subplot(ax)
            f.suptitle(f"Layer {j + 1}, Attention Head {i + 1}")

    plt.show()
    plt.close()


def subplot_glyph_(plot_object, glyph_data, n_features):
    x, y = [], []

    # Get x, y based on data features
    # touch has (x, y, [t], [p])
    for touch in chunker(glyph_data, n_features):
        if touch[0] > 0:
            x.append(touch[0])
            y.append(touch[1])

    # Must be the same size
    v = x if len(x) > len(y) else y
    [v.pop() for _ in range(abs(len(x) - len(y)))]

    xmin, xmax = [0, 1]
    ymin, ymax = [0, 1]  # min(y), max(y)

    try:
        # Turn off tick labels
        plot_object.set_yticklabels([])
        plot_object.set_xticklabels([])
        plot_object.set_ylim([ymin, ymax])
        plot_object.set_xlim([xmin, xmax])
        plot_object.invert_yaxis()  # Make plot appear upright

    except AttributeError:  # `plt` object
        # Turn off tick labels
        plot_object.gca().set_yticklabels([])
        plot_object.gca().set_xticklabels([])
        plot_object.gca().set_ylim([ymin, ymax])
        plot_object.set_xlim([xmin, xmax])
        plot_object.gca().invert_yaxis()  # Make plot appear upright

    plot_object.axis('equal')
    plot_object.axis('off')
    plot_object.plot(x, y, '-', linewidth=2.0)


def draw_word(data, subplot_size, n_features, granularity="glyph", split_token=-3):
    data = data[1:]  # Cut off bos token
    split_token = [float(split_token)] * n_features

    if granularity == "touch":
        data = data.tolist()

        split_chunks = [list(y) for x, y in itertools.groupby(
            data, lambda z: z == split_token) if not x]
        glyph_strokes = [list(itertools.chain.from_iterable(i))
                         for i in split_chunks]

        # Exclude very last glyph_stroke, as it contains padding and stuff
        _, ax = plt.subplots(1, len(glyph_strokes) - 1,
                             figsize=(15, 8))  # Everything on one row
        row_iter = [plt] if (len(glyph_strokes) - 1) == 1 else ax

        # Plot each stroke within the same subplot. Exclude padding
        for row, l in zip(row_iter, glyph_strokes):
            subplot_glyph_(row, l, n_features)

    else:
        # Everything on one row
        _, ax = plt.subplots(1, subplot_size, figsize=(15, 8))
        row_iter = [plt] if subplot_size == 1 else ax

        for i, row in enumerate(row_iter):
            glyph = data[i].tolist()  # Get glyph at index

            # Want to have the different strokes in the glyph
            # plotted individually within the same subplot
            if granularity == "glyph":
                g_chunks = [j for j in chunker(glyph, n_features)]
                split_chunks = [list(y) for x, y in itertools.groupby(
                    g_chunks, lambda z: z == split_token) if not x]
                glyph_strokes = [list(itertools.chain.from_iterable(i))
                                 for i in split_chunks]

                # Plot each stroke within the same subplot. Exclude padding
                for l in [i for i in glyph_strokes if len(set(i)) != 1]:
                    subplot_glyph_(row, l, n_features)

            # Plot strokes as they are
            else:
                subplot_glyph_(row, glyph, n_features)

    # Show plot
    plt.show()


def draw_expr(data, subplot_size, n_features, granularity="stroke", split_token=-3):
    split_token = [float(split_token)] * n_features

    if granularity == "touch":
        data = data.tolist()

        split_chunks = [list(y) for x, y in itertools.groupby(
            data, lambda z: z == split_token) if not x]
        glyph_strokes = [list(itertools.chain.from_iterable(i))
                         for i in split_chunks]

        # Exclude very last glyph_stroke, as it contains padding and stuff
        _, ax = plt.subplots(1, len(glyph_strokes) - 1,
                             figsize=(15, 8))  # Everything on one row
        row_iter = [plt] if (len(glyph_strokes) - 1) == 1 else ax

        # Plot each stroke within the same subplot. Exclude padding
        for row, l in zip(row_iter, glyph_strokes):
            subplot_glyph_(row, l, n_features)

    else:
        # Everything on one row
        _, ax = plt.subplots(1, subplot_size, figsize=(15, 8))
        row_iter = [plt] if subplot_size == 1 else ax

        for i, row in enumerate(row_iter):
            glyph = data[i].tolist()  # Get glyph at index

            # Want to have the different strokes in the glyph
            # plotted individually within the same subplot
            if granularity == "glyph":
                g_chunks = [j for j in chunker(glyph, n_features)]
                split_chunks = [list(y) for x, y in itertools.groupby(
                    g_chunks, lambda z: z == split_token) if not x]
                glyph_strokes = [list(itertools.chain.from_iterable(i))
                                 for i in split_chunks]

                # Plot each stroke within the same subplot. Exclude padding
                for l in [i for i in glyph_strokes if len(set(i)) != 1]:
                    subplot_glyph_(row, l, n_features)

            # Plot strokes as they are
            else:
                subplot_glyph_(row, glyph, n_features)

    # Show plot
    plt.show()


class SVGPath(object):
    """
       Generates a SVG format path given a tensor as input.
    """

    def __init__(self, input_tensor=None, src_pad_idx=-5, bos_idx=2, eos_idx=3):
        self.PAD_IDX = src_pad_idx
        self.BOS_IDX = bos_idx
        self.EOS_IDX = eos_idx
        self.path = self._remove_indexes(input_tensor) if input_tensor is not None else []

    def _remove_indexes(self, tensor: torch.Tensor):
        tensor = torch.flatten(tensor)
        assert tensor.size(0) == VECTOR_SIZE, "Error in input tensor dimension (should only contain VECTOR SIZE " \
                                              "elements) "
        bos_tensor = torch.zeros(tensor.size(-1), device=tensor.device) + self.BOS_IDX
        eos_tensor = torch.zeros(tensor.size(-1), device=tensor.device) + self.EOS_IDX
        pad_tensor = torch.zeros(tensor.size(-1), device=tensor.device) + self.PAD_IDX

        if torch.all(tensor.eq(bos_tensor)) or torch.all(tensor.eq(eos_tensor)) or torch.all(tensor.eq(pad_tensor)):
            return []
        return tensor[tensor != self.PAD_IDX].tolist()

    def isEmpty(self):
        return not self.path

    def drawPath(self, offset, max_height, max_width):
        if self.isEmpty():
            return ""
        svg_path_str = ""
        x = [elem for index, elem in enumerate(self.path) if not index % 2]
        y = [elem for index, elem in enumerate(self.path) if index % 2]
        assert len(x) == len(y), "Invalid x and y correspondence in input path"
        for i, (xi, yi) in enumerate(zip(x, y)):
            if i == 0:
                svg_path_str += "M"
            else:
                svg_path_str += "L"
            svg_path_str += str(offset + xi * max_width) + "," + str(yi * max_height) + " "
        return svg_path_str


def strokes_to_svg(x: torch.Tensor, size: dict, src_pad_idx, bos_idx, eos_idx, scale_factor=55):
    assert x.shape[-1] == VECTOR_SIZE and len(x.size()) == 2
    paths = []
    for p in x:
        new_path = SVGPath(p, src_pad_idx, bos_idx, eos_idx)
        if not new_path.isEmpty():
            paths.append(new_path)

    paths.append(SVGPath())
    paths.insert(0, SVGPath())
    WIDTH = scale_factor * len(paths)

    dividers = ""
    info = ""
    svgPath = "<path d=\" "

    def fun(j, pa):
        return pa.drawPath(offset=size['width'] * j, max_height=size['height'], max_width=size['width'])

    for i, p in enumerate(paths[:-1]):
        dividers = dividers + f"<line x1=\"{size['width'] * (i + 1)}\" y1=\"0\" x2=\"{size['width'] * (i + 1)}\" y2=\"{size['height']}\"/>" + "\n\t"

        info = info + f"<text x=\"{size['width'] * (i + 1)}\" y=\"{size['height'] * 0.95}\">{i}</text>" + "\n\t"

        svgPath = svgPath + str("" if p.isEmpty() else fun(i, p))

    svgPath += "\" />"

    TEMPLATE = f"<?xml version=\"1.0\" standalone=\"no\"?>\n\
        <svg width=\"{WIDTH}\" height=\"{(size['height'] / len(paths) * WIDTH / size['width'])}\" viewBox=\"0 0 {size['width'] * len(paths)} {size['height']}\" style=\"background-color:white\" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n\
        <title>Strokes</title>\n\
        <desc>Stroke Sequence</desc>\n\
        <g fill=\"#E0E0E0\" stroke=\"none\" stroke-width=\"{size['width'] / 30}\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n\
            <rect x=\"0\" y=\"0\" width=\"{size['width'] * len(paths)}\" height=\"{size['height']}\"/>\"\n\
        </g>\n\
        <g fill=\"#00D000\" stroke=\"none\" fill-opacity=\"0.58\">\n\
            <rect x=\"0\" y=\"0\" width=\"{size['width']}\" height=\"{size['height']}\"/>\"\n\
        </g>\n\
        <g fill=\"#D00000\" stroke=\"none\" fill-opacity=\"0.58\">\n\
            <rect x=\"{size['width'] * (len(paths) - 1)}\" y=\"0\" width=\"{size['width']}\" height=\"{size['height']}\"/>\"\n\
        </g>\n\
        <g fill=\"none\" stroke=\"#000000\" stroke-width=\"{size['width'] / 40}\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n\
            <rect x=\"0\" y=\"0\" width=\"{size['width'] * len(paths)}\" height=\"{size['height']}\"/>\"\n\
        </g>\n\
        <g fill=\"none\" stroke=\"#303030\" stroke-dasharray=\"{size['width'] / 18},{size['width'] / 18}\" stroke-width=\"{size['width'] / 50}\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n\
            {dividers}\n\
        </g>\n\
        <g fill=\"red\" font-size=\"{size['width'] / 60}em\" font-family=\"Arial\">\n\
            {info}\n\
        </g>\n\
        <g fill=\"none\" stroke=\"black\" stroke-width=\"{size['width'] / 100}\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n\
            {svgPath}\n\
        </g>\n\
    </svg>\n\
    "
    return TEMPLATE


def initialize_weights(m):
    """Initialize training weights"""
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        torch.nn.init.xavier_uniform_(m.weight.data)


class DatasetVocab(object):
    """
        Same as torchtext.vocab.Vocab object, but order of tokens is constant

        Defines a vocabulary object that will be used to numericalize a field.

        Attributes:
            freqs: A collections.Counter object holding the frequencies of tokens
                in the data used to build the DatasetVocab.
            stoi: A collections.defaultdict instance mapping token strings to
                numerical identifiers.
            itos: A list of token strings indexed by their numerical identifiers.
        """

    UNK = '<unk>'

    def __init__(self, counter, max_size=None, min_freq=1, specials=('<unk>', '<pad>'),
                 vectors=None, unk_init=None, vectors_cache=None, specials_first=True):
        """Create a DatasetVocab object from a collections.Counter.

            Args:
                counter: collections.Counter object holding the frequencies of
                    each value found in the data.
                max_size: The maximum size of the vocabulary, or None for no
                    maximum. Default: None.
                min_freq: The minimum frequency needed to include a token in the
                    vocabulary. Values less than 1 will be set to 1. Default: 1.
                specials: The list of special tokens (e.g., padding or eos) that
                    will be prepended to the vocabulary. Default: ['<unk'>, '<pad>']
                vectors: One of either the available pretrained vectors
                    or custom pretrained vectors (see DatasetVocab.load_vectors);
                    or a list of aforementioned vectors
                unk_init (callback): by default, initialize out-of-vocabulary word vectors
                    to zero vectors; can be any function that takes in a Tensor and
                    returns a Tensor of the same size. Default: 'torch.zeros'
                vectors_cache: directory for cached vectors. Default: '.vector_cache'
                specials_first: Whether to add special tokens into the vocabulary at first.
                    If it is False, they are added into the vocabulary at last.
                    Default: True.
            """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list()
        self.unk_index = None
        if specials_first:
            self.itos = list(specials)
            # only extend max size if specials are prepended
            max_size = None if max_size is None else max_size + len(specials)

        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])

        # Ensures tokens are always in alphabtical order, by commenting
        # words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        if DatasetVocab.UNK in specials:  # hard-coded for now
            unk_index = specials.index(DatasetVocab.UNK)  # position in list
            # account for ordering of specials, set variable
            self.unk_index = unk_index if specials_first else len(
                self.itos) + unk_index
            self.stoi = defaultdict(self._default_unk_index)
        else:
            self.stoi = defaultdict()

        if not specials_first:
            self.itos.extend(list(specials))

        # stoi is simply a reverse dict for itos
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})
        self.vectors = None
        assert unk_init is None and vectors_cache is None

    def _default_unk_index(self):
        return self.unk_index

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(DatasetVocab.UNK))

    def __getstate__(self):
        # avoid picking defaultdict
        attrs = dict(self.__dict__)
        # cast to regular dict
        attrs['stoi'] = dict(self.stoi)
        return attrs

    def __setstate__(self, state):
        if state.get("unk_index", None) is None:
            stoi = defaultdict()
        else:
            stoi = defaultdict(self._default_unk_index)
        stoi.update(state['stoi'])
        state['stoi'] = stoi
        self.__dict__.update(state)

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def lookup_indices(self, tokens):
        indices = [self.__getitem__(token) for token in tokens]
        return indices

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


# Ablation studies utilities

def remove_equal_sign(Tens):
    # first phase: find eos in tensor
    T = torch.clone(Tens).to(Tens.device)
    E = torch.empty(1, VECTOR_SIZE).fill_(EOS_IDX).to(Tens.device)
    diff = T.unsqueeze(2) - E.unsqueeze(1)
    dsum = torch.abs(diff).sum(-1)
    loc = (dsum == 0).nonzero()[0, 0].item()

    # second phase: copy eos tensor 2 positions before
    T[loc - 2, :] = E[0, :]

    # third phase : copy src_padding tensor in place of last glyph and eos
    T[loc - 1, :] = T[loc, :] = T[loc + 1, :]

    return T


def remove_brackets(Tens):
    # ending brackets is on the 3rd positions before eos, since the equal sign always requires 2 strokes
    # thus, once found the eos index, the bracket's one is loc-3
    # from loc-3 to loc included, we copy the next tensor

    # first phase: find eos in tensor
    T = torch.clone(Tens).to(Tens.device)
    E = torch.empty(1, VECTOR_SIZE).fill_(EOS_IDX).to(Tens.device)
    diff = T.unsqueeze(2) - E.unsqueeze(1)
    dsum = torch.abs(diff).sum(-1)
    loc = (dsum == 0).nonzero()[0, 0].item()

    # second phase: copy eos tensor 2 positions before
    for i in range(loc - 3, loc + 1):
        T[i, :] = T[i + 1, :]

    return T
