import time

import h5py
import random
import os
import numpy as np
import copy

import torch
import unicodedata
import multiprocessing

import tqdm
import string

from contextlib import suppress
from itertools import chain, repeat
from collections import Counter
from abc import ABC, abstractmethod
from models.scripts.defaults import VECTOR_SIZE, DECODER_OUTPUT_LENGTH, DB_PATH

from models.scripts.data_model import Subject, Glyph
from models.scripts.utils import create_db_session, to_secs, chunker, interpolate

CHARS = list(string.ascii_letters)
DIGITS = [i for i in string.digits]
LOWERCASE_LETTERS = [i for i in string.ascii_lowercase]
UPPERCASE_LETTERS = [i for i in string.ascii_uppercase]
MISC_CHARS = ['.', ',', '!', '?', '+', '-', '/', '*', '=', '(', ')', ]
NO_SPACE = [""]
NEW_LINE = ['\n']
BLANK_SPACE = [" "]

GRANULARITIES = ['glyph', 'stroke', 'touch']
EXPR_MODES = ['all', 'digits', 'alphabets']

DEF_AUGMENTATION = {'amount': 1,  # Amount to augment each glyph by
                    'combine': False,  # Determines if strategies should be combined
                    'strategies': [  # Define augmentation strategies along with their factors
                        ('scaleUp', 2),  # Scale glyph by 2%
                        ('scaleDown', 4),  # Reduce glyph by 4%
                        ('shiftX', 15),  # Shift along X axis by 15% (relative to client's viewport)
                        ('shiftY', 20),  # Shift along Y axis by 20% (relative to client's viewport)
                        ('squeezeX', 10),  # Squeeze along X axis by 10%
                        ('squeezeY', 10),  # Squeeze along Y axis by 10%
                    ]}


class SequenceGenerator(ABC):
    """
    Generates train, validation, and test set data

    param: vocab: vocabulary used to generate dataset
    param: allow_brackets: specify if complex expressions with brackets are allowed
    param: save_mode: specifies label to generate:
        ('unsolved': standard infix notation,
        'postfix': standard postfix notation,
        'marked_postfix': postfix but adds a separator mark at the end of the literals,
        'solved': saves results of the expressions)
    param: total_expressions: number of expressions to generate
    param: vector_size: number of touches to embed each stroke
    param: max_seq_len: Maximum length of an expression, default is 10
    param: padding_value (int): Value for padding the generated dataset to meet the required vector size (Def: -5)
    param: augmentation: dictionary of allowed glyph transformations to augment them
    param: train_split (float): Percentage of training set (defaults to 0.6, i.e. 60%)
    param: valid_split (float): Percentage of validation set (defaults to 0.2, i.e. 20%)
    param: scale_by_ar (bool): Flag to determine if y coordinate should be scaled by aspect ratio (Defaults to False)
    param: offset_delay (bool): Flag to determine if time delay between strokes should be considered (Defaults to False)
    param: granularity (str): Granularity of supervised dataset. (Def: 'glyph')
    param: end_of_glyph_value (int): Value to signify the end of a glyph in the generated dataset (Def: -1)
    param: end_of_stroke_value (int): Value to signify the end of a stroke in the generated dataset (Def: -2)
    param: include_time_feature (bool): Flag to indicate whether to include the time feature (Def: True)
    """

    CHUNK_SIZE = 8192  # 8kb
    SAMPLING_RATE = 10e-3  # 5ms
    NUM_PROCESSES = 3  # One process per batch

    def __init__(self,
                 vocab,
                 total_samples: int = 1000,
                 input_size: int = 99,
                 vector_size: int = VECTOR_SIZE,
                 db_path: str = DB_PATH,
                 padding_value: int = -5,
                 augmentation=None,
                 train_split: float = 0.6,
                 valid_split: float = 0.2,
                 use_subject: bool = True,
                 scale_by_ar: bool = True,
                 sample_data: bool = False,
                 offset_delay: bool = False,
                 granularity: str = "stroke",
                 end_of_glyph_value: int = -4,
                 end_of_stroke_value: int = -3,
                 expr_mode: str = "alphabets",
                 include_time_feature: bool = False,
                 include_pressure_feature: bool = False,
                 bos_idx=None,
                 eos_idx=None,
                 pad_idx=None,
                 fname=None
                 ):

        # Train and Validation split are within 0 and 1
        if augmentation is None:
            augmentation = {}
        assert (0 <= train_split <= 1), \
            "Train split should be between 0 and 1 (e.g 0.75)"
        assert (0 <= valid_split <= 1 - train_split), \
            f"Validation split should be between 0 and {1 - train_split}"

        # Ensure valid experiment mode and granularity are selected
        assert expr_mode in EXPR_MODES, f"Invalid experiment mode. Must be any of {EXPR_MODES}"
        assert granularity in GRANULARITIES, f"Invalid granularity. Must be any of {GRANULARITIES}"

        self.vocab = vocab
        self.total_samples = total_samples
        self.input_size = input_size
        self.expr_mode = expr_mode
        self.train_split = train_split
        self.vector_size = vector_size
        self.granularity = granularity
        self.augmentation = augmentation
        self.validation_split = valid_split

        # bos/eos/pad tokens idx
        self.bos_idx = bos_idx if bos_idx else self._get_token_id_from_vocab('<bos>')
        self.eos_idx = eos_idx if eos_idx else self._get_token_id_from_vocab('<eos>')
        self.pad_idx = pad_idx if pad_idx else self._get_token_id_from_vocab('<pad>')

        # Misc
        self.db_path = db_path
        self._should_augment = any(augmentation)

        # Calculate test split
        self.test_split = 1 - (train_split + valid_split)

        # Boolean flags
        self.use_subject = use_subject
        self.scale_by_ar = scale_by_ar
        self.sample_data = sample_data
        self.offset_delay = offset_delay
        self.include_time_feature = include_time_feature
        self.include_pressure_feature = include_pressure_feature

        # If time is included, features = (x,y,t) else (x,y)
        self.n_features = 3 if self.include_time_feature else 2
        # If position is included, features = (x,y,t,p)
        self.n_features = self.n_features + 1 if self.include_pressure_feature else self.n_features

        # Constant tuples
        self.padding_value = padding_value
        self.end_of_glyph_value = end_of_glyph_value
        self.end_of_stroke_value = end_of_stroke_value

        # Generated dataset
        self._x_test = []
        self._y_test = []
        self._x_train = []
        self._y_train = []
        self._x_valid = []
        self._y_valid = []

        self.avg_glyph_strokes = 2  # Avg of 2 strokes per glyph from statistical analysis
        self.avg_glyph_touches = 128  # Avg of 128 touches per glyph from statistical analysis

        self.dtype = None
        self.fname = fname
        self.fpath = None
        self.subjects = []

    def _get_token_id_from_vocab(self, token):
        """Get the token id from a vocabulary"""

        if type(self.vocab).__name__ == "Tokenizer":
            token_id = self.vocab.token_to_id(token)

        else:  # `torchtext.vocab.Vocab`
            token_id = self.vocab.stoi[token]

        return token_id

    def _to_gen(self, x, y, mode, hf_file=None):
        """
        Converts (x, y) data to a generator and iteratively consumes
        it. This is to account for very large datasets that would be
        eventually consumed by the PyTorch DataLoader for batching
        """

        # Y_count = 0
        # X_count = [0, 0, 0]
        for x_i, y_i in zip(x, y):
            yield np.array(x_i).tolist(), y_i.decode("utf-8")

        if hf_file:
            hf_file.close()  # Close the h5 file after

    def _pad(self, touches: list, size=None, padding=None):
        """Pad or chop off touches using the required size"""

        touches_length = len(touches)
        size = size or self.vector_size
        diff = abs(size - touches_length)

        # Vector size greater than touches?
        if size > touches_length:
            # Create padding list
            padding = padding or self.padding_value

            # Pad end of array up to vector size
            padded_touches = list(chain(touches, repeat(padding, diff)))

        # Touches greater than vector size?
        else:
            # Chop off difference to meet vector size
            padded_touches = list(chain(touches[0: (touches_length - diff)]))

        return padded_touches

    def _split(self, data_iter: list):
        """Splits a list of data_iter into the three sets"""

        # Get total length of data_iter
        data_iter_length = len(data_iter)

        train_index = int(self.train_split * data_iter_length)
        test_index = int(
            train_index + (self.validation_split * data_iter_length))

        # train set, validation set, test set
        t, v, ts = data_iter[0:train_index], data_iter[train_index: test_index], data_iter[test_index:]

        return t, v, ts

    def _generate_glyphs_from_sequence(self, seq: str, subject_choices, sc):
        pass

    def _tensorize_string(self, trg_string):
        """Convert a string into a tensor"""

        if type(self.vocab).__name__ == "ByteLevelBPETokenizer":
            tsor = torch.tensor(self.vocab.encode(trg_string).ids)

        else:  # `torchtext.vocab.Vocab`
            tsor = torch.stack([torch.tensor(self.vocab.stoi[j])
                                for j in trg_string])

        return tsor

    def _sample_touches(self, touches: list):
        """Sample touches using the defined sampling rate"""

        min_delta_t = SequenceGenerator.SAMPLING_RATE

        # Sample using minimum delta t
        count = 0  # Counter. Aliased as index
        touches_ = []  # Home for equally-sampled touches
        xn, yn, tn = None, None, None  # Variables for sampled touch

        # Split touches into list of touches grouped together
        touches_iter = list(chunker(touches, self.n_features))

        # Iterate until condition...
        while count < len(touches_iter):
            with suppress(IndexError):
                # Save curr and next touch
                x0, y0, t0 = touches_iter[count]
                x1, y1, t1 = touches_iter[count + 1]

                # If the first touch, then the next touch
                # is simply the next touch in the list
                if count == 0:
                    prev_ = [x0, y0]
                    next_ = [x1, y1]
                    t_start = t0
                    t_end = t1
                    t_next = t0 + min_delta_t

                # If not, then ensure that the sampled touch just
                # previously gotten has time more than the current
                # touch. If not, modify the state to take the current
                # touch as the next touch and the sampled touch as the
                # previous. Also decrement the counter. Continue as
                # usual, otherwise.
                else:
                    t_start = tn
                    prev_ = [xn, yn]
                    t_next = tn + min_delta_t

                    if t_next <= t0:
                        count -= 1
                        t_end = t0
                        next_ = [x0, y0]
                    else:
                        t_end = t1
                        next_ = [x1, y1]

                # Save the sampled touch
                touches_.append(prev_[0])
                touches_.append(prev_[1])

                tn = t_next  # Reassign the sampled touch to the next touch

                # Interpolate a touch position, given the time and the boundaries
                xn, yn = interpolate(prev_, next_, t_start, t_end, t_next)

            count += 1  # Increment the counter

        return touches_

    def _granularize(self, glyphs, session):
        """Expand `glyph`s into required granularity"""

        word_touches = []
        bos_vector = list(chain(repeat(self.bos_idx, self.vector_size)))
        eos_vector = list(chain(repeat(self.eos_idx, self.vector_size)))

        # Add beginning of sequence vector
        word_touches.append(bos_vector)

        stroke_position = 0  # Track position of stroke in glyphs

        # For each glyph...
        for char in glyphs:

            stroke_start_time = 0  # Default start time
            if self.granularity == "glyph":
                touches = []

            # Use serialized version of glyph. It's faster
            char = char.serialize(session=session) if not isinstance(
                char, dict) else char

            # Delays between strokes
            stroke_delays = char['stroke_delays'].split(" ")

            # For each stroke in the glyph...
            for stroke_index, stroke in enumerate(char['strokes']):

                if self.granularity == "stroke":
                    touches = []

                # For each touch in the stroke...
                min_x = float('inf')
                min_y = float('inf')
                max_x = 0
                max_y = 0
                for index, touch in enumerate(stroke['touches'], start=1):

                    if self.granularity == "touch":
                        touches = []

                    x = touch['x']
                    # Scale y (or not) by aspect ratio
                    y = touch['y']
                    if self.scale_by_ar:
                        ar = char['ar']
                        if ar > 1:
                            x *= ar
                        else:
                            y /= ar

                    min_x = min(x, min_x)
                    max_x = max(x, max_x)
                    min_y = min(y, min_y)
                    max_y = max(y, max_y)
                    # Get timestamp in seconds
                    t = touch['timestamp'] * 1e-15 + stroke_start_time  # (s)

                    # Add to touches list
                    touches.append(x)  # x
                    touches.append(y)  # y
                    touches.append(
                        t) if self.include_time_feature else None  # t
                    touches.append(
                        stroke_position) if self.include_pressure_feature else None  # p

                    # Add touch to sequence dimension
                    if self.granularity == 'touch':
                        word_touches.append(touches)

                        # Add end of stroke signal
                        if index == len(stroke['touches']):
                            word_touches.append(
                                [self.end_of_stroke_value for _ in range(self.n_features)])

                if max_x > 1:
                    new_touches = []
                    for i, t in enumerate(touches):
                        if i % self.n_features == 0:
                            new_touches.append(t - min_x + (min_x + (1 - max_x)) / 2)
                        else:
                            new_touches.append(t)
                    touches = new_touches
                if max_y > 1:
                    new_touches = []
                    for i, t in enumerate(touches):
                        if i % self.n_features == 1:
                            new_touches.append(t - min_y + (min_y + (1 - max_y)) / 2)
                        else:
                            new_touches.append(t)
                    touches = new_touches
                # Sample touches, if required. Time must also be included
                if self.sample_data and self.include_time_feature:
                    touches = self._sample_touches(touches)

                # If time delay should be offset
                if self.offset_delay:
                    # Get delay before next stroke
                    delay_to_next_stroke = 0
                    with suppress(IndexError):
                        delay_to_next_stroke = stroke_delays[stroke_index]

                    # Rewrite the stroke's start time to the delay plus time
                    stroke_start_time = (t + to_secs(delay_to_next_stroke))

                if self.granularity == 'glyph':
                    # Add end of stroke signal
                    for _ in range(self.n_features):
                        touches.append(self.end_of_stroke_value)

                # Add end of glyph signal, if end of glyph
                # Should also offset stroke start time for
                # the next glyph by average glyph delays
                # if (stroke_index + 1) == len(char.strokes):
                #     touches.append(self.end_of_glyph_value)

                # If granularity is stroke, pad
                # Save entire [padded] stroke vector
                if self.granularity == 'stroke':
                    # Pad all touches for the stroke
                    # to the required `vector size`
                    touches = self._pad(touches)

                    # Save to word ('Glyph')
                    word_touches.append(touches)

            # if self.granularity == 'touch':
            # Add end of glyph signal
            # word_touches.append([self.end_of_glyph_value for _ in range(self.n_features)])

            # Save entire [padded] glyph vector
            if self.granularity == 'glyph':
                # Pad all touches for the glyph
                # to the required `vector size`
                # touches is the concatenated touches
                # for all the strokes in the glyph
                touches = self._pad(touches)

                # Save to word ('Glyph')
                word_touches.append(touches)

            # Increment stroke posiiton used
            # when positon feature is enabled
            stroke_position += 1

        # Glyphs with multiple strokes should be accounted for
        if self.granularity == "glyph":
            max_size = self.input_size

        elif self.granularity == "stroke":
            max_size = self.input_size * self.avg_glyph_strokes

        elif self.granularity == "touch":
            max_size = self.input_size * self.avg_glyph_touches
        else:
            raise
        # Padding tensor/list
        padding = list(chain(repeat(self.padding_value, self.vector_size)))

        # Add end of sequence vector
        word_touches.append(eos_vector)

        # Pad dataset to the word max size
        word_touches = self._pad(word_touches, size=max_size, padding=padding)

        return word_touches  # Return the sequence dimension

    def _save_dataset(self, expanded_glyphs: list, chars_batch: list, mode: str = 'train'):
        """Save related dataset"""

        if mode == 'test':
            x_dataset, y_dataset = self._x_test, self._y_test
        elif mode == 'valid':
            x_dataset, y_dataset = self._x_valid, self._y_valid
        elif mode == 'train':
            x_dataset, y_dataset = self._x_train, self._y_train
        else:
            raise

        # Save the whole sequence
        for char_seq in chars_batch:
            y_dataset.append(char_seq.encode('utf-8'))

        # Add to list of generated datasets
        for glyph in expanded_glyphs:
            x_dataset.append(glyph)

    def _augment_glyphs(self, glyphs: list, session):
        """
        Augment a list of glyphs
        Converts the passed glyphs to their json representation
        and adds them to the augmented list. Then, using those
        json-represented glyphs, augments glyphs up to the requested
        amount and using the selected strategies.

        The json-representation workaround is because creating new glyphs
        from their SQLAlchemy representation has proved troublesome
        """

        augmented_glyphs = []  # Storage for augmented glyphs

        # Loop the requested amount of augmented glyphs
        for glyph in glyphs:

            # Should strategies be combined?
            # combine_strategies = self.augmentation['combine']

            # Create new glyph from chosen glyph
            new_glyph = copy.deepcopy(glyph.serialize(session))

            # Get the augmentation strategy and associated factor
            strategy, factor = random.choice(self.augmentation['strategies'])

            # Force decimal
            factor = (factor % 100) / 100

            # For each stroke...
            for stroke in new_glyph['strokes']:

                # For each touch...
                for touch in stroke['touches']:
                    if strategy == 'shiftX':
                        touch['x'] = touch['x'] + factor

                    elif strategy == 'shiftY':
                        touch['y'] = touch['y'] + factor

                    elif strategy == 'squeezeX':
                        aug_m = [[1 + factor, 0], [0, 1]]
                        x, y = np.array((touch['x'], touch['y'])).dot(
                            aug_m).tolist()
                        touch['x'], touch['y'] = x, y

                    elif strategy == 'squeezeY':
                        aug_m = [[1, 0], [0, 1 + factor]]
                        x, y = np.array((touch['x'], touch['y'])).dot(
                            aug_m).tolist()
                        touch['x'], touch['y'] = x, y

                    elif strategy == 'scaleUp':
                        aug_m = [[1 + factor, 0], [0, 1 + factor]]
                        x, y = np.array((touch['x'], touch['y'])).dot(
                            aug_m).tolist()
                        touch['x'], touch['y'] = x, y

                    elif strategy == 'scaleDown':
                        aug_m = [[1 - factor, 0], [0, 1 - factor]]
                        x, y = np.array((touch['x'], touch['y'])).dot(
                            aug_m).tolist()
                        touch['x'], touch['y'] = x, y

                    elif strategy == 'skew':
                        deg = factor * np.pi / 180  # To radians
                        aug_m = [[np.cos(deg), -(np.sin(deg))],
                                 [np.sin(deg), np.cos(deg)]]
                        x, y = np.array((touch['x'], touch['y'])).dot(
                            aug_m).tolist()
                        touch['x'], touch['y'] = x, y

                    # More
                    else:
                        pass

            # Save augmented glyph (spatial)
            augmented_glyphs.append(new_glyph)

        return augmented_glyphs

    def _load_dataset_from_cache(self, cache_file=None, mode='train'):
        """Loads dataset from a cache file"""

        if not cache_file:
            cache_file = self.fpath

        if mode == "train":
            X = "X_train"
            Y = "Y_train"

        elif mode == "valid":
            X = "X_valid"
            Y = "Y_valid"

        elif mode == "test":
            X = "X_test"
            Y = "Y_test"

        else:
            raise AttributeError(f"{mode} is an invalid mode")

        hf = h5py.File(cache_file, 'r')

        # Return generator
        return self._to_gen(hf[X], hf[Y], mode, hf)

    def get_learning_set_length(self, mode="train"):
        if mode == "train":
            Y = "Y_train"

        elif mode == "valid":
            Y = "Y_valid"

        elif mode == "test":
            Y = "Y_test"
        else:
            raise AttributeError(f"{mode} is an invalid mode")

        hf = h5py.File(self.fpath, 'r')
        return len(hf[Y])

    def get_all_subjects(self, session, mode=None, augment=False):
        """
        Get all subjects in dataset
        If `mode` is passed, then the corresponding subjects
        for that mode (e.g. `test`, or `train`) is returned.
        """

        if mode:
            if mode == 'test':
                index = 2
            elif mode == 'valid':
                index = 1
            else:  # Implicit train
                index = 0

            if self.subjects:
                return self.subjects[index]

        if self.subjects:
            return self.subjects

        # if here, subjects have not been initialized

        subjects_ = []

        if self.expr_mode == "digits":
            expr = [1, 4]
        elif self.expr_mode == "alphabets":
            expr = [1, 2, 3]  # Note that some subjects have missing characters.
        else:  # Implicit "all"
            expr = [1, 2, 3, 4]

        # Ascending order
        for subject in session.query(Subject):
            for gs in subject.glyph_sequences:
                if gs.experiment in expr and subject not in subjects_:
                    subjects_.append(subject)

        # Seeded shuffle
        random.seed(5050)
        random.shuffle(subjects_)

        # Split the [available] subjects into train, validation, and test subjects
        subjects = self._split(subjects_)

        if augment and self.expr_mode != "digits":
            for subject in session.query(Subject):
                for gs in subject.glyph_sequences:
                    if gs.experiment == 5 and subject not in subjects_ and random.random() < 0.3:  # take only 30% of all authors
                        subjects[0].append(subject)
            random.shuffle(subjects[0])

        # Ensure subjects are unique across each split mode
        assert len(set(subjects[0]) & set(subjects[2])) == 0
        assert len(set(subjects[0]) & set(subjects[1])) == 0
        assert len(set(subjects[1]) & set(subjects[2])) == 0

        self.subjects = subjects

        if mode:
            return subjects[index]

        print("Subjects are unique!\n")
        return subjects

    def cache_generated_dataset(self, fname=None):
        """Save generated  data to disk as a h5 file"""

        print("\nCaching...")

        if len(self._x_test) == 0:
            return "Caching failed. No generated data."

        if not fname:
            fpath = self.fpath
        else:
            fpath = fname

        y_test = np.array(self._y_test, dtype=self.dtype)
        y_valid = np.array(self._y_valid, dtype=self.dtype)
        y_train = np.array(self._y_train, dtype=self.dtype)

        with h5py.File(fpath, 'w') as hf:
            hf.create_dataset("Y_test", compression="gzip",
                              chunks=True, data=y_test)
            hf.create_dataset("Y_valid", compression="gzip",
                              chunks=True, data=y_valid)
            hf.create_dataset("Y_train", compression="gzip",
                              chunks=True, data=y_train)
            hf.create_dataset("X_test", compression="gzip",
                              chunks=True, data=np.array(self._x_test))
            hf.create_dataset("X_valid", compression="gzip",
                              chunks=True, data=np.array(self._x_valid))
            hf.create_dataset("X_train", compression="gzip",
                              chunks=True, data=np.array(self._x_train))

        print(f"Dataset saved to {fpath}.")

    def _create_dataset(self):
        """Create an empty dataset"""

        self._x_test = []
        self._y_test = []
        self._x_train = []
        self._y_train = []
        self._x_valid = []
        self._y_valid = []

    def _init_cache(self, fname=None):
        """Initialize cache"""

        print("\nInitializing cache...")

        if not fname:
            fname = self.fpath

        if os.path.exists(fname):
            os.remove(fname)

        with h5py.File(fname, 'w') as hf:
            if self.granularity == 'stroke':
                ms = (self.avg_glyph_strokes * self.input_size, self.vector_size)
            elif self.granularity == 'glyph':
                ms = (self.input_size, self.vector_size)
            elif self.granularity == 'touch':
                ms = (self.avg_glyph_touches * self.input_size, self.vector_size)
            else:
                raise

            # Create the actual datasets
            hf.create_dataset('X_test', compression="gzip",
                              chunks=True, maxshape=(None, *ms), shape=(0, *ms))
            hf.create_dataset('X_train', compression="gzip",
                              chunks=True, maxshape=(None, *ms), shape=(0, *ms))
            hf.create_dataset('X_valid', compression="gzip",
                              chunks=True, maxshape=(None, *ms), shape=(0, *ms))
            hf.create_dataset('Y_test', compression="gzip",
                              chunks=True, maxshape=(None,), shape=(0,), dtype=self.dtype)
            hf.create_dataset('Y_train', compression="gzip",
                              chunks=True, maxshape=(None,), shape=(0,), dtype=self.dtype)
            hf.create_dataset('Y_valid', compression="gzip",
                              chunks=True, maxshape=(None,), shape=(0,), dtype=self.dtype)

    def _update_cache(self):
        """
        Save generated dataset to filesystem as a .h5 file
        """

        y_test = np.array(self._y_test, dtype=self.dtype)
        y_valid = np.array(self._y_valid, dtype=self.dtype)
        y_train = np.array(self._y_train, dtype=self.dtype)

        x_test = np.array(self._x_test)
        x_train = np.array(self._x_train)
        x_valid = np.array(self._x_valid)

        # Add updates to h5 cache
        with h5py.File(self.fpath, 'a') as hf:
            if self._y_test:
                hf["Y_test"].resize((hf["Y_test"].shape[0] + y_test.shape[0]), axis=0)
                hf["X_test"].resize(hf["X_test"].shape[0] + x_test.shape[0], axis=0)
                hf["Y_test"][-y_test.shape[0]:] = y_test.astype(self.dtype)
                hf["X_test"][-x_test.shape[0]:] = x_test
            if self._y_train:
                hf["Y_train"].resize((hf["Y_train"].shape[0] + y_train.shape[0]), axis=0)
                hf["X_train"].resize((hf["X_train"].shape[0] + x_train.shape[0]), axis=0)
                hf["Y_train"][-y_train.shape[0]:] = y_train.astype(self.dtype)
                hf["X_train"][-x_train.shape[0]:] = x_train
            if self._y_valid:
                hf["Y_valid"].resize((hf["Y_valid"].shape[0] + y_valid.shape[0]), axis=0)
                hf["X_valid"].resize((hf["X_valid"].shape[0] + x_valid.shape[0]), axis=0)
                hf["Y_valid"][-y_valid.shape[0]:] = y_valid.astype(self.dtype)
                hf["X_valid"][-x_valid.shape[0]:] = x_valid

        # Reset for next batch
        self._create_dataset()

    def generate_from_cache(self, cache_file=None, mode=None):
        """Generate the train, validation, and test datasets from a cache file"""

        if not cache_file:
            cache_file = self.fpath

        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"{cache_file} does not exist.")

        print(f"Using cached dataset file in {cache_file}")

        if mode:
            if mode == 'test':
                return self._load_dataset_from_cache(cache_file, "test")
            elif mode == 'valid':
                return self._load_dataset_from_cache(cache_file, "valid")
            elif mode == 'train':
                return self._load_dataset_from_cache(cache_file, "train")
            else:
                raise

        else:
            test = self._load_dataset_from_cache(cache_file, "test")
            valid = self._load_dataset_from_cache(cache_file, "valid")
            train = self._load_dataset_from_cache(cache_file, "train")

        return train, valid, test

    @abstractmethod
    def generate(self):
        """
        Generate the train, validation, and test datasets.
        """

        pass


class ClassificationSequenceGenerator(SequenceGenerator):
    """
    Generates train, validation, and test set data

    param: vocab: vocabulary used to generate dataset
    param: allow_brackets: specify if complex expressions with brackets are allowed
    param: save_mode: specifies label to generate:
        ('unsolved': standard infix notation,
        'postfix': standard postfix notation,
        'marked_postfix': postfix but adds a separator mark at the end of the literals,
        'solved': saves results of the expressions)
    param: total_expressions: number of expressions to generate
    param: vector_size: number of touches to embed each stroke
    param: max_seq_len: Maximum length of an expression, default is 10
    param: padding_value (int): Value for padding the generated dataset to meet the required vector size (Def: -5)
    param: augmentation: dictionary of allowed glyph transformations to augment them
    param: train_split (float): Percentage of training set (defaults to 0.6, i.e. 60%)
    param: valid_split (float): Percentage of validation set (defaults to 0.2, i.e. 20%)
    param: scale_by_ar (bool): Flag to determine if y coordinate should be scaled by aspect ratio (Defaults to False)
    param: offset_delay (bool): Flag to determine if time delay between strokes should be considered (Defaults to False)
    param: granularity (str): Granularity of supervised dataset. (Def: 'glyph')
    param: end_of_glyph_value (int): Value to signify the end of a glyph in the generated dataset (Def: -1)
    param: end_of_stroke_value (int): Value to signify the end of a stroke in the generated dataset (Def: -2)
    param: include_time_feature (bool): Flag to indicate whether to include the time feature (Def: True)
    """

    def __init__(self, vocab, total_chars: int = 1000, input_size: int = 100, vector_size: int = 128,
                 db_path: str = "unified_schema.db", padding_value: int = -5, augmentation=None,
                 train_split: float = 0.6, valid_split: float = 0.2, use_subject: bool = True, scale_by_ar: bool = True,
                 sample_data: bool = False, offset_delay: bool = False, granularity: str = "stroke",
                 end_of_glyph_value: int = -4, end_of_stroke_value: int = -3, expr_mode: str = "alphabets",
                 include_time_feature: bool = False, include_pressure_feature: bool = False):

        # Train and Validation split are within 0 and 1
        super().__init__(vocab, total_chars, input_size, vector_size, db_path, padding_value, augmentation, train_split,
                         valid_split, use_subject, scale_by_ar, sample_data, offset_delay, granularity,
                         end_of_glyph_value, end_of_stroke_value, expr_mode, include_time_feature,
                         include_pressure_feature)

        self.dtype = np.dtype('S1')
        self.fname = f"classifications_{str(int(self.total_samples / 1000)) + 'k'}"
        self.fpath = os.path.join("cache", "classification_cache", f"{self.fname}.h5")

    def _generate_glyphs_from_sequence(self, seq: str, subject_choices, sc):
        """
        Generates glyphs corresponding to a given `expression`
        If `self.use_subject` is True, random subjects are selected
        from the dataset and used as the source of the glyphs
        """
        i = 0
        char = '#'
        for i, char in enumerate(seq):
            if char != '#':
                break
        # For each character in the expression
        # Round-robin all subjects in the training set.
        sc += 1  # Increment index
        if sc >= len(subject_choices):
            sc = 0  # Reset
        subject_to_use = None
        subj = subject_choices[sc]

        glyph_choices = [gl for gs in subj.glyph_sequences for gl in gs.glyphs if gl.ground_truth == char]
        if glyph_choices:
            subject_to_use = subj

        while not subject_to_use:

            sc += 1  # Increment index

            if sc >= len(subject_choices):
                sc = 0  # Reset

            subj = subject_choices[sc]
            glyph_choices = [gl for gs in subj.glyph_sequences for gl in gs.glyphs if gl.ground_truth == char]

            if glyph_choices:
                subject_to_use = subj

        glyph_choice = random.choice(glyph_choices)

        return list(seq[:i]) + [glyph_choice] + list(seq[i + 1:]), sc

    def _granularize(self, sequence: list, session):
        """Expand `glyph`s into required granularity"""

        expr_touches = []
        bos_vector = list(chain(repeat(self.bos_idx, self.vector_size)))
        eos_vector = list(chain(repeat(self.eos_idx, self.vector_size)))
        padding = list(chain(repeat(self.padding_value, self.vector_size)))

        # Add beginning of sequence vector
        expr_touches.append(bos_vector)

        stroke_position = 0  # Track position of stroke in glyphs

        # For each glyph...
        for char in sequence:
            if char == '#':
                expr_touches.append(padding)
                continue

            stroke_start_time = 0  # Default start time
            if self.granularity == "glyph":
                touches = []

            # Use serialized version of glyph. It's faster
            char = char.serialize(session=session) if not isinstance(
                char, dict) else char

            # Delays between strokes
            stroke_delays = char['stroke_delays'].split(" ")

            # For each stroke in the glyph...
            for stroke_index, stroke in enumerate(char['strokes']):

                if self.granularity == "stroke":
                    touches = []

                # For each touch in the stroke...
                for index, touch in enumerate(stroke['touches'], start=1):

                    if self.granularity == "touch":
                        touches = []

                    x = touch['x']

                    # Scale y (or not) by aspect ratio
                    y = touch['y'] / \
                        char['ar'] if self.scale_by_ar else touch['y']

                    # Get timestamp in seconds
                    t = touch['timestamp'] * 1e-15 + stroke_start_time  # (s)

                    # Add to touches list
                    touches.append(x)  # x
                    touches.append(y)  # y
                    touches.append(
                        t) if self.include_time_feature else None  # t
                    touches.append(
                        stroke_position) if self.include_pressure_feature else None  # p

                    # Add touch to sequence dimension
                    if self.granularity == 'touch':
                        expr_touches.append(touches)

                        # Add end of stroke signal
                        if index == len(stroke['touches']):
                            expr_touches.append(
                                [self.end_of_stroke_value for _ in range(self.n_features)])

                # If time delay should be offset
                if self.offset_delay:
                    # Get delay before next stroke
                    delay_to_next_stroke = 0
                    with suppress(IndexError):
                        delay_to_next_stroke = stroke_delays[stroke_index]

                    # Rewrite the stroke's start time to the delay plus time
                    stroke_start_time = (t + to_secs(delay_to_next_stroke))

                if self.granularity == 'glyph':
                    # Add end of stroke signal
                    for _ in range(self.n_features):
                        touches.append(self.end_of_stroke_value)

                # Add end of glyph signal, if end of glyph
                # Should also offset stroke start time for
                # the next glyph by average glyph delays
                # if (stroke_index + 1) == len(char.strokes):
                #     touches.append(self.end_of_glyph_value)

                # If granularity is stroke, pad
                # Save entire [padded] stroke vector
                if self.granularity == 'stroke':
                    # Pad all touches for the stroke
                    # to the required `vector size`
                    touches = self._pad(touches)

                    # Save to expression ('Glyph')
                    expr_touches.append(touches)

            # if self.granularity == 'touch':
            # Add end of glyph signal
            # expr_touches.append([self.end_of_glyph_value for _ in range(self.n_features)])

            # Save entire [padded] glyph vector
            if self.granularity == 'glyph':
                # Pad all touches for the glyph
                # to the required `vector size`
                # touches is the concatenated touches
                # for all the strokes in the glyph
                touches = self._pad(touches)

                # Save to expression ('Glyph')
                expr_touches.append(touches)

            # Increment stroke posiiton used
            # when positon feature is enabled
            stroke_position += 1

        # Glyphs with multiple strokes should be accounted for
        if self.granularity == "glyph":
            max_size = self.input_size

        elif self.granularity == "stroke":
            max_size = self.input_size * self.avg_glyph_strokes

        elif self.granularity == "touch":
            max_size = self.input_size * self.avg_glyph_touches

        else:
            raise

        # Add end of sequence vector
        expr_touches.append(eos_vector)

        # Pad dataset to the max_seq_len
        expr_touches = self._pad(expr_touches, size=max_size, padding=padding)

        return expr_touches  # Return the sequence dimension

    def generate_sequence_with_one_char(self):
        len_seq = int(self.input_size * (random.random() + 1))
        seq = '#' * len_seq
        pos = random.randint(0, len_seq)
        char = random.choice(CHARS)
        seq = seq[:pos] + char + seq[pos + 1:]
        return seq, char

    def generate(self):
        """
        Generate the train, validation, and test datasets.
        """

        self._init_cache(None)
        session = create_db_session(self.db_path)

        generated_sequences = []
        for _ in tqdm.tqdm(range(self.total_samples)):
            gen, lab = self.generate_sequence_with_one_char()
            generated_sequences.append((gen, lab))

        # Split expressions into train, validation, test
        train_seq, valid_seq, test_seq = self._split(generated_sequences)

        print("Generating datasets...\n")
        for (learning_set, mode) in [(test_seq, 'test'), (train_seq, 'train'), (valid_seq, 'valid')]:

            subjects = self.get_all_subjects(session, mode)

            sc = 0  # Subject counter index. Used to recycle subjects

            chunk_size = SequenceGenerator.CHUNK_SIZE if len(
                learning_set) > SequenceGenerator.CHUNK_SIZE else len(learning_set)

            # Range should have just one iteration if length of y is less than preset chunk size
            end_index = ((len(learning_set) // chunk_size) + 1) if len(
                learning_set) > SequenceGenerator.CHUNK_SIZE else 1

            with tqdm.tqdm(total=len(learning_set), desc=f"{mode.capitalize()} set progress") as pbar:
                for slice_index in tqdm.tqdm(range(end_index), disable=True):
                    next_slice = learning_set[slice_index * chunk_size: (slice_index + 1) * chunk_size]
                    et, etw = [], []  # Storage for generated touches and corresponding ground truth

                    for index, seq in enumerate(tqdm.tqdm(next_slice, disable=True)):
                        # 'boy' -> 'Glyph (b)', 'Glyph (o)', 'Glyph (y)'
                        glyphs, sc = self._generate_glyphs_from_sequence(seq[0], subjects, sc)
                        # Expand the glyphs into strokes or glyphs
                        char_touches = self._granularize(glyphs, session)
                        # Save expanded touches
                        et.append(char_touches)

                        # Save appropriate expression

                        label = seq[1].strip("'").strip('"').strip('[').strip(']')
                        etw.append(label)

                        if self._should_augment and mode == 'train':  # If augmenting...
                            for _ in range(self.augmentation['amount']):
                                aug_glyphs = self._augment_glyphs(glyphs, session)

                                # Expand the glyphs depending on granularity
                                aug_expr_touches = self._granularize(
                                    aug_glyphs, session)

                                # Save expanded touches
                                et.append(aug_expr_touches)
                                label = seq[1].strip("'").strip('"').strip('[').strip(']')
                                etw.append(label)
                        pbar.update(1)

                    assert len(et) == len(etw)
                    self._save_dataset(et, etw, mode=mode)
                    self._update_cache()
            print(f"Processed {mode} batch... (Total={len(learning_set)}).")

            # Save to class

            # Return generators to conserve memory

        return self._load_dataset_from_cache(mode='train'), self._load_dataset_from_cache(
            mode='valid'), self._load_dataset_from_cache(mode='test')


class WordGenerator:
    NUM_PROCESSES = 3

    def __init__(self, word_max_size: int = DECODER_OUTPUT_LENGTH, mode="all"):
        self.mode = mode
        self.word_max_size = word_max_size

        if mode == "digits":
            self.mode_words = DIGITS
        elif mode == "alphabets":
            self.mode_words = LOWERCASE_LETTERS + UPPERCASE_LETTERS + BLANK_SPACE
        else:  # all
            self.mode_words = DIGITS + MISC_CHARS + \
                              LOWERCASE_LETTERS + UPPERCASE_LETTERS + BLANK_SPACE

    @staticmethod
    def generate_word(size, source):
        return "".join(random.choice(source) for _ in range(size))

    @staticmethod
    def build_vocab(words, min_freq, source):
        counter = Counter()

        for word in words:
            for char in word:
                size = counter.get(char)

                if size is not None and size >= min_freq:
                    with suppress(ValueError):
                        source.remove(char)

                else:
                    counter.update(char)

        return counter, source

    def _preprocess_words(self, words):
        """Clip words that are more than required word max size"""

        # Use only unique words
        words = list(set(words))

        random.shuffle(words)

        # Skip words bigger than word max size.
        words = list(
            filter(lambda x: (len(x) + 2 <= self.word_max_size), words))

        return words

    @staticmethod
    def clean_sentence(sentence):
        """Clean a sentence based on our data"""

        sentence = sentence.strip()
        sentence = unicodedata.normalize("NFKD", sentence)

        rs = [
            (NEW_LINE[0], NO_SPACE[0]), ("–", BLANK_SPACE[0]), ("’", "'"), ("”", "'"), ("“", "'"),
            ("[", NO_SPACE[0]), ("]", NO_SPACE[0]), ("%", NO_SPACE[0]), ("#", NO_SPACE[0]),
            ("« ", "«"), (" »", "»"), ("é", "e"), ("ê", "e"), ("è", "e"), ("ë", "e"),
            ("ß", "ss"), ("„", "'"), ("“", "'"), ("ö", "o"), ("ü", "u"), ("ä", "a"),
            ("$", NO_SPACE[0]), ("ü", 'u'), ("ä", 'a'), ('ò', 'o'),
            ("à", "a"), ("â", "a"), ("ô", "o"), ("É", "E"), ("ç", "c"), ("     ", BLANK_SPACE[0]),
            ("    ", BLANK_SPACE[0]), ("   ", BLANK_SPACE[0]), ("  ", BLANK_SPACE[0]), ]

        for (old, new) in rs:
            sentence = sentence.replace(old, new)

        return sentence

    def _create_rand_words(self, min_freq):

        dataset = []
        reached = False
        source = copy.copy(self.mode_words)

        while 1:
            counter, source = WordGenerator.build_vocab(
                dataset, min_freq, source)
            glyph_count = list(counter.values())

            for i, j in enumerate(glyph_count, start=1):
                if j < min_freq:
                    break
                if i == len(glyph_count):
                    reached = True

            if reached and len(glyph_count) == len(self.mode_words):
                break
            else:
                for size in range(1, (self.word_max_size - 1)):
                    dataset.append(WordGenerator.generate_word(size, source))

        return dataset

    def generate_random_words(self, min_freq):
        """Generate random words"""

        # Ensures train, validation, and test have letters with equal distribution
        with multiprocessing.Pool(WordGenerator.NUM_PROCESSES) as pool:
            params = [(int(min_freq * 0.6),), (int(min_freq * 0.2),),
                      (int(min_freq * 0.2),), ]
            results = [pool.apply_async(
                self._create_rand_words, p) for p in params]
            pool.close()
            pool.join()

        return results[0].get() + results[1].get() + results[2].get()

    def generate_from_experiment_mode(self, min_freq):
        """Generate words from experiment characters (digits, alphabets, etc.)"""

        words = []

        for _ in range(min_freq):
            for char in self.mode_words:
                words.append(char)

        return words

    def generate_parallel_pair_from_parallel_file(self,
                                                  fpath,
                                                  delimiter="\t",
                                                  skip_tokens=["�", "#amp", "ampamp"]):
        """
        Generate parallel sentences from a parallel Dataset file.

        It is implied that the first part of the parallel dataset
        is in English. That is, it is an [e.g.] en-fr dataset
        """

        target_words = []  # Storage for target sentences
        source_words = []  # Storage for source sentences

        with open(fpath) as f:

            for sentence in f:
                valid_sentence = True
                sentence = WordGenerator.clean_sentence(sentence)

                for token in skip_tokens:
                    if token in sentence:
                        valid_sentence = False
                        break

                # Skip
                if not valid_sentence:
                    continue

                # Split into source and target sentences
                parallel_sentence = sentence.split(delimiter)

                # Should ideally be src/trg. Skip if not
                if len(parallel_sentence) != 2:
                    continue

                src, trg = parallel_sentence  # Split into English/French

                # Skip translation-less sentences (either way)
                if src in [NO_SPACE, NEW_LINE] or trg in [NO_SPACE, NEW_LINE]:
                    continue

                en_sentence = ""

                # The character must be in the glyphs
                # the experiment was carried out on e.g.
                # $ is not in the glyphs and must be excluded
                for word in src:
                    new_word = ""

                    for char in word:
                        if char not in self.mode_words:
                            continue

                        new_word += char
                    en_sentence += new_word

                # Filter valid words and re-write english sentence accordingly
                if en_sentence != "" and (len(trg) + 2) <= self.word_max_size and (
                        len(en_sentence) + 2) <= self.word_max_size:
                    src = en_sentence
                else:
                    continue

                source_words.append(src)
                target_words.append(trg)

        assert len(source_words) == len(target_words)
        print("Total =", len(source_words))

        return source_words, target_words

    def generate_from_file(self,
                           fpath,
                           words_only=False,
                           skip_tokens=["�", "#amp", "ampamp"]):
        """
        Load words or sentences from a Dataset file.

        if `words_only`, then return words instead of sentences
        `skip_tokens` are tokens whose sentences will be skipped
        if the tokens are found in the sentence
        """

        words = []  # Storage for valid words/sentences

        # Re-read file and cache
        with open(fpath, "r", encoding="utf-8") as f:
            # next(f)  # Skip potential header

            for sentence in f:
                valid_sentence = True
                sentence = WordGenerator.clean_sentence(sentence)

                for token in skip_tokens:
                    if token in sentence:
                        valid_sentence = False
                        break

                # Skip
                if not valid_sentence:
                    continue

                words_ = sentence.split(BLANK_SPACE[0])  # Split into words

                if words_only:
                    # The character must be in the glyphs
                    # the experiment was carried out on e.g.
                    # $ is not in the glyphs and must be excluded
                    for word in words_:
                        new_word = ""

                        for char in word:
                            if char not in self.mode_words:
                                continue
                            new_word += char

                        # Append only valid words
                        words.append(new_word) if new_word != "" else None

                else:  # Output sentences, and not just words
                    sentence_ = ""

                    # The character must be in the glyphs
                    # the experiment was carried out on e.g.
                    # $ is not in the glyphs and must be excluded
                    for word in sentence:
                        new_word = ""

                        for char in word:
                            if char not in self.mode_words:
                                continue

                            new_word += char
                        sentence_ += new_word

                    # Append only valid words
                    if sentence_ != "" and (len(sentence_) + 2) <= self.word_max_size:
                        words.append(sentence_)

        # Use unique words only if words (not sentences) are being outputted
        words = self._preprocess_words(words) if words_only else words

        return words


class WordDatasetGenerator(SequenceGenerator):
    """
    Generates train, validation, and test set data

    param: vector_size (int): size of each glyph vector
    param: words (list): List of words to generate dataset for (English words)
    param: train_split (float): Percentage of training set (defaults to 0.6, i.e. 60%)
    param: valid_split (float): Percentage of validation set (defaults to 0.2, i.e. 20%)
    param: scale_by_ar (bool): Flag to determine if y coordinate should be scaled by aspect ratio (Defaults to False)
    param: offset_delay (bool): Flag to determine if time delay between strokes should be considered (Defaults to False)
    param: granularity (str): Granularity of supervised dataset. Should be one of ['touch', 'stroke', 'glyph'] (Defaults to 'glyph')
    param: padding_value (int): Value to use for padding the generated dataset to meet the required vector size (Defaults to 0)
    param: end_of_glyph_value (int): Value to use to signify the end of a glyph in the generated dataset (Defaults to -1)
    param: end_of_stroke_value (int): Value to use to signify the end of a stroke in the generated dataset (Defaults to -2)
    param: include_time_feature (bool): Flag to indicate whether to include the time feature in the generated dataset (Defaults to True)
    """
    NUM_PROCESSES = 3  # One process per batch

    def __init__(self,
                 vocab,
                 vector_size: int = VECTOR_SIZE,
                 total_words=None,
                 words: list = [],
                 db_path: str = DB_PATH,
                 word_max_size: int = DECODER_OUTPUT_LENGTH,
                 padding_value: int = -5,
                 augmentation: dict = DEF_AUGMENTATION,
                 train_split: float = 0.6,
                 valid_split: float = 0.2,
                 use_subject: bool = True,
                 extended_dataset: bool = True,
                 scale_by_ar: bool = True,
                 sample_data: bool = False,
                 offset_delay: bool = False,
                 granularity: str = "stroke",
                 end_of_glyph_value: int = -4,
                 end_of_stroke_value: int = -3,
                 expr_mode: str = "alphabets",
                 include_time_feature: bool = False,
                 include_pressure_feature: bool = False,
                 bos_idx=None,
                 eos_idx=None,
                 pad_idx=None,
                 fname=None):

        # Train and Validation split are within 0 and 1
        super().__init__(vocab=vocab,
                         vector_size=vector_size,
                         db_path=db_path,
                         input_size=word_max_size,
                         total_samples=total_words,
                         padding_value=padding_value,
                         augmentation=augmentation,
                         train_split=train_split,
                         valid_split=valid_split,
                         use_subject=use_subject,
                         scale_by_ar=scale_by_ar,
                         sample_data=sample_data,
                         offset_delay=offset_delay,
                         granularity=granularity,
                         end_of_glyph_value=end_of_glyph_value,
                         end_of_stroke_value=end_of_stroke_value,
                         expr_mode=expr_mode,
                         include_time_feature=include_time_feature,
                         include_pressure_feature=include_pressure_feature,
                         bos_idx=bos_idx,
                         eos_idx=eos_idx,
                         pad_idx=pad_idx,
                         )
        self.extended_dataset = extended_dataset
        self.word_max_size = word_max_size
        self.mp_dict = multiprocessing.Manager().dict()
        self.dtype = np.dtype("S100")
        if total_words:
            self.total_words = min(total_words, len(words))
        else:
            self.total_words = len(words)
        self.words = words[:self.total_words]
        self.fname = fname if fname is not None else f"words_{self.granularity}_{self.word_max_size}_{self.total_words}"
        self.fpath = os.path.join("cache", "words_cache", f"{self.fname}.h5")

    def _generate_glyphs_from_sequence(self, word, subject_choices, sc):
        """
        Generates glyphs corresponding to a given `word`
        If `self.use_subject` is True, random subjects are selected
        from the dataset and used as the source of the glyphs
        """

        # Hold possible subject and glyphs
        subject_to_use = None
        subject_glyphs = []
        glyphs = []

        # For each character in the word

        found = False
        if self.use_subject:
            # Round-robin all subjects in the training set.
            # Ensures each subject is selected multiple times
            i = 0

            while not subject_to_use:
                i += 1
                if i > len(subject_choices):
                    break
                if sc >= len(subject_choices):
                    sc = 0  # Reset

                subj = subject_choices[sc]
                sc += 1  # Increment index

                # Get glyphs for subject
                subj_glyphs = [
                    j.ground_truth for i in subj.glyph_sequences for j in i.glyphs]

                if all(i in subj_glyphs for i in word if i != " "):
                    found = True
                    # Get glyphs for subject
                    subject_glyphs = [
                        j for i in subj.glyph_sequences for j in i.glyphs]
                    subject_to_use = subj  # Cache for remaining glyphs in `word`

            # Get glyphs for subject that corresponds to `char`
            if found:

                for char in word:
                    if char == " ":
                        continue

                    glyph_choices = [
                        i for i in subject_glyphs if i.ground_truth == char]

                    # Choose a glyph randomly
                    glyph_choice = random.choice(glyph_choices)

                    # Add to the glyphs list
                    glyphs.append(glyph_choice)

        if not self.use_subject or not found:

            for char in word:
                if char == " ":
                    continue

                glyph_choices = [gl for subj in subject_choices for gs in subj.glyph_sequences for gl in gs.glyphs if
                                 gl.ground_truth == char]

                # Choose a glyph randomly
                glyph_choice = random.choice(glyph_choices)

                # Add to the glyphs list
                glyphs.append(glyph_choice)

        return glyphs, sc

    def _tensorize_string(self, trg_string):
        """Convert a string into a tensor"""

        if type(self.vocab).__name__ == "ByteLevelBPETokenizer":
            tsor = torch.tensor(self.vocab.encode(trg_string).ids)

        else:  # `torchtext.vocab.Vocab`
            tsor = torch.stack([torch.tensor(self.vocab.stoi[j])
                                for j in trg_string])

        return tsor

    def _save_dataset(self, expanded_glyphs: list, words_batch: list, mode: str = 'train'):
        """Save related dataset"""

        if mode == 'test':
            x_dataset, y_dataset = self._x_test, self._y_test
        elif mode == 'valid':
            x_dataset, y_dataset = self._x_valid, self._y_valid
        elif mode == 'train':
            x_dataset, y_dataset = self._x_train, self._y_train
        else:
            raise

        # Save the whole word
        for word in words_batch:
            y_dataset.append(word.encode('utf-8'))

        # Add to list of generated datasets
        for glyph in expanded_glyphs:
            x_dataset.append(glyph)

    def _generate_word_pair(self, words, mode, mp_dict):
        """Generate src and trg pairs"""

        sc = 0
        wt, wtw = [], []

        session = create_db_session(self.db_path)
        subjects = self.get_all_subjects(session, mode)

        for word in tqdm.tqdm(words, desc=f"{mode.capitalize()} set progress"):
            # 'boy' -> 'Glyph (b)', 'Glyph (o)', 'Glyph (y)'
            glyphs, sc = self._generate_glyphs_from_sequence(word, subjects, sc)

            # Expand the glyphs into strokes or glyphs
            word_touches = self._granularize(glyphs, session)

            # Save expanded touches
            wt.append(word_touches)
            wtw.append(word)

            if self._should_augment:  # If augmenting...
                for _ in range(self.augmentation['amount']):
                    aug_glyphs = self._augment_glyphs(glyphs, session)

                    # Expand the glyphs depending on granularity
                    aug_word_touches = self._granularize(aug_glyphs, session)

                    # Save expanded touches
                    wt.append(aug_word_touches)
                    wtw.append(word)

        assert len(wt) == len(wtw)
        print(f"Processed {mode} batch... (Total={len(wtw)}).")

        # Save to multiprocessing dict for main process return and use
        self.mp_dict[mode] = (wt, wtw)

    def generate_src_from_trg_string(self, trg_string, subject, session, seed=None):
        """Generate source and target tensors from an input string"""

        g = []
        subj_glyphs = [j for i in subject.glyph_sequences for j in i.glyphs]

        if seed:
            random.seed(seed)
        for c in trg_string:
            if c == " ":
                continue  # Skip blanks
            try:
                ch = random.choice([i for i in subj_glyphs if i.ground_truth == c])
            except:
                print(c)
                print(sorted(list(set([i.ground_truth for i in subj_glyphs]))))
                break

            g.append(ch)

        # Expand the glyphs into strokes or glyphs
        src = torch.tensor(self._granularize(g, session))

        # Convert each char in the word to a tensor
        y = self._tensorize_string(trg_string)

        # Add bos/eos and pad up tokens up to word_max_size
        diff = (self.word_max_size - (y.shape[0])) - 2
        y_ = torch.cat([torch.tensor([self.bos_idx]),
                        y, torch.tensor([self.eos_idx]),
                        torch.tensor(list(repeat(torch.tensor([self.pad_idx]), diff)))], dim=0)

        # Because equally-padded tensors have float
        trg = torch.tensor(y_, dtype=torch.int64)

        # Return src and trg
        return src.unsqueeze(0), trg.unsqueeze(0)

    def generate(self, use_subprocess=False, trg_words=None):
        """
        Generate the train, validaton, and test datasets.

        If `trg_words` is passed, then they'll be used as the target words,
        while `self.words` will be used to generate the touch information.
        When using `trg_words`, it's implied that a machine translation data
        is to be generated. That is, strokes are generated for the english data
        (`self.words`), but the target language will be `trg_words`.

        `use_subprocess` should be used for relativelys small datasets
        """

        words = trg_words if trg_words else self.words
        use_subprocess = use_subprocess if not trg_words else False

        # Split words into train, validation, test
        train_words, valid_words, test_words = self._split(words)

        if use_subprocess:  # Moderately big datasets (optimized for speed)
            # Use separate process to generate the test and validation dataset
            test_p = multiprocessing.Process(target=self._generate_word_pair, args=(
                test_words, "test", self.mp_dict), daemon=True)
            valid_p = multiprocessing.Process(target=self._generate_word_pair, args=(
                valid_words, "valid", self.mp_dict), daemon=True)

            # Start all with short delay in between
            test_p.start()
            time.sleep(3)
            valid_p.start()

            # Wait for all to finish
            test_p.join()
            valid_p.join()

            # Once all are done, retrieve and save results
            for mode in ["test", "valid"]:
                wt, wtw = self.mp_dict[mode]
                self._save_dataset(wt, wtw, mode=mode)

            # Train generation has to be done in the main process (for large datasets) because
            # of the multiprocessing memory limit for child processes (~2GB). Test and Validation
            # sets are significantly smaller (60% less) than the training data and can be done in
            # a child process -------------------------------------------------------------------
            sc = 0
            mode = "train"
            wt, wtw = [], []

            session = create_db_session(self.db_path)
            subjects = self.get_all_subjects(session, mode, augment=self.extended_dataset)

            for word in tqdm.tqdm(train_words, desc=f"{mode.capitalize()} set progress"):
                # 'boy' -> 'Glyph (b)', 'Glyph (o)', 'Glyph (y)'
                glyphs, sc = self._generate_glyphs_from_sequence(
                    word, subjects, sc)

                # Expand the glyphs into strokes or glyphs
                word_touches = self._granularize(glyphs, session)

                # Save expanded touches
                wt.append(word_touches)
                wtw.append(word)

                if self._should_augment:  # If augmenting...
                    for _ in range(self.augmentation['amount']):
                        aug_glyphs = self._augment_glyphs(glyphs, session)

                        # Expand the glyphs depending on granularity
                        aug_word_touches = self._granularize(
                            aug_glyphs, session)

                        # Save expanded touches
                        wt.append(aug_word_touches)
                        wtw.append(word)

            assert len(wt) == len(wtw)
            print(f"Processed {mode} batch... (Total={len(wtw)}).")

            self._save_dataset(wt, wtw, mode=mode)

        else:  # Very big datasets. This will take time
            print("Generating datasets...\n")
            self._init_cache(None)
            session = create_db_session(self.db_path)

            for (learning_set, mode) in [(test_words, 'test'), (train_words, 'train'), (valid_words, 'valid')]:

                subjects = self.get_all_subjects(session, mode, augment=self.extended_dataset)

                sc = 0  # Subject counter index. Used to recycle subjects
                chunk_size = SequenceGenerator.CHUNK_SIZE if len(
                    learning_set) > SequenceGenerator.CHUNK_SIZE else len(learning_set)

                end_index = ((len(learning_set) // chunk_size) + 1) if len(
                    learning_set) > SequenceGenerator.CHUNK_SIZE else 1

                with tqdm.tqdm(total=len(learning_set), desc=f"{mode.capitalize()} set progress") as pbar:
                    for slice_index in tqdm.tqdm(range(end_index), disable=True):
                        next_slice = learning_set[slice_index * chunk_size: (slice_index + 1) * chunk_size]
                        wt, wtw = [], []  # Storage for generated touches and corresponding ground truth
                        for index, word in enumerate(tqdm.tqdm(next_slice, disable=True)):

                            # 'boy' -> 'Glyph (b)', 'Glyph (o)', 'Glyph (y)'
                            glyphs, sc = self._generate_glyphs_from_sequence(word, subjects, sc)
                            # Expand the glyphs into strokes or glyphs
                            word_touches = self._granularize(glyphs, session)
                            # Save expanded touches
                            wt.append(word_touches)
                            # Save appropriate word
                            wtw.append(word)

                            if self._should_augment and mode == 'train':  # If augmenting...
                                for _ in range(self.augmentation['amount']):
                                    aug_glyphs = self._augment_glyphs(glyphs, session)

                                    # Expand the glyphs depending on granularity
                                    aug_word_touches = self._granularize(
                                        aug_glyphs, session)

                                    # Save expanded touches
                                    wt.append(aug_word_touches)
                                    wtw.append(word)
                            pbar.update(1)
                        assert len(wt) == len(wtw)
                        self._save_dataset(wt, wtw, mode=mode)
                        self._update_cache()
                print(f"Processed {mode} batch... (Total={len(learning_set)}).")

        return self._load_dataset_from_cache(mode='train'), self._load_dataset_from_cache(
            mode='valid'), self._load_dataset_from_cache(mode='test')

    def add_training_words(self, words, quantity=None):
        if quantity:
            words = words[:quantity]

        session = create_db_session(self.db_path)
        subjects_ = []
        for subject in session.query(Subject):
            for gs in subject.glyph_sequences:
                if gs.experiment == 5 and subject not in subjects_:
                    subjects_.append(subject)
        wt, wtw = [], []
        for word in tqdm.tqdm(words):
            sc = random.randint(0, len(subjects_) - 1)
            glyphs, _ = self._generate_glyphs_from_sequence(word, subjects_, sc)
            word_touches = self._granularize(glyphs, session)
            wt.append(word_touches)
            # Save appropriate word
            wtw.append(word)

            if self._should_augment:  # If augmenting...
                for _ in range(self.augmentation['amount']):
                    aug_glyphs = self._augment_glyphs(glyphs, session)

                    # Expand the glyphs depending on granularity
                    aug_word_touches = self._granularize(
                        aug_glyphs, session)

                    # Save expanded touches
                    wt.append(aug_word_touches)
                    wtw.append(word)

        self._save_dataset(wt, wtw, mode='train')
        try:
            self._update_cache()
        except ValueError:
            print("Max dimension reached")
            return


class PatchedWordGenerator(WordDatasetGenerator):
    def __init__(self, **kvargs):
        self.granularity = "touch"
        super(PatchedWordGenerator, self).__init__(granularity=self.granularity, **kvargs)

    def _init_cache(self, fname=None):
        """Initialize cache"""

        print("\nInitializing cache...")

        if not fname:
            fname = self.fpath

        if os.path.exists(fname):
            os.remove(fname)

        with h5py.File(fname, 'w') as hf:
            ms = (self.input_size * self.avg_glyph_strokes, self.vector_size)

            # Create the actual datasets
            hf.create_dataset('X_test', compression="gzip",
                              chunks=True, maxshape=(None, *ms), shape=(0, *ms))
            hf.create_dataset('X_train', compression="gzip",
                              chunks=True, maxshape=(None, *ms), shape=(0, *ms))
            hf.create_dataset('X_valid', compression="gzip",
                              chunks=True, maxshape=(None, *ms), shape=(0, *ms))
            hf.create_dataset('Y_test', compression="gzip",
                              chunks=True, maxshape=(None,), shape=(0,), dtype=self.dtype)
            hf.create_dataset('Y_train', compression="gzip",
                              chunks=True, maxshape=(None,), shape=(0,), dtype=self.dtype)
            hf.create_dataset('Y_valid', compression="gzip",
                              chunks=True, maxshape=(None,), shape=(0,), dtype=self.dtype)

    def _granularize(self, glyphs, session):
        """Expand `glyph`s into required granularity"""

        word_touches = []
        last_touches = []
        bos_vector = list(chain(repeat(self.bos_idx, self.vector_size)))
        eos_vector = list(chain(repeat(self.eos_idx, self.vector_size)))

        # Add beginning of sequence vector
        word_touches.append(bos_vector)

        stroke_position = 0  # Track position of stroke in glyphs

        # For each glyph...
        for char in glyphs:

            stroke_start_time = 0  # Default start time

            # Use serialized version of glyph. It's faster
            char = char.serialize(session=session) if not isinstance(
                char, dict) else char

            # For each stroke in the glyph...
            for stroke_index, stroke in enumerate(char['strokes']):

                # For each touch in the stroke...
                min_x = float('inf')
                min_y = float('inf')
                max_x = 0
                max_y = 0
                stroke_touches = []
                for index, touch in enumerate(stroke['touches'], start=1):

                    x = touch['x']
                    # Scale y (or not) by aspect ratio
                    y = touch['y']
                    if self.scale_by_ar:
                        ar = char['ar']
                        if ar > 1:
                            x *= ar
                        else:
                            y /= ar

                    min_x = min(x, min_x)
                    max_x = max(x, max_x)
                    min_y = min(y, min_y)
                    max_y = max(y, max_y)

                    # Add to touches list
                    stroke_touches.append(x)  # x
                    stroke_touches.append(y)  # y
                    if self.include_time_feature:
                        t = touch['timestamp'] * 1e-15 + stroke_start_time  # (s)
                        stroke_touches.append(t)
                    if self.include_pressure_feature:
                        stroke_touches.append(stroke_position)

                if max_x > 1:
                    new_touches = []
                    for i, t in enumerate(stroke_touches):
                        if i % self.n_features == 0:
                            new_touches.append(t - min_x + (min_x + (1 - max_x)) / 2)
                        else:
                            new_touches.append(t)
                    stroke_touches = new_touches
                if max_y > 1:
                    new_touches = []
                    for i, t in enumerate(stroke_touches):
                        if i % self.n_features == 1:
                            new_touches.append(t - min_y + (min_y + (1 - max_y)) / 2)
                        else:
                            new_touches.append(t)
                    stroke_touches = new_touches

                for _ in range(self.n_features):
                    stroke_touches.append(self.end_of_stroke_value)

                # Sample touches, if required. Time must also be included
                if self.sample_data and self.include_time_feature:
                    stroke_touches = self._sample_touches(stroke_touches)

                if len(stroke_touches) > self.vector_size - len(last_touches):
                    last_touches = self._pad(last_touches)
                    word_touches.append(last_touches)
                    last_touches = stroke_touches.copy()

                else:
                    last_touches += stroke_touches

        if last_touches:
            last_touches = self._pad(last_touches)
            word_touches.append(last_touches)

        # Padding tensor/list
        padding = list(chain(repeat(self.padding_value, self.vector_size)))

        # Add end of sequence vector
        word_touches.append(eos_vector)

        # Pad dataset to the word max size
        word_touches = self._pad(word_touches, size=self.avg_glyph_strokes * self.input_size, padding=padding)

        return word_touches  # Return the sequence dimension
