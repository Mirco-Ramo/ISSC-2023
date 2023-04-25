import os
import sys
from tokenizers import Tokenizer
import torch.nn as nn

from pathlib import Path

import torch as torch
from torch.utils.data import DataLoader

BASE_DIR = Path(os.getcwd()).resolve()
print(BASE_DIR)

b_paths = [os.path.abspath(BASE_DIR), os.path.abspath(os.path.join(
    BASE_DIR, 'models')), os.path.abspath(os.path.join(BASE_DIR, 'models', 'scripts'))]
for b_path in b_paths:
    if b_path not in sys.path:
        sys.path.append(b_path)

from models.scripts.transformer.MultiLang import ConditionableTransformer
from models.scripts.transformer.utils import strokes_to_svg, seed_all, preprocess_with_lang
from models.scripts.generate_dataset import WordDatasetGenerator, WordGenerator
from models.scripts.defaults import Languages

VERSION = "no_unk15"
SEED = 2021
BATCH_SIZE = 256
EXPR_MODE = 'all'

seed_all(SEED)  # Reproducibility

TOKENIZER_FILE = os.path.join("word_sources", "tokenizer-big_multi-normalized.json")
VOCAB = Tokenizer.from_file(TOKENIZER_FILE)
VOCAB.add_special_tokens([f'<bos_{lang.name.lower()}>' for lang in Languages])

print(sorted(VOCAB.get_vocab()))
print(VOCAB.token_to_id("<pad>"))
print(VOCAB.token_to_id("<bos_en>"))
print(VOCAB.token_to_id("<bos_de>"))
print(VOCAB.token_to_id("<bos_fr>"))
print(VOCAB.token_to_id("<bos_it>"))
print(VOCAB.token_to_id("<bos_unk>"))
N_TOKENS = VOCAB.get_vocab_size()  # len(VOCAB)
print(f"Number of Tokens: {N_TOKENS}\n")

train_sets = []
valid_sets = []
test_sets = []

for lang in Languages:
    if lang == Languages.UNK:
        continue
    d_gen = WordDatasetGenerator(vocab=VOCAB,
                                 expr_mode=EXPR_MODE,
                                 fname=f"words_stroke_{lang.name.lower()}_full")
    train, valid, test = d_gen.generate_from_cache()
    train_sets.append(train)
    valid_sets.append(valid)
    test_sets.append(test)

assert len(train_sets) == len(valid_sets) == len(test_sets) == len(Languages) - 1

preprocessed_trains = []
preprocessed_valids = []
for i, lang in enumerate([l for l in Languages if l != Languages.UNK]):
    lower_name = lang.name.lower()
    d_gen = WordDatasetGenerator(vocab=VOCAB,
                                 expr_mode=EXPR_MODE,
                                 fname=f"words_stroke_{lower_name}_full")
    preprocessed_trains += preprocess_with_lang(train_sets[i], VOCAB, os.path.join(d_gen.fname, "train.pt"),
                                                total_len=d_gen.get_learning_set_length("train"),
                                                bos=VOCAB.token_to_id(f'<bos_{lower_name}>'))
    preprocessed_valids += preprocess_with_lang(valid_sets[i], VOCAB, os.path.join(d_gen.fname, "valid.pt"),
                                                total_len=d_gen.get_learning_set_length("valid"),
                                                bos=VOCAB.token_to_id(f'<bos_{lower_name}>'))

train_set = DataLoader(preprocessed_trains, batch_size=BATCH_SIZE, shuffle=True)
valid_set = DataLoader(preprocessed_valids, batch_size=BATCH_SIZE, shuffle=False)

model = ConditionableTransformer(VERSION, VOCAB, conv_layer_name='en-en11', encoder_name='en-en11', n_tokens=N_TOKENS, hid_dim=256, dec_heads=8, dec_layers=6, dec_pf_dim=256*3)
model.save_hyperparameters_to_json()
model.count_parameters()
model.requires_grad_(False)
model.decoder.requires_grad_(True)
model.enc_to_dec_proj.requires_grad_(True)
model.to(model.device)

LEARNING_RATE = 7e-4
criterion = nn.CrossEntropyLoss(ignore_index=VOCAB.token_to_id('<pad>'))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

model.train_loop(resume=False,
                 train_set=train_set,
                 valid_set=valid_set,
                 optimizer=optimizer,
                 scheduler=scheduler,
                 criterion=criterion,
                 n_epochs=4000)
