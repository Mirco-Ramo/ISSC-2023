import os
import sys
from pathlib import Path

import torch as torch
from tokenizers import Tokenizer
from torch import nn
from torch.utils.data import DataLoader

BASE_DIR = Path(os.getcwd()).resolve()
print(BASE_DIR)

b_paths = [os.path.abspath(BASE_DIR), os.path.abspath(os.path.join(
    BASE_DIR, 'models')), os.path.abspath(os.path.join(BASE_DIR, 'models', 'scripts'))]
for b_path in b_paths:
    if b_path not in sys.path:
        sys.path.append(b_path)

from models.scripts.transformer.BERT_decoder import BertTransformer
from models.scripts.transformer.utils import preprocess_dataset, seed_all, pad_bert_collate_fn
from models.scripts.generate_dataset import WordDatasetGenerator

VERSION = "bert17"
SEED = 2021
BATCH_SIZE = 128
EXPR_MODE = 'all'
seed_all(SEED)

VOCAB = Tokenizer.from_pretrained("bert-base-cased")
BOS_IDX = VOCAB.token_to_id('[CLS]')
EOS_IDX = VOCAB.token_to_id('[SEP]')
PAD_IDX = VOCAB.token_to_id("[PAD]")
UNK_IDX = VOCAB.token_to_id("[UNK]")

d_gen = WordDatasetGenerator(vocab=VOCAB,
                             expr_mode=EXPR_MODE,
                             eos_idx=3,
                             bos_idx=2,
                             pad_idx=PAD_IDX,
                             fname="words_stroke_en_full")

train, valid, test = d_gen.generate_from_cache()

print(VOCAB)
print("[PAD]:", VOCAB.token_to_id("[PAD]"))
N_TOKENS = VOCAB.get_vocab_size()  # len(VOCAB)
print(f"Number of Tokens: {N_TOKENS}\n")

train_set = DataLoader(preprocess_dataset(train, VOCAB, os.path.join(d_gen.fname, "train.pt"),
                                          total_len=d_gen.get_learning_set_length("train")), batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=lambda b: pad_bert_collate_fn(b, trg_pad_idx=PAD_IDX))
valid_set = DataLoader(preprocess_dataset(valid, VOCAB, os.path.join(d_gen.fname, "valid.pt"),
                                          total_len=d_gen.get_learning_set_length("valid")), batch_size=BATCH_SIZE,
                       shuffle=False, collate_fn=lambda b: pad_bert_collate_fn(b, trg_pad_idx=PAD_IDX))

model = BertTransformer(VERSION, VOCAB, n_tokens=N_TOKENS, encoder_name='en-en11',
                        decoder_name='new', conv_layer_name="en-en11", bos="[CLS]", eos="[SEP]", pad="[PAD]")
model.save_hyperparameters_to_json()
model.count_parameters()
model.requires_grad_(False)
model.decoder.enc_to_dec_proj.requires_grad_(True)
for layer in model.decoder.bert.encoder.layer:
    layer.crossattention.requires_grad_(True)
model.to(model.device)

LEARNING_RATE = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
criterion = nn.CrossEntropyLoss(ignore_index=VOCAB.token_to_id('[PAD]'))

model.train_loop(resume=False,
                 train_set=train_set,
                 valid_set=valid_set,
                 optimizer=optimizer,
                 scheduler=scheduler,
                 criterion=criterion,
                 n_epochs=4000)
