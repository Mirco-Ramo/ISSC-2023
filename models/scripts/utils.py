# Python 3.6+

import os
import csv
import torch
import pickle

from pprint import pprint
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.scripts.defaults import DEVICE
from models.scripts.data_model import Glyph
from Levenshtein import distance as lev

BLANK_SPACE = " "
DB_NAME = "digit_schema.db"
plt.rcParams["figure.figsize"] = (9, 7)
DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
OPERATORS = ['+', '-', '*', '/']
DECIMAL_NOTATION = ['.']
BRACKETS = ['(', ')']
EQUAL_SIGN = ['=']
SEPARATOR = [',']


# Helper functions
# ----------------

def pause_print(x):
    print(x)
    input()


def pause_pprint(x):
    pprint(x)
    input()


def to_secs(x): return abs(float(x)) * 1e-15


def to_millisecs(x): return abs(float(x)) * 1e-12


def delay_to_ms(time_delay):
    """Convert time delay to milliseconds"""
    return abs(float(time_delay)) * 1e-12


def chunker(seq, size):
    """Iterate through `seq` in steps of `size`"""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def interpolate(xy_1, xy_2, start_time, end_time, t):
    try:
        t = (t - start_time) / (end_time - start_time)
    except ZeroDivisionError:
        return tuple(xy_2)

    return tuple((1 - t) * x + t * y for x, y in zip(xy_1, xy_2))


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def plot_training(log_path):
    epoch = []
    valid_loss = []
    train_loss = []

    with open(log_path) as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        next(rows)  # Skip header

        for row in rows:
            epoch.append(float(row[0]))
            train_loss.append(float(row[1]))
            valid_loss.append(float(row[2]))

    plt.title('Train and Validation Loss per Epoch')
    plt.plot(epoch, valid_loss, "-b", label="valid loss")
    plt.plot(epoch, train_loss, "-r", label="train loss")

    plt.legend()
    plt.show()


def log_epoch(log_path, epoch, train_loss, valid_loss, test_loss=None):
    """Write log to a file"""

    epoch += 1

    if epoch == 1:
        with open(log_path, "w+") as lf:
            if not test_loss:
                lf.write("epoch,train_loss,validation_loss\n")
                lf.write(f"{epoch},{train_loss},{valid_loss}\n")
            else:
                lf.write("epoch,train_loss,validation_loss, test_loss\n")
                lf.write(f"{epoch},{train_loss},{valid_loss},{test_loss}\n")
    else:
        with open(log_path, "a") as lf:
            if not test_loss:
                lf.write(f"{epoch},{train_loss},{valid_loss}\n")
            else:
                lf.write(f"{epoch},{train_loss},{valid_loss},{test_loss}\n")


def load_checkpoint(checkpoint_path, model, strict=True, device=DEVICE, optimizer=None, scheduler=None):
    """Load model and optimizer from a checkpoint"""

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler'])

    model.load_state_dict(checkpoint['state_dict'], strict=strict)
    return model, optimizer, scheduler


def create_db_session(db_path=DB_NAME):
    """Create an SQL-Alchemy session"""

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"{db_path} does not exist")

    # Create session
    session = sessionmaker()
    session.configure(bind=create_engine(f"sqlite:///{db_path}"))

    return session()


def picklify(fpath: str, data: list):
    """Pickle `data` to `fpath`"""
    with open(fpath, 'wb') as f:
        pickle.dump(data, f)


def unpicklify(fpath: str):
    """Unpickle `fpath`"""
    with open(fpath, 'rb') as f:
        return pickle.load(f)


def get_distinct_glyphs(session):
    """Get all distinct glyphs"""

    glyphs = [glyph.ground_truth for glyph in session.query(Glyph.ground_truth)
    .distinct().order_by(Glyph.ground_truth).asc().all()]
    return glyphs


def Levenshtein_Normalized_distance(a, b):
    gld = lev(str(a), str(b))
    nld = (2 * gld) / (len(a) + len(b) + gld)
    return nld


def count_postfix_violations_no_separator(expression):
    counter = 0
    violations = 0
    for char in expression:
        if char in DIGITS:
            counter += 1
        elif char in DECIMAL_NOTATION:
            counter -= 1
            if counter < 0:
                violations += 1
        elif char in OPERATORS:
            counter -= 2
            if counter < 0:
                violations += 1
            counter += 1
    violations += abs(counter - 1)
    return violations
