import numpy as np
import os
import sys
from data_processing.training_data_generator import load_dictionaries
from util.helpers import make_dir_if_not_exists

import torch
import torch.nn as nn

class LoadData:
    def _deserialize(self, data_folder):
        train_ex = np.load(os.path.join(data_folder, 'examples-train.npy'))
        valid_ex = np.load(os.path.join(
            data_folder, 'examples-validation.npy'))
        test_ex = np.load(os.path.join(data_folder, 'examples-test.npy'))
        assert train_ex is not None and valid_ex is not None and test_ex is not None
        return train_ex, valid_ex, test_ex

    def __init__(self, data_folder, shuffle=True, load_only_dicts=False):
        self.rng = np.random.RandomState(1189)
        self.tl_dict, self.rev_tl_dict = load_dictionaries(data_folder)
        assert self.tl_dict is not None and self.rev_tl_dict is not None

        if load_only_dicts:
            return

        if not shuffle:
            self.train_ex, self.valid_ex, self.test_ex = self._deserialize(
                data_folder)

        else:
            try:
                self.train_ex, self.valid_ex, self.test_ex = self._deserialize(
                    os.path.join(data_folder, 'shuffled'))

                print("Successfully loaded shuffled data.")
                sys.stdout.flush()

            except IOError:
                print("Generating shuffled data...")
                sys.stdout.flush()

                self.train_ex, self.valid_ex, self.test_ex = self._deserialize(
                    data_folder)

                self.rng.shuffle(self.train_ex)
                self.rng.shuffle(self.valid_ex)
                self.rng.shuffle(self.test_ex)

                make_dir_if_not_exists(os.path.join(data_folder, 'shuffled'))

                np.save(os.path.join(data_folder, 'shuffled',
                                     'examples-train.npy'), self.train_ex)
                np.save(os.path.join(data_folder, 'shuffled',
                                     'examples-validation.npy'), self.valid_ex)
                np.save(os.path.join(data_folder, 'shuffled',
                                     'examples-test.npy'), self.test_ex)

    def get_raw_data(self):
        return self.train_ex, self.valid_ex, self.test_ex

    @classmethod
    def prepare_batch(self, sequences, msg=False):
        sequence_lengths = [len(seq) for seq in sequences]
        batch_size = len(sequences)
        max_sequence_length = max(sequence_lengths)

        if msg:
            print('max_sequence_length', max_sequence_length)

        # initialize with _pad_ = 0
        inputs_time_major = np.zeros(
            shape=[max_sequence_length, batch_size], dtype=np.int32)
        for i, seq in enumerate(sequences):
            for j, element in enumerate(seq):
                inputs_time_major[j, i] = element
        return [inputs_time_major, np.array(sequence_lengths)]

    def get_batch(self, start, end, which='train'):
        if which == 'train':
            X, Y = zip(*self.train_ex[start:end])
        elif which == 'valid':
            X, Y = zip(*self.valid_ex[start:end])
        elif which == 'test':
            X, Y = zip(*self.test_ex[start:end])
        else:
            raise ValueError('choose one of train/valid/test for which')
        return tuple(self.prepare_batch(X) + self.prepare_batch(Y))

    def get_tl_dictionary(self):
        return self.tl_dict

    def get_rev_tl_dictionary(self):
        return self.rev_tl_dict

    @property
    def data_size(self):
        return len(self.train_ex), len(self.valid_ex), len(self.test_ex)

    @property
    def vocabulary_size(self):
        return len(self.tl_dict)

def make_feed_dict(x, x_len, y, y_len):
    feed_dict = {
        "encoder_inputs": x,
        "encoder_inputs_length": x_len,

        "decoder_targets": y,
        "decoder_targets_length": y_len,
    }

    return feed_dict
