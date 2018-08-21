# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Modified to support pytorch Tensors
# ----------------------------------------------------------------------------
# Taken and adapted from:
# https://github.com/SeanNaren/deepspeech.pytorch/blob/master/decoder.py

import Levenshtein as Lev
import torch
from six.moves import xrange
import torch

class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.
    """

    def __init__(self, gt_preprocessor):
        # e.g. labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ#"
        self.labels = ''.join(map(str, gt_preprocessor.chars))
        self.int_to_char = gt_preprocessor.token2char
        self.blank_token = gt_preprocessor.blank_token

    def wer(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))

    def cer(self, s1, s2, no_spaces=False):
        """
        Computes the Character Error Rate, defined as the edit distance.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        if no_spaces:
            s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return Lev.distance(s1, s2)

    def decode(self, probs, remove_repetitions):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription
        """
        raise NotImplementedError


class GreedyDecoder(Decoder):
    def __init__(self, gt_preprocessor):
        super(GreedyDecoder, self).__init__(gt_preprocessor)

    def convert_to_strings(self, sequences, remove_repetitions):
        """Given a list of numeric sequences, returns the corresponding strings"""
        string_list = []
        for i in xrange(len(sequences)):
            string, tokens = self.process_string(sequences[i], remove_repetitions)
            string_list.append((tokens, string)) #encoded, decoded
        return string_list
    
    def process_string(self, sequence, remove_repetitions):
        char_string = ''
        tokens = []
        for i in range(len(sequence)):
            token = int(sequence[i])
            tokens.append(token)
            if token != self.blank_token:
                char = self.int_to_char[token]
                if remove_repetitions and i != 0 and (token == sequence[i-1]):
                    pass
                else:
                    char_string = char_string + char
        return char_string, tokens

    def decode(self, probs, remove_repetitions):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of seq_length x batch x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        _, max_probs = torch.max(probs.transpose(0, 1), 2)
        string_list = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)), remove_repetitions)
        return string_list