# coding=utf-8

import itertools as it
import json
import os
import shutil
import types
from collections import Counter
import numpy as np
import sqlite3
from keras.utils.np_utils import to_categorical
from scipy.sparse import csr_matrix
from tqdm import tqdm
from itertools import groupby
from keras.preprocessing.sequence import pad_sequences


TOKEN_START = 1
TOKEN_END = 2


class Text2Char(object):
    """
    This class converts text to char-level tokens. 
    """

    def __init__(self, max_length=None):
        """
        :param max_lenght: Maximum number of characters in each sentence 
                           Note that when max_lenght > 0 text will be padded and 
                           truncated if needed;
        """
                
        self.max_length = max_length                   

        alphabet = u' abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"´/\|_@#$£%^&*~`+-=<>()[]{}'
                
        chars_to_compensate = 3 # {0: '<pad>', 1: '<start>', 2: '<end>', 3: ' ', 4: 'a', ...}
        alphabet_dict = {x: i+chars_to_compensate for i, x in enumerate(alphabet)}
        self.alphabet_dict = alphabet_dict
        self.n_alphabet = len(alphabet) + chars_to_compensate

        self.reversed_alphabet_dict = {i: x for (x, i) in alphabet_dict.iteritems()}
        self.alphabet_dict.update(self._get_special_chars())
        
    def _get_special_chars(self,):

        special_chars  = []
        special_chars += [(u'áàâãä', u'a')]
        special_chars += [(u'éèêẽë', u'e')]
        special_chars += [(u'íìîĩï', u'i')]
        special_chars += [(u'óòôõö', u'o')]
        special_chars += [(u'úùûũü', u'u')]
        special_chars += [(u'ç', u'c')]
        special_chars += [(u'ñ', u'n')]

        special_chars_dict = {}
        for k, v in special_chars:            
            for _k in k:
                special_chars_dict[_k] = self.alphabet_dict[v]

        return special_chars_dict

    def encode_text(self, text):
        text = str(text).lower().strip().decode('utf-8')
        # text = unicode(text.lower().strip(), 'utf-8')
        text = text.lower().strip()
        token_chars = [TOKEN_START]
        for x in text:
            
            try:
                token_chars +=  [self.alphabet_dict[x]] 
            except KeyError:
                token_chars += [self.alphabet_dict[' ']]

        token_chars += [TOKEN_END]    

        return token_chars

    def encode_texts(self, texts):

        encoded_txts = []
        for txt in texts:
            t = self.encode_text(txt)
            encoded_txts.append(t)

        lengths = map(len, encoded_txts)
        msk = np.argsort(lengths)[::-1]

        if self.max_length is not None:
            encoded_txts = pad_sequences(encoded_txts, self.max_length, padding='post')

        encoded_txts = np.asarray(encoded_txts)
        lengths = np.asarray(lengths)
        encoded_txts, lenghts = encoded_txts[msk], lengths[msk]

        return np.asarray(encoded_txts).astype(np.int64), lengths        

    def decode_tokens(self, tokens):
        return ''.join([self.reversed_alphabet_dict[t] for t in tokens]).strip()

