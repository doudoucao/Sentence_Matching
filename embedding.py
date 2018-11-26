import numpy as np


def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.strip().split()
        # print(len(tokens))
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        return word_vec

print(load_word_vec('embedding/word_embedding.txt'))