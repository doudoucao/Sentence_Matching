import numpy as np
from torch.utils.data import Dataset


def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.strip().split()
        # print(len(tokens))
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim):
    print('loading word vectors....')
    embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))
    word_vec = load_word_vec('embedding/word_embedding.txt', word2idx)
    # print(word_vec.items())
    for word, i in word2idx.items():
        vec = word_vec.get(word.strip())
        # print(vec)
        if vec is not None:
            embedding_matrix[i] = vec
    return embedding_matrix


def load_char_vec(path, char2idx=None):
    fin = open(path, 'r', encoding='utf-8', errors='ignore')
    char_vec = {}
    for line in fin:
        chars = line.strip().split()
        if char2idx is None or chars[0] in char2idx.keys():
            char_vec[chars[0]] = np.asarray(chars[1:], dtype='float32')
    return char_vec


def build_char_embedding_matrix(char2idx, char_embed_dim):
    print('loading char vectors...')
    char_embedding_matrix = np.zeros((len(char2idx) + 2, char_embed_dim))
    char_vec = load_char_vec('embedding/char_embedding.txt', char2idx)
    for char, i in char2idx.items():
        vec = char_vec.get(char.strip())
        if vec is not None:
            char_embedding_matrix[i] = vec
    return char_embedding_matrix


class Tokenizer(object):
    def __init__(self, max_seq_len=None, max_char_len=None):
        self.max_seq_len = max_seq_len
        self.max_char_len = max_char_len
        self.word2idx = {}
        self.idx2word = {}
        self.char2idx = {}
        self.idx2char = {}
        self.idx = 1
        self.char_idx = 1

    def fit_on_text(self, text):
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def fit_on_char(self, text):
        chars = text.split()
        for char in chars:
            if char not in self.char2idx:
                self.char2idx[char] = self.char_idx
                self.idx2char[self.char_idx] = char
                self.char_idx += 1


    @staticmethod
    def pad_sequence(sequence, maxlen, dtype='int64', padding='pre', truncating='pre', value=0.):
        x = (np.ones(maxlen)*value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):]= trunc
        return x

    def text_to_sequence(self, text, reverse=False):
        words = text.split()
        unk_idx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unk_idx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        pad_and_trunc = 'post'
        if reverse:
            sequence = sequence[::-1]
        return Tokenizer.pad_sequence(sequence, self.max_seq_len, padding=pad_and_trunc, truncating=pad_and_trunc)

    def char_to_sequence(self, text):
        chars = text.split()
        unk_idx = len(self.char2idx) + 1
        sequence = [self.char2idx[char] if char in self.char2idx else unk_idx for char in chars]
        if len(sequence) == 0:
            sequence = [0]
        pad_and_trunc = 'post'
        return Tokenizer.pad_sequence(sequence, self.max_char_len, padding=pad_and_trunc, truncating=pad_and_trunc)


class CHIPDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class DatasetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        char = ''
        for fname in fnames:
            with open(fname, 'r', encoding='utf-8', errors='ignore') as fin:
                for line in fin:
                    content = line.split(',')
                    wid_1 = content[2]
                    wid_2 = content[3]
                    cid_1 = content[4]
                    cid_2 = content[5]
                    text_raw = wid_1 + ' ' + wid_2
                    char_raw = cid_1 + ' ' + cid_2
                    text += text_raw + " "
                    char += char_raw + " "
        return text, char

    @staticmethod
    def __read_data__(fname, tokenizer, type='train'):
        all_data = []
        with open(fname, encoding='utf-8', newline='\n', errors='ignore') as fin:
            for line in fin:
                content = line.split(',')
                premise = content[2].strip()
                hypothesis = content[3].strip()
                p_char = content[4].strip()
                h_char = content[5].strip()
                if type == 'train':
                    label = content[6]

                premise_indices = tokenizer.text_to_sequence(premise)
                hypothesis_indices = tokenizer.text_to_sequence(hypothesis)
                pchar_indices = tokenizer.char_to_sequence(p_char)
                hchar_indices = tokenizer.char_to_sequence(h_char)
                if type == 'train':
                    label = int(label)
                    data = {
                        'p': premise_indices,
                        'h': hypothesis_indices,
                        'p_char': pchar_indices,
                        'h_char': hchar_indices,
                        'label': label
                    }
                else:
                    data = {
                        'p': premise_indices,
                        'h': hypothesis_indices,
                        'p_char': pchar_indices,
                        'h_char': hchar_indices
                    }
                all_data.append(data)
        return all_data

    def __init__(self, embed_dim=300, char_embed_dim=300, max_seq_len=15, max_char_len=20):
        print('preparing dataset...')
        fname = {'train': './data/train.csv',
                 'test': './data/new_test.csv',
                 'val': './data/val.csv'}
        text, char = DatasetReader.__read_text__([fname['train'], fname['test'], fname['val']])
        tokenizer = Tokenizer(max_seq_len=max_seq_len, max_char_len=max_char_len)
        tokenizer.fit_on_text(text)
        tokenizer.fit_on_char(char)
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim)
        self.char_embedding_matrix = build_char_embedding_matrix(tokenizer.char2idx, char_embed_dim)
        self.train_data = CHIPDataset(DatasetReader.__read_data__(fname['train'], tokenizer, type='train'))
        self.test_data = CHIPDataset(DatasetReader.__read_data__(fname['test'], tokenizer, type='test'))
        self.val_data = CHIPDataset(DatasetReader.__read_data__(fname['val'], tokenizer, type='train'))