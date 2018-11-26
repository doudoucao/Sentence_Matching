from collections import Counter
import numpy as np



def read_data(filepath):
    with open(filepath, 'r') as input_data:
        ps, hs, p_cs, h_cs, labels = [], [], [], [], []

        # Ignore the headers on the first line of the file
        # next(input_data)

        for line in input_data:
            line = line.strip.split(',')
            p = line[2]
            h = line[3]
            p_c = line[4]
            h_c = line[5]
            label = line[6]

            # Each p and h is split into a list of words
            ps.append(p.rstrip().split())
            hs.append(h.rstrip().split())
            p_cs.append(p_c.rstrip().split())
            h_cs.append(h_c.rstrip().split())
            labels.append(label)

    return {
        'p': ps,
        'h': hs,
        'p_c': p_cs,
        'h_c':h_cs,
        'labels': labels
        }


def build_worddict(data, num_words=None):
    """
    Build a dictionary assocoating words from a set of premises and hypotheses to unique integer indices.
    :param data:
    :param num_words:
    :return:
    """

    words = []
    chars = []
    [words.extend(sentence) for sentence in data['p']]
    [words.extend(sentence) for sentence in data['h']]
    [chars.extend(sentence) for sentence in data['p_c']]
    [chars.extend(sentence) for sentence in data['h_C']]

    counts = Counter(words)
    if num_words is None:
        num_words = len(counts)

    worddict = {word[0]: i+4 for i, word in enumerate(counts.most_common(num_words))}

    worddict["_PAD_"] = 0
    worddict["_OOV_"] = 1
    worddict['_BOS_'] = 2
    worddict['_EOS_'] = 3

    return worddict


def words_to_indices(sentence, worddict):
    """
    Transform the words in a sentence to integer indices
    :param sentence:
    :param worddict:
    :return:
    """
    indices = [worddict['_BOS_ ']]
    for word in sentence:
        if word in worddict:
            index = worddict[word]
        else:
            index = worddict['_OOV_']
        indices.append(index)
    indices.append(worddict["_EOS_"])

    return indices


def transform_to_indices(data, worddict, labeldict):
    transformed_data = {'p': [], 'h': [], 'labels': []}

    for i, premise in enumerate(data['premises']):
        label = data['labels'][i]
        if label not in labeldict:
            continue

        transformed_data['labels'].append(labeldict[label])

        indices = words_to_indices(premise, worddict)
        transformed_data['p'].append(indices)

        indices = words_to_indices(data['h'][i], worddict)
        transformed_data['h'].append(indices)

    return transformed_data


def build_embedding_matrix(worddict, embeddings_file):
    embeddings = {}
    with open(embeddings_file, 'r', encoding='utf-8') as input_data:
        for line in input_data:
            line = line.split()
            word = line[0]

            if word in worddict:
                embeddings[word] = line[1: ]

    num_words = len(worddict)
    embedding_dim = len(list(embeddings.values())[0])
    embedding_matrix = np.zeros((num_words, embedding_dim))

    for word, i in worddict.items():
        if word in embeddings:
            embedding_matrix[i] = np.array(embeddings[word], dtype=float)

        else:
            if word == '_PAD_':
                continue

            embedding_matrix[i] = np.random.normal(size=(embedding_dim))

    return embedding_matrix


def process_data(inputdir, embedding_file, targetdir=False, num_words=None):

    train_file = ""
    val_file = ""
    test_file = ""
    pass














