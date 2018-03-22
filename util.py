import os
import pickle as pkl

import numpy as np


def build_dictionary(sentences, vocab=None, max_sent_len_=None):
    is_ext_vocab = True

    if vocab is None:
        is_ext_vocab = False
        vocab = {'<PAD>': 0, '<OOV>': 1}

    data_sentences = []
    max_sent_len = -1

    for sentence in sentences:
        words = []
        for word in sentence:
            if not is_ext_vocab and word not in vocab:
                vocab[word] = len(vocab)
            if word not in vocab:
                token_id = vocab['<OOV>']
            else:
                token_id = vocab[word]
            words.append(token_id)
        if len(words) > max_sent_len:
            max_sent_len = len(words)
        data_sentences.append(words)

    if max_sent_len_ is not None:
        max_sent_len = max_sent_len_

    enc_sentences = np.full([len(data_sentences), max_sent_len], vocab['<PAD>'], dtype=np.int32)

    sentence_lengths = []
    for i, sentence in enumerate(data_sentences):
        enc_sentences[i, 0:len(sentence)] = sentence
        sentence_lengths.append(len(sentence))

    sentence_lengths = np.array(sentence_lengths, dtype=np.int32)
    reverse_dictionary = dict(zip(vocab.values(), vocab.keys()))

    return vocab, reverse_dictionary, sentence_lengths, max_sent_len + 1, enc_sentences


def generate_pkl(train):
    if train:
        files = ['cyto.fasta', 'secreted.fasta', 'mito.fasta', 'nucleus.fasta']
        prefix = 'train_'
    else:
        files = ['blind.fasta']
        prefix = 'test_'

    data = []
    labels = []
    s = ''
    for fasta_file in files:
        with open(fasta_file, 'r') as f:
            lines = f.readlines()
        count = 0
        for l in lines:
            if l[0] == '>':
                if count > 0:
                    data.append(s)
                    labels.append(files.index(fasta_file))
                s = ''
            else:
                s += l[:-1]
                if count == len(lines) - 1:
                    data.append(s)
                    labels.append(files.index(fasta_file))
            count += 1
    with open(prefix + 'data.pkl', 'wb') as f:
        pkl.dump(data, f)
    with open(prefix + 'labels.pkl', 'wb') as f:
        pkl.dump(labels, f)


def get_data(train):
    if train:
        prefix = 'train_'
    else:
        prefix = 'test_'
    if not os.path.exists(prefix + 'data.pkl'):
        print('Generating data')
        generate_pkl(train)
    with open(prefix + 'data.pkl', 'rb') as f:
        data = pkl.load(f)
    with open(prefix + 'labels.pkl', 'rb') as f:
        labels = pkl.load(f)
    return data, labels


get_data(train=True)
get_data(train=False)
