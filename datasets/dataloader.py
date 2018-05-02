import numpy as np
import pickle

padding = '<PADDING>'


def read_data(fname, wordlist, targetlist, maxlen=20):
    file = open(fname, 'r')
    lines = file.readlines()
    sents = []
    targets = []
    ratings = []
    lens = []
    for i in range(len(lines) // 3):
        sent = lines[i * 3]
        target = lines[i * 3 + 1]
        rating = lines[i * 3 + 2]
        sent_list = []
        for word in sent.strip().split(' '):
            sent_list.append(wordlist[word])
        targets.append(targetlist[target])
        lens.append(len(sent_list))
        ratings.append(int(rating) + 1)
        while len(sent_list) < maxlen:
            sent_list.append(wordlist[padding])
        sent_list = sent_list[:maxlen]
        sents.append(sent_list)
    sents, targets, ratings, lens = np.asarray(sents), np.asarray(targets), np.asarray(ratings), np.asarray(lens)
    return sents, targets, ratings, lens


def load_worddict(train_fname, test_fname):
    train_file = open(train_fname, 'r')
    test_file = open(test_fname, 'r')
    # init
    word2id = {}
    word2id[padding] = 0
    tar2id = {}
    # load train
    lines = train_file.readlines()
    for i in range(len(lines) // 3):
        sent = lines[i * 3]
        target = lines[i * 3 + 1]
        # rating = lines[i * 3 + 2]
        for word in sent.strip().split():
            if word not in word2id:
                word2id[word] = len(word2id)
        if target not in tar2id:
            tar2id[target] = len(tar2id)
    # load test
    lines = test_file.readlines()
    for i in range(len(lines) // 3):
        sent = lines[i * 3]
        target = lines[i * 3 + 1]
        # rating = lines[i * 3 + 2]
        for word in sent.strip().split():
            if word not in word2id:
                word2id[word] = len(word2id)
        if target not in tar2id:
            tar2id[target] = len(tar2id)
    return word2id, tar2id


def load_embedding(emb_fname, read_from_file=False, save_fname='dic.pkl'):
    if read_from_file:
        dic = pickle.load(open(save_fname, 'rb'))
        return dic
    dic = {}
    emb_file = open(emb_fname, 'r')
    line = emb_file.readline()
    while line:
        try:
            parts = line.strip().split(' ')
            word = parts[0]
            vec = list(map(float, parts[1:]))
            dic[word] = np.array(vec)
        except:
            pass
        line = emb_file.readline()
    print('load %d words' % len(dic))
    pickle.dump(dic, open(save_fname, 'wb'))
    return dic


def load_wordvec(wordlist, emb_dic):
    num_word = len(wordlist) + 1
    dim_word = len(list(emb_dic.values())[0])
    embedding = np.random.normal(0, 0.01, [num_word, dim_word])
    not_found = 0
    for word, idx in wordlist.items():
        try:
            embedding[idx] = emb_dic[word]
        except:
            not_found += 1
    print('%d words not found' % not_found)
    return embedding
