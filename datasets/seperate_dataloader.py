import numpy as np
import pickle

padding = '<PADDING>'


def read_data(fname, wordlist, targetlist, maxlen=20):
    file = open(fname, 'r')
    sents = []
    left_sents = []
    right_sents = []
    lens = []
    left_lens = []
    right_lens = []
    targets = []
    polarities = []
    for line in file.readlines():
        txt, pos, polarity = line.strip().split('|||')
        words = txt.strip().split(' ')
        start, end = pos.strip().split(' ')
        start, end = int(start), int(end)
        polarity = int(polarity) + 1
        left_words = words[:start]
        right_words = words[end + 1:]
        targets.append(targetlist[' '.join(words[start:end + 1])])
        sent = [wordlist[k] for k in words]
        left_sent = [wordlist[k] for k in left_words]
        right_sent = [wordlist[k] for k in right_words]
        lens.append(len(sent))
        left_lens.append(len(left_sent))
        right_lens.append(len(right_sent))
        sents.append(pad_sent(sent, wordlist[padding], maxlen))
        left_sents.append(pad_sent(left_sent, wordlist[padding], maxlen))
        right_sents.append(pad_sent(right_sent, wordlist[padding], maxlen))
        polarities.append(polarity)
    return sents, left_sents, right_sents, lens, left_lens, right_lens, targets, polarities


def pad_sent(sent, padding, maxlen):
    while len(sent) < maxlen:
        sent.append(padding)
    return sent[:maxlen]


def load_worddict(train_fname, test_fname):
    train_file = open(train_fname, 'r')
    test_file = open(test_fname, 'r')
    # init
    word2id = {}
    target2id = {}
    word2id[padding] = len(word2id)
    # load train
    lines = train_file.readlines()
    for line in lines:
        txt, pos, polarity = line.strip().split('|||')
        words = txt.split(' ')
        start, end = pos.strip().split(' ')
        start, end = int(start), int(end)
        target = ' '.join(words[start:end + 1])
        if target not in target2id:
            target2id[target] = len(target2id)
        for word in words:
            if word not in word2id:
                word2id[word] = len(word2id)
    lines = test_file.readlines()
    for line in lines:
        txt, pos, polarity = line.strip().split('|||')
        words = txt.split(' ')
        start, end = pos.strip().split(' ')
        start, end = int(start), int(end)
        target = ' '.join(words[start:end + 1])
        if target not in target2id:
            target2id[target] = len(target2id)
        for word in words:
            if word not in word2id:
                word2id[word] = len(word2id)
    return word2id, target2id


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


if __name__ == '__main__':
    wordlist, targetlist = load_worddict('../data/restaurant/train_sep.txt', '../data/restaurant/test_sep.txt')
    sents, left_sents, right_sents, lens, left_lens, right_lens, targets, polarities = \
        read_data('../data/restaurant/train_sep.txt', wordlist, targetlist)
    print(sents)
    print(left_sents)
    print(right_sents)
    print(lens)
    print(left_lens)
    print(right_lens)
    print(targets)
    print(polarities)
