import xml.etree.ElementTree as ET
from nltk import word_tokenize


def process_category(fname, save_fname):
    fout = open(save_fname, 'w')
    dic = {'positive': 1, 'neutral': 0, 'negative': -1}
    tree = ET.parse(fname)
    root = tree.getroot()
    for sentence in root.findall('sentence'):
        txt = sentence.find('text').text.lower()
        words = word_tokenize(txt)
        txt = ' '.join(words)
        aspects = sentence.find('aspectCategories')
        for aspect in aspects.findall('aspectCategory'):
            a = aspect.get('category')
            if '/' in a:
                a = a.split('/')[-1]
            p = aspect.get('polarity')
            if p == 'conflict':
                continue
            p = dic[p]
            print(txt, file=fout)
            print(a, file=fout)
            print(p, file=fout)


def process_term(fname, save_fname):
    fout = open(save_fname, 'w')
    dic = {'positive': 1, 'neutral': 0, 'negative': -1}
    tree = ET.parse(fname)
    root = tree.getroot()
    bad_sent = 0
    for sentence in root.findall('sentence'):
        try:
            txt = sentence.find('text').text.lower()
            words = word_tokenize(txt)
            txt = ' '.join(words)
            aspects = sentence.find('aspectTerms')
            for aspect in aspects.findall('aspectTerm'):
                a = aspect.get('term')
                if '/' in a:
                    a = a.split('/')[-1]
                p = aspect.get('polarity')
                if p == 'conflict':
                    continue
                p = dic[p]
                print(txt, file=fout)
                print(a, file=fout)
                print(p, file=fout)
        except:
            bad_sent += 1
    print('bad sent %d' % bad_sent)


def process_seperate(fname, save_fname):
    fout = open(save_fname, 'w')
    dic = {'positive': 1, 'neutral': 0, 'negative': -1}
    tree = ET.parse(fname)
    root = tree.getroot()
    for sentence in root.findall('sentence'):
        try:
            txt = sentence.find('text').text.lower()
            aspects = sentence.find('aspectTerms')
            for aspect in aspects.findall('aspectTerm'):
                a = aspect.get('term')
                if '/' in a:
                    a = a.split('/')[-1]
                p = aspect.get('polarity')
                f = int(aspect.get('from'))
                t = int(aspect.get('to'))
                left_txt = txt[:f]
                target_txt = txt[f:t]
                if target_txt != a:
                    raise Exception('target not same')
                right_txt = txt[t:]
                if p == 'conflict':
                    continue
                p = dic[p]
                left_words, target_words, right_words = word_tokenize(left_txt), \
                                                        word_tokenize(target_txt), word_tokenize(right_txt)
                processed_txt = ' '.join(left_words + target_words + right_words)
                m = len(left_words)
                n = len(target_words)
                print('%s ||| %d %d ||| %d' % (processed_txt, m, m + n - 1, p), file=fout)
        except:
            pass


def process_seperate_twitter(fname, save_fname, split_tag='$T$'):
    fin = open(fname, 'r')
    fout = open(save_fname, 'w')
    lines = fin.readlines()
    for i in range(len(lines) // 3):
        sent = lines[i * 3].strip()
        target = lines[i * 3 + 1].strip()
        rating = lines[i * 3 + 2].strip()
        left, right = sent.split(split_tag)
        left_words, right_words = left.split(' '), right.split(' ')
        target_words = target.split(' ')
        total_sent = ' '.join(left_words + target_words + right_words)
        print('%s ||| %d %d ||| %s' % (total_sent, len(left_words), len(left_words) + len(target_words) - 1, rating),
              file=fout)


if __name__ == '__main__':
    process_category('../data/restaurant/Restaurants_Train.xml', '../data/restaurant/train_cat.txt')
    process_category('../data/restaurant/Restaurants_Test_Gold.xml', '../data/restaurant/test_cat.txt')
    process_term('../data/restaurant/Restaurants_Train.xml', '../data/restaurant/train.txt')
    process_term('../data/restaurant/Restaurants_Test_Gold.xml', '../data/restaurant/test.txt')
    process_term('../data/laptop/Laptops_Train.xml', '../data/laptop/train.txt')
    process_term('../data/laptop/Laptops_Test_Gold.xml', '../data/laptop/test.txt')
    process_seperate('../data/restaurant/Restaurants_Train.xml', '../data/restaurant/train_sep.txt')
    process_seperate('../data/restaurant/Restaurants_Test_Gold.xml', '../data/restaurant/test_sep.txt')
    process_seperate('../data/laptop/Laptops_Train.xml', '../data/laptop/train_sep.txt')
    process_seperate('../data/laptop/Laptops_Test_Gold.xml', '../data/laptop/test_sep.txt')
    # process_seperate_twitter('../data/twitter/train.data', '../data/twitter/train_sep.txt')
    # process_seperate_twitter('../data/twitter/test.data', '../data/twitter/test_sep.txt')
