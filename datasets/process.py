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
            txt = sentence.find('text').text
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


if __name__ == '__main__':
    process_category('../data/restaurant/Restaurants_Train.xml', '../data/restaurant/train_cat.txt')
    process_category('../data/restaurant/Restaurants_Test_Gold.xml', '../data/restaurant/test_cat.txt')
    process_term('../data/restaurant/Restaurants_Train.xml', '../data/restaurant/train.txt')
    process_term('../data/restaurant/Restaurants_Test_Gold.xml', '../data/restaurant/test.txt')
    process_term('../data/laptop/Laptops_Train.xml', '../data/laptop/train.txt')
    process_term('../data/laptop/Laptops_Test_Gold.xml', '../data/laptop/test.txt')
