import torch
import torch.nn as nn
from datasets.seperate_dataloader import read_data, load_embedding, load_worddict, load_wordvec
from models.bilstm_att_g import ABSA_Bilstm_Att_G
from models.basic import get_acc, get_score
import argparse
import sys
import time
import random
import numpy as np
from tensorboardX import SummaryWriter
from math import ceil

if __name__ == '__main__':
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model', type=str, default='bilstm_att_g')
    parser.add_argument('--seed', type=int, default=int(1000 * time.time()))
    parser.add_argument('--dim_word', type=int, default=300)
    parser.add_argument('--dim_hidden', type=int, default=150)
    parser.add_argument('--dim_att_hidden', type=int, default=100)
    parser.add_argument('--num_class', type=int, default=3)
    parser.add_argument('--maxlen', type=int, default=30)
    parser.add_argument('--dataset', type=str, default='laptop')
    parser.add_argument('--glove_file', type=str, default='./data/glove.840B.300d.txt')
    parser.add_argument('--optimizer', type=str, default='adagrad')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--batch', type=int, default=25)
    args, _ = parser.parse_known_args(argv)
    dataset_dic = {'restaurant': ['./data/restaurant/train_sep.txt', './data/restaurant/test_sep.txt'],
                   'laptop': ['./data/laptop/train_sep.txt', './data/laptop/test_sep.txt']}
    model_dic = {'bilstm_att_g': ABSA_Bilstm_Att_G}

    device = torch.device(args.device)
    # random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # load data
    train_fname, test_fname = dataset_dic[args.dataset]
    wordlist, targetlist = load_worddict(train_fname, test_fname)
    emb_dic = load_embedding(args.glove_file, read_from_file=True)
    wordemb = load_wordvec(wordlist, emb_dic)
    targetemb = load_wordvec(targetlist, emb_dic)
    train_data = read_data(train_fname, wordlist, targetlist, maxlen=args.maxlen)
    train_data = list(zip(*train_data))
    test_data = read_data(test_fname, wordlist, targetlist, maxlen=args.maxlen)
    test_data = list(zip(*test_data))
    # init model
    Model = model_dic[args.model]
    model = Model(dim_word=args.dim_word, dim_hidden=args.dim_hidden, dim_att_hidden=args.dim_att_hidden,
                  num_classification=args.num_class, wordemb=wordemb, targetemb=targetemb, device=device)
    model.to(device)
    # init loss
    cross_entropy = nn.CrossEntropyLoss()
    # train
    # summary writer
    writer = SummaryWriter('logs/%s/%s/%s' % (args.dataset, args.model, str(int(time.time()))))
    num_train = len(train_data)
    num_test = len(test_data)
    optim = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    for epoch in range(args.max_epoch):
        random.shuffle(train_data)
        random.shuffle(test_data)
        # train
        model.train()
        total = 0
        for i in range(ceil(num_train / args.batch)):
            model.zero_grad()
            batch_data = train_data[i * args.batch: (i + 1) * args.batch]
            sents, left_sents, right_sents, lens, left_lens, right_lens, targets, ratings = \
                list(map(np.array, list(zip(*batch_data))))
            sents, left_sents, right_sents, lens, left_lens, right_lens, targets, ratings = \
                torch.from_numpy(sents).long().to(device), \
                torch.from_numpy(left_sents).long().to(device), \
                torch.from_numpy(right_sents).long().to(device), \
                torch.from_numpy(lens).long().to(device), \
                torch.from_numpy(left_lens).long().to(device), \
                torch.from_numpy(right_lens).long().to(device), \
                torch.from_numpy(targets).long().to(device), \
                torch.from_numpy(ratings).long().to(device)
            logit = model(left_sents, right_sents, sents, targets, left_lens, right_lens, lens)
            loss = cross_entropy(logit, ratings)
            total += loss * len(batch_data)
            loss.backward()
            optim.step()
        loss = total / num_train
        writer.add_scalar('loss', loss, epoch)
        # eval
        model.eval()
        # eval on train
        total = 0
        logit_list = []
        rating_list = []
        for i in range(ceil(num_train / args.batch)):
            batch_data = train_data[i * args.batch: (i + 1) * args.batch]
            sents, left_sents, right_sents, lens, left_lens, right_lens, targets, ratings = \
                list(map(np.array, list(zip(*batch_data))))
            sents, left_sents, right_sents, lens, left_lens, right_lens, targets, ratings = \
                torch.from_numpy(sents).long().to(device), \
                torch.from_numpy(left_sents).long().to(device), \
                torch.from_numpy(right_sents).long().to(device), \
                torch.from_numpy(lens).long().to(device), \
                torch.from_numpy(left_lens).long().to(device), \
                torch.from_numpy(right_lens).long().to(device), \
                torch.from_numpy(targets).long().to(device), \
                torch.from_numpy(ratings).long().to(device)
            logit = model(left_sents, right_sents, sents, targets, left_lens, right_lens, lens)
            loss = cross_entropy(logit, ratings)
            acc = get_acc(logit, ratings)
            total += loss * len(batch_data)
            logit_list.append(logit)
            rating_list.append(ratings)
        train_loss = total / num_train
        train_acc, train_precision, train_recall, train_f1 = get_score(torch.cat(logit_list, dim=0).cpu().data.numpy(),
                                                                       torch.cat(rating_list, dim=0).cpu().data.numpy())
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('train_precision', train_precision, epoch)
        writer.add_scalar('train_recall', train_recall, epoch)
        writer.add_scalar('train_f1', train_f1, epoch)
        # eval on test
        total = 0
        logit_list = []
        rating_list = []
        for i in range(ceil(num_test / args.batch)):
            batch_data = test_data[i * args.batch: (i + 1) * args.batch]
            sents, left_sents, right_sents, lens, left_lens, right_lens, targets, ratings = \
                list(map(np.array, list(zip(*batch_data))))
            sents, left_sents, right_sents, lens, left_lens, right_lens, targets, ratings = \
                torch.from_numpy(sents).long().to(device), \
                torch.from_numpy(left_sents).long().to(device), \
                torch.from_numpy(right_sents).long().to(device), \
                torch.from_numpy(lens).long().to(device), \
                torch.from_numpy(left_lens).long().to(device), \
                torch.from_numpy(right_lens).long().to(device), \
                torch.from_numpy(targets).long().to(device), \
                torch.from_numpy(ratings).long().to(device)
            logit = model(left_sents, right_sents, sents, targets, left_lens, right_lens, lens)
            loss = cross_entropy(logit, ratings)
            acc = get_acc(logit, ratings)
            total += loss * len(batch_data)
            logit_list.append(logit)
            rating_list.append(ratings)
        test_loss = total / num_test
        test_acc, test_precision, test_recall, test_f1 = get_score(torch.cat(logit_list, dim=0).cpu().data.numpy(),
                                                                   torch.cat(rating_list, dim=0).cpu().data.numpy())
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)
        writer.add_scalar('test_precision', test_precision, epoch)
        writer.add_scalar('test_recall', test_recall, epoch)
        writer.add_scalar('test_f1', test_f1, epoch)
        print('epoch %2d :loss=%.4f, '
              'train_loss=%.4f, train_acc=%.4f, train_precision=%.4f, train_recall=%.4f,train_f1=%.4f,'
              ' test_loss=%.4f, test_acc=%.4f, test_precision=%.4f, test_recall=%.4f, test_f1=%.4f' %
              (epoch, loss, train_loss, train_acc, train_precision, train_recall, train_f1,
               test_loss, test_acc, test_precision, test_recall, test_f1))
        # show parameters
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch, bins='doane')
