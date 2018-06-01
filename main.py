import torch
import torch.nn as nn
from datasets import read_data, load_embedding, load_worddict, load_wordvec
from models import ABSA_Lstm, ABSA_Atae_Lstm, Model, ACSA_GCAE
from models.basic import get_acc
import argparse
import sys
import time
import random
import numpy as np
from tensorboardX import SummaryWriter
from models.basic import get_score

if __name__ == '__main__':
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model', type=str, default='acsa_gcae')
    parser.add_argument('--seed', type=int, default=int(1000 * time.time()))
    parser.add_argument('--dim_word', type=int, default=300)
    parser.add_argument('--dim_hidden', type=int, default=300)
    parser.add_argument('--num_class', type=int, default=3)
    parser.add_argument('--maxlen', type=int, default=30)
    parser.add_argument('--dataset', type=str, default='laptop')
    parser.add_argument('--glove_file', type=str, default='./data/glove.840B.300d.txt')
    parser.add_argument('--optimizer', type=str, default='adagrad')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--batch', type=int, default=25)
    # acsa model param
    parser.add_argument('--num_kernel', type=int, default=100)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--kernel_sizes', type=list, default=[3, 4, 5])
    args, _ = parser.parse_known_args(argv)
    dataset_dic = {'restaurant_cat': ['./data/restaurant/train_cat.txt', './data/restaurant/test_cat.txt'],
                   'restaurant': ['./data/restaurant/train.txt', './data/restaurant/test.txt'],
                   'laptop': ['./data/laptop/train.txt', './data/laptop/test.txt']}
    model_dic = {'lstm': ABSA_Lstm, 'atae_lstm': ABSA_Atae_Lstm, 'model': Model, 'acsa_gcae': ACSA_GCAE}

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
    if args.model == 'acsa_gcae':
        model = Model(dim_word=args.dim_word, num_kernel=args.num_kernel, num_classification=args.num_class,
                      maxlen=args.maxlen, dropout_rate=args.dropout_rate, kernel_sizes=args.kernel_sizes,
                      wordemb=wordemb, targetemb=targetemb, device=device)
    else:
        model = Model(dim_word=args.dim_word, dim_hidden=args.dim_hidden, num_classification=args.num_class,
                      maxlen=args.maxlen, wordemb=wordemb, targetemb=targetemb, device=device)
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
        for i in range(num_train // args.batch + 1):
            model.zero_grad()
            batch_data = train_data[i * args.batch: (i + 1) * args.batch]
            sent, target, rating, lens = list(zip(*batch_data))
            sent, target, rating, lens = np.array(sent), np.array(target), np.array(rating), np.array(lens)
            sent, target, rating, lens = torch.from_numpy(sent).long().to(device), \
                                         torch.from_numpy(target).long().to(device), \
                                         torch.from_numpy(rating).long().to(device), \
                                         torch.from_numpy(lens).long().to(device)
            logit = model(sent, target, lens)
            loss = cross_entropy(logit, rating)
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
        for i in range(num_train // args.batch + 1):
            batch_data = train_data[i * args.batch: (i + 1) * args.batch]
            sent, target, rating, lens = list(zip(*batch_data))
            sent, target, rating, lens = np.array(sent), np.array(target), np.array(rating), np.array(lens)
            sent, target, rating, lens = torch.from_numpy(sent).long().to(device), \
                                         torch.from_numpy(target).long().to(device), \
                                         torch.from_numpy(rating).long().to(device), \
                                         torch.from_numpy(lens).long().to(device)
            logit = model(sent, target, lens)
            loss = cross_entropy(logit, rating)
            acc = get_acc(logit, rating)
            total += loss * len(batch_data)
            logit_list.append(logit)
            rating_list.append(rating)
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
        for i in range(num_test // args.batch + 1):
            batch_data = test_data[i * args.batch: (i + 1) * args.batch]
            sent, target, rating, lens = list(zip(*batch_data))
            sent, target, rating, lens = np.array(sent), np.array(target), np.array(rating), np.array(lens)
            sent, target, rating, lens = torch.from_numpy(sent).long().to(device), \
                                         torch.from_numpy(target).long().to(device), \
                                         torch.from_numpy(rating).long().to(device), \
                                         torch.from_numpy(lens).long().to(device)
            logit = model(sent, target, lens)
            loss = cross_entropy(logit, rating)
            acc = get_acc(logit, rating)
            total += loss * len(batch_data)
            logit_list.append(logit)
            rating_list.append(rating)
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
