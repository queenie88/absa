import torch
import torch.nn as nn
from datasets import read_data, load_embedding, load_worddict, load_wordvec
from models import ABSA_Lstm
from models.basic import get_acc
import argparse
import sys
import time
import random
import numpy as np
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lstm')
    parser.add_argument('--seed', type=int, default=int(1000 * time.time()))
    parser.add_argument('--dim_word', type=int, default=300)
    parser.add_argument('--dim_hidden', type=int, default=300)
    parser.add_argument('--num_class', type=int, default=3)
    parser.add_argument('--maxlen', type=int, default=25)
    parser.add_argument('--dataset', type=str, default='restaurant')
    parser.add_argument('--glove_file', type=str, default='./data/glove.840B.300d.txt')
    parser.add_argument('--optimizer', type=str, default='adagrad')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--max_epoch', type=int, default=25)
    parser.add_argument('--batch', type=int, default=25)
    args, _ = parser.parse_known_args(argv)
    dataset_dic = {'restaurant': ['./data/restaurant/train.txt', './data/restaurant/test.txt']}
    model_dic = {'lstm': ABSA_Lstm}

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
    model = Model(dim_word=args.dim_word, dim_hidden=args.dim_hidden, num_classification=args.num_class,
                  maxlen=args.maxlen, batch=args.batch, wordemb=wordemb, targetemb=targetemb)
    # init loss
    cross_entropy = nn.CrossEntropyLoss()
    # train
    # summary writer
    writer = SummaryWriter('logs/{}/{}'.format(args.dataset, args.model))
    num_train = len(train_data)
    num_test = len(test_data)
    optim = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    for epoch in range(args.max_epoch):
        random.shuffle(train_data)
        random.shuffle(test_data)
        # train
        model.train()
        losses = []
        for i in range(num_train // args.batch):
            model.zero_grad()
            batch_data = train_data[i * args.batch: (i + 1) * args.batch]
            sent, target, rating, lens = list(zip(*batch_data))
            sent, target, rating, lens = np.array(sent), np.array(target), np.array(rating), np.array(lens)
            sent, target, rating, lens = torch.from_numpy(sent).long(), torch.from_numpy(target).long(), \
                                         torch.from_numpy(rating).long(), torch.from_numpy(lens).long()
            logit = model(sent, target, lens)
            loss = cross_entropy(logit, rating)
            losses.append(loss.data.numpy())
            loss.backward()
            optim.step()
        loss = sum(losses) / len(losses)
        writer.add_scalar('loss', loss, epoch)
        # eval
        model.eval()
        # eval on train
        train_losses = []
        train_acces = []
        for i in range(num_train // args.batch):
            batch_data = train_data[i * args.batch: (i + 1) * args.batch]
            sent, target, rating, lens = list(zip(*batch_data))
            sent, target, rating, lens = np.array(sent), np.array(target), np.array(rating), np.array(lens)
            sent, target, rating, lens = torch.from_numpy(sent).long(), torch.from_numpy(target).long(), \
                                         torch.from_numpy(rating).long(), torch.from_numpy(lens).long()
            logit = model(sent, target, lens)
            loss = cross_entropy(logit, rating)
            acc = get_acc(logit, rating)
            train_losses.append(loss)
            train_acces.append(acc)
        train_loss = sum(train_losses) / len(train_losses)
        train_acc = sum(train_acces) / len(train_acces)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        # eval on test
        test_losses = []
        test_acces = []
        for i in range(num_test // args.batch):
            batch_data = test_data[i * args.batch: (i + 1) * args.batch]
            sent, target, rating, lens = list(zip(*batch_data))
            sent, target, rating, lens = np.array(sent), np.array(target), np.array(rating), np.array(lens)
            sent, target, rating, lens = torch.from_numpy(sent).long(), torch.from_numpy(target).long(), \
                                         torch.from_numpy(rating).long(), torch.from_numpy(lens).long()
            logit = model(sent, target, lens)
            loss = cross_entropy(logit, rating)
            acc = get_acc(logit, rating)
            test_losses.append(loss)
            test_acces.append(acc)
        test_loss = sum(test_losses) / len(test_losses)
        test_acc = sum(test_acces) / len(test_acces)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)
        print('epoch %2d :loss=%.4f, train_loss=%.4f, train_acc=%.4f, test_loss=%.4f, test_acc=%.4f' %
              (epoch, loss, train_loss, train_acc, test_loss, test_acc))
        # show parameters
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch, bins='doane')
