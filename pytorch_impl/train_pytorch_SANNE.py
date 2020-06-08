#! /usr/bin/env python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(123)

import numpy as np
np.random.seed(123)
import time
import pickle as cPickle
from pytorch_model_SANNE import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
import statistics
from generate_random_walks import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

# Parameters
# ==================================================

parser = ArgumentParser("SANNE")
parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default="cora", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.005, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=3, type=int, help="Batch Size")
parser.add_argument("--num_epochs", default=50, type=int, help="Number of training epochs")
parser.add_argument("--model_name", default='cora', help="")
parser.add_argument('--sampled_num', default=512, type=int, help='')
parser.add_argument("--dropout", default=0.5, type=float, help="")
parser.add_argument("--num_heads", default=2, type=int, help="")
parser.add_argument("--num_self_att_layers", default=1, type=int, help="Number of self-attention layers")
parser.add_argument("--ff_hidden_size", default=16, type=int, help="The hidden size for the feedforward layer")
parser.add_argument("--num_neighbors", default=4, type=int, help="")
parser.add_argument('--fold_idx', type=int, default=1, help='The fold index. 0-9.')
parser.add_argument('--num_walks', type=int, default=3, help='')
parser.add_argument('--walk_length', type=int, default=8, help='')
args = parser.parse_args()

print(args)

walks = generate_random_walks(input='../data/'+args.dataset+'.Full.edgelist', num_walks=args.num_walks, walk_length=args.walk_length)
data_size = np.shape(walks)[0]
#cora,citeseer,pubmed
with open('../data/'+args.dataset+'.128d.feature.pickle', 'rb') as f:
    features_matrix = torch.from_numpy(cPickle.load(f)).to(device)
vocab_size = features_matrix.size(0)
feature_dim_size = features_matrix.size(1)

class Batch_Loader_RW(object):
    def __init__(self):

        self.dict_neighbors = {}
        with open('../data/' + args.dataset + '.Full.edgelist', 'r') as f:
            for line in f:
                lst_nodes = line.strip().split()
                if len(lst_nodes) == 2:
                    if int(lst_nodes[0]) not in self.dict_neighbors:
                        self.dict_neighbors[int(lst_nodes[0])] = []
                    self.dict_neighbors[int(lst_nodes[0])].append(int(lst_nodes[1]))

    def __call__(self):
        idxs = np.random.permutation(data_size)[:args.batch_size]
        context_nodes = []
        for walk in walks[idxs]:
            for node in walk:
                context_nodes.append(np.random.choice(self.dict_neighbors[node], args.num_neighbors, replace=True))
        return torch.from_numpy(walks[idxs]).to(device), torch.from_numpy(np.array(context_nodes)).view(-1).to(device)

batch_loader = Batch_Loader_RW()

print("Loading data... finished!")

model = SANNE(feature_dim_size=feature_dim_size, ff_hidden_size=args.ff_hidden_size,num_heads=args.num_heads,
                dropout=args.dropout, num_self_att_layers=args.num_self_att_layers,
                vocab_size=vocab_size, sampled_num=args.sampled_num,initialization=features_matrix,
                num_neighbors=args.num_neighbors, device=device).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # Adagrad?
num_batches_per_epoch = int((data_size - 1) / args.batch_size) + 1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_batches_per_epoch, gamma=0.1)

def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    for _ in range(num_batches_per_epoch):
        input_x, input_y = batch_loader()
        optimizer.zero_grad()
        logits = model(input_x, input_y)
        loss = torch.sum(logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    return total_loss

def evaluate(epoch, acc_write):
    model.eval() # Turn on the evaluation mode
    with torch.no_grad():
        # evaluating
        node_embeddings = model.ss.weight
        node_embeddings = node_embeddings.data.cpu().numpy()
        idxs_10_data_splits = open("../data/"+args.dataset+'.10sampledtimes', 'rb')
        # The evaluation process for each data split is to take the best score on the validation set from all hyper-parameter settings (to give the final score on the test set).
        for fold_idx in range(10):
            train_idxs, train_labels, val_idxs, val_labels, test_idxs, test_labels = cPickle.load(idxs_10_data_splits)

            train_embs = node_embeddings[list(train_idxs)]
            val_embs = node_embeddings[list(val_idxs)]
            test_embs = node_embeddings[list(test_idxs)]

            cls = LogisticRegression(solver="liblinear", tol=0.001)
            cls.fit(train_embs, train_labels)
            val_acc = cls.score(val_embs, val_labels)
            test_acc = cls.score(test_embs, test_labels)
            print('epoch ', epoch, ' fold_idx ', fold_idx, ' val_acc ', val_acc*100.0, ' test_acc ', test_acc*100.0)
            acc_write.write('epoch ' + str(epoch) + ' fold_idx ' + str(fold_idx) + ' val_acc ' + str(val_acc*100.0) + ' test_acc ' + str(test_acc*100.0) + '\n')

    return acc_write

"""main process"""
import os
out_dir = os.path.abspath(os.path.join(args.run_folder, "../runs_pytorch_SANNE", args.model_name))
print("Writing to {}\n".format(out_dir))
# Checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
acc_write = open(checkpoint_prefix + '_acc.txt', 'w')

cost_loss = []
for epoch in range(1, args.num_epochs + 1):
    epoch_start_time = time.time()
    train_loss = train()
    cost_loss.append(train_loss)
    print(train_loss)
    acc_write = evaluate(epoch, acc_write)
    if epoch > 5 and cost_loss[-1] > np.mean(cost_loss[-6:-1]):
        scheduler.step()

acc_write.close()