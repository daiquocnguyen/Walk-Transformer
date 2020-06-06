import warnings
warnings.filterwarnings('ignore')
import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict
from six import iteritems
import pickle as cPickle
import statistics
import numpy as np
from sklearn.linear_model import LogisticRegression
from utils import *


def main():
    parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--input', nargs='?', default='cora', help='Input')
    parser.add_argument('--output', nargs='?', default='cora', help='output')
    parser.add_argument('--tmpString', nargs='?', default='cora', help='')
    parser.add_argument("--run_folder", default="../runs_SANNE_ind/")
    parser.add_argument("--idx_time", default=1, type=int, help="From 1 to 10")
    args = parser.parse_args()

    lstfiles = []
    for root, dirs, files in os.walk(args.run_folder):
        for file in files:
            embeddings_file = os.path.join(root, file)
            if args.output not in str(embeddings_file):
                continue
            if args.input not in str(embeddings_file):
                continue
            if args.tmpString not in str(embeddings_file):
                continue
            lstfiles.append(embeddings_file)
    lstfiles = sorted(lstfiles, key=str.lower)

    tmpwrite = open('evaluation_ind_' + args.input + args.output + args.tmpString + '_accuracy.txt', 'w')
    tmpdata = open("data/" + args.output + '.10sampledtimes', 'rb')
    acc_resuls = []
    val_resuls = []

    for _ in range(args.idx_time):
        trainset, train_y, valset, val_y, testset, test_y = cPickle.load(tmpdata)

    tmpvalrel = 0.0
    tmpaccrel = 0.0
    tmpFile = ''

    for embeddings_file in lstfiles:
        try:
            with open(embeddings_file, 'rb') as f:
                features_matrix = cPickle.load(f)
                #features_matrix = cPickle.load(f)
                # features_matrix = cPickle.load(f)
        except:
            continue

        train_x = features_matrix[list(trainset)]
        val_x = features_matrix[list(valset)]
        test_x = features_matrix[list(testset)]

        cls = LogisticRegression(tol=0.001)
        cls.fit(train_x, train_y)
        ACC = cls.score(val_x, val_y)
        if tmpvalrel < ACC:
            tmpvalrel = ACC
            ACC = cls.score(test_x, test_y)
            tmpaccrel = ACC
            tmpFile = embeddings_file

    val_resuls.append(tmpvalrel)
    acc_resuls.append(tmpaccrel)
    tmpwrite.write(str(tmpFile) + "  -- eval:    " + str(tmpvalrel) + "    -- test:   " + str(tmpaccrel) + '\n')

    tmpwrite.close()
    tmpdata.close()


if __name__ == "__main__":
    sys.exit(main())
