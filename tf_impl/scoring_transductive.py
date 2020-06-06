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
    parser.add_argument("--run_folder", default="../runs_SANNE_trans/")
    args = parser.parse_args()

    lstfiles = []
    for root, dirs, files in os.walk(args.run_folder):
        for file in files:
            embeddings_file = os.path.join(root, file)
            if args.input not in str(embeddings_file):
                continue
            if args.tmpString not in str(embeddings_file):
                continue
            lstfiles.append(embeddings_file)
    lstfiles = sorted(lstfiles, key=str.lower)

    tmpwrite = open('evaluation_trans_' + args.output + args.tmpString + '_accuracy.txt', 'w')
    tmpdata = open("data/" + args.output + '.10sampledtimes', 'rb')
    acc_resuls = []
    val_resuls = []

    for _ in range(10):
        tmpvalrel = 0.0
        tmpaccrel = 0.0
        tmpFile = ''

        trainset, train_y, valset, val_y, testset, test_y = cPickle.load(tmpdata)

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

    tmpmean = statistics.mean(val_resuls)
    tmpstd = statistics.stdev(val_resuls)
    print(tmpmean, tmpstd)
    tmpwrite.write("10 times randomly eval: " + str(tmpmean) + "+-" + str(tmpstd))
    tmpwrite.write('\n\n')

    tmpmean = statistics.mean(acc_resuls)
    tmpstd = statistics.stdev(acc_resuls)
    print(tmpmean, tmpstd)
    tmpwrite.write("10 times randomly test: " + str(tmpmean) + "+-" + str(tmpstd))
    tmpwrite.write('\n---------------------------------------\n\n')

    tmpwrite.close()
    tmpdata.close()


if __name__ == "__main__":
    sys.exit(main())
