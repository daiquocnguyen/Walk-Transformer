import logging
from gensim.models.word2vec import Word2Vec
from gensim.models import doc2vec
import sys
import numpy as np
import pickle as cPickle
from utils import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser("TransfG", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--data", default="./data/", help="Data sources.")
parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--name", default="cora", help="Name of the dataset.")
args = parser.parse_args()

logger = logging.getLogger("doc2vec")

def getDocs():
    features, _ = load_data(str(args.name).split('.')[0])
    features_matrix, spars = preprocess_features(features)
    features_matrix = np.array(features_matrix, dtype=np.float32)
    print(features_matrix[0])

    with open('./data/' + args.name + '.content.txt', 'w') as f:
        for tmp in features_matrix:
            tmpDocs = []
            for i in range(len(tmp)):
                if tmp[i] > 0.:
                    tmpDocs.append(str(i))
            print(' '.join(tmpDocs), file=f)
            #print(tmpDocs)

def label_sentences(corpus):
    labeled = []
    for i, v in enumerate(corpus):
        label = 'id_' + str(i)
        labeled.append(doc2vec.LabeledSentence(v.split(), [label]))
    return labeled

def get_vectors(doc2vec_model, corpus_size, vectors_size):
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = 'id_' + str(i)
        vectors[i] = doc2vec_model.docvecs[prefix]
    return vectors

def runDoc2Vec():
    corpusLabeled = label_sentences(open('./data/' + args.name + '.content.txt', 'r').readlines())
    for i in range(128, 1500, 64):
        d2v = doc2vec.Doc2Vec(corpusLabeled, dm=0, vector_size=128, alpha=0.01, workers=20, epochs=i, min_count=0, sample=1e-5)
        d2v.save('../runs_TransfG/' + args.name + ".d2v.128d.model" + str(i))

if __name__ == '__main__':

    #runDoc2Vec()

    # d2v = doc2vec.Doc2Vec.load('./data/pubmed.d2v.128d.model')
    # features_matrix = get_vectors(d2v, d2v.corpus_count, d2v.vector_size)
    # features_matrix = np.array(features_matrix, dtype=np.float32)
    #
    # with open('./data/pubmed.128d.feature.pickle', 'wb') as f:
    #     cPickle.dump(features_matrix, f)

    pass
