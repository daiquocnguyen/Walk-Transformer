import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import pickle as cPickle

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    return features, labels

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

'''Generate a full list of edges'''
def generateTransEdgeList(file):
    transGraph = cPickle.load(open('./data/ind.' + file + '.graph', 'rb'), encoding='latin1')
    wri = open('./data/' + file + '.Full.edgelist', 'w')
    for tmp in transGraph:
        for _tmp in transGraph[tmp]:
            wri.write(str(tmp) + ' ' + str(_tmp))
            wri.write('\n')
    wri.close()

def creatBinaryTest_EdgePair(file):
    transGraph = cPickle.load(open('./data/ind.' + file + '.graph', 'rb'), encoding='latin1')
    lstkeys = list(transGraph.keys())
    wri = open('./graph/' + file + '.edge.binary.txt', 'w')
    for tmp in transGraph:
        tmpLst = set([x for x in lstkeys if x not in set(transGraph[tmp]) and x != tmp])
        for _tmp in transGraph[tmp]:
            wri.write(str(tmp) + ' ' + str(_tmp) + ' 1')
            wri.write('\n')
            idxsample = np.random.choice(tmpLst, 1, replace=False)
            wri.write(str(tmp) + ' ' + str(idxsample[0]) + ' -1')
            wri.write('\n')
    wri.close()

def getLabelList(idx, labels):
    tmpLst = []
    for tmp in labels[idx]:
        tmp1 = np.array(tmp).astype(int)
        try:
            tmpLst.append(list(tmp1).index(1))
        except:
            tmpLst.append(-1) #for citeseer dataset (there are some isolated nodes in the graph)
    return tmpLst

def sampleUniformRand(file):
    _, labels = load_data(file)

    lstLabels = getLabelList(range(len(labels)), labels)
    tmpdict = {}
    for i in range(len(labels)):
        if lstLabels[i] != -1:
            if lstLabels[i] not in tmpdict:
                tmpdict[lstLabels[i]] = []
            tmpdict[lstLabels[i]].append(i)

    idx_train = []
    for tmp in tmpdict:
        idx_train += list(np.random.choice(tmpdict[tmp], 20, replace=False))

    remainIdx = [idx for idx in range(len(labels)) if idx not in set(idx_train) and lstLabels[idx] != -1]

    idx_val = list(np.random.choice(remainIdx, 1000, replace=False))

    remainIdx = [idx for idx in set(remainIdx) if idx not in set(idx_val) and lstLabels[idx] != -1]

    idx_test = list(np.random.choice(remainIdx, 1000, replace=False))

    return idx_train, getLabelList(idx_train, labels), \
           idx_val, getLabelList(idx_val, labels), idx_test, getLabelList(idx_test, labels)

def generateInductiveEdgeList(file):
    transGraph = cPickle.load(open('./data/ind.' + file + '.graph', 'rb'), encoding='latin1')
    tmpdata = open(file + '.10sampledtimes', 'rb')
    for idx in range(10):
        _, _, _, _, idx_test, _ = cPickle.load(tmpdata)
        listTest = set(idx_test)
        wri = open('./data/' + file + '.ind.edgelist' + str(idx+1), 'w')
        for tmp in transGraph:
            if tmp not in listTest:
                for _tmp in transGraph[tmp]:
                    if _tmp not in listTest:
                        wri.write(str(tmp) + ' ' + str(_tmp))
                        wri.write('\n')
        wri.close()

def sampleUniformRand100(file):
    _, labels = load_data(file)

    lstLabels = getLabelList(range(len(labels)), labels)
    tmpdict = {}
    for i in range(len(labels)):
        if lstLabels[i] != -1:
            if lstLabels[i] not in tmpdict:
                tmpdict[lstLabels[i]] = []
            tmpdict[lstLabels[i]].append(i)

    idx_train = []
    tmpremaindict = {}
    for tmp in tmpdict:
        _lstidx = list(np.random.choice(tmpdict[tmp], 20, replace=False))
        idx_train += _lstidx
        tmpremaindict[tmp] = []
        for ii in tmpdict[tmp]:
            if ii not in _lstidx:
                if lstLabels[ii] != -1:
                    tmpremaindict[tmp].append(ii)

    remainIdx = [idx for idx in range(len(labels)) if idx not in set(idx_train) and lstLabels[idx] != -1]

    idx_val = []
    for tmp in tmpdict:
        idx_val += list(np.random.choice(tmpremaindict[tmp], 30, replace=False))

    remainIdx = [idx for idx in set(remainIdx) if idx not in set(idx_val) and lstLabels[idx] != -1]

    return idx_train, getLabelList(idx_train, labels), \
           idx_val, getLabelList(idx_val, labels), remainIdx, getLabelList(remainIdx, labels)


if __name__ == "__main__":
    #generateInductiveEdgeList('pubmed')
    pass