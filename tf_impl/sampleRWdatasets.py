import argparse
import numpy as np
import networkx as nx
import node2vec
import scipy.io as sio
from scipy.sparse import issparse
import pickle as cPickle
from utils import *

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Generate datasets.")
    parser.add_argument('--input', nargs='?', default='graph/karateRW.Full.edgelist', help='Input graph path')
    parser.add_argument('--output', nargs='?', default='graph/karateRW.pickle', help='Embeddings path')
    parser.add_argument('--walk-length', type=int, default=8, help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walks', type=int, default=128, help='Number of walks per source. Default is 10.')
    parser.add_argument('--p', type=float, default=1, help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=1, help='Inout hyperparameter. Default is 1.')
    parser.add_argument('--weighted', dest='weighted', action='store_true', help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)
    parser.add_argument('--directed', dest='directed', action='store_true', help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)
    return parser.parse_args()

args = parse_args()
print(args)

def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G

def sampleRandomWalks(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph()
    G = node2vec.Graph(nx_G, args.directed, 1, 1) #DeepWalk
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    walks = np.array(walks)

    with open(args.output, 'wb') as f:
        cPickle.dump(walks, f)

#for pos, ppi, blogcatalog
def generate_Edge_list(mat_file, output, variable_name="network", undirected=True):
    mat_varables = sio.loadmat(mat_file)
    mat_matrix = mat_varables[variable_name]
    return generateEdgelist(mat_matrix, output, undirected)

def generateEdgelist(x, output='graph/blogcatalog.edgelist', undirected=True):
    if issparse(x):
        wri = open(output, 'w')
        cx = x.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            #print(j, i)
            wri.write(str(j) + ' ' + str(i) + '\n')
        wri.close()
    else:
      raise Exception("Dense matrices not yet supported.")

if __name__ == "__main__":
    sampleRandomWalks(args)
