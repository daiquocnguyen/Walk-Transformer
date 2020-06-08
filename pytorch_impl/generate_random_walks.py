import torch
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

import numpy as np
np.random.seed(123)
import networkx as nx
import node2vec


def read_graph(input, weighted=False, directed=False):
    '''
    Reads the input network in networkx.
    '''
    if weighted:
        G = nx.read_edgelist(input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    return G

def generate_random_walks(input, num_walks, walk_length):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph(input)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)  #DeepWalk
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)

    return np.array(walks)
