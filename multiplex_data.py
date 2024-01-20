import numpy as np
import dgl
import pickle as pkl
from scipy import sparse
import torch
import scipy.io as scio
import networkx as nx
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def knn_graph(feat, topk, weight = False, loop = True):
    sim_feat = cosine_similarity(feat)
    sim_matrix = np.zeros(shape=(feat.shape[0], feat.shape[0]))

    inds = []
    for i in range(sim_feat.shape[0]):
        ind = np.argpartition(sim_feat[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    for i, vs in enumerate(inds):
        for v in vs:
            if v == i:
                pass
            else:
                if weight is True:
                    sim_matrix[i][v] = sim_feat[i][v]
                    sim_matrix[v][i] = sim_feat[v][i]
                else:
                    sim_matrix[i][v] = 1
                    sim_matrix[v][i] = 1

    sp_matrix = sparse.csr_matrix(sim_matrix)
    dgl_matrix = dgl.from_scipy(sp_matrix)
    if loop is True:
        dgl_matrix = dgl.add_self_loop(dgl_matrix)
    return dgl_matrix

def load_data(name: str):
    if name == "imdb5k":

        data = scio.loadmat("datasets/" + "imdb5k")
        adj_list = []
        label = data['label']
        attr = data['feature']
        num_nodes, feat_dim = attr.shape
        communities = label.shape[1]

        labels = np.argmax(label, axis=1)
        topo_mdm = dgl.from_scipy(sparse.csr_matrix(data['MDM']))
        topo_mam = dgl.from_scipy(sparse.csr_matrix(data['MAM']))

        graph = knn_graph(attr, 6, False)

        adj_list.append(topo_mdm)
        adj_list.append(topo_mam)
        adj_list.append(graph)
        attr = torch.from_numpy(attr).to(torch.float32)
        return num_nodes, feat_dim, communities, labels, adj_list, attr
    elif name == "amazon":
        data = pkl.load(open("datasets/" + name + ".pkl", 'rb'))
        adj_list = []
        label = data['label']
        attr = data['feature']
        num_nodes, feat_dim = attr.shape
        communities = label.shape[1]

        labels = np.argmax(label, axis=1)

        topo_ivi = dgl.from_scipy(sparse.csr_matrix(data['IVI']))
        topo_ibi = dgl.from_scipy(sparse.csr_matrix(data['IBI']))
        topo_ioi = dgl.from_scipy(sparse.csr_matrix(data['IOI']))

        graph = knn_graph(attr, 8, False)

        adj_list.append(topo_ivi)
        adj_list.append(topo_ibi)
        adj_list.append(topo_ioi)
        adj_list.append(graph)
        attr = torch.from_numpy(attr).to(torch.float32)

        return num_nodes, feat_dim, communities, labels, adj_list, attr
    elif name == "acm":
        data = scio.loadmat("datasets/" + name)
        # data = scio.loadmat("datasets/" + "ACM3025")
        adj_list = []
        label = data['label']
        attr = data['feature']
        num_nodes, feat_dim = attr.shape
        communities = label.shape[1]

        labels = np.argmax(label, axis=1)

        topo_plp = dgl.from_scipy(sparse.csr_matrix(data['PLP']))
        topo_pap = dgl.from_scipy(sparse.csr_matrix(data['PAP']))

        graph = knn_graph(attr, 2, False)




        adj_list.append(topo_plp)
        adj_list.append(topo_pap)
        adj_list.append(graph)
        attr = torch.from_numpy(attr).to(torch.float32)

        return num_nodes, feat_dim, communities, labels, adj_list, attr
    elif name == "dblp":
        data = scio.loadmat("datasets/" + name)
        adj_list = []
        label = data['label']
        attr = data['features']
        num_nodes, feat_dim = attr.shape
        communities = label.shape[1]

        labels = np.argmax(label, axis=1)

        topo_aptpa = dgl.from_scipy(sparse.csr_matrix(data['net_APTPA']))
        topo_apcpa = dgl.from_scipy(sparse.csr_matrix(data['net_APCPA']))
        topo_apa = dgl.from_scipy(sparse.csr_matrix(data['net_APA']))

        graph = knn_graph(attr, 10, False)
        adj_list.append(topo_aptpa)
        adj_list.append(topo_apcpa)
        adj_list.append(topo_apa)
        adj_list.append(graph)

        attr = torch.from_numpy(attr).to(torch.float32)
        return num_nodes, feat_dim, communities, labels, adj_list, attr
    elif name == "scholat":
        data = pkl.load(open("datasets/" + "scholat_multiplex.pkl", 'rb'))
        adj_list = []
        labels = data['label']
        attr = data['attr']
        num_nodes, feat_dim = attr.shape
        communities = len(np.unique(labels))


        topo_friends = dgl.from_scipy(data['friends'])
        topo_team = dgl.from_scipy(data['team'])
        topo_class = dgl.from_scipy(data['class'])


        graph = knn_graph(attr, 5, False)
        adj_list.append(topo_friends)
        adj_list.append(topo_team)
        adj_list.append(topo_class)
        adj_list.append(graph)

        attr = attr.to(torch.float32)
        return num_nodes, feat_dim, communities, labels, adj_list, attr
    else:
        print("ERROR")

if __name__ == "__main__":
    load_data("acm")