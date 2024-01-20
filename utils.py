# coding=utf-8
# Author: Jung
# Time: 2022/10/27 21:23
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import cosine_similarity
import dgl
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import NullFormatter
from sklearn import manifold
import community
import random
import pandas as pd
"""

    UTILS FOR MY RESEARCH

                        BEST J.

"""

""" 对称归一化邻接矩阵 D^(-0.5) @ A @ D^(-0.5) """
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

""" 对称归一化特征矩阵"""
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

""" 获得训练集, 测试集, 验证集. 输入为标签 """
def get_mask(y, train_ratio=0.6, test_ratio=0.2, device=None):
    if device is None:
        device = torch.device("cpu")
    train_indexes = list()
    test_indexes = list()
    val_indexes = list()
    npy = y.cpu().numpy()

    def get_sub_mask(sub_x_indexes):
        np.random.shuffle(sub_x_indexes)
        sub_train_count = int(len(sub_x_indexes) * train_ratio)
        sub_test_count = int(len(sub_x_indexes) * test_ratio)
        sub_train_indexes = sub_x_indexes[0:sub_train_count]
        sub_test_indexes = sub_x_indexes[sub_train_count:sub_train_count + sub_test_count]
        sub_val_indexes = sub_x_indexes[sub_train_count + sub_test_count:]
        return sub_train_indexes, sub_test_indexes, sub_val_indexes

    def flatten_np_list(np_list):
        total_size = sum([len(item) for item in np_list])
        result = np.ndarray(shape=total_size)
        last_i = 0
        for item in np_list:
            result[last_i:last_i + len(item)] = item
            last_i += len(item)
        return np.sort(result)
    # np.unique: 去除重复的数值
    for class_id in np.unique(npy):
        indexes = np.argwhere(npy == class_id).flatten().astype(int) # 获取ID为对应label的节点下标
        m, n, q = get_sub_mask(indexes)
        train_indexes.append(m)
        test_indexes.append(n)
        val_indexes.append(q)
    train_indexes = torch.LongTensor(flatten_np_list(train_indexes)).to(device)
    test_indexes = torch.LongTensor(flatten_np_list(test_indexes)).to(device)
    val_indexes = torch.LongTensor(flatten_np_list(val_indexes)).to(device)
    return train_indexes, test_indexes, val_indexes

""" 构建K近邻图. """
def knn_graph(feat, topk, weight = False, loop = True):
    """
    :param feat: 特征矩阵
    :param topk: 选取最相似的k个节点构建图
    :param weight: 是否保留节点权重（相似度）,默认为False
    :param loop: 是否构建自环
    :return: dgl_matrix
    """
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

    sp_matrix = sp.csr_matrix(sim_matrix)
    dgl_matrix = dgl.from_scipy(sp_matrix)
    if loop is True:
        dgl_matrix = dgl.add_self_loop(dgl_matrix)
    return dgl_matrix


def compute_nmi(labels, pred):
    return metrics.normalized_mutual_info_score(labels, pred)

def compute_ac(labels, pred):
    return metrics.accuracy_score(labels, pred)

def computer_f1(labels, pred):
    return metrics.f1_score(labels, pred, average='macro')

def computer_ari(labels, pred):
    return metrics.adjusted_rand_score(labels, pred)

""" 计算NMI, AC, F1, ARI"""
def computer_metrics(pred, labels):
    NMI = compute_nmi(labels, pred)
    ACC = compute_ac(labels, pred)
    F1 = computer_f1(labels, pred)
    ARI = computer_ari(labels, pred)
    return NMI, ACC, F1, ARI


""" 绘制社区对角线图, 对角线越密集社区结构越好"""
def plot_sparse_clustered_adjacency(A, num_coms, z, o, ax=None, markersize=0.25):
    """
    :param A: adjacency(ndarray)
    :param num_coms:  number of communities
    :param z: pred(ndarray)
    :param o: np.argsort(z)
    :param ax:
    :param markersize:
    :return:
    """
    if ax is None:
        ax = plt.gca()

    colors = sns.color_palette('hls', num_coms)
    sns.set_style('white')

    crt = 0
    for idx in np.where(np.diff(z[o]))[0].tolist() + [z.shape[0]]:
        ax.axhline(y=idx, linewidth=0.5, color='black', linestyle='--')
        ax.axvline(x=idx, linewidth=0.5, color='black', linestyle='--')
        crt = idx + 1

    ax.spy(A[o][:, o], markersize=markersize)
    ax.tick_params(axis='both', which='both', labelbottom='off', labelleft='off', labeltop='off')

""" 应用t-sne可视化社区 """
def visualization(epoch, emb, labels):
    """

    :param epoch: 无关紧要参数, 主要为了存储文件设置的编号
    :param emb: embedding
    :param labels: 标签
    :return:
    """
    # 具体参数可百度 https://blog.csdn.net/weixin_44387515/article/details/116117532
    tsne = manifold.TSNE(n_components=2, random_state=826, early_exaggeration = 50,init='pca', perplexity=50).fit_transform(
        emb.detach().numpy())
    tsne_min, tsne_max = tsne.min(0), tsne.max(0)
    tsne_norm = (tsne - tsne_min) / (tsne_max - tsne_min)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 设置节点颜色
    colors = ['darkorange', 'cornflowerblue', 'lightgreen', 'red', 'forestgreen','royalblue','c','lightcoral','wheat','cadetblue','m']
    xx = tsne_norm[:, 0]
    yy = tsne_norm[:, 1]

    for i in range(len(np.unique(labels))):
        ax.scatter(xx[labels == i], yy[labels == i], color=colors[i], s=10)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.savefig("vis/"+str(epoch)+".pdf")
    # plt.legend(loc='best', scatterpoints=1, fontsize=5)
    # plt.show()

""" 计算模块度  """
def modularity(adj: np.array, pred: np.array):
    """
    非重叠模块度
    :param adj: 邻接矩阵
    :param pred: 预测社区标签
    :return:
    """
    graph = nx.from_numpy_matrix(adj)
    part = pred.tolist()
    index = range(0, len(part))
    dic = zip(index, part)
    part = dict(dic)
    modur = community.modularity(part, graph)
    return modur

def swap_node_attr(g, feat, ratio: float):
    """
    以一定的比例交换节点属性
    :param g: dgl.DGLGraph
    :param ratio: 交换比例， [0, 1]
    :return:
    """
    n = g.num_nodes()  # 节点数
    num_swap = int(n * ratio * 0.5)
    for i in range(num_swap):
        a = random.randint(0, n - 1)
        b = random.randint(0, n - 1)
        if a == b:
            continue
        feat[[a, b], :] = feat[[b, a], :]
    return feat

def density(k, labels, A):
    communities = np.zeros(shape=(A.shape[0], k)) # n * k
    communities[range(A.shape[0]), labels] = 1
    communities = communities.dot(communities.T)
    row, col = np.diag_indices_from(communities)
    communities[row, col] = 0
    _density = communities * A
    return (_density.sum().sum() / (A.sum().sum())) * 0.5

# 备份
# def calculate_entropy(k, pred_labels, num_nodes, feat):
#     """
#     :param k: 社区个数
#     :param pred_labels: 预测社区
#     :param num_nodes: 节点的个数
#     :param feat: 节点属性
#     :return:
#     """
#     # 初始化两个矩阵
#     label_assemble = []
#     label_atts = []
#     for l in range (k):
#         label_assemble.append([])
#         label_atts.append([])
#     # 节点i属性社区j
#     for i, element in enumerate(pred_labels):
#         # 在社区j中存入节点i
#         label_assemble[element].append(i)
#
#     # 遍历每个社区中的节点
#     for i, node in enumerate(label_assemble):
#         # 遍历第i号社区中的所有节点
#         for u in node:
#             # 如果节点u的属性存在，则将属性j存入
#             label_atts[i] = label_atts[i] + [j for j in range(feat.shape[1]) if feat[u][j] != 0]
#     # 构建一个新矩阵p，大小为 k × m
#     p = np.zeros(shape=(k, feat.shape[1]))
#     # 遍历每个社区中具有的属性
#     for i, comty in enumerate(label_atts):
#         # 统计不同属性的个数,e.g., (10,20)表示属性10有20个
#         res = pd.value_counts(comty).sort_index()
#         # 统计社区i中总的属性数
#         num = len(label_atts[i])
#         # k / num 表示社区i中具有属性j的比重
#         for j, k in res.items():
#             ent = -(k / num) * np.log2(k/num)
#             p[i][j] = ent
#     __entropy = 0
#     # 遍历
#     for i, j in enumerate(p):
#         # len(label_assemble[i])表示统计社区i中节点的个数 / 总节点数目 * 社区i总节各节点所占比例的和(p[i].sum())
#         res = ( len(label_assemble[i]) / num_nodes) * p[i].sum()
#         __entropy += res
#
#     return __entropy


def calculate_entropy(k, pred_labels, feat):
    """
    :param k: 社区个数
    :param pred_labels: 预测社区
    :param num_nodes: 节点的个数
    :param feat: 节点属性
    :return:
    """
    # 初始化两个矩阵

    num_nodes = feat.shape[0]

    label_assemble = np.zeros(shape=(num_nodes, k))
    label_atts = np.zeros(shape=(k, feat.shape[1]))

    label_assemble[range(num_nodes), pred_labels] = 1
    label_assemble = label_assemble.T

    # 遍历每个社区
    for i in range(k):
        # 如果社区中的值大于0，则获得索引
        node_indx = np.where(label_assemble[i] > 0)
        # 获得索引下的所有属性
        node_feat = feat[node_indx]
        label_atts[i] = node_feat.sum(axis=0) # 向下加和

    __count_attrs = label_atts.sum(axis=1)
    __count_attrs = __count_attrs[:,np.newaxis]
    _tmp = label_atts / (__count_attrs + 1e-10)
    p = (_tmp) * - (np.log2(_tmp + 1e-10))

    p = p.sum(axis=1)
    label_assemble = label_assemble.sum(axis=1)
    __entropy = (label_assemble / num_nodes) * p
    return __entropy.sum()

if __name__ == "__main__":
    # 测试
    pred_label = [0, 1 ,1, 3 ,4 ,2 ,2 ,1 ,3 ,0]
    num = 10
    feat = [
        [1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
    feat  = np.array(feat)
    res = calculate_entropy(5, pred_label, num, feat)
    print(res)