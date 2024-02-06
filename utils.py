import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import get_train_val_test
from torch_geometric.utils import train_test_split_edges
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from deeprobust.graph.utils import *
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from torch_geometric.datasets import Planetoid


def get_dataset(name, normalize_features=False, transform=None, if_dpr=True):


    path = osp.join(osp.dirname(osp.realpath(__file__)), '/cpfs01/shared/public/ztl/SFGC-main/data/', name)

    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, name)
    elif name in ['ogbn-arxiv','ogbn-arxiv-xrt']:
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    else:
        raise NotImplementedError


    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform


    dpr_data = Pyg2Dpr(dataset)
    if name in ['ogbn-arxiv','ogbn-arxiv-xrt']:
        feat, idx_train = dpr_data.features, dpr_data.idx_train
        feat_train = feat[idx_train]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat = scaler.transform(feat)
        dpr_data.features = feat

    return dpr_data



class Pyg2Dpr(Dataset):
    def __init__(self, pyg_data, **kwargs):

        try:
            splits = pyg_data.get_idx_split() 
        except:
            pass

        dataset_name = pyg_data.name 
        pyg_data = pyg_data[0] 
        n = pyg_data.num_nodes 

        if dataset_name in ['ogbn-arxiv','ogbn-arxiv-xrt']:
            pyg_data.edge_index = to_undirected(pyg_data.edge_index, pyg_data.num_nodes)

        self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),
            (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))

        self.features = pyg_data.x.numpy() 
        self.labels = pyg_data.y.numpy() 

        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1) 

        if hasattr(pyg_data, 'train_mask'):

            self.idx_train = mask_to_index(pyg_data.train_mask, n) 
            self.idx_val = mask_to_index(pyg_data.val_mask, n) 
            self.idx_test = mask_to_index(pyg_data.test_mask, n)
            self.name = 'Pyg2Dpr' 
        else:
            try:
                self.idx_train = splits['train']
                self.idx_val = splits['valid'] 
                self.idx_test = splits['test'] 
                self.name = 'Pyg2Dpr' 
            except:
                self.idx_train, self.idx_val, self.idx_test = get_train_val_test(
                        nnodes=n, val_size=0.1, test_size=0.8, stratify=self.labels)




def mask_to_index(index, size):

    all_idx = np.arange(size)  
    return all_idx[index]  

def index_to_mask(index, size):

    mask = torch.zeros((size, ), dtype=torch.bool)  
    mask[index] = 1  
    return mask  



class Transd2Ind:
    # transductive setting to inductive setting

    def __init__(self, dpr_data, keep_ratio):

        idx_train, idx_val, idx_test = dpr_data.idx_train, dpr_data.idx_val, dpr_data.idx_test
        adj, features, labels = dpr_data.adj, dpr_data.features, dpr_data.labels
        self.nclass = labels.max()+1
        self.adj_full, self.feat_full, self.labels_full = adj, features, labels
        self.idx_train = np.array(idx_train)
        self.idx_val = np.array(idx_val)
        self.idx_test = np.array(idx_test)

        if keep_ratio < 1:
            idx_train, _ = train_test_split(idx_train,
                                            random_state=None,
                                            train_size=keep_ratio,
                                            test_size=1-keep_ratio,
                                            stratify=labels[idx_train])

        self.adj_train = adj[np.ix_(idx_train, idx_train)]
        self.adj_val = adj[np.ix_(idx_val, idx_val)]
        self.adj_test = adj[np.ix_(idx_test, idx_test)]
        print('size of adj_train:', self.adj_train.shape)
        print('#edges in adj_train:', self.adj_train.sum())

        self.labels_train = labels[idx_train]
        self.labels_val = labels[idx_val]
        self.labels_test = labels[idx_test]

        self.feat_train = features[idx_train]
        self.feat_val = features[idx_val]
        self.feat_test = features[idx_test]

        self.class_dict = None
        self.samplers = None
        self.class_dict2 = None

    def retrieve_class(self, c, num=256):

        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s'%i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)[:num]

    def retrieve_class_sampler(self, c, adj, transductive, num, args=None):

        # print(num)
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx

        if args.nlayers == 1:
            sizes = [15]
        if args.nlayers == 2:
            sizes = [10, 5]
            # sizes = [-1, -1]
        if args.nlayers == 3:
            sizes = [15, 10, 5]
        if args.nlayers == 4:
            sizes = [15, 10, 5, 5]
        if args.nlayers == 5:
            sizes = [15, 10, 5, 5, 5]

        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                self.samplers.append(NeighborSampler(adj,
                                    node_idx=node_idx,
                                    sizes=sizes, batch_size=num,
                                    num_workers=12, return_e_id=False,
                                    num_nodes=adj.size(0),
                                    shuffle=True))
        batch = np.random.permutation(self.class_dict2[c])[:num]
        # out = self.samplers[c].sample(batch)
        out = self.samplers[c].sample(torch.from_numpy(batch).long())
        return out
    

    def compute_pagerank_scores(self,adj):
        # Convert the sparse adjacency matrix to a NetworkX graph
        import networkx as nx
        import numpy as np
        from scipy.sparse import coo_matrix
        # G = nx.from_scipy_sparse_matrix(adj)
        adj_matrix = adj.to_scipy(layout='csr')  

        G = nx.DiGraph(adj_matrix) 

        # pagerank_scores = nx.pagerank(G)
        # Calculate the PageRank scores using NetworkX
        pagerank_scores = nx.pagerank(G)

        # Convert the dictionary of PageRank scores to a NumPy array
        num_nodes = adj.size(0)
        pagerank_array = np.zeros(num_nodes)
        for node, score in pagerank_scores.items():
            pagerank_array[node] = score

        return pagerank_array

    
    def retrieve_class_sampler_pagerank(self, c, adj, transductive, num=256, args=None):
        import torch
        import torch_geometric.utils as pyg_utils
        # Initialize PageRank centrality during the first execution
        if self.pagerank_scores is None:
            self.pagerank_scores = self.compute_pagerank_scores(adj)  # Custom function to compute PageRank centrality

        sizes = []
        if args.nlayers == 1:
            sizes = [30]
        if args.nlayers == 2:
            if args.dataset in ['reddit', 'flickr']:
                if args.option == 0:
                    sizes = [15, 8]
                if args.option == 1:
                    sizes = [20, 10]
                if args.option == 2:
                    sizes = [25, 10]
            else:
                sizes = [10, 5]

        if self.class_dict2 is None:
            print(sizes)
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx_train = np.array(self.idx_train)
                    idx = idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train == i]
                self.class_dict2[i] = idx

        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                if len(node_idx) == 0:
                    continue

                self.samplers.append(NeighborSampler(adj,
                                                    node_idx=node_idx,
                                                    sizes=sizes, batch_size=num,
                                                    num_workers=8, return_e_id=False,
                                                    num_nodes=adj.size(0),
                                                    shuffle=True))

        
        pagerank_scores_c = self.pagerank_scores[self.class_dict2[c]]
        pagerank_scores_c = torch.from_numpy(pagerank_scores_c)  
        top_nodes_idx = torch.argsort(pagerank_scores_c, descending=True)[:num]
        batch = self.class_dict2[c][top_nodes_idx]
        
        out = self.samplers[c].sample(batch)
        return out

    def retrieve_class_sampler_mutil(self, c_s, adj, transductive, num=256, args=None):
        sizes = []
        if args.nlayers == 1:
            sizes = [30]
        if args.nlayers == 2:
            if args.dataset in ['reddit', 'flickr']:
                if args.option == 0:
                    sizes = [15, 8]
                if args.option == 1:
                    sizes = [20, 10]
                if args.option == 2:
                    sizes = [25, 10]
            else:
                sizes = [10, 5]

        if self.class_dict2 is None:
            print(sizes)
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx_train = np.array(self.idx_train)
                    idx = idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train == i]
                self.class_dict2[i] = idx

        if self.samplers is None:
            self.samplers = {}
            for i in range(0, self.nclass, len(c_s)):
                # print(i)
                # classes = min(self.nclass, i + len(c_s))
                for j in range(len(c_s)):
                    if j == 0:
                        node_idx = torch.LongTensor(self.class_dict2[i + j])
                    else:
                        node_idx = torch.cat([node_idx, torch.LongTensor(self.class_dict2[i + j])], dim=0)

                if len(node_idx) == 0:
                    continue

                self.samplers[i] = (NeighborSampler(adj,
                                                     node_idx=node_idx,
                                                     sizes=sizes, batch_size=num*len(c_s),
                                                     num_workers=8, return_e_id=False,
                                                     num_nodes=adj.size(0),
                                                     shuffle=True))
        number = c_s[0]

        batches = []

        for i in range(number, number + len(c_s)):
            samples = np.random.permutation(self.class_dict2[i])[:num]
            batches.append(samples)

        batch = np.concatenate(batches, axis=0)

        out = self.samplers[number].sample(batch)
        return out


def match_loss(gw_syn, gw_real, args, device):

    dis = torch.tensor(0.0).to(device)  

    if args.dis_metric == 'ctrl': 

        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis +=  combined_distance(args,gwr, gws,1- args.beta,args.beta)

    elif args.dis_metric == 'mse':  
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)
        
    elif args.dis_metric == 'norm': 
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += norm_dis(gwr, gws)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')  

    return dis


def combined_distance(args,gwr, gws, alpha=0.2, beta=0.8):
    shape = gwr.shape

    # TODO: output node!!!!
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0
    
    if args.dataset in ['ogbn-arxiv']:
        gradient_sum = torch.sum(torch.abs(gwr))
        threshold = 50
        if gradient_sum < threshold:
            distance = alpha * (1 - F.cosine_similarity(gwr, gws, dim=-1)) + beta * torch.norm(gwr - gws, dim=-1)
            return torch.sum(distance)
        else:
            dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
            return torch.sum(dis_weight)
    elif args.dataset in ['reddit']:
        gradient_sum = torch.sum(torch.abs(gwr))
        threshold = 50
        if gradient_sum < threshold:
            distance = alpha * (1 - F.cosine_similarity(gwr, gws, dim=-1)) + beta * torch.norm(gwr - gws, dim=-1)
            return torch.sum(distance)
        else:
            dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
            return torch.sum(dis_weight)
    else:
        cosine_similarity = F.cosine_similarity(gwr, gws, dim=-1)
        euclidean_distance = torch.norm(gwr - gws, dim=-1)

        distance = alpha * (1 - cosine_similarity) + beta * euclidean_distance

        return torch.sum(distance)

def calc_f1(y_true, y_pred, is_sigmoid):

    if not is_sigmoid:
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def evaluate(output, labels, args):

    data_graphsaint = ['yelp', 'ppi', 'ppi-large', 'flickr', 'reddit', 'amazon']
    if args.dataset in data_graphsaint:
        labels = labels.cpu().numpy()
        output = output.cpu().numpy()
        if len(labels.shape) > 1:
            micro, macro = calc_f1(labels, output, is_sigmoid=True)
        else:
            micro, macro = calc_f1(labels, output, is_sigmoid=False)
        print("Test set results:", "F1-micro= {:.4f}".format(micro),
                "F1-macro= {:.4f}".format(macro))
    else:
        loss_test = F.nll_loss(output, labels)
        acc_test = accuracy(output, labels)
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
    return


from torchvision import datasets, transforms
def get_mnist(data_path):

    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]


    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no        augmentation
    dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    class_names = [str(c) for c in range(num_classes)]

    labels = []
    feat = []

    for x, y in dst_train:
        feat.append(x.view(1, -1))
        labels.append(y)
    feat = torch.cat(feat, axis=0).numpy()

    from utils_graphsaint import GraphData

    adj = sp.eye(len(feat))
    idx = np.arange(len(feat))
    dpr_data = GraphData(adj-adj, feat, labels, idx, idx, idx)

    from deeprobust.graph.data import Dpr2Pyg

    return Dpr2Pyg(dpr_data)

def regularization(adj, x, eig_real=None):

    # fLf
    loss = 0
    # loss += torch.norm(adj, p=1)
    loss += feature_smoothing(adj, x)
    return loss

def maxdegree(adj):

    n = adj.shape[0]
    return F.relu(max(adj.sum(1))/n - 0.5)

def sparsity2(adj):
    n = adj.shape[0]
    loss_degree = - torch.log(adj.sum(1)).sum() / n
    loss_fro = torch.norm(adj) / n
    return 0 * loss_degree + loss_fro

def sparsity(adj):

    n = adj.shape[0]  
    thresh = n * n * 0.01  
    return F.relu(adj.sum() - thresh)  



def feature_smoothing(adj, X):

    adj = (adj.t() + adj) / 2  
    rowsum = adj.sum(1)  
    r_inv = rowsum.flatten()  
    D = torch.diag(r_inv)  

    r_inv = r_inv + 1e-8  
    r_inv = r_inv.pow(-1/2).flatten()  
    r_inv[torch.isinf(r_inv)] = 0.  
    r_mat_inv = torch.diag(r_inv) 

    L = r_mat_inv @ (D - adj) @ r_mat_inv  

    XLXT = torch.matmul(torch.matmul(X.t(), L), X) 

    return torch.trace(XLXT)  

def row_normalize_tensor(mx):

    rowsum = mx.sum(1)
    
    r_inv = rowsum.pow(-1).flatten()
    
    r_mat_inv = torch.diag(r_inv)
   
    mx = r_mat_inv @ mx
    return mx


