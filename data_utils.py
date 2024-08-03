
import torch
from sklearn.metrics import roc_auc_score,f1_score
import dgl
import random
import GCL.augmentors as Aug

from GCL.augmentors.functional import dropout_adj
from GCL.augmentors.functional import sort_edge_index
EOS = 1e-10

from dgl.nn import EdgeWeightNorm
norm = EdgeWeightNorm(norm='both')


def gen_dgl_graph(index1, index2, edge_w=None, ndata=None):
    g = dgl.graph((index1, index2),num_nodes=ndata.shape[0])
    if edge_w is not None:
        g.edata['w'] = edge_w
    if ndata is not None:
        g.ndata['feature'] = ndata
    return g


def normalize(edge_index):
    """ normalizes the edge_index
    """
    adj_t = edge_index.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t




def normalize_adj(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. /(torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. /(torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size()).coalesce()


def eval_func(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred1 = y_pred.softmax(1)[:,1].detach().cpu().numpy()
    auc = roc_auc_score(y_true, y_pred1)
    f1 = f1_score(y_true, y_pred.softmax(1).data.cpu().numpy().argmax(axis=1), average="macro")
    return auc,f1



@torch.no_grad()
def evaluate_whole_graph( model,datasets_te):
    model.eval()
    auc_list=[]
    for i in range(len(datasets_te)):
        train_out = model.inference(datasets_te[i])
        train_auc,f1 = eval_func(datasets_te[i].ndata['label'], train_out)
        auc_list.append(train_auc)

    return auc_list




@torch.no_grad()
def evaluate_graph( model, dataset_tr):
    model.eval()
    train_out = model.inference(dataset_tr)
    train_auc,f1 = eval_func(dataset_tr.ndata['label'], train_out)


    return train_auc,f1




def graph_aug(args,dataset):
    g_list = []
    x = dataset.ndata['feature']
    edge_index = torch.cat((dataset.all_edges()[0].unsqueeze(0), dataset.all_edges()[1].unsqueeze(0)), dim=0)
    for k in range(args.K):
        x_k, edge_index_k = Graph_Editer(x, edge_index,dataset.ndata['label'])
        graph_k = dgl.graph((edge_index_k[0], edge_index_k[1]), num_nodes=x_k.shape[0])
        graph_k = dgl.add_self_loop(graph_k)
        graph_k.ndata['feature']=x_k
        g_list.append(graph_k)

    return g_list

def Graph_Editer(x, edge_index,labels):

    normal_node=torch.where(labels == 0)[0]
    abnormal_node = torch.where(labels == 1)[0]

    rate_feature= random.uniform(0, 0.5)
    rate_normal_normal_remove_edge=random.uniform(0, 1)
    rate_normal_normal_add_edge = random.uniform(0, 1)

    rate_normal_abnormal_remove_edge = random.uniform(0, 1)
    rate_normal_abnormal_add_edge = random.uniform(0, 1)

    rate_abnormal_abnormal_remove_edge = random.uniform(0, 1)
    rate_abnormal_abnormal_add_edge = random.uniform(0, 1)


    normal_index0 = torch.isin(edge_index[0], normal_node)
    normal_index1 = torch.isin(edge_index[1], normal_node)

    normal_index = normal_index0 &  normal_index1

    normal_normal_edge=edge_index[0:,normal_index]

    abnormal_index0 = torch.isin(edge_index[0], abnormal_node)
    abnormal_index1 = torch.isin(edge_index[1], abnormal_node)

    abnormal_index = abnormal_index0 & abnormal_index1

    abnormal_abnormal_edge = edge_index[0:, abnormal_index]

    normal_abnormal_index=(abnormal_index0 & normal_index1) | (abnormal_index1 & normal_index0)

    normal_abnormal_edge = edge_index[0:, normal_abnormal_index]


    normal_normal_new, edge_weights = dropout_adj(normal_normal_edge, edge_attr=None, p=rate_normal_normal_remove_edge)
    abnormal_abnormal_new, edge_weights = dropout_adj(abnormal_abnormal_edge, edge_attr=None, p=rate_abnormal_abnormal_remove_edge)
    normal_abnormal_new, edge_weights = dropout_adj(normal_abnormal_edge, edge_attr=None, p=rate_normal_abnormal_remove_edge)


    normal_normal_new = add_edge(normal_normal_new,normal_normal_edge.shape[1], normal_node,abnormal_node,ratio=rate_normal_normal_add_edge,flag=0)
    abnormal_abnormal_new = add_edge(abnormal_abnormal_new,abnormal_abnormal_edge.shape[1], normal_node,abnormal_node,ratio=rate_abnormal_abnormal_add_edge,flag=1)
    normal_abnormal_new = add_edge(normal_abnormal_new,normal_abnormal_edge.shape[1],normal_node,abnormal_node, ratio=rate_normal_abnormal_add_edge,flag=2)

    edge_index = torch.cat([normal_normal_new, abnormal_abnormal_new,normal_abnormal_new], dim=1)
    edge_index = sort_edge_index(edge_index)
    aug = Aug.Compose([Aug.FeatureMasking(pf=rate_feature)])
    x_aug, edge_index_aug,edge_weight_aug = aug(x, edge_index)
    return x_aug, edge_index



def add_edge(edge_index: torch.Tensor, num_edges1,normal_node: torch.Tensor,abnormal_node: torch.Tensor, ratio: float,flag: int) -> torch.Tensor:
    num_edges = edge_index.size()[1]
    num_add = int(num_edges1 * ratio)
    if flag==0 or flag==1:
        index=torch.randint(0, num_edges - 1, size=(2,num_add)).to(edge_index.device)
        new_edge_index = torch.cat((edge_index[0][index[0]].unsqueeze(0),edge_index[1][index[1]].unsqueeze(0)),dim=0).to(edge_index.device)
        edge_index = torch.cat([edge_index, new_edge_index], dim=1)
        edge_index = sort_edge_index(edge_index)
    else:
        index = torch.randint(0, num_edges - 1, size=(2, 3*num_add)).to(edge_index.device)
        abnormal_index0 = torch.isin(edge_index[0][index[0]], abnormal_node)
        abnormal_index1 = torch.isin(edge_index[1][index[1]], abnormal_node)

        normal_index0 = torch.isin(edge_index[0][index[0]], normal_node)
        normal_index1 = torch.isin(edge_index[1][index[1]], normal_node)

        index_ok=abnormal_index0 & normal_index1 | abnormal_index1 & normal_index0
        index=index[0:, index_ok][0:num_add]
        new_edge_index = torch.cat((edge_index[0][index[0]].unsqueeze(0), edge_index[1][index[1]].unsqueeze(0)),dim=0).to(edge_index.device)
        edge_index = torch.cat([edge_index, new_edge_index], dim=1)
        edge_index = sort_edge_index(edge_index)

    return edge_index




def graph_aug1(args,dataset):
    g_list = []
    x = dataset.ndata['feature']
    edge_index = torch.cat((dataset.all_edges()[0].unsqueeze(0), dataset.all_edges()[1].unsqueeze(0)), dim=0)
    for k in range(args.K):
        x_k, edge_index_k = Graph_Editer1(x, edge_index)
        graph_k = dgl.graph((edge_index_k[0], edge_index_k[1]), num_nodes=x_k.shape[0])
        graph_k = dgl.add_self_loop(graph_k)
        graph_k.ndata['feature'] = x_k

        g_list.append(graph_k)

    return g_list




def Graph_Editer1(x, edge_index):
    rate= random.uniform(0, 0.5)
    aug = Aug.Compose([Aug.EdgeRemoving(pe=rate), Aug.FeatureMasking(pf=rate)])
    x_aug, edge_index_aug,edge_weight_aug = aug(x, edge_index)
    return x_aug, edge_index_aug
