
import torch

import torch.nn as nn
import torch_geometric.datasets
import torchvision
from matplotlib import pyplot as plt
import torch_geometric.utils as tg_utils
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import networkx as nx

class SimpleGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, K, bias=True):
        super(SimpleGraphConvolution, self).__init__()
        self.layer = nn.Linear(in_features, out_features, bias)
        self.K = K

    def forward(self, data):
        x, edge_index = data
        if edge_index.shape[0] != x.shape[0]:  ## Compute S just once
            A = tg_utils.to_dense_adj(edge_index).squeeze()
            A_ = A + torch.eye(A.shape[-1])  ## Add self loop
            D_ = torch.sqrt(torch.sum(A_, dim=1) * torch.eye(A.shape[-1]))
            D__ = torch.inverse(D_)  ## very inefficient
            S = torch.matmul(torch.matmul(D__, A_), D__)
        else:
            S = edge_index
        S_k = torch.pow(S, self.K)
        H = torch.matmul(S_k, x)
        H_out = self.layer(H)
        return H_out, S_k



class SimpleGraphConvolutionalNetwork(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleGraphConvolutionalNetwork, self).__init__()
        self.model = nn.Sequential(SimpleGraphConvolution(in_features=in_features,
                                                           out_features=256, K=3))
        self.head = nn.Sequential(nn.Linear(in_features=256, out_features=out_features))

    def forward(self, x, edge_index):
        H, S = self.model((x, edge_index))
        out = self.head(H)
        return out
class GraphConvolutionalLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionalLayer, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
                                   nn.ReLU())
    def forward(self, data):
        x, edge_index = data
        if edge_index.shape[0] != x.shape[0]: ## Compute S just once
            A = tg_utils.to_dense_adj(edge_index).squeeze()
            A_ = A + torch.eye(A.shape[-1]) ## Add self loop
            D_ = torch.sqrt(torch.sum(A_, dim=1)*torch.eye(A.shape[-1]))
            D__ = torch.inverse(D_) ## very inefficient
            S = torch.matmul(torch.matmul(D__, A_), D__)
        else:
            S = edge_index
        H = torch.matmul(S, x)
        H_out = self.layer(H)
        return H_out, S



class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionalNetwork, self).__init__()
        self.model = nn.Sequential(GraphConvolutionalLayer(in_features=in_features,
                                                           out_features=16),
                                   GraphConvolutionalLayer(in_features=16,
                                                             out_features=16),
                                   GraphConvolutionalLayer(in_features=16,
                                                             out_features=16))
        self.head = nn.Sequential(nn.Linear(in_features=16, out_features=out_features))

    def forward(self, x, edge_index):
        H, S = self.model((x, edge_index))
        x = torch.matmul(S, H)
        out = self.head(x)
        return out


def visualize(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True,
                     node_color=color, cmap="Set2")
    plt.show()
'''

    Node Classification Task with Graph Convolutional Networks
'''
def test_1():
    '''
        KarateClub is a single Graph composed by
        - 34 nodes
        - 156 undirected and unweighted edges
        - Every node has a class of 4 associated
        - Every node has a feature vector of 1 element
    '''
    dataset = torch_geometric.datasets.KarateClub()
    dataloader = DataLoader(dataset)

    model = GraphConvolutionalNetwork(in_features=34, out_features=4)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_history = []
    n_epochs = 2000
    for n in tqdm(range(n_epochs)):
        for batch in dataloader:
            optim.zero_grad()
            x, edge_index = batch.x, batch.edge_index
            x = model(x, edge_index)
            loss = loss_fn(x, batch.y)
            loss.backward()
            optim.step()
            loss_history.append(loss.item())
    plt.plot(loss_history)
    plt.show()
    #data = torch_geometric.data.Data(x=dataset.data.x, edge_index=dataset.data.edge_index, n)

    g = torch_geometric.utils.to_networkx(dataset.data, to_undirected=True)
    visualize(g, dataset.data.y)


    output = model(dataset.data.x, dataset.data.edge_index)
    data = torch_geometric.data.Data(x=dataset.data.x, edge_index=dataset.data.edge_index)
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    visualize(g, color=torch.argmax(output, dim=1))
'''

    Node Classification Task with Graph Convolutional Networks
'''
def test_2():
    '''
            KarateClub is a single Graph composed by
            - 34 nodes
            - 156 undirected and unweighted edges
            - Every node has a class of 4 associated
            - Every node has a feature vector of 1 element
        '''
    dataset = torch_geometric.datasets.KarateClub()
    dataloader = DataLoader(dataset)

    model = SimpleGraphConvolutionalNetwork(in_features=34, out_features=4)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_history = []
    n_epochs = 2000
    for n in tqdm(range(n_epochs)):
        for batch in dataloader:
            optim.zero_grad()
            x, edge_index = batch.x, batch.edge_index
            x = model(x, edge_index)
            loss = loss_fn(x, batch.y)
            loss.backward()
            optim.step()
            loss_history.append(loss.item())
    plt.plot(loss_history)
    plt.show()
    # data = torch_geometric.data.Data(x=dataset.data.x, edge_index=dataset.data.edge_index, n)

    g = torch_geometric.utils.to_networkx(dataset.data, to_undirected=True)
    visualize(g, dataset.data.y)

    output = model(dataset.data.x, dataset.data.edge_index)
    data = torch_geometric.data.Data(x=dataset.data.x, edge_index=dataset.data.edge_index)
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    visualize(g, color=torch.argmax(output, dim=1))
if __name__ == '__main__':
    #test_1()
    test_2()
