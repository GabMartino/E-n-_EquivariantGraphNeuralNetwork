'''
    This wants to be a rough implementation of
    "E(n) Equivariant Graph Neural Networks" - Victor Garcia Satorras, Emiel Hoogeboom, Max Welling

    Gabriele Martino
'''
import networkx as nx
import torch
import torch.nn as nn
import torch_geometric as tg
import torch_geometric.data
from matplotlib import pyplot as plt
from tqdm import tqdm


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)

'''
    Equivariant Graph Convolutional Layer
    
    - m_ij = \phi(h_i, h_j, || x_i - x_j ||^2, aij)
    - m_i = \sum_N mij
    - h_i = \phi_h(h_i, m_i) # Output of the layer
    
    Following the paper it's possible to have 3 modalities
    1) Static
    2) Dynamic Positions of nodes ## This is the first proposal of the paper
        In this case we also have another output of the Layer:
        x_i = x_i + C \sum_N (x_i - x_j) \phi_x(m_ij)
    3) Dynamic Positions of nodes with momentum
        In this case we have a third output and a slight modification of the second
        v_i = \phi_v(h_i)v_i + C \sum_N (x_i - x_j) \phi_x(m_ij)
        
    ##### NOTICE IT'S NOT CLEAR FROM THE PAPER IF THE SUM HERE IS ACCORDING TO ALL THE DISTANCES IN THE GRAPH 
    #####OR ONLY THE NEIGHBOURS OF THE NODE
    -> For simplicity and following the message passing approach we'll just consider the neighbours
'''
class EGCL(nn.Module):

    def __init__(self, n_hidden_features, dim_coordinates=None, with_position=False, with_momentum=False):
        super(EGCL, self).__init__()
        '''
            We need to implement the 4 functions:
             \phi_e : equivariant function 
             \phi_x : function for the position update
             \phi_h : Final function
             \phi_v : velocity estimation function
        '''
        self.with_position = with_position
        self.with_momentum = with_momentum
        if with_momentum and not with_position:
            raise ValueError("It's not possible to have momentum if position is False")
        if with_position and dim_coordinates is None:
            raise ValueError("It's not possible to have dim_coordinates == None if with_position is True")

        self.phi_edge = nn.Sequential(nn.Linear(n_hidden_features*2, n_hidden_features),
                                      Swish(),
                                      nn.Linear(n_hidden_features, n_hidden_features),
                                      Swish())
        if self.with_position:
            self.phi_coordinate = nn.Sequential(nn.Linear(dim_coordinates, n_hidden_features),
                                                Swish(),
                                                nn.Linear(n_hidden_features, n_hidden_features))
        if self.with_momentum:
            self.phi_momentum = nn.Sequential(nn.Linear(2, n_hidden_features),
                                              Swish(),
                                              nn.Linear(n_hidden_features, n_hidden_features))

        self.phi_node = nn.Sequential(nn.Linear(n_hidden_features, n_hidden_features),
                                      Swish(),
                                      nn.Linear(n_hidden_features, n_hidden_features),
                                      )
    '''
        From the paper is not clarified how the inputs for the edge function are combined
        - m_ij = \phi(h_i, h_j, || x_i - x_j ||^2, aij)
        
        for simplicity we'll implement in this way
         m_ij = a_ij*||x_i - x_j||^2*\phi([h_i, h_j])
        So, using the concatenation of [h_i, h_j] as input of the NN and the distance and weight's edge as weight
    '''
    def forward(self, x):
        embedding, position, A = x
        if position.shape[0] != position.shape[1]:
            ## Compute the distance matrix
            distance_matrix = torch.cdist(position, position)
        else:
            distance_matrix = position
        ## Concat the features vector
        H_H = torch.concat([embedding, embedding], dim=1)
        ## pass through the NN
        H_H = self.phi_edge(H_H)
        ## multiply by the distance matrix and the edge weight
        M = torch.matmul(distance_matrix, torch.matmul(A, H_H))
        ## Sum from the paper
        H = embedding + self.phi_node(M)
        if self.with_momentum:
            pass
        elif self.with_position:
            pass

        return (H, distance_matrix, A)


class EnGNN(nn.Module):

    def __init__(self, input_features, hidden_features, out_features, dim_coordinates, edge_inference=False):
        super(EnGNN, self).__init__()
        self.embedding = nn.Linear(in_features=input_features,
                                   out_features=hidden_features)
        self.model = nn.Sequential(EGCL(n_hidden_features=hidden_features,
                                        dim_coordinates=dim_coordinates,
                                        with_position=False,
                                        with_momentum=False))
        if edge_inference:
            self.edge_inference_function = nn.Sequential(nn.Linear(in_features=input_features,
                                                                   out_features=1),
                                                         nn.Sigmoid())

        self.head = nn.Sequential(nn.Linear(in_features=hidden_features, out_features=hidden_features),
                                  Swish(),
                                  nn.Linear(in_features=hidden_features, out_features=hidden_features),
                                  tg.nn.SumAggregation(),
                                  nn.Linear(in_features=hidden_features, out_features=hidden_features),
                                  Swish(),
                                  nn.Linear(in_features=hidden_features,
                                            out_features=out_features),)

    def forward(self, x):
        embedding, position, A = x
        ## We used a linear embedding to change the dimension of the input features
        ## This is due to the fact that in the paper there's a sum for the the \phi_h so the dimension should be fixed for the operation
        embedding = self.embedding(embedding)
        H, _, _ = self.model((embedding, position, A))
        out = self.head(H)
        return out




def visualize(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True,
                     node_color=color, cmap="Set2")
    plt.show()


def test_1():
    from tensorboardX import SummaryWriter
    writer = SummaryWriter()
    '''
        QM9 Dataset composed by several molecules, each of them is a graph with a different number of atom (max 29)
        each node (atom) has
        - Feature vector that is an embedding of 11 elements
        - The edges can be of 4 type
    '''
    dataset = tg.datasets.QM9(root="../../Datasets/QM9")
    dataloader = torch_geometric.data.DataLoader(dataset, batch_size=1, shuffle=True)
    N_bond_type = 4

    model = EnGNN(input_features=11, dim_coordinates=3, hidden_features=128, out_features=19)
    model = model.to("cuda")
    loss_fun = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    n_epochs = 10
    for _ in tqdm(range(n_epochs)):
        for idx, batch in tqdm(enumerate(dataloader)):
            batch = batch.to("cuda")
            optim.zero_grad()
            _, bond_type_encoded = torch.where(batch.edge_attr)
            bond_type_encoded = (bond_type_encoded + 1)/N_bond_type
            embedding, positions, y = batch.x,  batch.pos, batch.y.squeeze()
            A = tg.utils.to_dense_adj(edge_index=batch.edge_index, edge_attr=bond_type_encoded).squeeze()
            out = model((embedding, positions, A)).squeeze()
            loss = loss_fun(out, y)
            loss.backward()
            optim.step()
            writer.add_scalar('training_loss', loss.item(), idx)


if __name__ == '__main__':
    test_1()
