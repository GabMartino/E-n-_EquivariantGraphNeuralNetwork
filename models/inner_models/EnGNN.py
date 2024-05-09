import torch
import torch.nn as nn
from torch_geometric.nn import SumAggregation


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)
class Phi_e(nn.Module):
    def __init__(self, n_input_features, n_hidden_features, n_output_features):
        super(Phi_e, self).__init__()
        n_input_features = n_input_features*2
        self.model = nn.Sequential(nn.Linear(n_input_features, n_hidden_features),
                                   Swish(),
                                   nn.Linear(n_hidden_features, n_output_features),
                                   Swish())
    '''
        We cannot assume that the concatenation of [h_i, h_j] must produce the same results than [h_j, h_i]
    '''
    def forward(self, Edges, Coordinates, Embeddings):
        ## Define the input [ a_{ij} || - ||^, h_i, h_j]
        ## Compute the distance matrix
        distance_matrix = torch.cdist(Coordinates, Coordinates, p=2)
        ## Combine the values of the weights
        w = Edges * distance_matrix
        ## Produce all the combinations of [h_i, h_j] also when j = i (it's possible to speed up this procedure)
        def concat_embed(embedding_a, embedding_b):
            return torch.cat((embedding_a, embedding_b))
        H = torch.vmap(func=torch.vmap(concat_embed, in_dims=(None, 0)), in_dims=(0, None))(Embeddings, Embeddings)
        H_hat = w[:, :, None] * H
        n_nodes = w.shape[0]
        H_hat = torch.flatten(H_hat, start_dim=0, end_dim=1)
        M = self.model(H_hat)
        M_unflattened = torch.unflatten(M, 0, (n_nodes, n_nodes, -1)).squeeze()
        return M_unflattened
class Phi_x(nn.Module):
    def __init__(self, n_input_features, n_hidden_features, n_output_features):
        super(Phi_x, self).__init__()
        self.model = nn.Sequential(nn.Linear(n_input_features, n_hidden_features),
                                   Swish(),
                                   nn.Linear(n_hidden_features, n_output_features))

    def forward(self, M):
        return self.model(M)
class Phi_h(nn.Module):
    def __init__(self, n_input_features, n_hidden_features, n_output_features):
        super(Phi_h, self).__init__()
        self.model = nn.Sequential(nn.Linear(n_input_features, n_hidden_features),
                                   Swish(),
                                   nn.Linear(n_hidden_features, n_output_features))

    def forward(self, H, M):
        input = torch.concatenate([H, M], dim=1)
        out = self.model(input)
        out = out + H
        return out
class EGCL(nn.Module):
    def __init__(self, n_input_features, n_hidden_features, n_output_features, with_velocity):
        super(EGCL, self).__init__()

        self.with_velocity = with_velocity
        if with_velocity:
            self.phi_v = nn.Sequential(nn.Linear(n_input_features, n_hidden_features),
                                       Swish(),
                                       nn.Linear(n_hidden_features, 1))
        self.phi_e = Phi_e(n_input_features=n_hidden_features,
                           n_hidden_features=n_hidden_features,
                           n_output_features=n_hidden_features)

        self.phi_x = Phi_x(n_input_features=n_hidden_features,
                           n_hidden_features=n_hidden_features,
                           n_output_features=1)

        self.phi_h = Phi_h(n_input_features=n_hidden_features*2,
                           n_hidden_features = n_hidden_features,
                           n_output_features=n_output_features)

    def forward(self, x):
        if self.with_velocity:
            edges, coordinates, embeddings, velocities = x
        else:
            edges, coordinates, embeddings = x
        M_edge = self.phi_e(edges, coordinates, embeddings)
        n_nodes = M_edge.shape[0]
        difference_position_fun = lambda x_i, x_j: x_i - x_j
        differences_of_positions = torch.vmap(func=torch.vmap(difference_position_fun, in_dims=(None, 0)), in_dims=(0, None))(coordinates, coordinates)

        M_edge_flatten = torch.flatten(M_edge, start_dim=0, end_dim=1)
        M_x = self.phi_x(M_edge_flatten)
        M_x_unflattened = torch.unflatten(M_x, 0, (n_nodes, n_nodes, -1)).squeeze()

        partials = (M_x_unflattened * (torch.ones((n_nodes, n_nodes), device=M_edge.device) - torch.eye(n=n_nodes, device=M_edge.device))).unsqueeze(-1)
        del M_x_unflattened
        if self.with_velocity:
            new_velocities = self.phi_v(embeddings)*velocities + (1/(n_nodes - 1))* torch.sum(differences_of_positions * partials, dim=0)
            new_coordinates = coordinates + new_velocities
        else:
            new_coordinates = coordinates + (1/(n_nodes - 1))* torch.sum(differences_of_positions * partials, dim=0)

        del differences_of_positions
        M_aggregated = torch.sum(M_edge * edges.unsqueeze(-1), dim=1)
        new_embeddings = self.phi_h(embeddings, M_aggregated)
        if self.with_velocity:
            return edges, new_coordinates, new_embeddings, new_velocities
        else:
            return edges, new_coordinates, new_embeddings
class EnGNN(nn.Module):
    def __init__(self, n_input_features, n_hidden_features, n_output_features, with_velocity=False):
        super(EnGNN, self).__init__()
        self.with_velocity = with_velocity
        self.embed = nn.Linear(n_input_features, n_hidden_features)
        self.model = nn.Sequential(EGCL(n_hidden_features, n_hidden_features, n_hidden_features, with_velocity),
                                   EGCL(n_hidden_features, n_hidden_features, n_hidden_features, with_velocity),
                                   EGCL(n_hidden_features, n_hidden_features, n_hidden_features, with_velocity),
                                   EGCL(n_hidden_features, n_hidden_features, n_hidden_features, with_velocity),
                                   EGCL(n_hidden_features, n_hidden_features, n_hidden_features, with_velocity),
                                   EGCL(n_hidden_features, n_hidden_features, n_hidden_features, with_velocity),
                                   EGCL(n_hidden_features, n_hidden_features, n_hidden_features, with_velocity))
        self.head_model_1 = nn.Sequential(nn.Linear(n_hidden_features, n_hidden_features),
                                        Swish(),
                                        nn.Linear(n_hidden_features, n_hidden_features))
        self.head_model_2 = nn.Sequential(nn.Linear(n_hidden_features, n_hidden_features),
                                        Swish(),
                                        nn.Linear(n_hidden_features, n_output_features))

        self.edge_inferring_model = nn.Sequential(nn.Linear(n_hidden_features, 1),
                                                  nn.Sigmoid())

    def forward(self, Edges, Coordinates, Embeddings, batch_pointer, velocities=None):
        Embeddings = self.embed(Embeddings)
        if self.with_velocity:
            edges, coordinates, embeddings, velocities = self.model((Edges, Coordinates, Embeddings, velocities))
        else:
            edges, coordinates, embeddings = self.model((Edges, Coordinates, Embeddings))
        embeddings = self.head_model_1(embeddings)
        ### Aggregation with pointer not implemented yet from torch geometric
        ranges = batch_pointer.unfold(0, 2, 1)
        embeddings_list = []
        for indeces in ranges:
            embeddings_list.append(torch.sum(embeddings[indeces[0]:indeces[1]], dim=0 ))
        ####################################################################à
        embeddings = torch.stack(embeddings_list)
        out = self.head_model_2(embeddings)
        return out