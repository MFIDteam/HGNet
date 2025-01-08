import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class NodeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 1],
                 dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        self.in_features = 256
        self.num_features_list = [256 * r for r in ratio]

        # num_features_list = [num_features_list * 2, num_features_list * 1]
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else self.in_features * 3,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            # dropout
            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        # node_feat [10, 32, 49]
        # edge_feat [10, 2, 32, 32]
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)

        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).unsqueeze(0).repeat(num_tasks, 2, 1, 1).cuda() # [10, 2, 32, 32]
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)  # [10, 2, 32, 32]
        aggr_feat = torch.bmm(torch.cat(torch.split(edge_feat, 1, 1), 2).squeeze(1), node_feat) # [10, 64, 49]
        node_feat = torch.cat([node_feat, torch.cat(aggr_feat.split(num_data, 1), -1)], -1).transpose(1, 2) # [10, 147, 32]

        # non-linear transform
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)

        return node_feat


class EdgeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 2, 1, 1],
                 separate_dissimilarity=False,
                 dropout=0.0):
        super(EdgeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                       out_channels=self.num_features_list[l],
                                                       kernel_size=1,
                                                       bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

        if self.separate_dissimilarity:
            # layers
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                # set layer
                layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                           out_channels=self.num_features_list[l],
                                                           kernel_size=1,
                                                           bias=False)
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                                )
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()

                if self.dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=self.dropout)

            layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                               out_channels=1,
                                               kernel_size=1)
            self.dsim_network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)

        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples) 计算相似性和不相似性
        sim_val = F.sigmoid(self.sim_network(x_ij))

        if self.separate_dissimilarity:
            dsim_val = F.sigmoid(self.dsim_network(x_ij))
        else:
            dsim_val = 1.0 - sim_val


        diag_mask = 1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 2, 1, 1).cuda()
        edge_feat = edge_feat * diag_mask
        merge_sum = torch.sum(edge_feat, -1, True)

        edge_feat = F.normalize(torch.cat([sim_val, dsim_val], 1) * edge_feat, p=1, dim=-1) * merge_sum
        force_edge_feat = torch.cat((torch.eye(node_feat.size(1)).unsqueeze(0),
                                     torch.zeros(node_feat.size(1), node_feat.size(1)).unsqueeze(0)), 0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1).cuda()

        edge_feat = edge_feat + force_edge_feat
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1).repeat(1, 2, 1, 1)

        return edge_feat


# DAF
class SpatialAttentionPooling(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(SpatialAttentionPooling, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):

        attention_weights = self.conv(x)  # [batch_size, 1, 7, 7]
        attention_weights = F.softmax(attention_weights.view(x.size(0), -1), dim=1)  # softmax
        attention_weights = attention_weights.view(x.size(0), 1, x.size(2), x.size(3))  # [batch_size, 1, 7, 7]

        weighted_x = x * attention_weights
        output = weighted_x.sum(dim=(2, 3))

        return output  # [batch_size, in_channels] = [10, 256]


class GraphNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 node_features,
                 edge_features,
                 num_layers,
                 ratio,
                 dropout=0.0):
        super(GraphNetwork, self).__init__()
        # set size
        self.in_features = in_features
        hide_channel = in_features // ratio
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = 0.2
        self.hidden_dim = hide_channel
        self.conv1 = nn.Conv2d(in_features, hide_channel, kernel_size=1, bias=False)
        self.ATT = SpatialAttentionPooling(in_channels=256)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.layer_last = nn.Sequential(nn.Linear(in_features=256,
                                                  out_features=10, bias=True),
                                        nn.BatchNorm1d(10))

        # for each layer
        for l in range(self.num_layers):
            # set edge to node
            edge2node_net = NodeUpdateNetwork(in_features=self.in_features if l == 0 else self.node_features,
                                              num_features=self.node_features,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            # set node to edge
            node2edge_net = EdgeUpdateNetwork(in_features=self.node_features,
                                              num_features=self.edge_features,
                                              separate_dissimilarity=False,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            self.add_module('edge2node_net{}'.format(l), edge2node_net)
            self.add_module('node2edge_net{}'.format(l), node2edge_net)

    def forward(self, x):  # input x: [[10, 2048, 7, 7]]

        # Channel compression x_cc: [[10, 32, 7, 7]]
        x_cc = self.conv1(x)

        # feature flatten x_ff: [[10, 32, 49]]
        x_ff = x_cc.flatten(-2, -1).transpose(1, 2)
        batch, channl, node_feature_dimension = x_ff.shape

        edge_initialization = torch.full((batch, 2, channl, channl), 0.5, device=x_ff.device)
        edge_initialization[:, 0, torch.arange(channl), torch.arange(channl)] = 1.0

        node_feat = x_ff
        edge_feat = edge_initialization

        for l in range(self.num_layers):
            # (1) Edge to node
            node_feat = self._modules[f'edge2node_net{l}'](node_feat, edge_feat)
            # (2) Node to edge
            edge_feat = self._modules[f'node2edge_net{l}'](node_feat, edge_feat)

        node_feat = node_feat.reshape(batch, node_feature_dimension, 7, 7)
        # node_feat = self.ATT(node_feat).squeeze(-1).squeeze(-1)
        node_feat = self.avg_pool(node_feat).squeeze(-1).squeeze(-1)

        node_feat_agg = self.layer_last(node_feat.view(batch, -1))

        return node_feat_agg