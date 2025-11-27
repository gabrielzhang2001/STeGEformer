import torch
from torch import nn
from .layers import TemporalGatedConv, GraphConvLayer, LGTEncoder
from .infographexplainer import InfoGraphExplainer


class STeGEformer(nn.Module):
    def __init__(self, train_config, data_config, info_config, blocks, num_nodes, edge_index, gso):
        super(STeGEformer, self).__init__()
        self.temporal_kernel_size = train_config['temporal_kernel_size']
        self.n_history = train_config['n_history']
        self.n_prediction = train_config['n_prediction']
        self.dropout_rate = train_config['dropout_rate']
        self.use_explanation = train_config['use_explanation']
        self.blocks = blocks
        self.num_nodes = num_nodes
        self.edge_index = edge_index

        self.temporal_gated_conv = TemporalGatedConv(self.temporal_kernel_size, self.blocks[0][0], self.blocks[1][0],
                                                     self.num_nodes)
        self.main_graph_conv = GraphConvLayer(self.blocks[1][0], self.blocks[1][1], gso)
        self.main_lgt_encoder = LGTEncoder(self.temporal_kernel_size, self.blocks[1][1], self.blocks[1][2], num_nodes)
        if self.use_explanation:
            self.info_graph_explainer = InfoGraphExplainer(self.blocks[1][0], data_config, info_config, self.dropout_rate)
            self.explainable_graph_conv = GraphConvLayer(self.blocks[1][0], self.blocks[1][1], gso)
            self.explainable_lgt_encoder = LGTEncoder(self.temporal_kernel_size, self.blocks[1][1], self.blocks[1][2], num_nodes)

        self.tc2_ln = nn.LayerNorm([num_nodes, self.blocks[1][2]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate)

        sequence_length_after_convs = self.n_history - (self.temporal_kernel_size - 1) * 2
        if sequence_length_after_convs == 0:
            self.fc1 = nn.Linear(in_features=blocks[1][2], out_features=blocks[2][0])
            self.fc2 = nn.Linear(in_features=blocks[2][0], out_features=blocks[3][0])
            self.tc1_ln = nn.LayerNorm([num_nodes, blocks[1][2]], eps=1e-12)
        elif sequence_length_after_convs > 0:
            self.output = OutputBlock(sequence_length_after_convs, blocks[1][2], blocks[2], blocks[3][0], num_nodes, self.dropout_rate)

    def forward(self, x, training=True):
        # x shape: [batch_size, channels, time_steps, num_nodes]
        batch_size, num_nodes = x.shape[0], x.shape[3]

        h_t = self.temporal_gated_conv(x)  # [batch_size, channels, time_steps, num_nodes]
        h_s1 = self.main_graph_conv(h_t, self.edge_index)
        h_s1 = self.relu(h_s1)
        h_t1 = self.main_lgt_encoder(h_s1)

        if self.use_explanation:
            edge_att = self.info_graph_explainer(h_t, self.edge_index, training)  # [num_edges, 1]

            # ignore self-loops
            self_loop_mask = (self.edge_index[0] == self.edge_index[1])
            non_self_loop_mask = ~self_loop_mask
            edge_att_filtered = edge_att[non_self_loop_mask]
            h_s2 = self.explainable_graph_conv(h_t, self.edge_index, edge_att)  # [batch_size, channels, time_steps, num_nodes]
            h_s2 = self.relu(h_s2)
            h_t2 = self.explainable_lgt_encoder(h_s2)

        else:
            edge_att_filtered = None
            h_t2 = torch.tensor(0.0)

        h = torch.add(h_t1, h_t2)

        h = self.tc2_ln(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.dropout(h)

        if hasattr(self, 'output'):
            x = self.output(h)
        else:
            x = self.tc1_ln(h.permute(0, 2, 3, 1))
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        return x.reshape(batch_size, self.n_prediction, num_nodes), edge_att_filtered

    def get_info_loss(self, att, epoch=None):
        return self.info_graph_explainer.info_loss(att, epoch)


class OutputBlock(nn.Module):
    def __init__(self, out_size, last_block__size, hidden_size, out_size_out, num_nodes, dropout_rate):
        super(OutputBlock, self).__init__()
        self.temporal_conv = TemporalGatedConv(out_size, last_block__size, hidden_size[0], num_nodes)
        self.fc1 = nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1])
        self.fc2 = nn.Linear(in_features=hidden_size[1], out_features=out_size_out)
        self.tc1_ln = nn.LayerNorm([num_nodes, hidden_size[0]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x).permute(0, 3, 1, 2)

        return x


class STeGE(nn.Module):
    def __init__(self, train_config, data_config, info_config, blocks, num_nodes, edge_index, gso):
        super(STeGE, self).__init__()
        self.temporal_kernel_size = train_config['temporal_kernel_size']
        self.n_history = train_config['n_history']
        self.n_prediction = train_config['n_prediction']
        self.dropout_rate = train_config['dropout_rate']
        self.use_explanation = train_config['use_explanation']
        self.blocks = blocks
        self.num_nodes = num_nodes
        self.edge_index = edge_index

        self.temporal_gated_conv = TemporalGatedConv(self.temporal_kernel_size, self.blocks[0][0], self.blocks[1][0],
                                                     self.num_nodes)
        self.main_graph_conv = GraphConvLayer(self.blocks[1][0], self.blocks[1][1], gso)
        if self.use_explanation:
            self.info_graph_explainer = InfoGraphExplainer(self.blocks[1][0], data_config, info_config, self.dropout_rate)
            self.explainable_graph_conv = GraphConvLayer(self.blocks[1][0], self.blocks[1][1], gso)

        self.tc2_ln = nn.LayerNorm([num_nodes, self.blocks[1][1]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate)

        sequence_length_after_convs = self.n_history - (self.temporal_kernel_size - 1)
        if sequence_length_after_convs == 0:
            self.fc1 = nn.Linear(in_features=blocks[1][1], out_features=blocks[2][0])
            self.fc2 = nn.Linear(in_features=blocks[2][0], out_features=blocks[3][0])
            self.tc1_ln = nn.LayerNorm([num_nodes, blocks[1][1]], eps=1e-12)
        elif sequence_length_after_convs > 0:
            self.output = OutputBlock(sequence_length_after_convs, blocks[1][1], blocks[2], blocks[3][0], num_nodes, self.dropout_rate)

    def forward(self, x, training=True):
        # x shape: [batch_size, channels, time_steps, num_nodes]
        batch_size, num_nodes = x.shape[0], x.shape[3]

        h_t = self.temporal_gated_conv(x)  # [batch_size, channels, time_steps, num_nodes]
        h_s1 = self.main_graph_conv(h_t, self.edge_index)
        h_s1 = self.relu(h_s1)
        if self.use_explanation:
            edge_att = self.info_graph_explainer(h_t, self.edge_index, training)  # [num_edges, 1]

            # ignore self-loops
            self_loop_mask = (self.edge_index[0] == self.edge_index[1])
            non_self_loop_mask = ~self_loop_mask
            edge_att_filtered = edge_att[non_self_loop_mask]
            h_s2 = self.explainable_graph_conv(h_t, self.edge_index, edge_att)  # [batch_size, channels, time_steps, num_nodes]
            h_s2 = self.relu(h_s2)
        else:
            edge_att_filtered = None
            h_s2 = torch.tensor(0.0)

        h = torch.add(h_s1, h_s2)

        h = self.tc2_ln(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.dropout(h)

        if hasattr(self, 'output'):
            x = self.output(h)
        else:
            x = self.tc1_ln(h.permute(0, 2, 3, 1))
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        return x.reshape(batch_size, self.n_prediction, num_nodes), edge_att_filtered

    def get_info_loss(self, att, epoch=None):
        return self.info_graph_explainer.info_loss(att, epoch)