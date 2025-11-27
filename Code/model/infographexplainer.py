import torch
import torch.nn as nn
from torch_sparse import transpose
from torch_geometric.utils import sort_edge_index
from torch_geometric.nn import InstanceNorm


def reorder_like(from_edge_index, to_edge_index, values):
    from_edge_index, values = sort_edge_index(from_edge_index, values)
    ranking_score = to_edge_index[0] * (to_edge_index.max()+1) + to_edge_index[1]
    ranking = ranking_score.argsort().argsort()
    if not (from_edge_index[:, ranking] == to_edge_index).all():
        raise ValueError("Edges in from_edge_index and to_edge_index are different, impossible to match both.")
    return values[ranking]


class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [B, C, T, N]
        x = x.permute(0, 3, 2, 1)  # [B, N, T, C]
        B, N, T, C = x.shape

        queries = self.query(x).view(B * N, T, C)  # [B*N, T, C]
        keys = self.key(x).view(B * N, T, C)  # [B*N, T, C]
        values = self.value(x).view(B * N, T, C)  # [B*N, T, C]

        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) / (C ** 0.5)  # [B*N, T, T]
        attn_weights = self.softmax(attn_scores)  # [B*N, T, T]

        weighted = torch.bmm(attn_weights, values).view(B, N, T, C)  # [B, N, T, C]
        output = weighted.sum(dim=2)  # [B, N, C]
        return output


class BatchSequential(nn.Sequential):
    def forward(self, inputs):
        for module in self._modules.values():
            inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout=0.0, bias=True):
        layers = []
        for i in range(1, len(channels)):
            layers.append(nn.Linear(channels[i-1], channels[i], bias=bias))
            if i != len(channels) - 1:  # Not the last layer
                layers.append(InstanceNorm(channels[i]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        super().__init__(*layers)


class ExtractorMLP(nn.Module):
    def __init__(self, hidden_size, data_config, dropout_p=0.3):
        super().__init__()
        dataset_name = data_config['dataset_name']

        # f1||f2 -> h1 -> h2 -> 1
        channels = {
            'pemsd7-m':   [hidden_size * 2, hidden_size * 4, hidden_size, 1],
            'pems-bay':   [hidden_size * 2, hidden_size * 4, hidden_size, 1],
        }[dataset_name]

        self.mlp = MLP(channels=channels, dropout=dropout_p, bias=True)

    def forward(self, emb, edge_index):
        row, col = edge_index  # [E], [E]
        f1, f2 = emb[row], emb[col]  # [E, C]
        f12 = torch.cat([f1, f2], dim=-1)  # [E, 2C]
        att_log_logits = self.mlp(f12)  # [E, 1]
        return att_log_logits


class InfoGraphExplainer(nn.Module):
    def __init__(self, hidden_dim, data_config, info_config, dropout_rate):
        super(InfoGraphExplainer, self).__init__()
        self.extractor = ExtractorMLP(hidden_dim, data_config, dropout_p=dropout_rate)
        self.fix_r = info_config['fix_r']
        self.init_r = info_config['init_r']
        self.decay_interval = info_config['decay_interval']
        self.decay_r = info_config['decay_r']
        self.final_r = info_config['final_r']
        self.temp = info_config['temp']
        self.temporal_attention = TemporalAttention(hidden_dim)

    def forward(self, h_in, edge_index, training=True):
        emb = self.temporal_attention(h_in)
        emb_mean = torch.mean(emb, dim=0)  # [num_nodes, channels]

        # ignore self-loops
        self_loop_mask = (edge_index[0] == edge_index[1])
        non_self_loop_mask = ~self_loop_mask

        att_log_logits = self.extractor(emb_mean, edge_index[:, non_self_loop_mask])
        att = self.concrete_sample(att_log_logits, training=training)

        all_att = torch.ones(edge_index.size(1), 1, device=att.device)
        all_att[non_self_loop_mask] = att

        trans_idx, trans_val = transpose(edge_index, all_att, None, None, coalesced=False)
        trans_val_perm = reorder_like(trans_idx, edge_index, trans_val)
        edge_att = (all_att + trans_val_perm) / 2

        return edge_att

    def concrete_sample(self, logits, training=True):
        if training:
            random_noise = torch.empty_like(logits).uniform_(1e-10, 1-1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((logits + random_noise) / self.temp).sigmoid()
        else:
            att_bern = (logits).sigmoid()

        return att_bern

    def info_loss(self, att, epoch):
        r = self.fix_r if self.fix_r else self.get_r(epoch, self.init_r, self.final_r, self.decay_interval, self.decay_r)
        info_loss = (att * torch.log(att / r + 1e-6) + (1 - att) * torch.log((1 - att) / (1 - r + 1e-6) + 1e-6)).mean()

        return info_loss, r

    def get_r(self, epoch, init_r, final_r, decay_interval, decay_r):
        if self.fix_r:
            return self.fix_r
        r = init_r - epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r

        return r