import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x = x

        return x


class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1,
                 bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0,
                                           dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)

        return result


class TemporalGatedConv(nn.Module):
    def __init__(self, Kt, c_in, c_out, n_vertex):
        super(TemporalGatedConv, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(Kt, 1),
                                        enable_padding=False, dilation=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        x_causal_conv = self.causal_conv(x)

        x_p = x_causal_conv[:, : self.c_out, :, :]
        x_q = x_causal_conv[:, -self.c_out:, :, :]

        x = torch.mul((x_p + x_in), torch.sigmoid(x_q))

        return x


class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, gso):
        super(GraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, edge_index=None, edge_weight=None):
        x = torch.permute(x, (0, 2, 3, 1))  # [batch_size, time_steps, num_nodes, channels]

        if edge_index is not None and edge_weight is not None:
            batch_size, time_steps, num_nodes, features = x.shape

            if edge_weight.dim() == 1:
                edge_weight = edge_weight.unsqueeze(-1)  # [num_edges, 1]
            elif edge_weight.dim() > 2:
                edge_weight = edge_weight.squeeze()

            weighted_adj = torch.zeros(batch_size, num_nodes, num_nodes,
                                       device=x.device)  # [batch_size, num_nodes, num_nodes]
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]

            if edge_weight.dim() > 1:
                edge_weight = edge_weight.squeeze()  # [num_edges,]

            weighted_adj[:, src_nodes, dst_nodes] = edge_weight
            gso_expanded = self.gso.unsqueeze(0).expand(batch_size, -1, -1)
            adjusted_gso = gso_expanded * weighted_adj

            first_mul = torch.einsum('bmn,btnc->btmc', adjusted_gso, x)
            second_mul = torch.einsum('bthi,ij->bthj', first_mul, self.weight)
        else:
            first_mul = torch.einsum('hi,btij->bthj', self.gso, x)
            second_mul = torch.einsum('bthi,ij->bthj', first_mul, self.weight)

        graph_conv = torch.add(second_mul, self.bias)

        return graph_conv


class GraphConvLayer(nn.Module):
    def __init__(self, c_in, c_out, gso):
        super(GraphConvLayer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gso = gso
        self.align = Align(c_in, c_out)
        self.graph_conv = GraphConv(c_out, c_out, gso)

    def forward(self, x, edge_index=None, edge_weight=None):
        x_gc_in = self.align(x)
        x_gc = self.graph_conv(x_gc_in, edge_index, edge_weight)
        x_gc = x_gc.permute(0, 3, 1, 2)  # [batch_size, channels, time_steps, num_nodes]
        x_gc_out = torch.add(x_gc, x_gc_in)
        return x_gc_out


class PositionEncoder(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionEncoder, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        pe = self.pe[:seq_len, :].to(x.device)
        x = x + pe

        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Multi-head attention
        src2 = self.multi_head_attention(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # Feedforward
        src2 = self.feedforward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src


class LGTEncoder(nn.Module):
    def __init__(self, temporal_kernel_size, c_in, c_out, num_nodes):
        super(LGTEncoder, self).__init__()
        self.temporal_kernel_size = temporal_kernel_size
        self.c_in = c_in
        self.c_out = c_out
        self.num_nodes = num_nodes
        self.current_epoch = 0

        self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=c_out, kernel_size=(temporal_kernel_size, 1),
                                        enable_padding=False, dilation=1)

        self.position_encoder = PositionEncoder(d_model=c_out)
        self.transformer = TransformerBlock(d_model=c_out, nhead=8, dim_feedforward=c_out)

        self.no_improvement_count = 0
        self.best_val_loss = float('inf')

    def forward(self, x):
        x_causal_conv = self.causal_conv(x)  # [batch_size, c_out, time_steps, num_nodes]
        batch_size, d_model, time_steps, num_nodes = x_causal_conv.shape

        x_position_in = x_causal_conv.reshape(time_steps, -1, d_model)  # [timestep, batch_size * n_vertex, c_out]
        x_position_out = self.position_encoder(x_position_in)

        x_transformer_out = self.transformer(x_position_out)

        x = x_transformer_out.reshape(batch_size, d_model, time_steps, num_nodes)

        return x
