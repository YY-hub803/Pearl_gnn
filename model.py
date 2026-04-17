import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.data import Batch, Data


class LSTMLayer(nn.Module):
    def __init__(self, input_size,hidden_size,num_layer):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers=num_layer,batch_first=True, bidirectional=False)
    def forward(self, x):
        B, N, T, _ = x.shape
        x_in = x.reshape(B * N, T, -1)
        lstm_out,_ = self.lstm(x_in)
        return lstm_out.reshape(B, N, T, self.hidden_size)

class GCNLayer(nn.Module):
    def __init__(self, input_size, hidden_size,edge_index):
        super().__init__()
        self.edge_index = edge_index
        self.conv = GATConv(input_size, hidden_size,add_self_loops=False)
    def forward(self, x):
        # x: [B, T, N, F]
        B, N, T, nF = x.shape
        x_in = x.permute(0, 2,1,3)
        data_list = []
        for b in range(B):
            for t in range(T):
                data = Data(x=x_in[b,t], edge_index=self.edge_index)
                data_list.append(data)
        batch = Batch.from_data_list(data_list)
        gcn_out = self.conv(batch.x, batch.edge_index)
        gcn_out = gcn_out.view(B,T,N,-1)
        return F.gelu(gcn_out).reshape(B,N,T,-1)

class GcnLstmModel(nn.Module):
    def __init__(self, nx,ny, edge_index,hidden_size, pred_len,num_layer,drop_rate,device):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.pred_len = pred_len
        self.edge_index = edge_index.to(device)
        self.fc = nn.Linear(nx, hidden_size)
        self.conv1 = GCNLayer(hidden_size,hidden_size,self.edge_index)
        self.conv2 = GCNLayer(hidden_size,hidden_size,self.edge_index)
        self.lstm = LSTMLayer(hidden_size,hidden_size,num_layer)
        self.norm_l = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, self.pred_len*self.ny)
    def forward(self, x):
        B, N, T, _ = x.shape
        x_h = self.fc(x)
        gout1 = self.conv1(x_h)
        gout2 = self.conv2(gout1)
        lout1 = self.lstm(gout2)
        h = self.norm_l(lout1)
        out = self.dense(h[:,:,-1:,:])
        return out.reshape(B,N, self.pred_len, self.ny)

#-----------------------------------------------------------------------------------------------------------------------
class PhysicsGuidedGCN(nn.Module):
    def __init__(self, in_features, hidden_size,dropout,lag_emb_dim=8):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size

        self.in_proj = nn.Linear(in_features, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.lag_embed = nn.Embedding(256, lag_emb_dim)

        self.ctx_mlp = nn.Sequential(
            nn.Linear(in_features, lag_emb_dim),
            nn.GELU(),
            nn.Linear(lag_emb_dim, lag_emb_dim),
        )

        self.lag_gate = nn.Sequential(
            nn.Linear(lag_emb_dim * 2, lag_emb_dim),
            nn.GELU(),
            nn.Linear(lag_emb_dim, 1),
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def row_normalize(A, eps=1e-8):
        # A: [B, K, N, N]
        row_sum = A.sum(dim=-1, keepdim=True).clamp_min(eps)
        return A / row_sum

    def forward(self, x,A_list):

        B, N, T, F_in = x.shape
        K, N1, N2 = A_list.shape

        assert N1 == N and N2 == N, f"A_list shape should be [B,K,{N},{N}], got {A_list.shape}"
        device = x.device
        dtype = x.dtype
        # 邻接矩阵规范化
        A = A_list.to(device=device, dtype=dtype)
        A = A.unsqueeze(0).repeat(B, 1, 1, 1)
        A = self.row_normalize(A)  # [B, K, N, N]

        h = self.in_proj(x)  # [B, N, T, H]
        h_t = h.permute(0, 2, 1, 3)  # [B, T, N, H]

        # 构造全局上下文（每个样本一个向量），用于动态 lag 权重
        ctx = x.mean(dim=(1, 2))  # [B, Fin]
        ctx_feat = self.ctx_mlp(ctx)  # [B, D]

        lag_idx = torch.arange(K, device=device)
        if K > self.lag_embed.num_embeddings:
            raise ValueError(f"max_lag={K} exceeds lag_embed size={self.lag_embed.num_embeddings}")
        lag_feat = self.lag_embed(lag_idx)  # [K, D]

        out = torch.zeros_like(h_t)  # [B, T, N, H]
        for lag in range(K):
            #  自身当前时刻
            if lag == 0:
                h_lag = h_t
            #  其他站点过去时刻
            else:
                h_lag = torch.roll(h_t, shifts=lag, dims=1)
                h_lag[:, :lag] = 0.0

            #  聚合操作
            # A[:, lag]: [B, N, N]
            agg = torch.einsum('bij,btjh->btih', A[:, lag], h_lag)  # [B, T, N, H]
            lag_b = lag_feat[lag].unsqueeze(0).expand(B, -1)  # [B, D]
            gate_in = torch.cat([lag_b, ctx_feat], dim=-1)  # [B, 2D]
            alpha = torch.sigmoid(self.lag_gate(gate_in))  # [B, 1]
            alpha = alpha.view(B, 1, 1, 1)  # broadcast 到 [B,T,N,H]
            out = out + alpha * agg

        # 残差 + 归一化 + dropout
        out = self.out_proj(out)
        out = self.dropout(out)
        # out = self.norm(out)
        out = self.norm(out + h_t)  # residual
        # 回到 [B, N, T, H]
        out = out.permute(0, 2, 1, 3)
        return F.gelu(out)

class PhysicsSTNNModel(nn.Module):
    def __init__(self, nx,ny,hidden_size, pred_len,num_layer,drop_rate,device):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.pred_len = pred_len
        self.fc = nn.Linear(nx, hidden_size)
        self.conv1 = PhysicsGuidedGCN(hidden_size,hidden_size,drop_rate)
        self.conv2 = PhysicsGuidedGCN(hidden_size,hidden_size,drop_rate)
        self.lstm = LSTMLayer(hidden_size,hidden_size,num_layer)
        self.dense = nn.Linear(hidden_size, self.pred_len*self.ny)
        self.norm_l = nn.LayerNorm(hidden_size)

    def forward(self, x,A_list):
        B, N, T, _ = x.shape
        x_h = self.fc(x)
        gout1 = self.conv1(x_h,A_list)
        gout2 = self.conv2(gout1,A_list)
        lout1 = self.lstm(gout2)
        l_hn = self.norm_l(lout1)
        out = self.dense(l_hn[:,:,-1:,:])
        return out.reshape(B,N, self.pred_len, self.ny)