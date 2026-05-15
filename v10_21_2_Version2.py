# v10_21_unified_time_basis.py
# 使用 PeriodicTimeEncoder 作为时间基底，节点学习系数形成 Node-level 时间动态

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

# =========================
# 配置
# =========================
CONFIG = {
    "DATA_PATH": "./bronx_synthetic_data_v13/bronx_st_v13.npz",
    #"DATA_PATH": "./bronx_synthetic_data_v19_real/bronx_st_real.npz",
    "OUTPUT_DIR": "./output_v10_21_time_basis_new",
    #"OUTPUT_DIR": "./output_v10_21_time_basis_real",
    "HIST_LEN": 12,
    "PRED_LEN": 1,
    "STEPS_PER_DAY": 24,

    "GCN_HID": 32,
    "LSTM_HID": 32,

    # time_enc 维度重新启用，用作时间基底
    "TIME_EMB_DIM":16,

    "POI_RANK": 10,
    "NODE_EMB_DIM": 10,

    "WEATHER_EMB_DIM": 16,
    "WEATHER_ATTN_HEADS": 2,
    "ATTR_FUSION_DIM": 32,

    "TRAIN_RATIO": 0.6,
    "VAL_RATIO": 0.2,
    "BATCH_SIZE": 32,
    "EPOCHS": 150,
    "LR": 1e-3,
    "WEIGHT_DECAY": 1e-5,
    "SEED": 42,
    "PATIENCE": 50,

    "WEATHER_SPARSITY_TARGET": 0.55,
    "WEATHER_SPARSITY_WEIGHT": 0.25,
    "WEATHER_TEMP_INIT": 1.5,
    "WEATHER_TEMP_FINAL": 0.3,
    "WEATHER_RANKING_WEIGHT": 0.08,
    "WEATHER_RECALL_WEIGHT": 0.03,

    "LAMBDA_GC_POI_LOCAL": 3e-5,
    "LAMBDA_GC_POI_NEIGHBOR": 2e-5,
    "POI_NEIGHBOR_CONSIST_WEIGHT": 0.02,
    "LAMBDA_GC_TRAFFIC": 1e-3,
    "LAMBDA_ADAPT_SPARSE": 5e-4,
    "LAMBDA_TIME_SMOOTH": 5e-3,

    # 时间动态不再单独加平滑/对比 loss，只依靠结构与主任务
    "THRESH_TRAFFIC": 0.01,
    "THRESH_WEATHER": 0.45,
    "THRESH_POI": 0.04,

    "CURRICULUM_STAGE1_END": 40,
    "CURRICULUM_STAGE2_END": 100,
}


# =========================
# Utils
# =========================
def seed_everything(s=42):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_data(path):
    d = np.load(path, allow_pickle=True)
    files = set(d.files)
    def get_optional(key, default):
        return d[key] if key in files else default

    # 必备字段（真实数据也应提供）
    X = d["X"]
    W = d["W"]
    poi = d["poi"]
    A = d["A"]

    # 可选字段（仅合成数据/带真值数据才有）
    N = X.shape[1]
    D = poi.shape[1]
    return {
        "X": X,
        "W": W,
        "poi": poi,
        "A": A,

        # --- 真值/监督相关（缺失则为 None） ---
        "GC_traffic": get_optional("GC_traffic", None),
        "GC_poi_local": get_optional("GC_poi_local", None),
        "GC_poi_neighbor": get_optional("GC_poi_neighbor", None),
        "GC_weather": get_optional("GC_weather", None),
        "Weather_Gamma": get_optional("Weather_Gamma", None),
        "True_Time_Pattern": get_optional("True_Time_Pattern", None),
        "True_Node_Pattern": get_optional("True_Node_Pattern", None),  # (N, 24)
        "POI_Local_Weights": get_optional("POI_Local_Weights", None),
        "POI_Neighbor_Weights": get_optional("POI_Neighbor_Weights", None),

        # --- 缺真值时用来保证代码健壮的默认值（不代表真值） ---
        "_DEFAULT_gamma": np.zeros(N, dtype=float),
        "_DEFAULT_poi_local_w": np.zeros((D, N), dtype=float),
        "_DEFAULT_poi_neighbor_w": np.zeros((D, N, N), dtype=float),
    }


def metrics(pred, true):
    p, t = pred.flatten(), true.flatten()
    return (
        precision_score(t, p, zero_division=0),
        recall_score(t, p, zero_division=0),
        f1_score(t, p, zero_division=0),
        p.mean(),
        t.mean(),
    )


def weight_correlation(learned, true):
    l_flat = learned.flatten()
    t_flat = true.flatten()
    if l_flat.std() < 1e-8 or t_flat.std() < 1e-8:
        return 0.0
    return np.corrcoef(l_flat, t_flat)[0, 1]


def build_unified_true_poi(gc_local, gc_neighbor, w_local, w_neighbor):
    D, N = gc_local.shape
    gc_unified = gc_neighbor.copy()
    w_unified = w_neighbor.copy()
    for d in range(D):
        np.fill_diagonal(gc_unified[d], gc_local[d])
        np.fill_diagonal(w_unified[d], w_local[d])
    return gc_unified, w_unified


class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best = None
        self.stop = False

    def __call__(self, val):
        if self.best is None or val < self.best - 1e-4:
            self.best = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# =========================
# Dataset
# =========================
class TrafficDataset(Dataset):
    def __init__(self, X, W, poi, hist_len, pred_len, start, end, steps_per_day):
        self.X, self.W, self.poi = X, W, poi
        self.hist_len, self.pred_len, self.steps_per_day = hist_len, pred_len, steps_per_day
        self.indices = list(range(start + hist_len - 1, end))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t = self.indices[i]
        Xh = self.X[t - self.hist_len + 1: t + 1]
        Wh = self.W[t - self.hist_len + 1: t + 1]
        H, N = Xh.shape
        feat = np.transpose(
            np.concatenate(
                [Xh[..., None], Wh[..., None], np.repeat(self.poi[None], H, 0)],
                -1,
            ),
            (1, 0, 2),
        )
        target_t = t + self.pred_len
        return (
            torch.tensor(feat, dtype=torch.float32),
            torch.tensor(target_t % self.steps_per_day, dtype=torch.long),
            torch.tensor(self.X[target_t], dtype=torch.float32),
            torch.tensor(target_t, dtype=torch.long),
        )


def create_loaders(X, W, poi, cfg):
    T = X.shape[0]
    pred_len = cfg["PRED_LEN"]
    t_eff = T - pred_len
    train_end = int(t_eff * cfg["TRAIN_RATIO"])
    val_end = int(t_eff * (cfg["TRAIN_RATIO"] + cfg["VAL_RATIO"]))
    train_ds = TrafficDataset(
        X, W, poi, cfg["HIST_LEN"], pred_len, 0, train_end, cfg["STEPS_PER_DAY"]
    )
    val_ds = TrafficDataset(
        X, W, poi, cfg["HIST_LEN"], pred_len, train_end, val_end, cfg["STEPS_PER_DAY"]
    )
    # 剩余时段为测试集（训练/早停未使用，仅用于事后更长对比曲线）
    test_ds = TrafficDataset(
        X, W, poi, cfg["HIST_LEN"], pred_len, val_end, t_eff, cfg["STEPS_PER_DAY"]
    )
    return (
        DataLoader(train_ds, cfg["BATCH_SIZE"], shuffle=True),
        DataLoader(val_ds, cfg["BATCH_SIZE"]),
        DataLoader(test_ds, cfg["BATCH_SIZE"]),
    )


# =========================
# Model Components
# =========================
class Sparsemax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        sorted_x, _ = torch.sort(x, dim=self.dim, descending=True)
        k = torch.arange(1, x.size(self.dim) + 1, device=x.device).view(
            [1] * (x.dim() - 1) + [-1]
        )
        cumsum = torch.cumsum(sorted_x, dim=self.dim)
        is_gt = (1 + k * sorted_x > cumsum).float()
        k_z = is_gt.sum(dim=self.dim, keepdim=True)
        tau = (cumsum.gather(self.dim, (k_z - 1).long().clamp(min=0)) - 1) / k_z
        return torch.clamp(x - tau, min=0)


class NodeAdaptiveTimeEncoder(nn.Module):
    """
    改进版节点时间编码器：
    1. 使用线性系数 (Linear Coeffs) 替代相位/振幅，解决非凸优化导致的相位死锁。
    2. 提供频域平滑正则化 (Frequency Regularization)，抑制高频噪声。
    """

    def __init__(self, steps_per_day, emb_dim, num_nodes, num_freqs=4):
        super().__init__()
        self.steps = steps_per_day
        self.emb_dim = emb_dim
        self.num_nodes = num_nodes
        self.num_freqs = num_freqs

        # 基础频率：1x, 2x, ..., num_freqs x / day
        freqs = 2 * np.pi * torch.arange(1, num_freqs + 1).float() / steps_per_day
        self.register_buffer("freqs", freqs)

        # --- 核心修改：线性系数 ---
        # 形状: (num_nodes, 2 * num_freqs)
        # 含义: [sin(w1), sin(w2)..., cos(w1), cos(w2)...] 的系数
        # 初始化: 使用较小的标准差 (0.02) 防止初始输出落入 tanh 饱和区
        self.coeffs = nn.Parameter(torch.randn(num_nodes, 2 * num_freqs) * 0.02)

        self.proj = nn.Linear(2 * num_freqs, emb_dim)

    def forward(self, t):
        # t: (B,)
        # 1. 构建傅里叶基底 (B, F)
        angles = t.float().unsqueeze(-1) * self.freqs
        sin_base = torch.sin(angles)
        cos_base = torch.cos(angles)

        # 拼接: (B, 2F)
        bases = torch.cat([sin_base, cos_base], dim=-1)

        # 2. 节点特异性加权 (Node-Specific Weighting)
        # bases: (B, 1, 2F)
        # coeffs: (1, N, 2F)
        # node_fourier: (B, N, 2F)
        node_fourier = bases.unsqueeze(1) * self.coeffs.unsqueeze(0)

        # 3. 映射到 Embedding
        return self.proj(node_fourier)

    def get_smoothness_loss(self):
        """
        频域平滑约束：
        对高频分量的系数施加更大的惩罚。
        原理：时间序列越平滑，其高频分量的能量越低。
        """
        # 频率权重向量
        # 假设 F=4, 我们希望惩罚 w3, w4 远大于 w1, w2
        # 权重构造: [1, 4, 9, 16] (平方级增长，强力抑制高频)
        F = self.num_freqs
        w_vec = torch.arange(1, F + 1, device=self.coeffs.device, dtype=torch.float) ** 2

        # 拼接对应 sin 和 cos 部分: [1, 4, 9, 16, 1, 4, 9, 16]
        w_vec = torch.cat([w_vec, w_vec])

        # 计算加权 L2 正则
        # self.coeffs shape: (N, 2F)
        # weighted_coeffs: (N, 2F)
        loss = (self.coeffs ** 2) * w_vec.unsqueeze(0)

        return loss.mean()

class FourierTimeEncoder(nn.Module):
    """
    更复杂的时间模式表示：
    使用多频率的 sin/cos 作为时间基底，再映射到 TIME_EMB_DIM 维度。
    """
    def __init__(self, steps_per_day, emb_dim, num_freqs=4):
        super().__init__()
        self.steps = steps_per_day
        self.emb_dim = emb_dim
        self.num_freqs = num_freqs

        # frequencies: 1x, 2x, ..., num_freqs x per day
        freqs = 2 * np.pi * torch.arange(1, num_freqs + 1).float() / steps_per_day
        self.register_buffer("freqs", freqs)  # shape: (num_freqs,)

        # 将 2 * num_freqs 维的 Fourier 特征映射到 emb_dim
        self.proj = nn.Linear(2 * num_freqs, emb_dim)

    def forward(self, t):
        """
        t: (B,) int64
        return: (B, emb_dim)
        """
        # angles: (B, num_freqs)
        angles = t.float().unsqueeze(-1) * self.freqs  # (B,1)*(num_freqs,) -> (B,num_freqs)
        sin_feat = torch.sin(angles)                  # (B, num_freqs)
        cos_feat = torch.cos(angles)                  # (B, num_freqs)

        fourier = torch.cat([sin_feat, cos_feat], dim=-1)  # (B, 2*num_freqs)
        return self.proj(fourier)                           # (B, emb_dim)


class PeriodicTimeEncoder(nn.Module):
    """
    周期时间编码：输出 φ(t) 作为时间基底
    """
    def __init__(self, steps_per_day, emb_dim):
        super().__init__()
        self.steps = steps_per_day
        self.emb_dim = emb_dim

        self.register_buffer(
            "freq",
            2 * np.pi * torch.arange(1, 5).float() / steps_per_day,
        )
        self.emb = nn.Embedding(steps_per_day, emb_dim // 2)
        self.proj = nn.Linear(8, emb_dim // 2)
        self.fuse = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, t):
        """
        t: (B,) or (steps,) long
        return: (B, emb_dim) or (steps, emb_dim)
        """
        angles = t.float().unsqueeze(-1) * self.freq  # (B, 4)
        four = self.proj(torch.cat([torch.sin(angles), torch.cos(angles)], -1))
        return self.fuse(torch.cat([self.emb(t), four], -1))


class WeatherGate(nn.Module):
    def __init__(self, num_nodes, target=0.55):
        super().__init__()
        self.target = target
        self.scores = nn.Parameter(torch.randn(num_nodes) * 0.5)
        self.thresh = nn.Parameter(torch.tensor(0.0))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, temp=1.0):
        g = (
            F.softplus(self.scores / temp - torch.sigmoid(self.thresh) * 2 + 0.5)
            * self.scale.abs()
        )
        return g / (g.max() + 1e-8)

    def sparsity_loss(self, temp):
        return ((self.forward(temp) > 0.25).float().mean() - self.target) ** 2

    def mask(self, th=0.25):
        with torch.no_grad():
            return (self.forward(0.3) > th).int().cpu().numpy()

    def ranking_loss(self, gamma):
        g = self.forward(1.0)
        v = gamma > 0
        if v.sum() < 2:
            return torch.tensor(0.0, device=g.device)
        vg, vgam = g[v], gamma[v]
        loss = torch.tensor(0.0, device=g.device)
        n = vg.shape[0]
        for _ in range(min(50, n * (n - 1) // 2)):
            i, j = np.random.choice(n, 2, replace=False)
            if vgam[i] > vgam[j]:
                loss += F.relu(0.1 - (vg[i] - vg[j]))
            elif vgam[j] > vgam[i]:
                loss += F.relu(0.1 - (vg[j] - vg[i]))
        return loss / 50

    def recall_loss(self, gamma, temp):
        g = self.forward(temp)
        m = (gamma > 0).float() * (0.5 + 0.5 * gamma / (gamma.max() + 1e-8))
        return -torch.sum(m * torch.log(g + 1e-8)) / (m.sum() + 1e-8)


class WeatherAttention(nn.Module):
    def __init__(self, hist_len, num_nodes, emb_dim, heads=2, target=0.55):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.pos = nn.Parameter(torch.randn(hist_len, emb_dim) * 0.1)
        self.attn = nn.MultiheadAttention(
            emb_dim, heads, batch_first=True, dropout=0.1
        )
        self.gate = WeatherGate(num_nodes, target)
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, w, temp=1.0):
        """
        w: (B, N, H)
        """
        B, N, H = w.shape
        e = self.enc(w.unsqueeze(-1)) + self.pos  # (B,N,H,emb)
        flat = e.view(B * N, H, -1)
        out, attn = self.attn(flat[:, -1:], flat, flat)
        g = self.gate(temp)
        return (
            self.proj(out.squeeze(1).view(B, N, -1) * g.unsqueeze(0).unsqueeze(-1)),
            attn.view(B, N, H),
            g,
        )


class POILocalModule(nn.Module):
    def __init__(self, poi_dim, num_nodes):
        super().__init__()
        self.beta = nn.Parameter(torch.randn(poi_dim, num_nodes) * 0.5)

    def forward(self, poi_static):
        # poi_static: (B, N, D)
        return poi_static * self.beta.T.unsqueeze(0)

    def get_weights(self):
        return self.beta

    def l1_loss(self):
        return self.beta.abs().mean()

    def get_mask(self, threshold=0.02):
        with torch.no_grad():
            return (self.beta.abs() > threshold).int().cpu().numpy()


class POINeighborModule(nn.Module):
    def __init__(self, num_nodes, poi_dim, rank=10, adj_mask=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.poi_dim = poi_dim
        self.U = nn.Parameter(torch.randn(poi_dim, num_nodes, rank) * 0.3)
        self.V = nn.Parameter(torch.randn(poi_dim, num_nodes, rank) * 0.3)
        self.scale = nn.Parameter(torch.ones(poi_dim) * 0.5)
        if adj_mask is not None:
            self.register_buffer("adj_mask", adj_mask.float())
        else:
            self.adj_mask = None
        self.register_buffer("diag_mask", 1.0 - torch.eye(num_nodes))

    def get_weights(self):
        W = torch.einsum("dnr,dmr->dnm", self.U, self.V)
        W = W * torch.sigmoid(self.scale).view(-1, 1, 1)
        W = W * self.diag_mask.unsqueeze(0)
        if self.adj_mask is not None:
            W = W * self.adj_mask.unsqueeze(0)
        return W

    def forward(self, poi_static):
        """
        poi_static: (B, N, D)
        """
        W = self.get_weights()        # (D,N,N)
        poi_t = poi_static.permute(0, 2, 1)  # (B,D,N)
        effect = torch.einsum("dij,bdj->bdi", W, poi_t)
        return effect.permute(0, 2, 1)       # (B,N,D)

    def consistency_loss(self):
        if self.adj_mask is None:
            return torch.tensor(0.0)
        return (self.get_weights() * (1.0 - self.adj_mask.unsqueeze(0))).pow(2).mean()

    def l1_loss(self):
        return self.get_weights().abs().mean()

    def get_mask(self, threshold=0.02):
        with torch.no_grad():
            return (self.get_weights().abs() > threshold).int().cpu().numpy()


class AttrFusion(nn.Module):
    def __init__(self, traffic_dim, weather_dim, poi_dim, out_dim):
        super().__init__()
        self.t_proj = nn.Linear(traffic_dim, out_dim)
        self.w_proj = nn.Linear(weather_dim, out_dim)
        self.p_proj = nn.Linear(poi_dim, out_dim)
        self.inter = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, t, w, p):
        tp, wp, pp = self.t_proj(t), self.w_proj(w), self.p_proj(p)
        cat = torch.cat([tp, wp, pp], -1)
        g = self.gate(cat)
        return g * self.inter(cat) + (1 - g) * tp


class AdaptiveGNN(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, A, emb_dim=10):
        super().__init__()
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.gate = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.2)
        self.e1 = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.e2 = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.smax = Sparsemax(dim=1)
        self.l1 = nn.Linear(in_dim, out_dim)
        self.l2 = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.last_adapt = None

    def forward(self, X):
        """
        X: (B, N, F)
        """
        B = X.size(0)
        Ap = self.A * torch.sigmoid(self.gate)
        Aa = self.smax(self.e1 @ self.e2.T)
        self.last_adapt = Aa
        At = Ap + Aa
        h = F.gelu(torch.bmm(At.unsqueeze(0).expand(B, -1, -1), self.l1(X)))
        h = torch.bmm(At.unsqueeze(0).expand(B, -1, -1), self.l2(h))
        return F.gelu(self.norm(h))

    def get_adj(self, th):
        with torch.no_grad():
            total = (self.A * torch.sigmoid(self.gate)) + self.smax(self.e1 @ self.e2.T)
            return (total.cpu().numpy() > th).astype(int)


# =========================
# Main Model
# =========================
class STGNN(nn.Module):
    """
    使用 time_enc(t) 作为时间基底 φ(t)，每个节点 i 学一个系数向量 a_i：
      s_i(t) = a_i^T φ(t)， node_factor_i(t) = 1 + tanh(s_i(t))
    然后用 node_factor_i(t) 去调节该节点的属性特征（例如 POI 编码）。
    """

    def __init__(self, num_nodes, poi_dim, A, cfg, gamma=None,
                 true_node_pattern=None):
        super().__init__()
        self.cfg = cfg
        self.num_nodes = num_nodes
        self.poi_dim = poi_dim

        adj_mask = torch.tensor(A > 0, dtype=torch.float32)
        self.register_buffer("adj_mask", adj_mask)
        self.register_buffer(
            "gamma",
            torch.tensor(gamma, dtype=torch.float32)
            if gamma is not None else torch.zeros(num_nodes),
        )

        # 时间基底编码器 φ(t)
        # 节点自适应时间编码器：φ_i(t) 直接是 (B,N, TIME_EMB_DIM)
        self.time_enc = NodeAdaptiveTimeEncoder(
            cfg["STEPS_PER_DAY"],
            cfg["TIME_EMB_DIM"],
            num_nodes,
            num_freqs=4,
        )
        # shared readout，把时间 embedding 压成一个标量 s_i(t)
        self.node_time_readout = nn.Parameter(
            torch.randn(num_nodes, cfg["TIME_EMB_DIM"]) * 0.1
        )
        # 存一下真实节点时间模式（仅用于评估，不参与训练）
        if true_node_pattern is not None:
            tnp = torch.tensor(true_node_pattern, dtype=torch.float32)
            steps = cfg["STEPS_PER_DAY"]
            tnp = tnp[:, :steps]
            self.register_buffer("true_node_pattern", tnp)
        else:
            self.register_buffer("true_node_pattern",
                                 torch.zeros(num_nodes, cfg["STEPS_PER_DAY"]))

        self.weather = WeatherAttention(
            cfg["HIST_LEN"],
            num_nodes,
            cfg["WEATHER_EMB_DIM"],
            cfg["WEATHER_ATTN_HEADS"],
            cfg["WEATHER_SPARSITY_TARGET"],
        )
        self.poi_local = POILocalModule(poi_dim, num_nodes)
        self.poi_neighbor = POINeighborModule(
            num_nodes, poi_dim, cfg["POI_RANK"], adj_mask
        )
        self.poi_enc = nn.Sequential(
            nn.Linear(poi_dim, cfg["ATTR_FUSION_DIM"]),
            nn.GELU(),
            nn.Linear(cfg["ATTR_FUSION_DIM"], cfg["ATTR_FUSION_DIM"]),
        )
        self.attr = AttrFusion(
            1,
            cfg["WEATHER_EMB_DIM"],
            cfg["ATTR_FUSION_DIM"],
            cfg["ATTR_FUSION_DIM"],
        )
        self.gnn = AdaptiveGNN(
            num_nodes,
            cfg["ATTR_FUSION_DIM"],
            cfg["GCN_HID"],
            A,
            cfg["NODE_EMB_DIM"],
        )
        # 不再拼接 te，LSTM 只吃 GCN_HID
        self.lstm = nn.LSTM(
            cfg["GCN_HID"],
            cfg["LSTM_HID"],
            batch_first=True,
            num_layers=2,
            dropout=0.1,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg["LSTM_HID"], cfg["LSTM_HID"] // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg["LSTM_HID"] // 2, 1),
        )

    def forward(self, hist, tidx, temp=1.0):
        """
        hist: (B, N, H, C)
        tidx: (B,) 当前预测时间的时间索引 (0..STEPS_PER_DAY-1)
        """
        B, N, H, _ = hist.shape
        x, w, poi = hist[..., 0], hist[..., 1], hist[:, :, 0, 2:]

        we, _, _ = self.weather(w, temp)

        # 静态 POI 结构效应
        eff_l = self.poi_local(poi)
        eff_n = self.poi_neighbor(poi)
        poi_eff = eff_l + eff_n                    # (B, N, D)

        # POI 编码
        pe = self.poi_enc(poi_eff)                 # (B, N, F_attr)

        phi = self.time_enc(tidx)  # (B,N,D_te)

        # node_time_readout: (N, D_te)
        # 扩成 (1,N,D_te) 后与 phi 相乘，再在最后一维求和：
        s = (phi * self.node_time_readout.unsqueeze(0)).sum(dim=-1)  # (B,N)

        node_factor = 1.0 + torch.tanh(s)  # (B,N)

        # 用节点时间因子调节节点属性特征
        # ✅ 1) 调制 traffic 历史序列
        x_mod = x * node_factor.unsqueeze(-1)  # (B,N,H)

        # ✅ 2) 调制 weather embedding
        we_mod = we * node_factor.unsqueeze(-1)  # (B,N,W_emb)

        # （POI 也可以继续调制）
        pe_mod = pe * node_factor.unsqueeze(-1)  # (B,N,F_attr)

        gs = []
        for t in range(H):
            aug = self.attr(x_mod[:, :, t:t + 1], we_mod, pe_mod)
            g_out = self.gnn(aug)  # ✅ 不要传 adjs_time
            gs.append(g_out)

        seq = torch.cat(gs, dim=2).view(B * N, H, -1)   # (B*N, H, GCN_HID)
        out, _ = self.lstm(seq)
        return self.head(out[:, -1]).view(B, N)

    def get_unified_poi_weights(self):
        with torch.no_grad():
            D, N = self.poi_dim, self.num_nodes
            unified = self.poi_neighbor.get_weights().clone()
            local_w = self.poi_local.get_weights()
            for d in range(D):
                unified[d].fill_diagonal_(0)
                unified[d] += torch.diag(local_w[d])
            return unified.cpu().numpy()

    def get_node_time_patterns(self):
        """
        利用节点自适应时间编码，生成每个节点的时间模式：
          s_i(t) = time_readout( φ_i(t) )
          pattern_i(t) = tanh( s_i(t) )
        return: (N, steps)
        """
        steps = self.cfg["STEPS_PER_DAY"]
        device = next(self.parameters()).device
        with torch.no_grad():
            t = torch.arange(steps, device=device, dtype=torch.long)
            phi = self.time_enc(t)  # (T,N,D)
            # 原来是 self.time_readout(phi)
            s = (phi * self.node_time_readout.unsqueeze(0)).sum(dim=-1)  # (T,N)
            patt = torch.tanh(s).T  # (N,T)
            return patt.cpu().numpy()

    def analyze(self, cfg):
        with torch.no_grad():
            node_patterns = self.get_node_time_patterns()  # (N, steps)
            static_weights = self.get_unified_poi_weights()
            return (
                self.gnn.get_adj(cfg["THRESH_TRAFFIC"]),
                self.weather.gate.mask(cfg["THRESH_WEATHER"]),
                self.poi_local.get_mask(cfg["THRESH_POI"]),
                self.poi_neighbor.get_mask(cfg["THRESH_POI"]),
                node_patterns,
                static_weights,
            )

    def weather_sens(self):
        with torch.no_grad():
            return self.weather.gate(0.3).cpu().numpy()


# =========================
# Training
# =========================
def get_temp(ep, tot, ti, tf):
    return tf + 0.5 * (ti - tf) * (1 + np.cos(np.pi * ep / tot))


def get_scales(ep, cfg):
    s1, s2 = cfg["CURRICULUM_STAGE1_END"], cfg["CURRICULUM_STAGE2_END"]
    if ep <= s1:
        return 0.3 + 0.7 * ep / s1, 0.3 + 0.7 * ep / s1
    elif ep <= s2:
        return 1.0, 0.6 + 0.4 * (ep - s1) / (s2 - s1)
    return 1.0, 1.0


def train_epoch(model, opt, loader, dev, ep, cfg):
    model.train()
    total_loss, count = 0.0, 0
    temp = get_temp(ep, cfg["EPOCHS"], cfg["WEATHER_TEMP_INIT"], cfg["WEATHER_TEMP_FINAL"])
    ws, ps = get_scales(ep, cfg)
    rg = min(1.0, ep / (cfg["EPOCHS"] * 0.1))

    for hist, tidx, label, _target_t in loader:
        hist, tidx, label = hist.to(dev), tidx.to(dev), label.to(dev)
        opt.zero_grad()
        pred = model(hist, tidx, temp)
        mse = F.mse_loss(pred, label)

        # ... 原有的正则化项 ...
        lt = cfg["LAMBDA_GC_TRAFFIC"] * rg * torch.sigmoid(model.gnn.gate).sum()
        la = (
            cfg["LAMBDA_ADAPT_SPARSE"] * rg * model.gnn.last_adapt.abs().sum()
            if model.gnn.last_adapt is not None else 0
        )
        lws = cfg["WEATHER_SPARSITY_WEIGHT"] * ws * model.weather.gate.sparsity_loss(temp)
        lwr = (
                cfg["WEATHER_RANKING_WEIGHT"] * rg * ws
                * model.weather.gate.ranking_loss(model.gamma)
        )
        lwrc = (
                cfg["WEATHER_RECALL_WEIGHT"] * rg * ws
                * model.weather.gate.recall_loss(model.gamma, temp)
        )
        lpl = cfg["LAMBDA_GC_POI_LOCAL"] * rg * ps * model.poi_local.l1_loss()
        lpn = cfg["LAMBDA_GC_POI_NEIGHBOR"] * rg * ps * model.poi_neighbor.l1_loss()
        lpc = (
                cfg["POI_NEIGHBOR_CONSIST_WEIGHT"] * rg * ps
                * model.poi_neighbor.consistency_loss()
        )

        # === 新增：时间平滑约束 ===
        # 随着 rg (ramp-up) 逐渐加入，避免一开始就限制死了
        lsmooth = cfg["LAMBDA_TIME_SMOOTH"] * rg * model.time_enc.get_smoothness_loss()

        # === 新增步骤 2: 锚点相关性约束 (Anchor Correlation Loss) ===
        # 核心思想：模型学到的 pattern 应该与真实的交通流量趋势(label)正相关

        # 1. 获取模型对当前时刻 tidx 的时间因子预测
        #    我们需要调用 model 内部组件来获取。
        #    最简单的方式是直接用 model.time_enc(tidx) 算一遍
        #    (B, N, D)
        phi = model.time_enc(tidx)
        #    (B, N) - 计算出的时间因子 s
        s = (phi * model.node_time_readout.unsqueeze(0)).sum(dim=-1)
        learned_factor = torch.tanh(s)  # (B, N)

        # 2. 获取真实的交通流量 label (B, N)
        #    对其进行标准化，减去均值
        target_traffic = label

        # 3. 计算 Pearson 相关性 (负相关性作为 Loss)
        #    我们要最大化 learned_factor 和 target_traffic 的相关性
        #    Corr = Cov(x, y) / (Std(x) * Std(y))

        # 沿 Batch 维度 (dim=0) 计算相关性，得到每个节点的瞬时相关度
        # 为了数值稳定，加上 1e-6
        lf_centered = learned_factor - learned_factor.mean(dim=0, keepdim=True)
        tt_centered = target_traffic - target_traffic.mean(dim=0, keepdim=True)

        cov = (lf_centered * tt_centered).mean(dim=0)  # (N,)
        std_lf = learned_factor.std(dim=0) + 1e-6
        std_tt = target_traffic.std(dim=0) + 1e-6

        correlation = cov / (std_lf * std_tt)

        # Loss = 1 - 平均相关性 (范围 0~2)
        # 只有在前 50 epoch 强力引导，防止后期限制模型的非线性能力
        anchor_weight = 0.1 if ep < 50 else 0.01
        l_anchor = anchor_weight * (1.0 - correlation.mean())


        loss = mse + lt + la + lws + lwr + lwrc + lpl + lpn + lpc + lsmooth + l_anchor

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        total_loss += loss.item() * hist.size(0)
        count += hist.size(0)

    return total_loss / count


def evaluate(model, loader, dev):
    model.eval()
    total, count = 0, 0
    with torch.no_grad():
        for h, t, l, _target_t in loader:
            h, t, l = h.to(dev), t.to(dev), l.to(dev)
            total += F.mse_loss(model(h, t, 0.3), l, reduction="sum").item()
            count += h.size(0) * h.size(1)
    return total / count


def collect_val_predictions(model, loader, dev, cfg):
    """验证集逐样本：全局时间索引 target_t、真实流量 y_true、预测 y_pred，用于事后画连续时段对比曲线。"""
    model.eval()
    temp = float(cfg.get("WEATHER_TEMP_FINAL", 0.3))
    all_t, all_true, all_pred = [], [], []
    with torch.no_grad():
        for h, tidx, label, target_t in loader:
            h, tidx, label = h.to(dev), tidx.to(dev), label.to(dev)
            pred = model(h, tidx, temp)
            all_t.append(target_t.numpy().astype(np.int64))
            all_true.append(label.detach().cpu().numpy().astype(np.float64))
            all_pred.append(pred.detach().cpu().numpy().astype(np.float64))
    target_t = np.concatenate(all_t, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    return target_t, y_true, y_pred


# =========================
# Main
# =========================
def main():
    seed_everything(CONFIG["SEED"])
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    data = load_data(CONFIG["DATA_PATH"])
    X, W, poi, A = data["X"], data["W"], data["poi"], data["A"]
    ttr = data["GC_traffic"]
    tpl = data["GC_poi_local"]
    tpn = data["GC_poi_neighbor"]
    tw = data["GC_weather"]
    tgam = data["Weather_Gamma"]
    true_node_pattern = data["True_Node_Pattern"]  # (N, 24) or None
    w_local = data["POI_Local_Weights"]
    w_neighbor = data["POI_Neighbor_Weights"]

    has_truth = all(v is not None for v in [ttr, tpl, tpn, tw, tgam, true_node_pattern, w_local, w_neighbor])
    if has_truth:
        gc_unified, w_unified = build_unified_true_poi(tpl, tpn, w_local, w_neighbor)
    else:
        gc_unified, w_unified = None, None

    print(f"  Data: X={X.shape}, POI={poi.shape}")
    if has_truth:
        print(
            f"  True densities: Local={tpl.mean():.3f}, "
            f"Neighbor={tpn.mean():.3f}, Unified={gc_unified.mean():.3f}"
        )
    else:
        print("  (Real-data mode) No ground-truth causal graphs/weights found in npz. "
              "Disable truth-supervised losses/evaluations.")

    if np.abs(X.mean()) > 0.1:
        X = (X - X.mean()) / (X.std() + 1e-6)
        W = (W - W.mean()) / (W.std() + 1e-6)

    train_loader, val_loader, test_loader = create_loaders(X, W, poi, CONFIG)

    print("\n" + "=" * 70)
    print("STGNN V10.21 Time-Basis Node Dynamics")
    print("=" * 70)

    model = STGNN(
        X.shape[1],
        poi.shape[1],
        A,
        CONFIG,
        tgam if tgam is not None else data["_DEFAULT_gamma"],
        true_node_pattern=true_node_pattern,
    ).to(dev)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=50, T_mult=2
    )
    es = EarlyStopping(CONFIG["PATIENCE"])

    best_val, best_state = float("inf"), None
    for ep in range(1, CONFIG["EPOCHS"] + 1):
        loss = train_epoch(model, opt, train_loader, dev, ep, CONFIG)
        val = evaluate(model, val_loader, dev)
        sched.step()
        es(val)
        if val < best_val:
            best_val = val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0:
            print(f"Ep {ep:03d} | L: {loss:.5f} | V: {val:.5f}")
        if es.stop:
            print(f"Early stop @ {ep}")
            break

    if best_state:
        model.load_state_dict(best_state)

        ckpt_path = os.path.join(CONFIG["OUTPUT_DIR"], "best_model.pt")
        torch.save(best_state, ckpt_path)
        print(f"已保存最优权重: {ckpt_path}")

        train_eval_loader = DataLoader(
            train_loader.dataset, CONFIG["BATCH_SIZE"], shuffle=False
        )
        tr_t, tr_y, tr_p = collect_val_predictions(model, train_eval_loader, dev, CONFIG)
        n_train = int(tr_t.shape[0])

        tv, yv, pv = collect_val_predictions(model, val_loader, dev, CONFIG)
        n_val = int(tv.shape[0])

        n_test_ds = len(test_loader.dataset)
        if n_test_ds > 0:
            tt, yt, pt = collect_val_predictions(model, test_loader, dev, CONFIG)
            n_test = int(tt.shape[0])
        else:
            tt = yt = pt = None
            n_test = 0

        parts_t = [tr_t, tv]
        parts_y = [tr_y, yv]
        parts_p = [tr_p, pv]
        parts_s = [
            np.zeros(n_train, dtype=np.int8),
            np.ones(n_val, dtype=np.int8),
        ]
        if n_test > 0:
            parts_t.append(tt)
            parts_y.append(yt)
            parts_p.append(pt)
            parts_s.append(np.full(n_test, 2, dtype=np.int8))

        target_t = np.concatenate(parts_t, axis=0)
        y_true = np.concatenate(parts_y, axis=0)
        y_pred = np.concatenate(parts_p, axis=0)
        split = np.concatenate(parts_s, axis=0)

        val_series_path = os.path.join(CONFIG["OUTPUT_DIR"], "val_flow_predictions.npz")
        np.savez(
            val_series_path,
            target_t=target_t,
            y_true=y_true,
            y_pred=y_pred,
            split=split,
            split_version=np.int8(2),
            steps_per_day=np.int32(CONFIG["STEPS_PER_DAY"]),
            pred_len=np.int32(CONFIG["PRED_LEN"]),
        )
        print(
            f"已保存训练+验证+测试流量序列: {val_series_path} "
            f"[合计={target_t.shape[0]}（训练 {n_train} + 验证 {n_val} + 测试 {n_test}）, "
            f"节点数={y_true.shape[1]}]；split_version=2: 0=训练 1=验证 2=测试"
        )

        # === Analysis Output (no truth required) ===
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        pA, pW, pPL, pPN, node_patterns, learned_unified = model.analyze(CONFIG)

        # 保存结果为 .npz 文件
        output_file_path = os.path.join(CONFIG["OUTPUT_DIR"], "causal_analysis_results.npz")
        np.savez(
            output_file_path,
            traffic_analysis=pA,
            weather_analysis=pW,
            poi_local_analysis=pPL,
            poi_neighbor_analysis=pPN,
            node_patterns=node_patterns,
            learned_unified=learned_unified
        )

        print(f"因果分析结果已保存至：{output_file_path}")

        # === Truth-based evaluation (synthetic/with labels only) ===
        if has_truth:
            print("\n[Separate Evaluation]")
            for nm, p, tr in [
                ("Traffic", pA, ttr),
                ("Weather", pW, tw),
                ("POI Local", pPL, tpl),
                ("POI Neighbor", pPN, tpn),
            ]:
                pr, rc, f1, pd, td = metrics(p, tr)
                print(
                    f"[{nm:12s}] F1: {f1:.4f} | P: {pr:.4f} | R: {rc:.4f} | D: {pd:.3f}/{td:.3f}"
                )

            ws = model.weather_sens()
            vi = tgam > 0
            cw = np.corrcoef(ws[vi], tgam[vi])[0, 1] if vi.sum() > 1 else 0
            print(f"   >> W Intensity Corr: {cw:.4f}")

            steps = min(24, node_patterns.shape[1], true_node_pattern.shape[1])
            N_use = min(node_patterns.shape[0], true_node_pattern.shape[0])

            raw_corrs, norm_corrs = [], []
            for i in range(N_use):
                lp = node_patterns[i, :steps]
                tp = true_node_pattern[i, :steps]
                if lp.std() < 1e-6 or tp.std() < 1e-6:
                    continue

                c_raw = np.corrcoef(lp, tp)[0, 1]
                lp_n = (lp - lp.mean()) / (lp.std() + 1e-8)
                tp_n = (tp - tp.mean()) / (tp.std() + 1e-8)
                c_norm = np.corrcoef(lp_n, tp_n)[0, 1]

                if not np.isnan(c_raw):
                    raw_corrs.append(c_raw)
                if not np.isnan(c_norm):
                    norm_corrs.append(c_norm)

            avg_raw_node_corr = np.mean(raw_corrs) if raw_corrs else 0.0
            avg_norm_node_corr = np.mean(norm_corrs) if norm_corrs else 0.0
            print(f"[Node Time Pattern] Avg Raw Corr : {avg_raw_node_corr:.4f}")
            print(f"                    Avg Norm Corr: {avg_norm_node_corr:.4f}")

            print("\n[Unified POI Evaluation]")
            D, N = pPL.shape
            pred_unified = np.zeros((D, N, N))
            for d in range(D):
                pred_unified[d] = pPN[d].copy()
                np.fill_diagonal(pred_unified[d], pPL[d])

            pr, rc, f1, pd, td = metrics(pred_unified, gc_unified)
            print(
                f"[POI Unified ] F1: {f1:.4f} | P: {pr:.4f} | R: {rc:.4f} | D: {pd:.3f}/{td:.3f}"
            )

            w_corr = weight_correlation(learned_unified, w_unified)
            print(f"[POI Weights ] Static Correlation: {w_corr:.4f}")

            print("   >> Per POI Type (Static):")
            poi_names = ["Dining", "Shopping", "Office", "Residential"]
            for d in range(D):
                f1_d = f1_score(
                    gc_unified[d].flatten(),
                    (pred_unified[d] != 0).astype(int).flatten(),
                    zero_division=0,
                )
                w_corr_d = weight_correlation(learned_unified[d], w_unified[d])
                name = poi_names[d] if d < len(poi_names) else f"Type{d}"
                print(
                    f"      {name:12s}: F1={f1_d:.4f}, Static W-Corr={w_corr_d:.4f}"
                )

            vis_path = os.path.join(CONFIG["OUTPUT_DIR"], "time_pattern_compare.npz")
            np.savez(
                vis_path,
                node_patterns=node_patterns,
                true_node_pattern=true_node_pattern,
            )
            print(f"Saved time pattern compare data to: {vis_path}")

            print("\n" + "=" * 70)
            _, _, f1tr, _, _ = metrics(pA, ttr)
            _, _, f1w, _, _ = metrics(pW, tw)
            _, _, f1poi, _, _ = metrics(pred_unified, gc_unified)
            print(f"Traffic F1: {f1tr:.4f}")
            print(f"Weather F1: {f1w:.4f} | Intensity Corr: {cw:.4f}")
            print(f"POI Unified F1: {f1poi:.4f} | Static W-Corr: {w_corr:.4f}")
            print(f"Node Time Pattern Avg Norm Corr: {avg_norm_node_corr:.4f}")
            print("=" * 70)
        else:
            print("(Real-data mode) Saved causal analysis only. No truth-based evaluation performed.")


if __name__ == "__main__":
    main()
