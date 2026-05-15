"""
PDC v22 — 情境先验 + lag 级 Granger 打分 + 聚合成图
====================================================
相对 v21：记忆池不再返回 base_graph；改为 prototype_lag_prior / prototype_edge_prior 等先验，
经 LagGrangerScorer 得到 (i,j,ℓ) 分数，LagContextModulator 做情境差分调制后沿 ℓ 聚合为 causal_adj。

改法评析（设计层面）：
- 方向正确：把「检索模板图 + 图编辑」与「可解释的时滞贡献」叙事对齐，先验与数据驱动分工更清晰。
- 注意：轻量 pairwise 打分仍是端到端可微启发式，并非经典 Granger 的受限/全模型似然比；若要可检验性需另加 ablation 损失或显著性检验。
- 风险：去掉强 base_graph 后训练初期更不稳定，故保留 WEAK_BASE_FUSION 并支持按 epoch 退火到 0。
"""
import copy
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdc_causal_v11 as base
import v10_21_2_Version2 as v1021


CONFIG = copy.deepcopy(base.CONFIG)
CONFIG.update({
    "DATA_PATH": "./bronx_synthetic_data_v19_pinghua_d199/bronx_st_real.npz",
    "OUTPUT_DIR": "./output_pdc_v22_granger_pinghua_d199",
    "NODE_DYNAMIC_FREQS": 4,
    "CONTEXT_ATTN_HEADS": 4,
    "CONTEXT_ATTN_DROPOUT": 0.05,
    "CONTEXT_ATTN_REVERSE_PENALTY": 1.6,
    "PROTOTYPE_COUNT": 8,
    "PROTOTYPE_TOP_K": 3,
    # True: 检索权重为全体原型的 softmax（每图都有非零贡献）；False: 仅 TOP_K 个原型非零（易在「时间+天气」查询下塌到 1～2 个图）
    "PROTOTYPE_DENSE_RETRIEVAL": True,
    "PROTOTYPE_TEMP": 0.7,
    "PROTO_CTX_DIM": 64,
    "DELTA_HIDDEN": 64,
    "LAMBDA_PROTO_DIVERSITY": 1e-3,
    "LAMBDA_PROTO_USAGE": 1e-3,
    "LAMBDA_DELTA_SPARSE": 1e-3,
    # lag_gate 稀疏（替代原 delta 边编辑稀疏）
    "LAMBDA_LAG_GATE_SPARSE": 1e-3,
    "LAMBDA_CONTEXT_SEP": 1e-4,
    "LAMBDA_GRAPH_L1": 0.0,
    "LAMBDA_RANK": 0.0,
    "TOPK_RATIO": 0.10,
    "BURN_IN_EPOCHS": 10,
    "PROTO_MOMENTUM": 0.98,
    "CONFIDENCE_TEMP": 0.5,
    "EDGE_STATS_MOMENTUM": 0.95,
    "DELTA_LIMIT": 0.35,
    "CAUSAL_WARMUP_EPOCHS": 20,
    # 早期与静态路网 A 混合；从 epoch 0 线性退火至 WEAK_BASE_FUSION_END（分析期可为 0 得纯 Granger 聚合图）
    "WEAK_BASE_FUSION": 0.10,
    "WEAK_BASE_FUSION_END": 0.0,
    "WEAK_BASE_FUSION_RAMP_EPOCHS": 40,
    "GRANGER_NUM_LAGS": 4,
    "GRANGER_HIDDEN": 32,
    # True: 原型检索仅用外生随 t 变的量（时间+天气），POI 仅参与五源支路 q_ctx/q_node（方案B）
    "REGIME_USE_EXOG_ONLY": True,
    # 情境路由 router_regime 上天气 logit 偏置，与五源 router 的 WEATHER_SOURCE_BOOST 独立；默认 0 避免切换支路也被天气独大
    "REGIME_WEATHER_BOOST": 0.0,
    # True: 原型检索 query 用 cat(时间向量, 天气向量, 逐维乘积 t⊙w)+MLP，可区分「周末+暴雨」与单独周末/单独雨天；False: 仅用 time/weather 两 logits 凸组合（无法表达 AND 式联合）
    "REGIME_JOINT_INTERACTION": True,
})


class ContextEncoder(nn.Module):
    def __init__(self, cfg, num_nodes: int, spatial_dim: int, node_dynamic_dim: int):
        super().__init__()
        context_dim = cfg["CONTEXT_DIM"]
        weather_context_dim = cfg.get("WEATHER_CONTEXT_DIM", 10)
        self.context_dim = context_dim
        self.num_sources = 5
        self.weather_source_boost = cfg.get("WEATHER_SOURCE_BOOST", 1.0)

        self.time_proj = nn.Linear(cfg["TIME_EMB_DIM"], context_dim)
        self.weather_norm = nn.LayerNorm(weather_context_dim)
        self.weather_proj = nn.Sequential(
            nn.Linear(weather_context_dim, context_dim), nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        self.spatial_norm = nn.LayerNorm(spatial_dim)
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_dim, context_dim), nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        self.traffic_proj = nn.Sequential(
            nn.Linear(1, context_dim), nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        self.node_dynamic_norm = nn.LayerNorm(node_dynamic_dim)
        self.node_dynamic_proj = nn.Sequential(
            nn.Linear(node_dynamic_dim, context_dim), nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        num_heads = max(1, math.gcd(context_dim, cfg.get("CONTEXT_ATTN_HEADS", 4)))
        self.attn = nn.MultiheadAttention(
            embed_dim=context_dim,
            num_heads=num_heads,
            dropout=cfg.get("CONTEXT_ATTN_DROPOUT", 0.0),
            batch_first=True,
        )
        reverse_penalty = cfg.get("CONTEXT_ATTN_REVERSE_PENALTY", 1.6)
        bias = torch.zeros(self.num_sources, self.num_sources, dtype=torch.float32)
        cause_idx = [0, 1, 2]
        effect_idx = [3, 4]
        for q in cause_idx:
            for k in effect_idx:
                bias[q, k] = -float(reverse_penalty)
        self.register_buffer("context_attn_bias", bias)

        self.attn_norm = nn.LayerNorm(context_dim)
        self.ffn = nn.Sequential(
            nn.Linear(context_dim, context_dim * 2), nn.GELU(),
            nn.Linear(context_dim * 2, context_dim),
        )
        self.ffn_norm = nn.LayerNorm(context_dim)

        self.router = nn.Sequential(
            nn.Linear(context_dim * self.num_sources, context_dim), nn.GELU(),
            nn.Linear(context_dim, self.num_sources),
        )
        # 情境切换专用：仅时间+天气两路，与五源 router 参数完全独立
        self.num_regime_sources = 2
        self.router_regime = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim), nn.GELU(),
            nn.Linear(context_dim, self.num_regime_sources),
        )
        self.regime_joint_interaction = cfg.get("REGIME_JOINT_INTERACTION", True)
        self.regime_joint = nn.Sequential(
            nn.Linear(context_dim * 3, context_dim * 2), nn.GELU(),
            nn.Linear(context_dim * 2, context_dim),
        )
        self.regime_weather_boost = cfg.get("REGIME_WEATHER_BOOST", 0.0)
        self.proto_ctx_dim = cfg.get("PROTO_CTX_DIM", context_dim)
        self.global_proj = nn.Sequential(
            nn.Linear(context_dim, self.proto_ctx_dim), nn.GELU(),
            nn.Linear(self.proto_ctx_dim, self.proto_ctx_dim),
        )
        # 融合后的时间/天气向量（context_dim）→ prototype_ctx 维，不含 POI
        self.regime_proj = nn.Sequential(
            nn.Linear(context_dim, self.proto_ctx_dim * 2), nn.GELU(),
            nn.Linear(self.proto_ctx_dim * 2, self.proto_ctx_dim),
        )
        self.node_proj = nn.Sequential(
            nn.Linear(context_dim, context_dim), nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        self.last_source_weights = None
        self.last_regime_weights = None
        self.last_context_interaction_matrix = None

    def forward(self, time_feat, weather_stats, traffic_level, spatial_summary, node_dynamic_summary):
        bsz, num_nodes, _ = spatial_summary.shape
        time_query = self.time_proj(time_feat).unsqueeze(1).expand(-1, num_nodes, -1)
        weather_query = self.weather_proj(self.weather_norm(weather_stats)).unsqueeze(1).expand(-1, num_nodes, -1)
        spatial_query = self.spatial_proj(self.spatial_norm(spatial_summary))
        traffic_query = self.traffic_proj(traffic_level)
        dynamic_query = self.node_dynamic_proj(self.node_dynamic_norm(node_dynamic_summary))

        queries = torch.stack([time_query, weather_query, spatial_query, traffic_query, dynamic_query], dim=2)
        attn_in = queries.reshape(bsz * num_nodes, self.num_sources, self.context_dim)
        attn_out, attn_w = self.attn(
            attn_in, attn_in, attn_in,
            attn_mask=self.context_attn_bias,
            need_weights=True,
            average_attn_weights=False,
        )
        attended = self.attn_norm(attn_in + attn_out)
        attended = self.ffn_norm(attended + self.ffn(attended))
        attended = attended.view(bsz, num_nodes, self.num_sources, self.context_dim)
        flat = attended.reshape(bsz, num_nodes, -1)

        logits = self.router(flat)
        logits[:, :, 1] = logits[:, :, 1] + self.weather_source_boost
        source_weights = F.softmax(logits, dim=-1)
        node_ctx = (source_weights.unsqueeze(-1) * attended).sum(dim=2)
        global_ctx = node_ctx.mean(dim=1)
        # q_ctx：全局上下文 → lag 调制 / Granger 先验偏置；q_node：节点级 lag 调制
        q_ctx = self.global_proj(global_ctx)
        # 情境切换：batch 级时间 / 天气；不含 POI/交通/动态
        t_vec = self.time_proj(time_feat)
        w_vec = self.weather_proj(self.weather_norm(weather_stats))
        if self.regime_joint_interaction:
            # t⊙w 提供显式交叉项，经 MLP 可逼近「周末+下雨」等与单独周末、单独雨天不同的检索向量
            tw = t_vec * w_vec
            regime_feat = self.regime_joint(torch.cat([t_vec, w_vec, tw], dim=-1))
            q_ctx_switch = self.regime_proj(regime_feat)
            self.last_regime_weights = None
        else:
            regime_logits = self.router_regime(torch.cat([t_vec, w_vec], dim=-1))
            regime_logits[:, 1] = regime_logits[:, 1] + float(self.regime_weather_boost)
            regime_w = F.softmax(regime_logits, dim=-1)
            fused_regime = regime_w[:, 0:1] * t_vec + regime_w[:, 1:2] * w_vec
            q_ctx_switch = self.regime_proj(fused_regime)
            self.last_regime_weights = regime_w.detach()

        q_node = self.node_proj(node_ctx)
        self.last_source_weights = source_weights.detach()
        self.last_context_interaction_matrix = attn_w.view(bsz, num_nodes, attn_w.size(1), self.num_sources, self.num_sources).mean(dim=(0, 1, 2)).detach()
        return q_ctx_switch, q_ctx, q_node, source_weights

    def get_context_interaction_matrix(self):
        if self.last_context_interaction_matrix is None:
            return None
        return self.last_context_interaction_matrix.cpu().numpy()


class PrototypeCausalMemory(nn.Module):
    """情境记忆：仅返回先验与 prompt，不返回答案图。"""

    def __init__(
        self,
        num_nodes: int,
        context_dim: int,
        hidden_dim: int,
        num_prototypes: int,
        num_lags: int,
        top_k: int,
        temp: float,
        dense_retrieval: bool = False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        self.num_lags = num_lags
        self.top_k = top_k
        self.temp = temp
        self.dense_retrieval = dense_retrieval

        self.prototype_ctx = nn.Parameter(torch.randn(num_prototypes, context_dim) * 0.05)
        self.prototype_prompt = nn.Parameter(torch.randn(num_prototypes, hidden_dim) * 0.05)
        self.prototype_lag_prior = nn.Parameter(torch.randn(num_prototypes, num_lags) * 0.05)
        self.prototype_edge_prior_logits = nn.Parameter(torch.randn(num_prototypes, num_nodes, num_nodes) * 0.05)
        self.register_buffer("diag_mask", 1.0 - torch.eye(num_nodes))
        self.last_weights = None

    def get_proto_edge_prior(self):
        return torch.sigmoid(self.prototype_edge_prior_logits) * self.diag_mask.unsqueeze(0)

    def retrieve(self, q_ctx):
        sim = torch.matmul(F.normalize(q_ctx, dim=-1), F.normalize(self.prototype_ctx, dim=-1).T)
        dense = F.softmax(sim / max(self.temp, 1e-6), dim=-1)
        if self.dense_retrieval:
            weights = dense
        else:
            k = min(self.top_k, self.num_prototypes)
            topv, topi = torch.topk(sim, k=k, dim=-1)
            sparse = torch.zeros_like(sim)
            sparse.scatter_(-1, topi, F.softmax(topv / max(self.temp, 1e-6), dim=-1))
            weights = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)

        proto_edge_prior = self.get_proto_edge_prior()
        lag_prior = torch.matmul(weights, self.prototype_lag_prior)
        edge_prior = torch.einsum("bk,kij->bij", weights, proto_edge_prior)
        base_prompt = torch.matmul(weights, self.prototype_prompt)
        ctx_ref = torch.matmul(weights, self.prototype_ctx)
        self.last_weights = weights.detach()
        return {
            "weights": weights,
            "dense_weights": dense,
            "ctx_ref": ctx_ref,
            "lag_prior": lag_prior,
            "edge_prior": edge_prior,
            "base_prompt": base_prompt,
            "proto_lag_bank": self.prototype_lag_prior,
            "proto_edge_prior_bank": proto_edge_prior,
        }

    def diversity_loss(self):
        dev = self.prototype_ctx.device
        eye = torch.eye(self.num_prototypes, device=dev)
        ctx = F.normalize(self.prototype_ctx, dim=-1)
        ctx_gram = ctx @ ctx.T
        lag = F.normalize(self.prototype_lag_prior, dim=-1)
        lag_gram = lag @ lag.T
        ep = self.get_proto_edge_prior().view(self.num_prototypes, -1)
        ep = F.normalize(ep, dim=-1)
        ep_gram = ep @ ep.T
        prompt = F.normalize(self.prototype_prompt, dim=-1)
        prompt_gram = prompt @ prompt.T
        return (
            ((ctx_gram - eye) ** 2).mean()
            + ((lag_gram - eye) ** 2).mean()
            + ((ep_gram - eye) ** 2).mean()
            + ((prompt_gram - eye) ** 2).mean()
        )


class LagGrangerScorer(nn.Module):
    """轻量 (i,j,ℓ) 打分：源节点在滞后 ℓ 的流量 + 目标节点上下文，叠加上情境先验偏置。"""

    def __init__(self, cfg, num_nodes: int, node_dim: int, ctx_dim: int, max_lags: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.max_lags = max_lags
        d = cfg.get("GRANGER_HIDDEN", 32)
        self.src_mlp = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.lag_emb = nn.Embedding(max_lags, d)
        self.tgt_mlp = nn.Linear(node_dim + ctx_dim, d)
        self.score_mlp = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 1))
        self.bias_edge = nn.Parameter(torch.tensor(0.5))
        self.bias_lag = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_hist, q_ctx, q_node, lag_prior, edge_prior):
        """
        x_hist: (B, N, H) 流量通道
        q_ctx: (B, Dg), q_node: (B, N, Dn)
        lag_prior: (B, L), edge_prior: (B, N, N)
        """
        bsz, num_nodes, hist_len = x_hist.shape
        L = min(self.max_lags, max(0, hist_len - 1))
        if L == 0:
            z = x_hist.new_zeros(bsz, num_nodes, num_nodes, 1)
            return z, torch.sigmoid(z)

        scores = []
        for ell in range(L):
            idx = -(ell + 1)
            sv = x_hist[:, :, idx].unsqueeze(-1)
            sj = self.src_mlp(sv) + self.lag_emb.weight[ell].view(1, 1, -1)
            tgt_in = torch.cat([q_node, q_ctx.unsqueeze(1).expand(-1, num_nodes, -1)], dim=-1)
            ti = self.tgt_mlp(tgt_in)
            ti_e = ti.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            sj_e = sj.unsqueeze(1).expand(-1, num_nodes, -1, -1)
            pair = torch.cat([ti_e, sj_e], dim=-1)
            s = self.score_mlp(pair).squeeze(-1)
            scores.append(s)
        lag_score = torch.stack(scores, dim=-1)

        lp = lag_prior[:, :L].unsqueeze(1).unsqueeze(2)
        bias = self.bias_edge * edge_prior.unsqueeze(-1) + self.bias_lag * torch.tanh(lp)
        lag_score = lag_score + bias
        lag_gate = torch.sigmoid(lag_score)
        return lag_score, lag_gate


class LagContextModulator(nn.Module):
    """用 q_ctx、q_node 与 ctx_ref 的差分调制各滞后分数，不直接改邻接矩阵。"""

    def __init__(self, cfg, node_dim: int, ctx_dim: int, num_lags: int):
        super().__init__()
        self.num_lags = num_lags
        h = cfg.get("DELTA_HIDDEN", cfg["CONTEXT_DIM"])
        gin = 2 * ctx_dim + num_lags
        self.global_mod = nn.Sequential(nn.Linear(gin, h), nn.GELU(), nn.Linear(h, num_lags))
        nin = node_dim + ctx_dim
        self.node_mod = nn.Sequential(nn.Linear(nin, h), nn.GELU(), nn.Linear(h, num_lags))

    def forward(self, q_ctx, q_node, ctx_ref, lag_prior, raw_lag_score):
        ctx_diff = q_ctx - ctx_ref
        B, N, L = q_node.shape[0], q_node.shape[1], raw_lag_score.shape[-1]
        lag_slice = lag_prior[:, :L]
        g = torch.cat([q_ctx, ctx_diff, lag_slice], dim=-1)
        lag_scale_global = torch.sigmoid(self.global_mod(g)).view(B, 1, 1, L)
        n_in = torch.cat([q_node, ctx_diff.unsqueeze(1).expand(-1, N, -1)], dim=-1)
        lag_scale_node = torch.sigmoid(self.node_mod(n_in)).unsqueeze(2)
        modulated = raw_lag_score * lag_scale_global * lag_scale_node
        return {
            "lag_scale_global": lag_scale_global,
            "lag_scale_node": lag_scale_node,
            "modulated_lag_score": modulated,
            "ctx_diff": ctx_diff,
        }


class ContextPromptDelta(nn.Module):
    def __init__(self, global_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim * 2, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, q_ctx, ctx_ref):
        ctx_diff = q_ctx - ctx_ref
        return self.net(torch.cat([q_ctx, ctx_diff], dim=-1))


class CausalOnlyGNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, A: np.ndarray, weak_base_fusion: float = 0.10):
        super().__init__()
        self.l1 = nn.Linear(in_dim, out_dim)
        self.l2 = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.base_adj = nn.Parameter(torch.tensor(A, dtype=torch.float32), requires_grad=False)
        self.weak_base_fusion = weak_base_fusion
        self.last_adj = None

    def forward(self, x, causal_adj, weak_fusion: Optional[float] = None):
        wf = self.weak_base_fusion if weak_fusion is None else float(weak_fusion)
        base = self.base_adj.to(x.device).unsqueeze(0)
        mixed_adj = (1.0 - wf) * causal_adj + wf * base
        mixed_adj = mixed_adj / (mixed_adj.amax(dim=(1, 2), keepdim=True) + 1e-8)
        self.last_adj = mixed_adj.detach()
        hidden = F.gelu(torch.bmm(mixed_adj, self.l1(x)))
        hidden = F.gelu(self.norm(torch.bmm(mixed_adj, self.l2(hidden))))
        return hidden


class EdgeStatistics(nn.Module):
    def __init__(self, num_nodes: int, momentum: float = 0.95):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("edge_mean", torch.zeros(num_nodes, num_nodes))
        self.register_buffer("edge_sq_mean", torch.zeros(num_nodes, num_nodes))
        self.register_buffer("edge_repeatability", torch.zeros(num_nodes, num_nodes))
        self.register_buffer("updates", torch.tensor(0.0))
        self.register_buffer("diag_mask", 1.0 - torch.eye(num_nodes))

    @torch.no_grad()
    def update(self, adj: torch.Tensor, proto_weights: Optional[torch.Tensor] = None):
        mean_adj = adj.mean(dim=0)
        sq_adj = (adj ** 2).mean(dim=0)
        m = self.momentum
        self.edge_mean.mul_(m).add_((1.0 - m) * mean_adj)
        self.edge_sq_mean.mul_(m).add_((1.0 - m) * sq_adj)
        if proto_weights is not None:
            peaked = proto_weights.max(dim=-1).values.mean()
            self.edge_repeatability.mul_(m).add_((1.0 - m) * peaked * (mean_adj > mean_adj.mean()).float())
        self.updates.add_(1.0)

    def confidence(self, proto_conf: torch.Tensor):
        var = (self.edge_sq_mean - self.edge_mean ** 2).clamp(min=0.0)
        stability = torch.exp(-4.0 * var)
        repeatability = torch.clamp(self.edge_repeatability, 0.0, 1.0)
        confidence = 0.50 * proto_conf + 0.30 * stability.unsqueeze(0) + 0.20 * repeatability.unsqueeze(0)
        confidence = confidence * self.diag_mask.unsqueeze(0)
        return confidence.clamp(0.0, 1.0)


class PDC_CausalV22(nn.Module):
    def __init__(self, num_nodes: int, poi_dim: int, A: np.ndarray, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_nodes = num_nodes
        self.poi_dim = poi_dim
        self.hidden_dim = cfg["GCN_HID"]
        self.num_granger_lags = cfg.get("GRANGER_NUM_LAGS", 4)
        self.register_buffer("diag_mask", 1.0 - torch.eye(num_nodes))

        self.time_enc = base.AdvancedTimeEncoder(cfg["STEPS_PER_DAY"], cfg["TIME_EMB_DIM"], num_nodes)
        self.node_dynamic_time_enc = v1021.NodeAdaptiveTimeEncoder(
            cfg["STEPS_PER_DAY"], cfg["TIME_EMB_DIM"], num_nodes, num_freqs=cfg.get("NODE_DYNAMIC_FREQS", 4)
        )
        self.node_time_readout = nn.Parameter(torch.randn(num_nodes, cfg["TIME_EMB_DIM"]) * 0.1)
        self.weather = base.WeatherAttention(
            cfg["HIST_LEN"], num_nodes, cfg["WEATHER_EMB_DIM"],
            cfg["WEATHER_ATTN_HEADS"], cfg["WEATHER_SPARSITY_TARGET"]
        )
        self.poi_encoder = nn.Sequential(
            nn.Linear(poi_dim, cfg["ATTR_FUSION_DIM"]), nn.GELU(),
            nn.Linear(cfg["ATTR_FUSION_DIM"], cfg["ATTR_FUSION_DIM"]),
        )
        self.attr = base.AttrFusion(1, cfg["WEATHER_EMB_DIM"], cfg["ATTR_FUSION_DIM"], cfg["ATTR_FUSION_DIM"])

        node_dynamic_dim = cfg["TIME_EMB_DIM"] * 2 + 2
        self.context_encoder = ContextEncoder(cfg, num_nodes, cfg["ATTR_FUSION_DIM"], node_dynamic_dim)
        proto_dim = cfg.get("PROTO_CTX_DIM", cfg["CONTEXT_DIM"])
        self.prototype_memory = PrototypeCausalMemory(
            num_nodes=num_nodes,
            context_dim=proto_dim,
            hidden_dim=cfg["GCN_HID"],
            num_prototypes=cfg.get("PROTOTYPE_COUNT", 8),
            num_lags=self.num_granger_lags,
            top_k=cfg.get("PROTOTYPE_TOP_K", 3),
            temp=cfg.get("PROTOTYPE_TEMP", 0.7),
            dense_retrieval=cfg.get("PROTOTYPE_DENSE_RETRIEVAL", True),
        )
        self.edge_stats = EdgeStatistics(num_nodes, momentum=cfg.get("EDGE_STATS_MOMENTUM", 0.95))
        self.lag_granger = LagGrangerScorer(
            cfg, num_nodes=num_nodes,
            node_dim=cfg["CONTEXT_DIM"], ctx_dim=proto_dim,
            max_lags=self.num_granger_lags,
        )
        self.lag_modulator = LagContextModulator(
            cfg, node_dim=cfg["CONTEXT_DIM"], ctx_dim=proto_dim, num_lags=self.num_granger_lags,
        )
        self.prompt_delta = ContextPromptDelta(proto_dim, cfg["GCN_HID"])
        self.gnn = CausalOnlyGNN(
            in_dim=cfg["ATTR_FUSION_DIM"],
            out_dim=cfg["GCN_HID"],
            A=A,
            weak_base_fusion=cfg.get("WEAK_BASE_FUSION", 0.10),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=4,
            dim_feedforward=self.hidden_dim * 2,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.node_prompt_mod = base.NodeCausalPromptModulator(self.hidden_dim, poi_dim)
        self.node_dynamic_prompt_proj = nn.Sequential(
            nn.Linear(cfg["TIME_EMB_DIM"], self.hidden_dim), nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.film = base.FiLM(self.hidden_dim, self.hidden_dim, init_std=0.02)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(self.hidden_dim // 2, 1),
        )

        self.last_proto_weights = None
        self.last_edge_prior = None
        self.last_causal_adj = None
        self.last_ctx_diff = None
        self.last_source_weights = None
        self.last_q_ctx_switch = None
        self.last_lag_score = None
        self.last_lag_gate = None
        self.last_modulated_lag = None

    def _rank_loss(self, pred_adj, target_adj, ratio=0.10):
        b, n, _ = pred_adj.shape
        offdiag = (1.0 - torch.eye(n, device=pred_adj.device)).bool().view(1, -1)
        pred_flat = pred_adj.view(b, -1)[:, offdiag.squeeze(0)]
        tgt_flat = target_adj.view(b, -1)[:, offdiag.squeeze(0)]
        num_edges = pred_flat.size(1)
        k = max(1, int(num_edges * ratio))
        top_idx = torch.topk(tgt_flat, k=k, dim=1).indices
        bottom_idx = torch.topk(-tgt_flat, k=k, dim=1).indices
        top_pred = torch.gather(pred_flat, 1, top_idx).mean(dim=1)
        bottom_pred = torch.gather(pred_flat, 1, bottom_idx).mean(dim=1)
        return F.relu(0.1 - (top_pred - bottom_pred)).mean()

    def forward(self, hist, time_idx, temp=1.0, mask_ratio=0.0, epoch: int = 0, target_graph=None):
        bsz, num_nodes, hist_len, _ = hist.shape
        x, w, poi = hist[..., 0], hist[..., 1], hist[:, :, 0, 2:]

        weather_emb, _ = self.weather(w, temp)
        poi_emb = self.poi_encoder(poi)
        _, time_base = self.time_enc(time_idx)
        hour_idx = time_idx[:, 0]
        node_dynamic_emb = self.node_dynamic_time_enc(hour_idx)

        if hist_len >= 2:
            node_dynamic_signal = x[:, :, -1] - x[:, :, -2]
        else:
            node_dynamic_signal = torch.zeros(bsz, num_nodes, device=hist.device, dtype=x.dtype)
        node_factor = 1.0 + torch.tanh(node_dynamic_signal)
        weather_stats = base.build_weather_context_features(w)
        traffic_level = x.mean(dim=2, keepdim=True)
        dynamic_summary = torch.cat([
            node_dynamic_emb,
            node_dynamic_emb.mean(dim=1, keepdim=True).expand(-1, num_nodes, -1),
            node_dynamic_signal.unsqueeze(-1),
            node_factor.unsqueeze(-1),
        ], dim=-1)

        q_ctx_switch, q_ctx, q_node, source_weights = self.context_encoder(
            time_feat=time_base,
            weather_stats=weather_stats,
            traffic_level=traffic_level,
            spatial_summary=poi_emb,
            node_dynamic_summary=dynamic_summary,
        )
        proto_q = q_ctx_switch if self.cfg.get("REGIME_USE_EXOG_ONLY", True) else q_ctx
        proto_info = self.prototype_memory.retrieve(proto_q)
        edge_gate = self.edge_stats.confidence(proto_info["edge_prior"])

        warm = max(1, self.cfg.get("WEAK_BASE_FUSION_RAMP_EPOCHS", 40))
        w0 = self.cfg.get("WEAK_BASE_FUSION", 0.10)
        w1 = self.cfg.get("WEAK_BASE_FUSION_END", 0.0)
        weak_fusion = w0 + (w1 - w0) * min(1.0, float(epoch) / warm)

        lag_score, lag_gate = self.lag_granger(
            x_hist=x,
            q_ctx=q_ctx,
            q_node=q_node,
            lag_prior=proto_info["lag_prior"],
            edge_prior=proto_info["edge_prior"],
        )
        L_eff = lag_score.shape[-1]
        L_full = self.num_granger_lags
        if L_eff < L_full:
            pad = lag_score.new_zeros(bsz, num_nodes, num_nodes, L_full - L_eff)
            lag_score = torch.cat([lag_score, pad], dim=-1)
            lag_gate = torch.cat([lag_gate, pad], dim=-1)

        lag_mod = self.lag_modulator(
            q_ctx, q_node, proto_info["ctx_ref"], proto_info["lag_prior"], lag_score,
        )
        modulated = lag_mod["modulated_lag_score"]
        modulated_gate = torch.sigmoid(modulated)
        edge_from_lags = modulated_gate.sum(dim=-1) * self.diag_mask.unsqueeze(0)
        causal_adj = edge_from_lags / (edge_from_lags.amax(dim=(1, 2), keepdim=True) + 1e-8)

        ctx_diff = lag_mod["ctx_diff"]
        combined_prompt = proto_info["base_prompt"] + self.prompt_delta(q_ctx, proto_info["ctx_ref"])

        x_mod = x * node_factor.unsqueeze(-1)
        weather_mod = weather_emb * node_factor.unsqueeze(-1)
        seq_features = []
        for t in range(hist_len):
            fused_attr = self.attr(x_mod[:, :, t:t + 1], weather_mod, poi_emb)
            graph_out = self.gnn(fused_attr, causal_adj, weak_fusion=weak_fusion)
            seq_features.append(graph_out)

        batch_node_seq = torch.stack(seq_features, dim=2).view(bsz * num_nodes, hist_len, self.hidden_dim)
        if self.training and mask_ratio > 0:
            mask = torch.rand(bsz * num_nodes, hist_len, device=hist.device) < mask_ratio
            batch_node_seq[mask] = 0.0

        out = self.transformer(batch_node_seq)
        final_feat = out[:, -1, :]
        traffic_node_mean = x.mean(dim=2).unsqueeze(-1)
        node_prompt = self.node_prompt_mod(proto_info["base_prompt"], poi, traffic_node_mean)
        node_prompt = node_prompt + combined_prompt.unsqueeze(1).expand(-1, num_nodes, -1).reshape(bsz * num_nodes, -1)
        node_prompt = node_prompt + self.node_dynamic_prompt_proj(node_dynamic_emb.reshape(bsz * num_nodes, -1))
        final_feat = self.film(final_feat, node_prompt)
        pred = self.head(final_feat).view(bsz, num_nodes)

        self.last_proto_weights = proto_info["weights"].detach()
        self.last_edge_prior = proto_info["edge_prior"].detach()
        self.last_causal_adj = causal_adj.detach()
        self.last_ctx_diff = ctx_diff.detach()
        self.last_source_weights = source_weights.detach()
        self.last_q_ctx_switch = q_ctx_switch.detach()
        self.last_lag_score = lag_score.detach()
        self.last_lag_gate = modulated_gate.detach()
        self.last_modulated_lag = modulated.detach()

        aux = {
            "proto_weights": proto_info["weights"],
            "edge_prior": proto_info["edge_prior"],
            "lag_prior": proto_info["lag_prior"],
            "causal_adj": causal_adj,
            "ctx_diff": ctx_diff,
            "lag_score": lag_score,
            "lag_gate": modulated_gate,
            "modulated_lag": modulated,
            "source_weights": source_weights,
            "regime_weights": self.context_encoder.last_regime_weights,
            "edge_gate": edge_gate,
            "q_ctx": q_ctx,
            "q_ctx_switch": q_ctx_switch,
        }
        if target_graph is not None:
            aux["graph_l1"] = F.l1_loss(causal_adj, target_graph)
            aux["rank_loss"] = self._rank_loss(causal_adj, target_graph, ratio=self.cfg.get("TOPK_RATIO", 0.10))
        return pred, aux

    @torch.no_grad()
    def update_edge_statistics(self):
        if self.last_causal_adj is None:
            return
        self.edge_stats.update(self.last_causal_adj, self.last_proto_weights)

    def prototype_diversity_loss(self):
        return self.prototype_memory.diversity_loss()

    def delta_sparsity_loss(self):
        if self.last_lag_gate is None:
            return torch.tensor(0.0, device=self.prototype_memory.prototype_ctx.device)
        return self.last_lag_gate.mean()

    def context_separation_loss(self):
        if self.last_proto_weights is None or self.last_ctx_diff is None:
            return torch.tensor(0.0, device=self.prototype_memory.prototype_ctx.device)
        peaked = self.last_proto_weights.max(dim=-1).values
        return (1.0 - peaked).mean() + 0.1 * self.last_ctx_diff.norm(dim=-1).mean()

    def get_node_dynamic_patterns(self):
        steps = self.cfg["STEPS_PER_DAY"]
        device = self.node_time_readout.device
        with torch.no_grad():
            t = torch.arange(steps, device=device, dtype=torch.long)
            phi = self.node_dynamic_time_enc(t)
            signal = (phi * self.node_time_readout.unsqueeze(0)).sum(dim=-1)
            return torch.tanh(signal).transpose(0, 1).cpu().numpy()

    def analyze(self, cfg):
        with torch.no_grad():
            weather_gate = self.weather.gate(0.3).cpu().numpy()
            node_dynamic_patterns = self.get_node_dynamic_patterns()
            lag_score_m = None
            lag_gate_m = None
            lag_peak = None
            if self.last_lag_score is not None:
                ls = self.last_lag_score
                lag_score_m = ls.mean(dim=0).cpu().numpy()
                lag_peak = ls.argmax(dim=-1).float().mean().item()
            if self.last_lag_gate is not None:
                lag_gate_m = self.last_lag_gate.mean(dim=0).cpu().numpy()
            return {
                "weather_gate": weather_gate,
                "node_dynamic_patterns": node_dynamic_patterns,
                "context_interaction_matrix": self.context_encoder.get_context_interaction_matrix(),
                "prototype_weights": None if self.last_proto_weights is None else self.last_proto_weights.mean(dim=0).cpu().numpy(),
                "prototype_lag_prior_bank": self.prototype_memory.prototype_lag_prior.detach().cpu().numpy(),
                "prototype_edge_prior_bank": self.prototype_memory.get_proto_edge_prior().detach().cpu().numpy(),
                "final_context_adj": None if self.last_causal_adj is None else self.last_causal_adj.mean(dim=0).cpu().numpy(),
                "edge_prior_mean": None if self.last_edge_prior is None else self.last_edge_prior.mean(dim=0).cpu().numpy(),
                "lag_score_mean": lag_score_m,
                "lag_gate_mean": lag_gate_m,
                "lag_peak_index": lag_peak,
                "ctx_diff_norm": None if self.last_ctx_diff is None else self.last_ctx_diff.norm(dim=-1).mean().item(),
                "q_ctx_switch_mean": None if self.last_q_ctx_switch is None else self.last_q_ctx_switch.mean(dim=0).cpu().numpy(),
                "source_usage": None if self.last_source_weights is None else self.last_source_weights.mean(dim=(0, 1)).cpu().numpy(),
                "regime_source_usage": None
                if self.context_encoder.last_regime_weights is None
                else self.context_encoder.last_regime_weights.mean(dim=0).cpu().numpy(),
                "edge_mean": self.edge_stats.edge_mean.cpu().numpy(),
                "edge_var": (self.edge_stats.edge_sq_mean - self.edge_stats.edge_mean ** 2).clamp(min=0.0).cpu().numpy(),
                "edge_repeatability": self.edge_stats.edge_repeatability.cpu().numpy(),
                "edge_stability": torch.exp(
                    -4.0 * (self.edge_stats.edge_sq_mean - self.edge_stats.edge_mean ** 2).clamp(min=0.0)
                ).cpu().numpy(),
            }


@dataclass
class EpochStats:
    train_loss: float
    pred_loss: float
    proto_entropy: float
    delta_sparse: float


def train_epoch(model, opt, loader, dev, epoch, cfg) -> EpochStats:
    model.train()
    total_loss = total_pred = total_entropy = total_delta = 0.0
    count = 0
    temp = 1.0 - 0.7 * (epoch / max(cfg["EPOCHS"], 1))
    mask_ratio = cfg["MASK_RATIO"] if epoch < cfg["EPOCHS"] * 0.8 else 0.0

    for batch in loader:
        if len(batch) >= 5:
            hist, time_idx, label, _, target_graph = batch[:5]
            target_graph = target_graph.to(dev)
        else:
            hist, time_idx, label = batch[:3]
            target_graph = None

        hist = hist.to(dev)
        time_idx = time_idx.to(dev)
        label = label.to(dev)

        opt.zero_grad()
        pred, aux = model(hist, time_idx, temp=temp, mask_ratio=mask_ratio, epoch=epoch, target_graph=target_graph)
        pred_loss = F.mse_loss(pred, label)
        loss = pred_loss
        loss = loss + cfg["WEATHER_SPARSITY_WEIGHT"] * model.weather.gate.sparsity_loss(temp)
        loss = loss + cfg.get("LAMBDA_PROTO_DIVERSITY", 1e-3) * model.prototype_diversity_loss()
        loss = loss + cfg.get("LAMBDA_LAG_GATE_SPARSE", cfg.get("LAMBDA_DELTA_SPARSE", 1e-3)) * model.delta_sparsity_loss()
        loss = loss + cfg.get("LAMBDA_CONTEXT_SEP", 1e-4) * model.context_separation_loss()

        proto_entropy = -(aux["proto_weights"] * torch.log(aux["proto_weights"] + 1e-8)).sum(dim=-1).mean()
        k_proto = cfg.get("PROTOTYPE_COUNT", 8)
        if cfg.get("PROTOTYPE_DENSE_RETRIEVAL", True):
            target_entropy = math.log(max(2, k_proto))
        else:
            target_entropy = math.log(max(2, min(k_proto, cfg.get("PROTOTYPE_TOP_K", 3) + 1)))
        loss = loss + cfg.get("LAMBDA_PROTO_USAGE", 1e-3) * F.relu(torch.tensor(target_entropy, device=dev) - proto_entropy).pow(2)

        if target_graph is not None:
            loss = loss + cfg.get("LAMBDA_GRAPH_L1", 0.0) * aux["graph_l1"]
            loss = loss + cfg.get("LAMBDA_RANK", 0.0) * aux["rank_loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        model.update_edge_statistics()

        bsz = hist.size(0)
        total_loss += loss.item() * bsz
        total_pred += pred_loss.item() * bsz
        total_entropy += proto_entropy.item() * bsz
        total_delta += model.delta_sparsity_loss().item() * bsz
        count += bsz

    return EpochStats(
        train_loss=total_loss / max(count, 1),
        pred_loss=total_pred / max(count, 1),
        proto_entropy=total_entropy / max(count, 1),
        delta_sparse=total_delta / max(count, 1),
    )


def evaluate(model, loader, dev, cfg):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            hist, time_idx, label = batch[:3]
            hist = hist.to(dev)
            time_idx = time_idx.to(dev)
            label = label.to(dev)
            pred, _ = model(hist, time_idx, temp=0.3, mask_ratio=0.0, epoch=cfg["EPOCHS"], target_graph=None)
            total += F.mse_loss(pred, label, reduction="sum").item()
            count += hist.size(0) * hist.size(1)
    return total / max(count, 1)


def main():
    base.seed_everything(CONFIG["SEED"])
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("PDC v22 - regime priors + lag Granger aggregation")
    data = base.load_data(CONFIG["DATA_PATH"])
    X, W, poi, A = data["X"], data["W"], data["poi"], data["A"]
    if np.abs(X.mean()) > 0.1:
        X = (X - X.mean()) / (X.std() + 1e-6)
        W = (W - W.mean()) / (W.std() + 1e-6)

    train_loader, val_loader = base.create_loaders(X, W, poi, CONFIG)
    model = PDC_CausalV22(X.shape[1], poi.shape[1], A, CONFIG).to(dev)
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50)
    stopper = base.EarlyStopping(CONFIG["PATIENCE"])

    best_val = float("inf")
    best_state = None
    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        train_stats = train_epoch(model, opt, train_loader, dev, epoch, CONFIG)
        val_loss = evaluate(model, val_loader, dev, CONFIG)
        sched.step()
        stopper(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(CONFIG["OUTPUT_DIR"], "best_model.pt"))

        if epoch % 10 == 0:
            proto_usage = None
            if model.last_proto_weights is not None:
                proto_usage = model.last_proto_weights.mean(dim=0).detach().cpu().numpy()
            source_usage = None
            if model.last_source_weights is not None:
                source_usage = model.last_source_weights.mean(dim=(0, 1)).detach().cpu().numpy()
            regime_u = None
            if model.context_encoder.last_regime_weights is not None:
                regime_u = model.context_encoder.last_regime_weights.mean(dim=0).detach().cpu().numpy()
            if model.context_encoder.regime_joint_interaction:
                regime_str = "joint(t,w,t*w)"
            else:
                regime_str = f"[time,weather]={None if regime_u is None else np.round(regime_u, 3)}"
            print(
                f"Ep {epoch:03d} | Train {train_stats.train_loss:.5f} | Val {val_loss:.5f} | "
                f"Pred {train_stats.pred_loss:.5f} | ProtoH {train_stats.proto_entropy:.4f} | "
                f"LagG {train_stats.delta_sparse:.5f} | "
                f"ProtoUsage={None if proto_usage is None else np.round(proto_usage, 3)} | "
                f"SourceUsage={None if source_usage is None else np.round(source_usage, 3)} | "
                f"Regime={regime_str}"
            )

        if stopper.stop:
            print(f"Early stop @ Ep {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(dev)

    with torch.no_grad():
        sample_h, sample_t, _, _ = next(iter(val_loader))
        model(sample_h.to(dev), sample_t.to(dev), temp=0.3, mask_ratio=0.0, epoch=CONFIG["EPOCHS"])

    analysis = model.analyze(CONFIG)
    out_path = os.path.join(CONFIG["OUTPUT_DIR"], "res_v22_granger.npz")
    np.savez(
        out_path,
        prototype_weights=analysis["prototype_weights"],
        prototype_lag_prior_bank=analysis["prototype_lag_prior_bank"],
        prototype_edge_prior_bank=analysis["prototype_edge_prior_bank"],
        edge_prior_mean=analysis["edge_prior_mean"] if analysis["edge_prior_mean"] is not None else np.array(0.0),
        final_context_adj=analysis["final_context_adj"],
        lag_score_mean=analysis["lag_score_mean"] if analysis["lag_score_mean"] is not None else np.array(0.0),
        lag_gate_mean=analysis["lag_gate_mean"] if analysis["lag_gate_mean"] is not None else np.array(0.0),
        lag_peak_index=np.array(analysis["lag_peak_index"] if analysis["lag_peak_index"] is not None else -1.0),
        weather_gate=analysis["weather_gate"],
        node_dynamic_patterns=analysis["node_dynamic_patterns"],
        context_interaction_matrix=analysis["context_interaction_matrix"],
        source_usage=analysis["source_usage"],
        regime_source_usage=analysis["regime_source_usage"],
        edge_mean=analysis["edge_mean"],
        edge_var=analysis["edge_var"],
        edge_repeatability=analysis["edge_repeatability"],
        edge_stability=analysis["edge_stability"],
        ctx_diff_norm=np.array(analysis["ctx_diff_norm"] if analysis["ctx_diff_norm"] is not None else 0.0),
        q_ctx_switch_mean=analysis["q_ctx_switch_mean"],
        best_val_mse=np.array(best_val),
    )
    print(f"Best Val MSE: {best_val:.6f}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()