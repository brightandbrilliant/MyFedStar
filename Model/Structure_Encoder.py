# structure_encoders.py
import math
from typing import List, Literal, Sequence, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import (
    degree as pyg_degree,
    to_scipy_sparse_matrix,
    k_hop_subgraph,
    to_networkx,
)
import numpy as np
from scipy import sparse as sp
import networkx as nx


# ------------------------------
# Base class
# ------------------------------
class BaseStructureEncoder(nn.Module):
    """
    所有结构信息编码器的基类。
    规定统一接口，forward 接收单个 PyG 图，返回节点级结构向量 [num_nodes, feat_dim]。
    """
    def __init__(self):
        super().__init__()

    def forward(self, graph: Data) -> torch.Tensor:
        """
        Args:
            graph: PyG Data, 至少包含 edge_index、num_nodes；若有 x 则沿用其 device。
        Returns:
            torch.FloatTensor: [num_nodes, feature_dim]
        """
        raise NotImplementedError

    @staticmethod
    def _infer_device(graph: Data) -> torch.device:
        if hasattr(graph, "x") and graph.x is not None:
            return graph.x.device
        return torch.device("cpu")


# ------------------------------
# 1) Random-Walk Encoder
# ------------------------------
class RWEncoder(BaseStructureEncoder):
    """
    随机游走结构编码器（对角 RW 统计）：
    - 构造列归一化转移矩阵 RW = A * D^{-1}（无向图）。
    - 计算 M^t 的对角线（t=1..num_steps），拼接成特征。
    与很多图位置编码（如 FedStar 的实现片段）一致。

    Args:
        num_steps: t 的最大步数（>=1）
        add_identity: 是否把 t=0 的单位对角（恒为1）也加入特征
        eps: 数值稳定项，防止度为0
        standardize: 是否对每列特征做标准化（均值0方差1）
    """
    def __init__(
        self,
        num_steps: int,
        add_identity: bool = False,
        eps: float = 1e-12,
        standardize: bool = False,
    ):
        super().__init__()
        assert num_steps >= 1, "num_steps must be >= 1"
        self.num_steps = num_steps
        self.add_identity = add_identity
        self.eps = eps
        self.standardize = standardize

    @torch.no_grad()
    def forward(self, graph: Data) -> torch.Tensor:
        device = self._infer_device(graph)
        num_nodes = graph.num_nodes

        # 构造稀疏邻接与列归一化转移矩阵 RW = A * D^{-1}
        A: sp.csr_matrix = to_scipy_sparse_matrix(graph.edge_index, num_nodes=num_nodes).astype(np.float64)
        deg = np.asarray(A.sum(axis=0)).ravel()  # 列度
        deg = np.maximum(deg, self.eps)
        D_inv = sp.diags(1.0 / deg)  # 列归一化
        M = (A @ D_inv).tocsr()

        feats: List[torch.Tensor] = []
        if self.add_identity:
            feats.append(torch.ones(num_nodes, dtype=torch.float32))

        # 逐步幂乘，提取 diag(M^t)
        M_power = M.copy()
        for _t in range(self.num_steps):
            diag = torch.from_numpy(M_power.diagonal().astype(np.float32))
            feats.append(diag)
            # 下一步
            M_power = (M_power @ M).tocsr()

        X = torch.stack(feats, dim=1)  # [N, (num_steps + add_identity)]
        if self.standardize:
            # 对列标准化（稳定性：加 eps）
            mean = X.mean(dim=0, keepdim=True)
            std = X.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            X = (X - mean) / std

        return X.to(device)


# ------------------------------
# 2) Degree Encoder
# ------------------------------
class DegreeEncoder(BaseStructureEncoder):
    """
    节点度编码器：
    支持三种模式：
    - 'onehot': one-hot（维度 = max_degree），度会先 clip 到 [1, max_degree]，再减 1 作为 index。
    - 'embed': 使用 nn.Embedding 把度（clip 后）映射到 embedding 空间（emb_dim）。
    - 'numeric': 直接返回一个实数特征（可选 log/标准化）。

    Args:
        max_degree: onehot/embed 模式的最大度阈值（度会 clip 到该值）
        mode: 'onehot' | 'embed' | 'numeric'
        emb_dim: embed 模式下的嵌入维度
        normalize_numeric: numeric 模式下是否做归一化到 [0, 1]
        log_transform: numeric 模式下是否做 log(1 + d) 以减小长尾
    """
    def __init__(
        self,
        max_degree: int,
        mode: Literal["onehot", "embed", "numeric"] = "onehot",
        emb_dim: Optional[int] = None,
        normalize_numeric: bool = True,
        log_transform: bool = False,
    ):
        super().__init__()
        assert max_degree >= 1, "max_degree must be >= 1"
        self.max_degree = max_degree
        self.mode = mode
        self.normalize_numeric = normalize_numeric
        self.log_transform = log_transform

        if self.mode == "embed":
            assert emb_dim is not None and emb_dim > 0, "emb_dim must be set for 'embed' mode"
            # 索引范围: 0..max_degree-1
            self.embedding = nn.Embedding(num_embeddings=max_degree, embedding_dim=emb_dim)

    @torch.no_grad()
    def forward(self, graph: Data) -> torch.Tensor:
        device = self._infer_device(graph)
        N = graph.num_nodes

        deg = pyg_degree(graph.edge_index[0], num_nodes=N).to(torch.long)  # [N]
        # 将0度节点映射到最小桶，也可独立处理；这里与 FedStar 类似：clip 到 [1, max_degree] 再减 1
        deg_clip = deg.clamp(min=1, max=self.max_degree) - 1  # [0 .. max_degree-1]

        if self.mode == "onehot":
            X = torch.zeros(N, self.max_degree, dtype=torch.float32)
            X[torch.arange(N), deg_clip] = 1.0
            return X.to(device)

        elif self.mode == "embed":
            X = self.embedding(deg_clip.to(self.embedding.weight.device))
            return X.to(device)

        else:  # numeric
            d = deg.to(torch.float32)
            if self.log_transform:
                d = torch.log1p(d)
            if self.normalize_numeric:
                # 归一化到 [0,1]：除以 max_degree
                d = d / float(self.max_degree)
                d = d.clamp(0.0, 1.0)
            return d.view(-1, 1).to(device)


# ------------------------------
# 3) Subgraph (k-hop ego) Encoder
# ------------------------------
class SubgraphEncoder(BaseStructureEncoder):
    """
    k-hop 子图统计特征编码器（每个节点一个 k-hop ego 子图，计算统计量）：

    默认统计集合（可配置）：
      - 'num_nodes'     : 子图节点数
      - 'num_edges'     : 子图边数（无向边记一次）
      - 'density'       : 2E / (n(n-1))，n<2 时为 0
      - 'avg_deg'       : 子图平均度（2E/n）
      - 'max_deg'       : 子图最大度
      - 'min_deg'       : 子图最小度
      - 'clustering'    : 使用 NetworkX 的局部聚类系数（目标节点在子图中的 clustering）
      - 'assortativity' : 子图的度同配性（皮尔逊），nx.degree_assortativity_coefficient

    注意：逐节点提取子图开销较大，适合中小图或离线缓存；如需大图可减少统计项或降低 k。
    你也可以把 `stats` 列表裁剪成轻量组合。

    Args:
        k_hop: k 跳邻域
        stats: 统计项名称列表
        standardize: 是否按列做标准化（均值0方差1）
        safe_nan_to_num: 对可能返回 NaN 的统计量（如 assortativity），是否置 0
    """
    SUPPORTED: Sequence[str] = (
        "num_nodes",
        "num_edges",
        "density",
        "avg_deg",
        "max_deg",
        "min_deg",
        "clustering",
        "assortativity",
    )

    def __init__(
        self,
        k_hop: int,
        stats: Optional[Sequence[str]] = None,
        standardize: bool = False,
        safe_nan_to_num: bool = True,
    ):
        super().__init__()
        assert k_hop >= 1, "k_hop must be >= 1"
        self.k_hop = k_hop
        self.stats = list(stats) if stats is not None else [
            "num_nodes",
            "num_edges",
            "density",
            "avg_deg",
            "max_deg",
            "min_deg",
            "clustering",
        ]
        # 校验
        for s in self.stats:
            if s not in self.SUPPORTED:
                raise ValueError(f"Unsupported stat: {s}")
        self.standardize = standardize
        self.safe_nan_to_num = safe_nan_to_num

    @torch.no_grad()
    def forward(self, graph: Data) -> torch.Tensor:
        device = self._infer_device(graph)
        N = graph.num_nodes
        ei = graph.edge_index

        # 预先转换成 NetworkX 以便统计聚类系数等（对每个子图再切片更方便）
        # 注意：to_networkx 会构图一次，之后子图用 induced_subgraph 切
        G_full = to_networkx(graph, to_undirected=True)

        feats = []
        for center in range(N):
            # 提取 k-hop 子图（包含中心点）
            nodes, sub_ei, _, _ = k_hop_subgraph(
                center,
                self.k_hop,
                ei,
                relabel_nodes=True,
                num_nodes=N,
            )
            # 子图节点映射：nodes 是原图索引；我们需要在 full graph 里取诱导子图
            G_sub = G_full.subgraph(nodes.tolist()).copy()

            n = G_sub.number_of_nodes()
            e = G_sub.number_of_edges()  # 无向图计一次

            # 统计特征
            vals = []
            for s in self.stats:
                if s == "num_nodes":
                    vals.append(float(n))
                elif s == "num_edges":
                    vals.append(float(e))
                elif s == "density":
                    if n <= 1:
                        vals.append(0.0)
                    else:
                        vals.append(2.0 * e / (n * (n - 1)))
                elif s == "avg_deg":
                    vals.append(0.0 if n == 0 else 2.0 * e / max(n, 1))
                elif s == "max_deg":
                    vals.append(0.0 if n == 0 else float(max(dict(G_sub.degree()).values())))
                elif s == "min_deg":
                    vals.append(0.0 if n == 0 else float(min(dict(G_sub.degree()).values())))
                elif s == "clustering":
                    # 局部聚类系数：以中心点在子图中的 local clustering
                    # center 在子图中的新索引不一定为 0，因此用原 id
                    if n <= 2:
                        vals.append(0.0)
                    else:
                        # NetworkX clustering 返回字典或标量
                        c = nx.clustering(G_sub, nodes=[center]).get(center, 0.0)
                        vals.append(float(c))
                elif s == "assortativity":
                    # 度同配性：小图时可能为 NaN
                    try:
                        coeff = nx.degree_assortativity_coefficient(G_sub)
                        if math.isnan(coeff) and self.safe_nan_to_num:
                            coeff = 0.0
                        vals.append(float(coeff))
                    except Exception:
                        vals.append(0.0)
                else:
                    raise RuntimeError(f"Unhandled stat: {s}")

            feats.append(torch.tensor(vals, dtype=torch.float32))

        X = torch.stack(feats, dim=0)  # [N, F]
        if self.standardize and X.numel() > 0:
            mean = X.mean(dim=0, keepdim=True)
            std = X.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            X = (X - mean) / std

        return X.to(device)


# ------------------------------
# 工厂函数
# ------------------------------
def build_structure_encoder(name: str, **kwargs) -> BaseStructureEncoder:
    """
    工厂方法，根据名称构造对应的结构编码器。
    name ∈ {'rw', 'degree', 'subgraph'}
    """
    name = name.lower()
    if name == "rw":
        return RWEncoder(**kwargs)
    elif name == "degree":
        return DegreeEncoder(**kwargs)
    elif name == "subgraph":
        return SubgraphEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown structure encoder type: {name}")



