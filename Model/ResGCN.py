import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ResGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        """
        构建一个带有残差连接的GCN模型。

        Args:
            input_dim (int): 输入节点特征的维度。
            hidden_dim (int): 隐藏层中GCN卷积的输出维度。
            output_dim (int): 最后一层GCN卷积的输出维度。
            num_layers (int): GCN层的总数量。
            dropout (float): Dropout 比率。
        """
        super(ResGCN, self).__init__()

        # 确保至少有两层，才能有中间层进行残差连接的尝试
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2 for a meaningful ResidualGCN.")

        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()

        # 第一层：输入维度 -> 隐藏维度
        self.layers.append(GCNConv(input_dim, hidden_dim))

        # 中间层：隐藏维度 -> 隐藏维度。这些层将应用残差连接。
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        # 最后一层：隐藏维度 -> 输出维度。通常不应用残差连接和激活函数。
        self.layers.append(GCNConv(hidden_dim, output_dim))

        # 如果 input_dim != hidden_dim，我们需要一个额外的线性层来使第一层的输出维度
        # 与后续层的输入维度（hidden_dim）匹配，以便进行残差连接。
        # 这里的残差连接只在 hidden_dim -> hidden_dim 的层之间进行。
        # 或者，更复杂的残差设计可以在第一层后也使用一个投影层来对齐维度。
        # 在这个实现中，我们只在 hidden_dim -> hidden_dim 的 GCNConv 层中实现残差。
        # 如果需要将第一层输出也与后续残差连接，可以添加一个额外的线性层进行维度匹配，
        # 或者设计成 input_dim == hidden_dim。

        # 为了简单起见和保持 GCNConv 的通用性，我们假定残差只发生在
        # (hidden_dim, hidden_dim) 的卷积层之间。
        # 对于第一层，它的输出维度可能与 hidden_dim 不同（如果 output_dim 不同）。
        # 如果你希望所有层都能残差，你可能需要确保所有 GCNConv 的输出维度相同，
        # 或者使用额外的线性层来匹配维度。

        # 鉴于你的原始GCN设计，最后一层的 output_dim 可能不同于 hidden_dim，
        # 所以最后一层不适合直接进行残差连接，它更像是一个最终的投影层。
        # 残差连接将主要应用于中间的隐藏层。

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 节点的特征矩阵，形状通常为 [num_nodes, input_dim]。
            edge_index (torch.Tensor): 图的边索引，形状为 [2, num_edges]。

        Returns:
            torch.Tensor: 经过GCN传播后的最终节点嵌入，形状为 [num_nodes, output_dim]。
        """

        # 第一层 (Input_dim -> Hidden_dim)
        # 这一层通常不直接参与残差连接，因为它改变了维度。
        x = self.layers[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 中间层 (Hidden_dim -> Hidden_dim)，应用残差连接
        # 从第二层开始到倒数第二层
        for i in range(1, self.num_layers - 1):
            # 保存当前层的输入，作为残差连接的跳跃路径
            identity = x

            conv = self.layers[i]
            x = conv(x, edge_index)

            # 检查维度是否匹配以进行残差连接
            # 在这里，因为都是 hidden_dim -> hidden_dim，所以维度是匹配的
            # 如果 conv 的输出维度和 identity 的维度不同，你需要一个线性层来对齐
            # 例如: x = F.relu(x + self.linear_skip(identity))

            x = F.relu(x + identity)  # 残差连接：当前层输出 + 上一层输入
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 最后一层 (Hidden_dim -> Output_dim)
        # 最后一层通常不应用激活函数和残差连接，因为它生成最终的嵌入
        x = self.layers[-1](x, edge_index)

        return x
