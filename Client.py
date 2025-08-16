import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling
import torch.optim as optim
import torch.nn.functional as F


class Client:
    def __init__(self, client_id, data, feature_encoder, structure_encoders, decoder, device, lr=1e-4, weight_decay=1e-5):
        """
        初始化客户端。
        - structure_encoders: 多个结构通道编码器的列表
        """
        self.client_id = client_id
        self.data = data.to(device)
        self.feature_encoder = feature_encoder.to(device)
        self.structure_encoders = [enc.to(device) for enc in structure_encoders]
        self.decoder = decoder.to(device)
        self.device = device

        params = list(self.feature_encoder.parameters()) + list(self.decoder.parameters())
        for enc in self.structure_encoders:
            params += list(enc.parameters())

        self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def get_parameters(self):
        """
        返回客户端当前模型的参数（包括特征通道和所有结构通道的编码器、解码器）。
        """
        return {
            "feature_encoder": self.feature_encoder.state_dict(),
            "structure_encoders": [enc.state_dict() for enc in self.structure_encoders],
            "decoder": self.decoder.state_dict()
        }

    def set_parameters(self, parameters):
        """
        从服务器下发参数并更新本地模型。
        """
        self.feature_encoder.load_state_dict(parameters["feature_encoder"])
        for enc, enc_params in zip(self.structure_encoders, parameters["structure_encoders"]):
            enc.load_state_dict(enc_params)
        self.decoder.load_state_dict(parameters["decoder"])

    def extract_structure_features(self):
        """
        根据本地图数据提取并融合多种结构特征。
        """
        struct_features = []
        for enc in self.structure_encoders:
            struct_features.append(enc(self.data))
        self.data.structure_x = torch.cat(struct_features, dim=1)  # [N, sum(feature_dims)]

    def local_train(self, epochs):
        self.feature_encoder.train()
        for enc in self.structure_encoders:
            enc.train()
        self.decoder.train()

        for _ in range(epochs):
            self.optimizer.zero_grad()

            # 特征通道
            z_feat = self.feature_encoder(self.data.x, self.data.edge_index)  # [N, d_f]

            # 结构通道（多编码器融合）
            struct_features = []
            for enc in self.structure_encoders:
                struct_features.append(enc(self.data))  # [N, d_i]
            z_struct = torch.cat(struct_features, dim=1)  # [N, sum(d_i)]

            # 融合特征和结构
            z = torch.cat([z_feat, z_struct], dim=1)  # [N, d_f + sum(d_i)]

            # 正负边索引
            pos_edge_index = self.data.edge_index
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=self.data.num_nodes,
                num_neg_samples=pos_edge_index.size(1)
            )

            # 按边取节点嵌入
            pos_out = self.decoder(z[pos_edge_index[0]], z[pos_edge_index[1]])
            neg_out = self.decoder(z[neg_edge_index[0]], z[neg_edge_index[1]])

            # BCE Loss
            pos_label = torch.ones(pos_out.size(0), device=self.device)
            neg_label = torch.zeros(neg_out.size(0), device=self.device)
            loss = F.binary_cross_entropy_with_logits(
                torch.cat([pos_out, neg_out]).view(-1),
                torch.cat([pos_label, neg_label])
            )

            loss.backward()
            self.optimizer.step()

    def evaluate(self):
        pass

