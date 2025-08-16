import torch
import torch.nn as nn
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
        """
        在本地进行训练。
        - 使用特征通道编码器对节点特征编码
        - 使用多个结构通道编码器对结构特征编码
        - 融合两个通道的嵌入后输入解码器
        - 计算链接预测损失并更新模型参数
        """
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

            # 融合通道
            z = torch.cat([z_feat, z_struct], dim=1)  # [N, d_f + sum(d_i)]

            # 链接预测
            pos_edge_index = self.data.train_pos_edge_index
            neg_edge_index = self.data.train_neg_edge_index

            pos_out = self.decoder(z, pos_edge_index)
            neg_out = self.decoder(z, neg_edge_index)

            # 损失函数（BCE）
            pos_label = torch.ones(pos_out.size(0), device=self.device)
            neg_label = torch.zeros(neg_out.size(0), device=self.device)
            loss = F.binary_cross_entropy_with_logits(
                torch.cat([pos_out, neg_out]),
                torch.cat([pos_label, neg_label])
            )

            loss.backward()
            self.optimizer.step()

    def evaluate(self):
        pass

