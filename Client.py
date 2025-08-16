import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

class Client:
    def __init__(self, client_id, data, feature_encoder, structure_encoders, decoder, device, lr=1e-4, weight_decay=1e-5):
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

        # 初始化结构特征缓存
        self.extract_structure_features()

    def get_parameters(self):
        return {
            "feature_encoder": self.feature_encoder.state_dict(),
            "structure_encoders": [enc.state_dict() for enc in self.structure_encoders],
            "decoder": self.decoder.state_dict()
        }

    def set_parameters(self, parameters):
        self.feature_encoder.load_state_dict(parameters["feature_encoder"])
        for enc, enc_params in zip(self.structure_encoders, parameters["structure_encoders"]):
            enc.load_state_dict(enc_params)
        self.decoder.load_state_dict(parameters["decoder"])

    def extract_structure_features(self):
        """
        提取结构特征，并缓存到 self.data.structure_x。
        训练阶段直接复用，无需重复计算。
        """
        struct_features = []
        for enc in self.structure_encoders:
            struct_features.append(enc(self.data))
        self.data.structure_x = torch.cat(struct_features, dim=1)  # [N, sum(feature_dims)]

    def local_train(self, epochs):
        self.feature_encoder.train()
        for enc in self.structure_encoders:
            enc.eval()  # 结构编码器不再训练，使用缓存
        self.decoder.train()

        # 缓存的结构特征
        z_struct = self.data.structure_x  # [N, sum(d_i)]

        for _ in range(epochs):
            self.optimizer.zero_grad()

            # 特征通道
            z_feat = self.feature_encoder(self.data.x, self.data.edge_index)  # [N, d_f]

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

    def evaluate(self, use_test=False):
        """
        在验证集或测试集上评估当前客户端模型。
        Args:
            use_test (bool): True 则使用测试集，否则使用验证集
        Returns:
            acc, recall, precision, f1
        """
        self.feature_encoder.eval()
        for enc in self.structure_encoders:
            enc.eval()
        self.decoder.eval()

        with torch.no_grad():
            # 特征通道
            z_feat = self.feature_encoder(self.data.x, self.data.edge_index)

            # 结构通道
            struct_features = []
            for enc in self.structure_encoders:
                struct_features.append(enc(self.data))
            z_struct = torch.cat(struct_features, dim=1)

            # 融合
            z = torch.cat([z_feat, z_struct], dim=1)

            # 选择验证集或测试集
            if use_test:
                pos_edge_index = self.data.test_pos_edge_index
                neg_edge_index = self.data.test_neg_edge_index
            else:
                pos_edge_index = self.data.val_pos_edge_index
                neg_edge_index = self.data.val_neg_edge_index

            pos_out = self.decoder(z, pos_edge_index).sigmoid()
            neg_out = self.decoder(z, neg_edge_index).sigmoid()

            y_true = torch.cat([
                torch.ones(pos_out.size(0), device=self.device),
                torch.zeros(neg_out.size(0), device=self.device)
            ])
            y_pred = torch.cat([pos_out, neg_out])
            y_pred_labels = (y_pred > 0.5).long()

            # 计算评估指标
            acc = accuracy_score(y_true.cpu(), y_pred_labels.cpu())
            recall = recall_score(y_true.cpu(), y_pred_labels.cpu(), zero_division=0)
            precision = precision_score(y_true.cpu(), y_pred_labels.cpu(), zero_division=0)
            f1 = f1_score(y_true.cpu(), y_pred_labels.cpu(), zero_division=0)

        return acc, recall, precision, f1
