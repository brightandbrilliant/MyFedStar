import os
import torch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected

from Client import Client
from Server import Server
from Model.GraphSage import GraphSAGE
from Model.ResMLP import ResMLP
from Model.Structure_Encoder import DegreeEncoder, RWEncoder


# =============================
# 工具函数
# =============================
def split_client_data(data, val_ratio=0.1, test_ratio=0.1, device="cuda"):
    """
    对单个图数据划分 train/val/test 边。
    """
    data = data.to(device)
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        neg_sampling_ratio=1.0,
    )
    train_data, val_data, test_data = transform(data)

    val_mask = val_data.edge_label.bool()
    test_mask = test_data.edge_label.bool()
    train_data.val_pos_edge_index = val_data.edge_label_index[:, val_mask]
    train_data.val_neg_edge_index = val_data.edge_label_index[:, ~val_mask]
    train_data.test_pos_edge_index = test_data.edge_label_index[:, test_mask]
    train_data.test_neg_edge_index = test_data.edge_label_index[:, ~test_mask]

    return train_data


# -----------------------------
# 结构 encoder 输出维度计算
# -----------------------------
def get_struct_encoder_out_dim(encoder_cls, params):
    if encoder_cls == DegreeEncoder:
        mode = params.get("mode", "onehot")
        if mode == "onehot":
            return params["max_degree"]
        elif mode == "embed":
            return params["emb_dim"]
        else:  # numeric
            return 1
    elif encoder_cls == RWEncoder:
        return params["num_steps"] + (1 if params.get("add_identity", False) else 0)
    else:
        raise ValueError(f"Unknown encoder class: {encoder_cls}")


# -----------------------------
# 初始化客户端
# -----------------------------
def load_clients(data_paths, encoder_params, decoder_params, struct_encoder_params, training_params, device):
    clients = []
    for client_id, path in enumerate(data_paths):
        raw_data = torch.load(path)
        data = split_client_data(raw_data, device=device)

        feature_encoder = GraphSAGE(**encoder_params)

        structure_encoders = []
        struct_out_dim_total = 0
        for encoder_cls, params in struct_encoder_params:
            enc = encoder_cls(**params)
            structure_encoders.append(enc)
            struct_out_dim_total += get_struct_encoder_out_dim(encoder_cls, params)

        decoder_in_dim = encoder_params["output_dim"] + struct_out_dim_total
        decoder = ResMLP(input_dim=decoder_in_dim * 2, **decoder_params)

        client = Client(
            client_id=client_id,
            data=data,
            feature_encoder=feature_encoder,
            structure_encoders=structure_encoders,
            decoder=decoder,
            device=device,
            lr=training_params["lr"],
            weight_decay=training_params["weight_decay"],
        )
        clients.append(client)
    return clients


# -----------------------------
# 全局模型评估
# -----------------------------
def evaluate_global_model(server, clients, use_test=False):
    metrics = []
    for client in clients:
        acc, recall, precision, f1 = client.evaluate(use_test=use_test)
        metrics.append((acc, recall, precision, f1))
        print(f"[Client {client.client_id}] Acc={acc:.4f}, Recall={recall:.4f}, "
              f"Prec={precision:.4f}, F1={f1:.4f}")
    avg = torch.tensor(metrics).mean(dim=0).tolist()
    print(f"\n===> Global Avg: Acc={avg[0]:.4f}, Recall={avg[1]:.4f}, "
          f"Prec={avg[2]:.4f}, F1={avg[3]:.4f}")
    return avg


# =============================
# 主函数
# =============================
def main():
    # -------- Step 1: 参数配置 --------
    data_dir = "Parsed_dataset/wd"
    pyg_data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_params = {
        "input_dim": torch.load(pyg_data_files[0]).x.shape[1],
        "hidden_dim": 128,
        "output_dim": 64,
        "num_layers": 3,
        "dropout": 0.4,
    }
    decoder_params = {"hidden_dim": 128, "num_layers": 3, "dropout": 0.3}

    # 每个结构 encoder 独立参数
    struct_encoder_params = [
        (DegreeEncoder, {"max_degree": 50, "mode": "onehot"}),  # 输出维度自动计算
        (RWEncoder, {"num_steps": 10, "add_identity": False}),   # 输出维度 = num_steps + add_identity
    ]

    training_params = {"lr": 0.001, "weight_decay": 1e-4, "local_epochs": 5}
    num_rounds = 50
    eval_interval = 5

    # -------- Step 2: 初始化客户端 & 全局模型 --------
    clients = load_clients(
        data_paths=pyg_data_files,
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        struct_encoder_params=struct_encoder_params,
        training_params=training_params,
        device=device,
    )

    # 初始化全局模型
    init_params = clients[0].get_parameters()

    structure_encoders_global = []
    struct_out_dim_total = 0
    for encoder_cls, params in struct_encoder_params:
        enc = encoder_cls(**params)
        structure_encoders_global.append(enc)
        struct_out_dim_total += get_struct_encoder_out_dim(encoder_cls, params)

    server = Server(
        feature_encoder=GraphSAGE(**encoder_params),
        structure_encoders=structure_encoders_global,
        decoder=ResMLP(input_dim=(encoder_params["output_dim"] + struct_out_dim_total) * 2,
                       **decoder_params),
        device=device,
    )
    server.set_global_parameters(init_params)

    # -------- Step 3: 联邦训练循环 --------
    print("\n================ Federated Training Start ================")
    best_f1 = -1
    best_state = None

    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Round {rnd} ---")

        for client in clients:
            client.local_train(training_params["local_epochs"])
            print(f"Client {client.client_id} finished local training.")

        server.aggregate_all_weights(clients)
        server.distribute_parameters(clients)

        if rnd % eval_interval == 0:
            avg_acc, avg_recall, avg_prec, avg_f1 = evaluate_global_model(server, clients, use_test=False)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_state = server.get_global_parameters()
                print("===> New best model saved")

    # -------- Step 4: 最终评估 --------
    print("\n================ Federated Training Finished ================")
    if best_state is not None:
        server.set_global_parameters(best_state)
        server.distribute_parameters(clients)

    print("\n================ Final Evaluation ================")
    evaluate_global_model(server, clients, use_test=True)


if __name__ == "__main__":
    main()
