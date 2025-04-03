# 启动脚本
import yaml
import torch
from data.datasets import load_mnist, load_data
from data.partition import split_iid
from client import Client
from server import Server
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # 加载配置
    config = load_config("configs/params.yaml")
    acc_list = []
    loss_list = []
    # 准备数据

    # train_set, test_set = load_mnist(config['paths']['data_dir'])
    train_set, test_set = load_data(config['paths']['data_dir'], config['dataset']['name'])
    client_datasets = split_iid(train_set, config['federated']['num_clients'])
    test_loader = DataLoader(test_set, batch_size=config['federated']['batch_size'], shuffle=False)
    # 初始化服务器和客户端
    server = Server(config)
    clients = [Client(i, data, config) for i, data in enumerate(client_datasets)]
    total_samples = len(train_set)
    # 训练循环
    for round in range(config['federated']['global_rounds']):
        # 选择客户端
        selected = np.random.choice(
            clients,
            size=int(len(clients) * config['federated']['client_rate']),
            replace=False
        )

        # 客户端训练
        updates = []
        for client in selected:
            global_weights = server.get_global_weights()
            local_weights = client.train(global_weights)
            updates.append(local_weights)

        # 聚合更新
        server.aggregate(updates, total_samples)

        # 打印日志
        epsilon, delta = server.get_privacy_cost()
        print(f"Round {round + 1}: ε={epsilon:.4f}, δ={delta}")
        # 检查隐私预算
        if server.accountant.is_budget_exhausted():
            print(f"隐私预算耗尽! ε={epsilon:.4f} > {config['privacy']['max_epsilon']}")
            break
        # 每一轮都要测试模型
        loss, acc = server.test(test_loader)
        acc_list.append(acc)
        loss_list.append(loss)
        print(f"Round {round + 1}: accuracy={acc_list[-1]:.4f}, loss={loss_list[-1]:.4f}")

    # 绘图
    if config['federated']['plot']:
        plt.plot(acc_list)
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        # acc_of_模型_轮数_数据集_差分隐私方法_隐私预算.png
        plt.savefig(f"picNOautoclipper/acc_of_{config['model']['name']}_{config['dataset']['name']}_{config['federated']['global_rounds']}_"
                    f"{config['privacy']['method']}_{config['privacy']['max_epsilon']}_noiseMultiplier_{config['privacy']['noise_multiplier']}.png")
        plt.show()
        plt.plot(loss_list)
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.savefig(f"picNOautoclipper/loss_of_{config['model']['name']}_{config['dataset']['name']}_{config['federated']['global_rounds']}_"
                    f"{config['privacy']['method']}_{config['privacy']['max_epsilon']}_noiseMultiplier_{config['privacy']['noise_multiplier']}.png")
        plt.show()
        torch.save(server.global_model.state_dict(), f"modelsNOautoclipper/model_of_{config['model']['name']}_{config['dataset']['name']}_{config['federated']['global_rounds']}_"
                    f"{config['privacy']['method']}_{config['privacy']['max_epsilon']}_noiseMultiplier_{config['privacy']['noise_multiplier']}.pth")



if __name__ == "__main__":
    main()