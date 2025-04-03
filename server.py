# 服务端代码
import copy
import torch
import numpy as np
from privacy import PrivacyAccountant, AutoClipper
from models import build_model

class Server:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 全局模型
        self.global_model = build_model(config).to(self.device)

        # 隐私会计
        self.accountant = PrivacyAccountant(
            delta=config['privacy']['delta'],
            use_rdp=config['privacy']['use_rdp'],
            max_epsilon=config['privacy']['max_epsilon']
        )
        self.clipper = AutoClipper(
            gamma=config['privacy']['clipping_gamma'],
            R_init=config['privacy']['R_init'],
            momentum=config['privacy']['momentum']
        )

    def aggregate(self, client_updates, total_samples):
        """模型聚合"""
        averaged_weights = {}

        # 计算加权平均
        total = len(client_updates)
        for key in client_updates[0].keys():
            averaged_weights[key] = sum(
                [update[key].cpu() for update in client_updates]
            ) / total

        # 更新全局模型
        self.global_model.load_state_dict(averaged_weights)

        # 更新隐私预算
        sampling_rate = self.config['federated']['client_rate']
        # total_samples = len(train_set)
        self.accountant.update(batch_size=self.config['federated']['batch_size'],
            dataset_size = sampling_rate * total_samples,
            steps=1,
            noise_multiplier=self.config['privacy']['noise_multiplier'],
            current_R=self.clipper.get_current_R()
        )

    def get_global_weights(self):
        return copy.deepcopy(self.global_model.state_dict())

    def get_privacy_cost(self):
        return self.accountant.get_epsilon()

    def test(self, test_loader):  # 参数名改为 test_loader 更准确
        self.global_model.eval()
        test_loss = 0.0  # 明确使用浮点类型
        correct = 0.0  # 初始化为浮点数类型

        with torch.no_grad():
            for data, target in test_loader:  # 确保遍历的是 DataLoader
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)

                # 计算损失（保持原有逻辑）
                test_loss += torch.nn.functional.cross_entropy(
                    output, target, reduction='sum'
                ).item()  # 立即转换为 Python 浮点数

                # 计算正确预测数（转换为浮点）
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()  # 转换为 Python 浮点数

        # 数据集总样本数（确保是整数）
        total_samples = len(test_loader.dataset)

        # 最终计算（使用浮点运算）
        test_loss /= total_samples
        accuracy = correct / total_samples  # 保存到新变量

        return test_loss, accuracy  # 直接返回浮点数