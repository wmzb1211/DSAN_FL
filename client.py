# 客户端代码
import torch
from torch.utils.data import DataLoader
from models import build_model
from privacy import DPOptimizer, AutoClipper, DPOptimizerWithoutClipper

class Client:
    def __init__(self, client_id, data, config):
        self.id = client_id
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化本地模型
        self.model = build_model(config).to(self.device)

        # 准备数据
        self.loader = DataLoader(
            data,
            batch_size=config['federated']['batch_size'],
            shuffle=True
        )

        # 初始化优化器，有sgd和adam两种选择
        if config['training']['optimizer'] == 'adam':
            base_optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config['training']['learning_rate']
            )
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                base_optimizer,
                step_size=config['training']['lr_step_size'],  # 步长：每 step_size 轮衰减
                gamma=config['training']['lr_decay']  # 衰减比例
            )
            clipper = AutoClipper(
                gamma=config['privacy']['clipping_gamma'],
                R_init=config['privacy']['R_init'],
                momentum=config['privacy']['momentum']
            )
        else:
            # 默认为sgd
            base_optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config['federated']['learning_rate'],
                momentum=config['federated']['momentum']
            )
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                base_optimizer,
                step_size=config['training']['lr_step_size'],  # 步长：每 step_size 轮衰减
                gamma=config['training']['lr_decay']  # 衰减比例
            )
            clipper = AutoClipper(
                gamma=config['privacy']['clipping_gamma'],
                R_init=config['privacy']['R_init'],
                momentum=config['privacy']['momentum']
            )
        if config['privacy']['use_autoclipping'] == True:
            self.optimizer = DPOptimizer(
                base_optimizer,
                self.config['privacy']['noise_multiplier'],
                clipper=clipper,
                lr_scheduler=lr_scheduler
                )
        else:
            self.optimizer = DPOptimizerWithoutClipper(
                base_optimizer,
                self.config['privacy']['noise_multiplier'],
                self.config['privacy']['clip_norm'],
                lr_scheduler=lr_scheduler
            )

    def train(self, global_weights):
        """本地训练"""
        self.model.load_state_dict(global_weights)
        self.model.train()

        for _ in range(self.config['federated']['local_epochs']):
            for X, y in self.loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = torch.nn.functional.cross_entropy(output, y)
                loss.backward()
                self.optimizer.step()

        return self.model.state_dict()