from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import tqdm
import os

# ==========================================
# 1. 设备配置与参数设置
# ==========================================
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
print(f"Device: {device}")

# 超参数
batch_size = 256
fp16_precision = True
temperature = 0.5
num_epoches = 100

# ==========================================
# 2. 数据加载 (只加载原始频域文件，无需增强视图)
# ==========================================
print("Loading frequency domain dataset...")
freq_data = np.load('./datasets/awf1_freq.npz')

x_train = freq_data['x']  # 频域特征
y_train = freq_data['y']  # 标签

print(f"Freq data shape: {x_train.shape}")
print(f"Labels shape: {y_train.shape}")

num_classes = len(np.unique(y_train))
print(f"Number of classes: {num_classes}")

# ==========================================
# 3. 模型定义 (Backbone & Projection Head)
# ==========================================
class DFNet(nn.Module):
    def __init__(self, out_dim):
        super(DFNet, self).__init__()
        kernel_size = 8
        channels = [1, 32, 64, 128, 256]
        conv_stride = 1
        pool_stride = 4
        pool_size = 8

        self.conv1 = nn.Conv1d(1, 32, kernel_size, stride=conv_stride)
        self.conv1_1 = nn.Conv1d(32, 32, kernel_size, stride=conv_stride)

        self.conv2 = nn.Conv1d(32, 64, kernel_size, stride=conv_stride)
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size, stride=conv_stride)

        self.conv3 = nn.Conv1d(64, 128, kernel_size, stride=conv_stride)
        self.conv3_3 = nn.Conv1d(128, 128, kernel_size, stride=conv_stride)

        self.conv4 = nn.Conv1d(128, 256, kernel_size, stride=conv_stride)
        self.conv4_4 = nn.Conv1d(256, 256, kernel_size, stride=conv_stride)

        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(256)

        self.max_pool_1 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.dropout4 = nn.Dropout(p=0.1)

        self.fc = nn.Linear(2560, out_dim)

    def weight_init(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.zero_()

    def forward(self, inp):
        x = inp
        # ==== first block ====
        x = F.pad(x, (3,4))
        x = F.elu((self.conv1(x)))
        x = F.pad(x, (3,4))
        x = F.elu(self.batch_norm1(self.conv1_1(x)))
        x = F.pad(x, (3, 4))
        x = self.max_pool_1(x)
        x = self.dropout1(x)

        # ==== second block ====
        x = F.pad(x, (3,4))
        x = F.relu((self.conv2(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm2(self.conv2_2(x)))
        x = F.pad(x, (3,4))
        x = self.max_pool_2(x)
        x = self.dropout2(x)

        # ==== third block ====
        x = F.pad(x, (3,4))
        x = F.relu((self.conv3(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm3(self.conv3_3(x)))
        x = F.pad(x, (3,4))
        x = self.max_pool_3(x)
        x = self.dropout3(x)

        # ==== fourth block ====
        x = F.pad(x, (3,4))
        x = F.relu((self.conv4(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm4(self.conv4_4(x)))
        x = F.pad(x, (3,4))
        x = self.max_pool_4(x)
        x = self.dropout4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class DFsimCLR(nn.Module):
    def __init__(self, df, out_dim):
        super(DFsimCLR, self).__init__()

        self.backbone = df
        self.backbone.weight_init()
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.BatchNorm1d(dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def forward(self, inp):
        out = self.backbone(inp)
        return out

# ==========================================
# 4. 数据加载器 (修改：单个样本 + 标签)
# ==========================================
class FreqTrainData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

# ==========================================
# 5. 监督对比学习 (Supervised Contrastive Loss)
# ==========================================
class SupervisedCLR(object):
    def __init__(self, **args):
        self.model = args['model']
        self.optimizer = args['optimizer']
        self.scheduler = args['scheduler']
        self.fp16_precision = args['fp16_precision']
        self.num_epoches = args['num_epoches']
        self.batch_size = args['batch_size']
        self.device = args['device']
        self.temperature = args['temperature']
        self.log_every_n_step = 100

    def supervised_contrastive_loss(self, features, labels):
        """
        Supervised Contrastive Loss (SupCon)
        同类样本推近，异类样本拉远
        L = -1/|P(i)| * sum_{p in P(i)} log( exp(z_i·z_p/τ) / sum_{a in A(i)} exp(z_i·z_a/τ) )
        """
        batch_size = features.shape[0]

        # L2归一化
        features = F.normalize(features, dim=1)

        # 相似度矩阵 (batch, batch)
        sim = torch.matmul(features, features.T) / self.temperature

        # 根据标签构建正样本掩码
        labels = labels.unsqueeze(1)
        pos_mask = torch.eq(labels, labels.T).float().to(self.device)  # (batch, batch)

        # 去掉自身的对角线
        mask_self = torch.eye(batch_size, dtype=torch.bool).to(self.device)
        pos_mask = pos_mask[~mask_self].view(batch_size, -1)   # (batch, batch-1)
        sim_masked = sim[~mask_self].view(batch_size, -1)      # (batch, batch-1)

        # log-sum-exp 分母: sum over all a in A(i)
        log_sim = torch.exp(sim_masked)
        log_prob = sim_masked - torch.log(log_sim.sum(1, keepdim=True))

        # 正样本的 log-probability 均值
        pos_count = pos_mask.sum(1).clamp(min=1)  # 防止除零
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_count

        loss = -mean_log_prob_pos.mean()
        return loss

    def train(self, train_loader):
        best_acc = 0
        scaler = GradScaler(enabled=self.fp16_precision)

        n_iter = 0
        print("Start Supervised Contrastive Learning for %d epochs" % self.num_epoches)

        for epoch_counter in range(self.num_epoches + 1):
            with tqdm.tqdm(train_loader, unit='batch') as tepoch:
                for data, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch_counter}")

                    self.model.train()

                    # data shape: (Batch, Length) -> (Batch, 1, Length)
                    data = data.view(data.size(0), 1, data.size(1))
                    data = data.float().to(self.device)
                    labels = labels.long().to(self.device)

                    with autocast(enabled=self.fp16_precision):
                        features = self.model(data)
                        loss = self.supervised_contrastive_loss(features, labels)

                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    if n_iter % self.log_every_n_step == 0:
                        tepoch.set_postfix(loss=loss.item())
                    n_iter += 1

            if epoch_counter >= 10:
                self.scheduler.step()

            # 保存模型
            if epoch_counter % 20 == 0 and epoch_counter > 0:
                os.makedirs('./checkpoints/WFTFC/', exist_ok=True)
                torch.save(self.model.state_dict(), f'./checkpoints/WFTFC/WFTFC_freq_supcon_epoch_{epoch_counter}.pth.tar')

# ==========================================
# 6. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 创建数据集和数据加载器
    train_dataset = FreqTrainData(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 获取频域特征的长度
    input_feature_dim = x_train.shape[1]
    print("input feature:", input_feature_dim)

    # 初始化模型
    df = DFNet(out_dim=512)
    model = DFsimCLR(df, out_dim=128).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    supclr = SupervisedCLR(
               model = model,
               optimizer = optimizer,
               scheduler = scheduler,
               fp16_precision = fp16_precision,
               device = device,
               temperature = temperature,
               num_epoches = 101,
               batch_size = batch_size)
    supclr.train(train_loader)