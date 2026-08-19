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
batch_size = 128
fp16_precision = True
temperature = 0.5
num_epoches = 100
alpha = 0.8  # 时域和频域自监督损失的权重
beta = 0.2   # 时频一致性损失的权重

# ==========================================
# 2. 数据加载 (去掉增强视图，使用监督对比学习)
# ==========================================
print("Loading Time and Frequency domain datasets...")

# --- 加载时域数据 (只需原始数据) ---
time_data_orig = np.load('./datasets/awf1.npz')
x_time = time_data_orig['feature']

# --- 加载频域数据 (只需原始数据) ---
freq_data_orig = np.load('./datasets/awf1_freq.npz')
x_freq = freq_data_orig['x']

# --- 加载标签 ---
y_train = time_data_orig['label']

print(f"Time data shape: {x_time.shape}")
print(f"Freq data shape: {x_freq.shape}")

num_classes = len(np.unique(y_train))
print(f"Number of classes: {num_classes}")

# ==========================================
# 3. 模型定义 (Backbone & Projection Head)
# ==========================================
class DFNet(nn.Module):
    def __init__(self, out_dim, input_feature_dim):
        super(DFNet, self).__init__()
        kernel_size = 8
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

        # 动态计算展平后的维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_feature_dim)
            out = self._forward_features(dummy_input)
            flattened_dim = out.view(1, -1).size(1)

        self.fc = nn.Linear(flattened_dim, out_dim)
        self.weight_init()

    def _forward_features(self, x):
        x = F.pad(x, (3,4))
        x = F.elu((self.conv1(x)))
        x = F.pad(x, (3,4))
        x = F.elu(self.batch_norm1(self.conv1_1(x)))
        x = F.pad(x, (3, 4))
        x = self.max_pool_1(x)
        x = self.dropout1(x)

        x = F.pad(x, (3,4))
        x = F.relu((self.conv2(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm2(self.conv2_2(x)))
        x = F.pad(x, (3,4))
        x = self.max_pool_2(x)
        x = self.dropout2(x)

        x = F.pad(x, (3,4))
        x = F.relu((self.conv3(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm3(self.conv3_3(x)))
        x = F.pad(x, (3,4))
        x = self.max_pool_3(x)
        x = self.dropout3(x)

        x = F.pad(x, (3,4))
        x = F.relu((self.conv4(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.batch_norm4(self.conv4_4(x)))
        x = F.pad(x, (3,4))
        x = self.max_pool_4(x)
        x = self.dropout4(x)
        return x

    def weight_init(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, inp):
        x = self._forward_features(inp)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, out_dim)
        )

    def forward(self, x):
        return self.projector(x)

class DualDomainModel(nn.Module):
    def __init__(self, time_input_dim, freq_input_dim, out_dim=128):
        super(DualDomainModel, self).__init__()
        # 时域特征提取器 (Backbone)
        self.time_encoder = DFNet(out_dim=512, input_feature_dim=time_input_dim)
        # 频域特征提取器 (Backbone)
        self.freq_encoder = DFNet(out_dim=512, input_feature_dim=freq_input_dim)

        # 投影头 (Projection Head)
        self.time_projector = ProjectionHead(input_dim=512, out_dim=out_dim)
        self.freq_projector = ProjectionHead(input_dim=512, out_dim=out_dim)

    def forward(self, x_time, x_freq):
        h_time = self.time_encoder(x_time)
        h_freq = self.freq_encoder(x_freq)

        z_time = self.time_projector(h_time)
        z_freq = self.freq_projector(h_freq)

        return z_time, z_freq

# ==========================================
# 4. 数据加载器 (去掉增强视图，加入标签)
# ==========================================
class DualDomainDataset(Dataset):
    def __init__(self, x_time, x_freq, y):
        self.x_time = x_time
        self.x_freq = x_freq
        self.y = y

    def __getitem__(self, index):
        return self.x_time[index], self.x_freq[index], self.y[index]

    def __len__(self):
        return len(self.x_time)

# ==========================================
# 5. 监督对比学习训练逻辑
# ==========================================
class SupervisedDualCLR(object):
    def __init__(self, **args):
        self.model = args['model']
        self.optimizer = args['optimizer']
        self.scheduler = args['scheduler']
        self.fp16_precision = args['fp16_precision']
        self.num_epoches = args['num_epoches']
        self.batch_size = args['batch_size']
        self.device = args['device']
        self.temperature = args['temperature']
        self.alpha = args['alpha']
        self.beta = args['beta']
        self.log_every_n_step = 100

    def supervised_contrastive_loss(self, features, labels):
        """
        Supervised Contrastive Loss (SupCon)
        L = -1/|P(i)| * sum_{p in P(i)} log( exp(z_i·z_p/τ) / sum_{a in A(i)} exp(z_i·z_a/τ) )
        """
        batch_size = features.shape[0]

        # L2归一化
        features = F.normalize(features, dim=1)

        # 相似度矩阵 (batch, batch)
        sim = torch.matmul(features, features.T) / self.temperature

        # 根据标签构建正样本掩码
        labels = labels.unsqueeze(1)
        pos_mask = torch.eq(labels, labels.T).float().to(self.device)

        # 去掉自身的对角线
        mask_self = torch.eye(batch_size, dtype=torch.bool).to(self.device)
        pos_mask = pos_mask[~mask_self].view(batch_size, -1)
        sim_masked = sim[~mask_self].view(batch_size, -1)

        # log-sum-exp 分母
        log_sim = torch.exp(sim_masked)
        log_prob = sim_masked - torch.log(log_sim.sum(1, keepdim=True))

        # 正样本的 log-probability 均值
        pos_count = pos_mask.sum(1).clamp(min=1)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_count

        loss = -mean_log_prob_pos.mean()
        return loss

    def train(self, train_loader):
        best_acc = 0
        scaler = GradScaler(enabled=self.fp16_precision)
        n_iter = 0

        print(f"Start Dual-Domain Supervised Contrastive Learning for {self.num_epoches} epochs")
        print(f"  alpha={self.alpha} (time+freq supcon), beta={self.beta} (cross-consistency)")

        for epoch_counter in range(self.num_epoches + 1):
            epoch_loss_time = 0.0
            epoch_loss_freq = 0.0
            epoch_loss_cons = 0.0
            num_batches = 0

            with tqdm.tqdm(train_loader, unit='batch') as tepoch:
                for x_time, x_freq, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch_counter}")
                    self.model.train()

                    # 增加通道维度 (Batch, Length) -> (Batch, 1, Length)
                    x_time = x_time.view(x_time.size(0), 1, x_time.size(1)).float().to(self.device)
                    x_freq = x_freq.view(x_freq.size(0), 1, x_freq.size(1)).float().to(self.device)
                    labels = labels.long().to(self.device)

                    with autocast(enabled=self.fp16_precision):
                        # 前向传播，获取时域和频域的投影特征
                        z_time, z_freq = self.model(x_time, x_freq)

                        # 1. 时域监督对比损失
                        loss_time = self.supervised_contrastive_loss(z_time, labels)

                        # 2. 频域监督对比损失
                        loss_freq = self.supervised_contrastive_loss(z_freq, labels)

                        # 3. 时频一致性损失 (Cross-modal SupCon)
                        # 将时域和频域的原始样本拼在一起，同类跨域互为正样本
                        z_cross = torch.cat([z_time, z_freq], dim=0)
                        cross_labels = torch.cat([labels, labels], dim=0)
                        loss_consistency = self.supervised_contrastive_loss(z_cross, cross_labels)

                        # 总损失
                        loss = self.alpha * (loss_time + loss_freq) + self.beta * loss_consistency

                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    epoch_loss_time += loss_time.item()
                    epoch_loss_freq += loss_freq.item()
                    epoch_loss_cons += loss_consistency.item()
                    num_batches += 1

                    if n_iter % self.log_every_n_step == 0:
                        tepoch.set_postfix(
                            loss=loss.item(),
                            t=loss_time.item(),
                            f=loss_freq.item(),
                            c=loss_consistency.item()
                        )
                    n_iter += 1

            if epoch_counter >= 10:
                self.scheduler.step()

            # 打印 epoch 平均损失
            avg_t = epoch_loss_time / num_batches
            avg_f = epoch_loss_freq / num_batches
            avg_c = epoch_loss_cons / num_batches
            print(f"  Epoch {epoch_counter}: loss_time={avg_t:.4f}, loss_freq={avg_f:.4f}, loss_cons={avg_c:.4f}")

            # 保存模型
            if epoch_counter % 20 == 0 and epoch_counter > 0:
                os.makedirs('./checkpoints/WFTFC/', exist_ok=True)
                torch.save(self.model.state_dict(), f'./checkpoints/WFTFC/WFTFC_dualsupcon_epoch_{epoch_counter}.pth.tar')

# ==========================================
# 6. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 创建数据集和数据加载器
    train_dataset = DualDomainDataset(x_time, x_freq, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    time_input_dim = x_time.shape[1]
    freq_input_dim = x_freq.shape[1]
    print(f"Time input feature dim: {time_input_dim}")
    print(f"Freq input feature dim: {freq_input_dim}")

    # 初始化双域模型
    model = DualDomainModel(time_input_dim, freq_input_dim, out_dim=128).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    trainer = SupervisedDualCLR(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        fp16_precision=fp16_precision,
        device=device,
        temperature=temperature,
        num_epoches=num_epoches,
        batch_size=batch_size,
        alpha=alpha,
        beta=beta
    )
    trainer.train(train_loader)