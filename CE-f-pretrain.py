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
num_epoches = 100
learning_rate = 0.0005

# ==========================================
# 2. 数据加载
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
# 3. 模型定义 (直接做分类的 Backbone + Classifier)
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

# ==========================================
# 4. 数据加载器
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
# 5. 交叉熵分类训练
# ==========================================
def accuracy(output, target):
    """计算 Top-1 准确率"""
    with torch.no_grad():
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).float().sum().item()
        return correct / target.size(0) * 100.0

class CrossEntropyTrainer(object):
    def __init__(self, **args):
        self.model = args['model']
        self.optimizer = args['optimizer']
        self.scheduler = args['scheduler']
        self.fp16_precision = args['fp16_precision']
        self.num_epoches = args['num_epoches']
        self.device = args['device']
        self.criterion = nn.CrossEntropyLoss()
        self.log_every_n_step = 100

    def train(self, train_loader):
        best_acc = 0
        scaler = GradScaler(enabled=self.fp16_precision)

        n_iter = 0
        print("Start Cross-Entropy training for %d epochs" % self.num_epoches)

        for epoch_counter in range(self.num_epoches + 1):
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0

            with tqdm.tqdm(train_loader, unit='batch') as tepoch:
                for data, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch_counter}")

                    self.model.train()

                    # data shape: (Batch, Length) -> (Batch, 1, Length)
                    data = data.view(data.size(0), 1, data.size(1))
                    data = data.float().to(self.device)
                    labels = labels.long().to(self.device)

                    with autocast(enabled=self.fp16_precision):
                        output = self.model(data)
                        loss = self.criterion(output, labels)

                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    epoch_loss += loss.item()
                    acc = accuracy(output, labels)
                    epoch_acc += acc
                    num_batches += 1

                    if n_iter % self.log_every_n_step == 0:
                        tepoch.set_postfix(loss=loss.item(), acc=f"{acc:.2f}%")
                    n_iter += 1

            # 每个 epoch 结束后 step scheduler
            if epoch_counter >= 10:
                self.scheduler.step()

            # 打印 epoch 统计
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            print(f"  Epoch {epoch_counter} avg: loss={avg_loss:.4f}, acc={avg_acc:.2f}%")
            best_acc = max(best_acc, avg_acc)

            # 保存模型
            if epoch_counter % 20 == 0 and epoch_counter > 0:
                os.makedirs('./checkpoints/ce/', exist_ok=True)
                torch.save(self.model.state_dict(), f'./checkpoints/ce/WFTFC_freq_ce_epoch_{epoch_counter}.pth.tar')

        print(f"Best avg accuracy: {best_acc:.2f}%")

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

    # 初始化模型：直接输出 num_classes 做分类
    model = DFNet(out_dim=num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    trainer = CrossEntropyTrainer(
               model = model,
               optimizer = optimizer,
               scheduler = scheduler,
               fp16_precision = fp16_precision,
               num_epoches = 101,
               device = device)
    trainer.train(train_loader)