"""
提取 Sinkhorn 预训练模型在 AWF1 源域上的时域和频域特征。
使用 DualDomainModel 的 time_encoder 和 freq_encoder backbone（经过 fc 层后的 512-dim 特征）。
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import os
import matplotlib
matplotlib.use('Agg')  # 非交互后端，避免GUI问题
import matplotlib.pyplot as plt

# ==========================================
# 1. 设备配置
# ==========================================
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Device: {device}")

# ==========================================
# 2. 加载 AWF1 数据（源域）
# ==========================================
print("Loading AWF1 source domain data...")

# 时域
time_data = np.load('./datasets/awf1.npz')
x_time = time_data['feature']      # (N, 5000)
y = time_data['label']             # (N,)

# 频域
freq_data = np.load('./datasets/awf1_freq.npz')
x_freq = freq_data['x']            # (N, 2500)

print(f"Time data shape: {x_time.shape}")
print(f"Freq data shape: {x_freq.shape}")
print(f"Labels shape: {y.shape}")
num_classes = len(np.unique(y))
print(f"Number of classes: {num_classes}")

# ==========================================
# 3. 模型定义
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


class DualDomainModel(nn.Module):
    def __init__(self, time_input_dim, freq_input_dim, out_dim=128):
        super(DualDomainModel, self).__init__()
        self.time_encoder = DFNet(out_dim=512, input_feature_dim=time_input_dim)
        self.freq_encoder = DFNet(out_dim=512, input_feature_dim=freq_input_dim)
        self.time_projector = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, out_dim)
        )
        self.freq_projector = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, out_dim)
        )

    def forward(self, x_time, x_freq):
        h_time = self.time_encoder(x_time)
        h_freq = self.freq_encoder(x_freq)
        z_time = self.time_projector(h_time)
        z_freq = self.freq_projector(h_freq)
        return h_time, h_freq, z_time, z_freq  # 返回 backbone 特征和投影特征


# ==========================================
# 4. 加载 Sinkhorn 预训练权重
# ==========================================
time_input_dim = x_time.shape[1]   # 5000
freq_input_dim = x_freq.shape[1]   # 2500

model = DualDomainModel(time_input_dim, freq_input_dim, out_dim=128).to(device)
model.eval()

# 加载 checkpoint
checkpoint_path = './checkpoints/WFTFC/WFTFC_dualsupcon_sinkhorn_epoch_80.pth.tar'
print(f"Loading checkpoint: {checkpoint_path}")
state = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state, strict=False)
print("Checkpoint loaded.")

# 打印哪些键被跳过（projector 不是 strict=False 也会被检查）
missing, unexpected = model.load_state_dict(state, strict=False)
if len(unexpected) > 0:
    print(f"Unexpected keys (projectors skipped): {len(unexpected)}")
if len(missing) > 0:
    print(f"Missing keys: {missing}")


# ==========================================
# 5. 提取特征
# ==========================================
print("\nExtracting features...")
batch_size = 256

class SimpleDataset(Dataset):
    def __init__(self, x_time, x_freq, y):
        self.x_time = x_time
        self.x_freq = x_freq
        self.y = y
    def __getitem__(self, idx):
        return self.x_time[idx], self.x_freq[idx], self.y[idx]
    def __len__(self):
        return len(self.y)

dataset = SimpleDataset(x_time, x_freq, y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

all_h_time = []
all_h_freq = []
all_z_time = []
all_z_freq = []
all_labels = []

with torch.no_grad():
    for xt, xf, lb in loader:
        xt = xt.view(xt.size(0), 1, xt.size(1)).float().to(device)
        xf = xf.view(xf.size(0), 1, xf.size(1)).float().to(device)
        h_t, h_f, z_t, z_f = model(xt, xf)
        all_h_time.append(h_t.cpu())
        all_h_freq.append(h_f.cpu())
        all_z_time.append(z_t.cpu())
        all_z_freq.append(z_f.cpu())
        all_labels.append(lb)

h_time = torch.cat(all_h_time, dim=0).numpy()   # (N, 512)
h_freq = torch.cat(all_h_freq, dim=0).numpy()   # (N, 512)
z_time = torch.cat(all_z_time, dim=0).numpy()   # (N, 128)
z_freq = torch.cat(all_z_freq, dim=0).numpy()   # (N, 128)
labels = torch.cat(all_labels, dim=0).numpy()

print(f"\nExtracted feature shapes:")
print(f"  Backbone: h_time {h_time.shape}, h_freq {h_freq.shape}")
print(f"  Projection: z_time {z_time.shape}, z_freq {z_freq.shape}")

# ==========================================
# 6. 打印每个类别的特征统计（前5类）
# ==========================================
print("\n=== Feature Statistics (mean ± std per class, first 5 classes) ===")
for c in sorted(np.unique(labels))[:5]:
    mask = labels == c
    print(f"\nClass {c} (n={mask.sum()}):")
    print(f"  h_time:   mean={h_time[mask].mean():.4f}, std={h_time[mask].std():.4f}")
    print(f"  h_freq:   mean={h_freq[mask].mean():.4f}, std={h_freq[mask].std():.4f}")
    print(f"  z_time:   mean={z_time[mask].mean():.4f}, std={z_time[mask].std():.4f}")
    print(f"  z_freq:   mean={z_freq[mask].mean():.4f}, std={z_freq[mask].std():.4f}")

# 整体统计
print("\n=== Overall Feature Statistics ===")
print(f"h_time:   mean={h_time.mean():.4f}, std={h_time.std():.4f}, min={h_time.min():.4f}, max={h_time.max():.4f}")
print(f"h_freq:   mean={h_freq.mean():.4f}, std={h_freq.std():.4f}, min={h_freq.min():.4f}, max={h_freq.max():.4f}")
print(f"z_time:   mean={z_time.mean():.4f}, std={z_time.std():.4f}, min={z_time.min():.4f}, max={z_time.max():.4f}")
print(f"z_freq:   mean={z_freq.mean():.4f}, std={z_freq.std():.4f}, min={z_freq.min():.4f}, max={z_freq.max():.4f}")

# ==========================================
# 7. t-SNE 可视化（每个域随机选 2000 个样本）
# ==========================================
print("\nRunning t-SNE visualization...")
np.random.seed(42)
n_vis = min(2000, len(labels))
idx_vis = np.random.choice(len(labels), n_vis, replace=False)

# 对 4 组特征分别做 t-SNE
for feat_name, feat_data in [('h_time (backbone, 512d)', h_time),
                               ('h_freq (backbone, 512d)', h_freq),
                               ('z_time (projection, 128d)', z_time),
                               ('z_freq (projection, 128d)', z_freq)]:
    print(f"  t-SNE on {feat_name}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    feat_2d = tsne.fit_transform(feat_data[idx_vis])

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(feat_2d[:, 0], feat_2d[:, 1],
                         c=labels[idx_vis], cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(scatter, label='Class')
    plt.title(f't-SNE of {feat_name}\nAWF1 Source Domain (Sinkhorn Pretrain)')
    plt.tight_layout()
    safe_name = feat_name.split('(')[0].strip().replace(' ', '_')
    plt.savefig(f'./tsne_awf1_{safe_name}.png', dpi=150)
    plt.close()
    print(f"    Saved tsne_awf1_{safe_name}.png")

# ==========================================
# 8. 保存特征到磁盘（后续微调/迁移可用）
# ==========================================
os.makedirs('./features/', exist_ok=True)
np.savez('./features/awf1_sinkhorn_features.npz',
         h_time=h_time, h_freq=h_freq,
         z_time=z_time, z_freq=z_freq,
         labels=labels)
print("\nFeatures saved to ./features/awf1_sinkhorn_features.npz")
print("Done!")