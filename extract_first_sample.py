"""
提取 源域AWF1 第一个样本的时域和频域特征（分别打印）
使用 Sinkhorn 预训练模型
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# ==========================================
# 1. 设备
# ==========================================
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Device: {device}")

# ==========================================
# 2. 模型定义
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
        return h_time, h_freq, z_time, z_freq


# ==========================================
# 3. 加载数据和模型
# ==========================================
print("Loading AWF1 data...")
time_data = np.load('./datasets/awf1.npz')
freq_data = np.load('./datasets/awf1_freq.npz')

x_time_all = time_data['feature']
x_freq_all = freq_data['x']
y_all = time_data['label']

print(f"Time: {x_time_all.shape}, Freq: {x_freq_all.shape}, Labels: {y_all.shape}")

time_input_dim = x_time_all.shape[1]
freq_input_dim = x_freq_all.shape[1]

# 构建模型并加载权重
model = DualDomainModel(time_input_dim, freq_input_dim, out_dim=128).to(device)
model.eval()

checkpoint = torch.load('./checkpoints/WFTFC/WFTFC_dualsupcon_sinkhorn_epoch_80.pth.tar',
                         map_location=device)
model.load_state_dict(checkpoint, strict=False)
print("Checkpoint loaded.")

# ==========================================
# 4. 提取第一个样本的特征
# ==========================================
print("\n" + "="*70)
print("         第一个样本特征提取")
print("="*70)

# 取第一个样本
idx = 0
xt = torch.from_numpy(x_time_all[idx]).float().view(1, 1, -1).to(device)
xf = torch.from_numpy(x_freq_all[idx]).float().view(1, 1, -1).to(device)
label = int(y_all[idx])

print(f"\n样本索引: {idx}")
print(f"标签 (类别): {label}")
print(f"时域输入形状: {xt.shape}")
print(f"频域输入形状: {xf.shape}")

with torch.no_grad():
    h_time, h_freq, z_time, z_freq = model(xt, xf)

h_time_np = h_time.cpu().numpy()[0]  # (512,)
h_freq_np = h_freq.cpu().numpy()[0]  # (512,)
z_time_np = z_time.cpu().numpy()[0]  # (128,)
z_freq_np = z_freq.cpu().numpy()[0]  # (128,)

# ---- 时域特征 ----
print("\n" + "-"*70)
print(f"【时域特征 h_time】 (512维)")
print("-"*70)
print(f"  形状: {h_time_np.shape}")
print(f"  数据类型: {h_time_np.dtype}")
print(f"  统计: min={h_time_np.min():.6f}, max={h_time_np.max():.6f}, "
      f"mean={h_time_np.mean():.6f}, std={h_time_np.std():.6f}")
print(f"  前20个元素: {np.array2string(h_time_np[:20], precision=4, separator=', ', suppress_small=True)}")
print(f"  后20个元素: {np.array2string(h_time_np[-20:], precision=4, separator=', ', suppress_small=True)}")
# 打印全部 512 维
print(f"\n  完整512维向量:")
print(np.array2string(h_time_np, precision=4, separator=', ', suppress_small=True, max_line_width=120))

# ---- 频域特征 ----
print("\n" + "-"*70)
print(f"【频域特征 h_freq】 (512维)")
print("-"*70)
print(f"  形状: {h_freq_np.shape}")
print(f"  数据类型: {h_freq_np.dtype}")
print(f"  统计: min={h_freq_np.min():.6f}, max={h_freq_np.max():.6f}, "
      f"mean={h_freq_np.mean():.6f}, std={h_freq_np.std():.6f}")
print(f"  前20个元素: {np.array2string(h_freq_np[:20], precision=4, separator=', ', suppress_small=True)}")
print(f"  后20个元素: {np.array2string(h_freq_np[-20:], precision=4, separator=', ', suppress_small=True)}")
# 打印全部 512 维
print(f"\n  完整512维向量:")
print(np.array2string(h_freq_np, precision=4, separator=', ', suppress_small=True, max_line_width=120))

# ---- 时域投影特征 ----
print("\n" + "-"*70)
print(f"【时域投影特征 z_time】 (128维)")
print("-"*70)
print(f"  统计: min={z_time_np.min():.6f}, max={z_time_np.max():.6f}, "
      f"mean={z_time_np.mean():.6f}, std={z_time_np.std():.6f}")
print(f"  向量: {np.array2string(z_time_np, precision=4, separator=', ', suppress_small=True)}")

# ---- 频域投影特征 ----
print("\n" + "-"*70)
print(f"【频域投影特征 z_freq】 (128维)")
print("-"*70)
print(f"  统计: min={z_freq_np.min():.6f}, max={z_freq_np.max():.6f}, "
      f"mean={z_freq_np.mean():.6f}, std={z_freq_np.std():.6f}")
print(f"  向量: {np.array2string(z_freq_np, precision=4, separator=', ', suppress_small=True)}")

# ---- 跨模态对比 ----
print("\n" + "="*70)
print("         跨模态对比")
print("="*70)

# 余弦相似度
from numpy.linalg import norm
cos_sim_h = np.dot(h_time_np, h_freq_np) / (norm(h_time_np) * norm(h_freq_np))
cos_sim_z = np.dot(z_time_np, z_freq_np) / (norm(z_time_np) * norm(z_freq_np))
l2_dist_h = norm(h_time_np - h_freq_np)
l2_dist_z = norm(z_time_np - z_freq_np)

print(f"  时域 vs 频域 — 特征空间:")
print(f"    h_time vs h_freq:  cosine_sim={cos_sim_h:.4f}, L2_dist={l2_dist_h:.4f}")
print(f"    z_time vs z_freq:  cosine_sim={cos_sim_z:.4f}, L2_dist={l2_dist_z:.4f}")

# 保存到文件以便查看
np.savez('./features/first_sample_features.npz',
         h_time=h_time_np, h_freq=h_freq_np,
         z_time=z_time_np, z_freq=z_freq_np,
         label=label)
print("\n特征已保存至 ./features/first_sample_features.npz")
print("Done!")
"""
骨干网络特征（h_time、h_freq——经全连接层后为512维）的方差更大
（标准差约0.45），而投影特征（z_time、z_freq——128维）由于经过投影头和
归一化处理，分布更为紧凑（标准差约0.08，以0为中心）。
"""