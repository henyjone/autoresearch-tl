# 模型架构

## 1. Baseline MLP 设计

当前 baseline 是一个 4 层全连接神经网络（Multi-Layer Perceptron），结构简洁，易于理解和修改。

### 网络结构

```
输入层：  64 维归一化特征向量
隐藏层1： Linear(64 → 256) → GELU → Dropout(0.1)
隐藏层2： Linear(256 → 256) → GELU → Dropout(0.1)
隐藏层3： Linear(256 → 256) → GELU → Dropout(0.1)
隐藏层4： Linear(256 → 256) → GELU → Dropout(0.1)
输出层：  Linear(256 → 1) → squeeze → 标量 TL 预测（dB）
```

### 代码定义

```python
@dataclass
class TLNetConfig:
    n_features: int = 64   # MAX_FEATURES，固定不变
    hidden_dim: int = 256
    n_layers:   int = 4
    dropout:    float = 0.1


class TLNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        in_dim = config.n_features
        for i in range(config.n_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.GELU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim
        layers.append(nn.Linear(config.hidden_dim, 1))
        self.net = nn.Sequential(*layers)
```

### 参数量

```
4层 × (64→256 或 256→256) + 输出层：
  层1：  64 × 256 + 256   =  16,640
  层2~4：256 × 256 + 256  =  65,792 × 3 = 197,376
  输出：  256 × 1 + 1     =     257
  总计：约 214,273 参数（~214K）
```

---

## 2. 模型接口约束

**此接口由 `prepare.py` 中的 `evaluate_rmse` 决定，不得更改签名语义。**

```python
def forward(self, x, targets=None):
    """
    Args:
        x:       (B, MAX_FEATURES) — 已归一化的输入特征，float32，在 CUDA 上
        targets: (B,) — TL 值（dB，未归一化），float32，在 CUDA 上；或 None

    Returns:
        若 targets 不为 None：返回标量 MSE loss（用于训练）
        若 targets 为 None：返回预测值张量 (B,)，单位 dB
    """
    pred = self.net(x).squeeze(-1)   # (B,)
    if targets is not None:
        return F.mse_loss(pred, targets)
    return pred
```

评估时 `evaluate_rmse` 直接调用 `model(x)`（不传 targets），因此 `forward` 在推理模式下必须返回 `(B,)` 形状的张量。

---

## 3. 超参数说明

以下超参数在 `train.py` 顶部直接定义，Agent 可自由修改：

```python
# 模型架构
HIDDEN_DIM   = 256    # 隐藏层宽度（可调：128 / 256 / 512 / 1024）
N_LAYERS     = 4      # 隐藏层数（可调：2 ~ 8）
DROPOUT      = 0.1    # Dropout 比率（可调：0.0 ~ 0.3）

# 优化
BATCH_SIZE    = 1024  # 训练批大小（可调：256 ~ 4096）
LEARNING_RATE = 1e-3  # 峰值学习率（可调：1e-4 ~ 3e-3）
WEIGHT_DECAY  = 1e-4  # AdamW 权重衰减（可调：0 ~ 1e-3）
ADAM_BETAS    = (0.9, 0.999)  # Adam 动量参数

# 学习率调度
WARMUP_RATIO   = 0.05  # 线性预热阶段占总时间的比例
WARMDOWN_RATIO = 0.30  # 余弦衰减阶段占总时间的比例
FINAL_LR_FRAC  = 0.01  # 终止学习率 = LEARNING_RATE × FINAL_LR_FRAC
```

### 超参数选择建议

| 超参数 | 增大的效果 | 减小的效果 | 建议方向 |
|--------|-----------|-----------|---------|
| HIDDEN_DIM | 表达能力↑，VRAM↑，每步耗时↑ | 更快迭代，但欠拟合 | 先试 512 |
| N_LAYERS | 深度特征提取↑，梯度消失风险↑ | 更快，但容量不足 | 配合残差连接 |
| BATCH_SIZE | GPU 利用率↑，每步时间↓，步数↑ | 泛化略好，但步数↓ | 2048 / 4096 |
| LEARNING_RATE | 收敛快，但不稳定 | 稳定但收敛慢 | 随架构调整 |
| WARMUP_RATIO | 训练初期更稳定 | 可能早期震荡 | 保持 0.05 |
| WARMDOWN_RATIO | 最终收敛更好 | 可能欠收敛 | 0.3 ~ 0.5 |

---

## 4. 学习率调度

采用三段式调度：**线性预热 → 恒定 → 余弦衰减**

```python
def get_lr_multiplier(progress):
    """progress: 0.0（开始）→ 1.0（结束）"""
    if progress < WARMUP_RATIO:
        # 线性预热：0 → 1
        return progress / WARMUP_RATIO

    elif progress < 1.0 - WARMDOWN_RATIO:
        # 恒定阶段：峰值学习率
        return 1.0

    else:
        # 余弦衰减：1 → FINAL_LR_FRAC
        warmdown_progress = (1.0 - progress) / WARMDOWN_RATIO
        return FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * 0.5 * (
            1 + math.cos(math.pi * (1 - warmdown_progress))
        )
```

```
LR 曲线示意（TIME_BUDGET=300s，WARMUP=5%，WARMDOWN=30%）：

LR
│       ┌─────────────────────────┐
│      /│                         │╲
│     / │                         │  ╲_______
│    /  │                         │
└───────────────────────────────────────► 时间
     0  15s          210s         300s
     └─预热─┘└──────恒定──────┘└─衰减─┘
```

`progress` 基于实际训练时间（排除启动和编译开销）与 `TIME_BUDGET` 的比值计算，因此不受每步耗时影响。

---

## 5. 训练循环设计

### 时间预算控制

```python
# 前 5 步（step < 5）不计入训练时间，排除 torch.compile 的热身耗时
if step > 5:
    total_training_time += dt

# 达到时间预算后退出（至少完成 5 步）
if step > 5 and total_training_time >= TIME_BUDGET:
    break
```

### EMA 损失平滑

```python
ema_beta = 0.95
smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
# 偏差修正
debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
```

EMA 平滑后的损失用于日志展示（`~rmse` 近似值），不影响梯度计算。

### 快速失败机制

```python
if math.isnan(train_loss_f) or train_loss_f > 1e6:
    print("FAIL")
    exit(1)
```

若损失出现 NaN 或爆炸（> 10⁶），立即终止并返回非零退出码，使 Agent 能快速检测崩溃。

### GC 冻结优化

```python
if step == 0:
    gc.collect()
    gc.freeze()
    gc.disable()
```

第一步后冻结 GC，避免训练中途垃圾回收导致的延迟抖动，提升计时精度。

### Batch 预取

```python
# 当前步计算的同时，预取下一批数据（CPU→GPU 重叠）
x, y, epoch = next(train_loader)  # 在 optimizer.step() 之后
```

`make_dataloader` 使用 `pin_memory` + `non_blocking=True` 实现异步传输。

---

## 6. 训练输出格式

### 实时日志（循环中打印，`\r` 覆盖同一行）

```
step 01234 (45.2%) | loss: 12.3456 | ~rmse: 3.51 dB | lr: 0.001000 | dt: 8ms | epoch: 3 | remaining: 164s
```

| 字段 | 含义 |
|------|------|
| `step` | 当前训练步数 |
| `(%)` | 训练时间进度 |
| `loss` | EMA 平滑后的 MSE 损失（dB²） |
| `~rmse` | 近似 RMSE（=√loss，dB） |
| `lr` | 当前实际学习率 |
| `dt` | 上一步耗时（毫秒） |
| `epoch` | 当前数据 epoch |
| `remaining` | 剩余训练时间（秒） |

### 训练完成后输出（固定格式，供 Agent grep）

```
---
val_rmse:         3.456789     # 验证集 RMSE（dB），越低越好
training_seconds: 300.1        # 纯训练时间（不含启动）
total_seconds:    305.2        # 总耗时（含启动、编译、评估）
peak_vram_mb:     1234.5       # 峰值显存占用（MB）
total_samples_M:  12.3         # 训练总样本数（百万）
num_steps:        12058        # 总训练步数
num_params_K:     214.3        # 模型参数量（千）
hidden_dim:       256
n_layers:         4
```

Agent 通过以下命令提取关键指标：

```bash
grep "^val_rmse:\|^peak_vram_mb:" run.log
```

---

## 7. Agent 可探索的优化方向

以下方向来自 `program.md` 的架构提示，按照探索优先级排序：

### 7.1 残差连接（Residual Connections）★★★

```python
class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
    def forward(self, x):
        return x + self.net(x)  # 跳跃连接
```

残差连接使梯度更容易传播，支持更深的网络（6~8 层不再退化）。

### 7.2 特征工程（Feature Engineering）★★★

利用保留位 `[48:64]` 加入物理导出特征：

```python
# 在 forward() 中预处理，或在数据预处理阶段固化
log_freq   = torch.log10(x[:, IDX_FREQ])         # 对数频率
log_range  = torch.log10(x[:, IDX_RANGE] + 1e-6) # 对数距离
wavelength = 1500.0 / x[:, IDX_FREQ]             # 波长（m）
thorp_coef = compute_thorp(x[:, IDX_FREQ])       # Thorp 系数
```

log_freq 和 log_range 是最重要的物理特征（TL ≈ 20·log10(R)）。

### 7.3 分支编码器（Separate Branch Encoders）★★

```
几何特征 [0:8]  → MLP_geo (8 → 64)  ──┐
SSP 特征 [8:28] → MLP_ssp (20 → 64) ──┼─→ concat(192) → MLP_fusion → TL
地形特征 [28:48]→ MLP_bathy(20 → 64)──┘
```

分支编码器让每个物理子系统独立学习，避免 SSP/地形信息被几何特征淹没。

### 7.4 SSP 注意力（Attention over SSP）★

```python
# 将 20 个 SSP 深度样本视为序列
ssp = x[:, 8:28].unsqueeze(-1)  # (B, 20, 1)
# 加入深度位置编码后做自注意力
```

注意力机制可学习不同深度对 TL 的相对重要性（声道轴附近权重更高）。

### 7.5 归一化层（Normalization Layers）★★

```python
# 在 Linear 和 GELU 之间插入
nn.LayerNorm(hidden_dim)   # 或
nn.BatchNorm1d(hidden_dim) # 或
# RMSNorm（无偏置，更高效）
```

BatchNorm 对大 batch_size 效果好；LayerNorm 更稳定但略慢。

### 7.6 损失函数替换（Alternative Loss Functions）★★

```python
# Huber loss（对异常值鲁棒）
loss = F.huber_loss(pred, targets, delta=5.0)

# MAE（更鲁棒，收敛稍慢）
loss = F.l1_loss(pred, targets)

# log-cosh（介于 MSE 和 MAE 之间）
loss = torch.log(torch.cosh(pred - targets)).mean()
```

合成数据中 TL 分布右偏（极大值可能达 200 dB），Huber loss 能减少极端值对训练的影响。

### 7.7 更宽/更深的网络★

```
baseline:  4×256  ~214K 参数
尝试方向:  6×512  ~1.6M 参数（5 分钟内仍可收敛）
          8×128  ~130K 参数（更快每步，更多步数）
          3×1024 ~3.2M 参数（高容量，需更大 batch）
```

在 RTX 5090 上，1~2M 参数模型 5 分钟内通常可以充分训练。
