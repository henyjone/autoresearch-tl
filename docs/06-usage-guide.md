# 使用指南

## 1. 环境准备

### 前置要求

- Python 3.10+
- NVIDIA GPU（推荐 RTX 5090，最低 8 GB VRAM）
- CUDA 12.8+
- [uv](https://docs.astral.sh/uv/) 包管理器

### 安装 uv（如未安装）

```bash
# Windows（PowerShell）
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 安装依赖

```bash
cd C:\Users\Administrator\Documents\autoresearch-tl
uv sync
```

uv 会自动解析 `pyproject.toml`，从 PyTorch cu128 源下载正确的 GPU 版本（约 2~3 GB）。

### 验证安装

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
# 期望输出示例：
# 2.5.1+cu128
# NVIDIA GeForce RTX 5090
```

---

## 2. 数据生成

### 生成 Phase 1 合成数据

```bash
uv run prepare.py
```

默认生成 20 万训练样本 + 5 万验证样本，存储于 `~/.cache/autoresearch-tl/data/`。

典型输出：

```
Cache directory: C:\Users\<用户名>\.cache\autoresearch-tl
Generating 200,000 training samples...
  Generated 50000/200000 samples
  Generated 100000/200000 samples
  Generated 150000/200000 samples
  Generated 200000/200000 samples
  Training data generated in 28.3s
Generating 50,000 validation samples...
  Validation data generated in 7.1s

Data saved to C:\Users\<用户名>\.cache\autoresearch-tl\data
  Training:   ...train.npz (200,000 samples)
  Validation: ...val.npz (50,000 samples)
  Stats:      ...stats.npz

Data statistics:
  TL range:  [10.0, 200.0] dB
  TL mean:   72.4 dB
  TL std:    24.8 dB
  Freq range: [10, 9998] Hz
  Range:     [1.0, 198.3] km

Done! Ready to train.
```

**数据已存在时**（重复运行 prepare.py）会跳过生成，直接退出。若需重新生成，先删除缓存目录：

```bash
# Windows PowerShell
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\autoresearch-tl"
```

### 小数据集（快速测试）

```bash
uv run prepare.py --num-train 10000 --num-val 5000
```

适用于调试数据管道，不适合正式训练。

---

## 3. 训练

### 启动训练

```bash
uv run train.py
```

训练将运行固定 5 分钟（`TIME_BUDGET = 300` 秒），完成后打印最终结果。

### 保存日志（推荐）

```bash
uv run train.py > run.log 2>&1
```

将所有输出重定向到 `run.log`，避免终端滚动丢失信息。

### 监控训练进度

在另一个终端窗口：

```bash
# 实时查看最新日志行（近似当前步）
Get-Content run.log -Wait -Tail 3

# 或等训练结束后查看关键指标
Select-String "^val_rmse:|^peak_vram_mb:" run.log
```

### 训练完成后的输出示例

```
---
val_rmse:         4.523187
training_seconds: 300.1
total_seconds:    342.7
peak_vram_mb:     187.3
total_samples_M:  14.2
num_steps:        13867
num_params_K:     214.3
hidden_dim:       256
n_layers:         4
```

---

## 4. 启动 Agent 自主实验

### 4.1 创建实验分支

```bash
cd C:\Users\Administrator\Documents\autoresearch-tl
git checkout -b autoresearch-tl/mar23
```

分支命名规范：`autoresearch-tl/<日期标签>`，如 `mar23`、`apr01`。

### 4.2 初始化 results.tsv

```bash
# 创建带表头的空 TSV（注意：制表符分隔）
echo "commit`tval_rmse`tmemory_gb`tstatus`tdescription" > results.tsv
```

或手动创建 `results.tsv`，内容为：

```
commit	val_rmse	memory_gb	status	description
```

**重要**：`results.tsv` 不纳入 git 追踪，保持为未暂存文件。

### 4.3 验证数据存在

```bash
ls "$env:USERPROFILE\.cache\autoresearch-tl\data\"
# 应看到：train.npz  val.npz  stats.npz
```

若文件不存在，先运行 `uv run prepare.py`。

### 4.4 Agent 读取指令文件

将以下内容提供给 Agent（LLM）：

```
请读取以下文件，然后按照 program.md 的规则开始自主实验循环：
- program.md（实验规则与物理直觉）
- prepare.py（固定不变，了解数据格式和评估函数）
- train.py（当前模型，你将修改此文件）

当前分支：autoresearch-tl/mar23
数据路径：~/.cache/autoresearch-tl/data/（已确认存在）
results.tsv：已初始化（只有表头）

第一次运行请先建立 baseline（不修改 train.py，直接运行）。
之后开始无限实验循环，直到我手动停止。
```

### 4.5 实验循环协议

Agent 按以下流程循环运行（来自 `program.md`）：

```
1. 查看当前 git 状态（分支/commit）
2. 修改 train.py（实验想法）
3. git commit
4. uv run train.py > run.log 2>&1
5. 读取结果：grep "^val_rmse:\|^peak_vram_mb:" run.log
6. 若结果为空（崩溃）：tail -n 50 run.log 查看错误，尝试修复
7. 记录到 results.tsv
8. 若 val_rmse 改进：保留 commit（前进）
   若 val_rmse 未改进：git reset --hard HEAD~1（回滚）
9. 回到步骤 1，永不停止
```

---

## 5. 训练完成后加载和使用模型

### 5.1 保存模型（在 train.py 末尾添加）

```python
# 在 train.py 末尾，Final evaluation 之后添加：
import os

model_dir = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-tl", "models")
os.makedirs(model_dir, exist_ok=True)

# 保存模型权重和配置
save_path = os.path.join(model_dir, "tl_model.pt")
torch.save({
    "model_state_dict": model.state_dict(),
    "config": asdict(config),
    "val_rmse": val_rmse,
}, save_path)
print(f"Model saved to {save_path}")
```

### 5.2 推理示例：预测单个场景的 TL

```python
import torch
import numpy as np
from prepare import (
    FeatureStats, MAX_FEATURES,
    IDX_FREQ, IDX_SRC_DEPTH, IDX_RCV_DEPTH, IDX_RANGE,
    IDX_WATER_DEPTH_SRC, IDX_WATER_DEPTH_RCV,
    IDX_BOTTOM_SS, IDX_BOTTOM_DENSITY,
    IDX_SSP_START, IDX_BATHY_START,
    SSP_DEPTHS, generate_ssp_profile, generate_bathymetry_profile,
)
from train import TLNet, TLNetConfig

# --- 1. 加载归一化统计量 ---
stats = FeatureStats.from_data()

# --- 2. 加载模型 ---
import os
model_path = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-tl", "models", "tl_model.pt")
checkpoint = torch.load(model_path, map_location="cuda")

config = TLNetConfig(**checkpoint["config"])
model = TLNet(config).cuda()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded model, val_rmse={checkpoint['val_rmse']:.4f} dB")

# --- 3. 构建查询特征向量 ---
def make_query(freq_hz, src_depth_m, rcv_depth_m, range_km,
               water_depth_m, bottom_type="sand",
               ssp_type="thermocline", bathy_type="flat"):
    """构建单个查询的 64 维特征向量。"""
    from prepare import BOTTOM_TYPES
    import numpy as np

    rng = np.random.default_rng(seed=0)
    features = np.zeros(MAX_FEATURES, dtype=np.float32)

    features[IDX_FREQ]            = freq_hz
    features[IDX_SRC_DEPTH]       = src_depth_m
    features[IDX_RCV_DEPTH]       = rcv_depth_m
    features[IDX_RANGE]           = range_km
    features[IDX_WATER_DEPTH_SRC] = water_depth_m
    features[IDX_WATER_DEPTH_RCV] = water_depth_m  # 假设平坦地形

    bt = BOTTOM_TYPES[bottom_type]
    features[IDX_BOTTOM_SS]      = bt["soundspeed"]
    features[IDX_BOTTOM_DENSITY] = bt["density"]

    ssp = generate_ssp_profile(ssp_type, water_depth_m, rng)
    features[IDX_SSP_START:IDX_SSP_START+20] = ssp

    bathy = generate_bathymetry_profile(bathy_type, water_depth_m, water_depth_m, 20, rng)
    features[IDX_BATHY_START:IDX_BATHY_START+20] = bathy

    return features

# --- 4. 执行预测 ---
query = make_query(
    freq_hz=500,          # 500 Hz
    src_depth_m=50,       # 声源深度 50 m
    rcv_depth_m=100,      # 接收器深度 100 m
    range_km=50,          # 距离 50 km
    water_depth_m=1000,   # 水深 1000 m
    bottom_type="sand",   # 砂质海底
    ssp_type="thermocline",  # 温跃层 SSP
    bathy_type="flat",    # 平坦地形
)

x = torch.from_numpy(query).unsqueeze(0).cuda()  # (1, 64)
x_norm = stats.normalize(x)

with torch.no_grad():
    tl_pred = model(x_norm)  # (1,)

print(f"预测 TL = {tl_pred.item():.2f} dB")
```

### 5.3 批量推理（高吞吐量）

```python
# 批量预测 N 个场景
queries = np.stack([make_query(...) for _ in range(1000)])  # (N, 64)
x = torch.from_numpy(queries).cuda()
x_norm = stats.normalize(x)

with torch.no_grad():
    tl_preds = model(x_norm)  # (N,)

print(f"批量预测 {len(queries)} 个场景")
print(f"TL 范围: [{tl_preds.min():.1f}, {tl_preds.max():.1f}] dB")
print(f"TL 均值: {tl_preds.mean():.1f} dB")
```

RTX 5090 上批量推理速度参考：
- batch_size=1：< 0.5 ms/次（含 CPU-GPU 传输）
- batch_size=1024：约 1 ms/批（~1 µs/次）

---

## 6. results.tsv 解读

实验结束后，`results.tsv` 记录了完整的实验历史：

```
commit	val_rmse	memory_gb	status	description
a1b2c3d	5.234567	0.8	keep	baseline MLP 4x256 GELU
b2c3d4e	4.891234	0.9	keep	add residual connections
c3d4e5f	5.100000	0.9	discard	switch ReLU: worse
d4e5f6g	4.523456	1.1	keep	log freq+range features in reserved slots
e5f6g7h	0.000000	0.0	crash	8x1024 MLP: OOM
f6g7h8i	4.489012	1.1	keep	huber loss delta=5
```

**解读要点**：

- 按时间顺序排列，代表 Agent 的探索路径
- `keep` 行的 commit hash 对应当前 git 历史中的实际 commit（可 `git checkout` 还原）
- `discard` 行的 commit hash 已被 `git reset` 移除，仅作记录用
- `crash` 行 val_rmse=0.0，memory_gb=0.0，表示运行失败
- 最后一个 `keep` 行就是当前 baseline（当前分支 HEAD）

**查找最佳结果**：

```bash
# 在 PowerShell 中按 val_rmse 排序（排除 crash）
Import-Csv results.tsv -Delimiter "`t" |
    Where-Object { $_.status -ne "crash" } |
    Sort-Object val_rmse |
    Select-Object -First 5 |
    Format-Table

# 或用 Python
python -c "
import pandas as pd
df = pd.read_csv('results.tsv', sep='\t')
print(df[df.status != 'crash'].sort_values('val_rmse').head(10))
"
```

---

## 7. 常见问题排查

### Q1：`uv run train.py` 报错 `FileNotFoundError: stats.npz`

**原因**：未生成数据，或数据目录不存在。

**解决**：

```bash
uv run prepare.py
```

---

### Q2：`torch.compile` 长时间无响应（> 5 分钟）

**原因**：首次编译需要 30~120 秒，属正常现象（不计入 TIME_BUDGET）。

**解决**：等待即可。若超过 5 分钟仍无输出，检查 GPU 驱动版本是否支持 CUDA 12.8。

```bash
nvidia-smi  # 查看驱动版本和 CUDA 版本
```

---

### Q3：训练输出 `FAIL` 并退出

**原因**：loss 出现 NaN 或爆炸（> 10⁶），通常由以下原因引起：
- 学习率过大（LEARNING_RATE > 0.01 时风险增加）
- 模型权重初始化问题（如自定义初始化不当）
- 批归一化与小 batch size 的不兼容

**解决**：
1. 降低 `LEARNING_RATE`（例如从 1e-3 降至 3e-4）
2. 检查是否有除零操作或 log(0) 等数值不稳定点
3. 添加梯度裁剪：

```python
# 在 loss.backward() 之后，optimizer.step() 之前添加
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### Q4：`CUDA out of memory`

**原因**：模型或 batch 过大（在 RTX 5090 的 32 GB 上不太可能，但仍有可能）。

**解决**：
1. 减小 `BATCH_SIZE`（从 1024 降至 512 或 256）
2. 减小模型规模（`HIDDEN_DIM` 或 `N_LAYERS`）
3. 检查是否有意外的大张量未释放（自定义前向中的中间变量）

```bash
# 查看当前 GPU 显存占用
nvidia-smi
```

---

### Q5：val_rmse 比训练 ~rmse 高很多（过拟合）

**原因**：模型容量过大，或训练集太小。

**解决**：
1. 增大 `DROPOUT`（从 0.1 到 0.2~0.3）
2. 添加 `WEIGHT_DECAY`（从 1e-4 增至 1e-3）
3. 增大训练集：`uv run prepare.py --num-train 500000`

---

### Q6：每步耗时（dt）异常高（> 100 ms）

**原因**：可能是 GC 未冻结，或 CPU-GPU 同步开销。

**检查**：
- 确认 `gc.disable()` 已执行（在 step 0 之后）
- 确认 `torch.cuda.synchronize()` 仅用于计时（不要在每步都同步）
- 检查是否有 Python 端的大型循环在 forward 中运行

---

### Q7：git reset 之后 results.tsv 内容丢失

**原因**：若误将 `results.tsv` 纳入 git，reset 会还原文件内容。

**解决**：确保 `results.tsv` 从未被 `git add`。若已追踪，移除追踪：

```bash
git rm --cached results.tsv
echo "results.tsv" >> .gitignore
git commit -m "untrack results.tsv"
```

---

### Q8：Windows 上 `grep` 命令不可用

`program.md` 中的 grep 命令是 Linux/macOS 语法。在 Windows PowerShell 中等价操作：

```powershell
# 替代 grep "^val_rmse:\|^peak_vram_mb:" run.log
Select-String "^val_rmse:|^peak_vram_mb:" run.log

# 替代 tail -n 50 run.log
Get-Content run.log -Tail 50
```

若使用 Git Bash（推荐），`grep` 和 `tail` 可直接使用。

---

## 8. 目录结构速查

```
autoresearch-tl/                     # 项目根目录
├── prepare.py                       # 数据生成 + 评估（固定，不可改）
├── train.py                         # 模型 + 训练循环（Agent 修改此文件）
├── program.md                       # 实验规则与物理提示（只读）
├── pyproject.toml                   # 依赖定义
├── uv.lock                          # 锁定的依赖版本
├── results.tsv                      # 实验记录（不纳入 git）
├── run.log                          # 最近一次训练日志（不纳入 git）
└── docs/
    ├── 01-project-overview.md
    ├── 02-data-design.md
    ├── 03-model-architecture.md
    ├── 04-autoresearch-adaptation.md
    ├── 05-experiment-log.md
    └── 06-usage-guide.md            # 本文件

~/.cache/autoresearch-tl/data/       # 数据缓存（自动生成）
    ├── train.npz                    # 训练集
    ├── val.npz                      # 验证集
    └── stats.npz                    # 归一化统计
```
