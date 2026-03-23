# 框架改造说明

## 1. 原版 autoresearch 的工作原理

**autoresearch** 是 Andrej Karpathy 设计的一套 LLM 自主实验框架，原版用于语言模型架构研究：

```
目标：最小化字符级语言模型的验证集 bits-per-byte（val_bpb）
任务类型：NLP，序列建模
输入：文本 token 序列（离散整数）
输出：下一个 token 的概率分布
损失：交叉熵 → bits-per-byte
```

**核心工作流**：

1. Agent 读取 `program.md` 获取规则和物理直觉提示
2. 在专用 git 分支上修改 `train.py`（模型/超参数/训练逻辑）
3. 固定 5 分钟时间预算运行实验
4. 对比 val_bpb：改进则保留 commit，退步则 `git reset` 回滚
5. 结果记录到 `results.tsv`，无限循环直到人工停止

---

## 2. 改造的核心变化

### NLP → 水下声学回归

| 维度 | 原版 autoresearch | autoresearch-tl |
|------|-------------------|-----------------|
| **任务类型** | 序列建模（语言模型） | 标量回归（TL 预测） |
| **输入** | 离散 token 序列（变长） | 固定 64 维连续特征向量 |
| **输出** | 词汇表上的概率分布 | 单个标量（TL，dB） |
| **损失函数** | 交叉熵 | MSE（baseline）/ 可替换 |
| **评估指标** | val_bpb（越低越好） | val_rmse（dB，越低越好） |
| **数据来源** | 文本语料库 | 解析物理公式生成 |
| **特征表示** | Embedding 查表 | 连续数值 + 归一化 |
| **输入维度** | 变长序列 × 词汇表 | 固定 64 维 |

### 离散 → 连续

原版需要 tokenizer（将文本转换为整数 token），模型首层是 Embedding 层。

本项目改为连续特征向量，首层直接是 `nn.Linear(64, hidden_dim)`，无需 embedding。

### val_bpb → val_rmse

```python
# 原版评估：交叉熵 → bits-per-byte
val_bpb = cross_entropy_loss / math.log(2)

# 本项目评估：均方误差 → RMSE（dB）
val_rmse = math.sqrt(mean_squared_error_over_validation_set)
```

单位为 dB，物理含义直观：`val_rmse = 3.5` 意味着模型预测误差约 ±3.5 dB（均方根）。

---

## 3. 保留的核心机制

### 3.1 固定时间预算

```python
TIME_BUDGET = 300  # 秒，定义在 prepare.py（固定不变）
```

所有实验使用相同的 5 分钟训练时间，排除"训练更久自然更好"的混淆因素。不同实验的对比是公平的：相同时间预算下，谁的 val_rmse 更低谁胜出。

计时从第 6 步开始（跳过 `torch.compile` 热身），确保各实验的有效计算时间一致。

### 3.2 keep/discard 实验循环

```
LOOP:
  修改 train.py → git commit → uv run train.py > run.log
  读取 val_rmse
  if val_rmse < 上一次 best:
      keep（保留 commit，前进）
  else:
      discard（git reset --hard HEAD~1，回到上次 best）
```

这是一个严格的爬山算法（hill climbing）：每次只接受改进，拒绝退步。

### 3.3 results.tsv 实验台账

```
commit	val_rmse	memory_gb	status	description
a1b2c3d	5.234567	0.8	keep	baseline MLP 4x256 GELU
b2c3d4e	4.891234	0.9	keep	add residual connections
c3d4e5f	5.100000	0.9	discard	switch ReLU, worse
d4e5f6g	0.000000	0.0	crash	OOM with 8x1024
```

- `results.tsv` **不纳入 git 追踪**（留在工作目录未暂存状态），作为独立的实验日志
- `status` 字段：`keep`（保留）/ `discard`（回滚）/ `crash`（崩溃）
- `memory_gb`：`peak_vram_mb / 1024`，监控显存消耗

### 3.4 git 分支管理

```bash
# 每次独立实验在专用分支上进行
git checkout -b autoresearch-tl/mar23

# 实验成功（改进）
git commit -m "add residual connections: val_rmse 5.23 → 4.89"

# 实验失败（退步）
git reset --hard HEAD~1
```

分支名格式：`autoresearch-tl/<tag>`，tag 通常为日期（如 `mar23`）。这使得不同实验轮次互不干扰，可以随时回溯历史。

---

## 4. 去掉的组件

### 4.1 Flash Attention / CUDA Kernels

原版 autoresearch 包含手写 CUDA kernel（Flash Attention 变体），用于加速自注意力计算。

**为何去掉**：本项目 baseline 是 MLP，不含注意力机制。若 Agent 探索 SSP 注意力分支，可使用 `torch.nn.MultiheadAttention`（PyTorch 内置，支持 `scaled_dot_product_attention` 后端自动选择 Flash Attention）。

### 4.2 rustbpe / tokenizer

原版包含 Rust 实现的 BPE tokenizer（将文本转换为 token）。

**为何去掉**：本项目输入是连续数值特征向量，不需要 tokenizer。`FeatureStats`（归一化统计类）承担了 tokenizer 的"预处理"角色。

### 4.3 数据流水线的差异

原版：文本文件 → tokenizer → token 整数数组 → GPU Embedding

本项目：`prepare.py` 生成 `.npz` 文件 → `FeatureStats.normalize()` → GPU 连续特征矩阵

---

## 5. 接口映射表

| 原版组件 | 原版作用 | 本项目对应 | 本项目作用 |
|---------|---------|-----------|-----------|
| `Tokenizer` | 文本 → token ID，词汇表管理 | `FeatureStats` | 特征归一化统计（mean/std） |
| `make_dataloader(tokenizer, B, split)` | 返回 token batch | `make_dataloader(stats, B, split)` | **接口保持一致**，返回归一化特征 batch |
| `evaluate_bpb(model, tokenizer, B)` | 计算 val bits-per-byte | `evaluate_rmse(model, stats, B)` | 计算 val RMSE（dB） |
| `model(x) -> logits` | 返回 token 概率 | `model(x) -> pred` | 返回 TL 预测（dB），shape `(B,)` |
| `model(x, targets) -> loss` | 交叉熵 loss | `model(x, targets) -> loss` | **接口保持一致**，MSE loss |
| `val_bpb` | 评估指标（越低越好） | `val_rmse` | 评估指标（越低越好，dB） |
| `results.tsv` 列 `val_bpb` | 记录语言模型困惑度 | `results.tsv` 列 `val_rmse` | 记录 TL 预测误差 |

`make_dataloader` 和 `forward(x, targets=None)` 的调用签名有意保持与原版一致，降低 Agent 的适应成本。

---

## 6. Windows + RTX 5090 兼容性说明

### 6.1 CUDA 版本

RTX 5090（Blackwell 架构，sm_120）需要 CUDA 12.8+。依赖配置：

```toml
# pyproject.toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]

[[tool.uv.index]]
name = "pytorch-cu128"
url  = "https://download.pytorch.org/whl/cu128"
explicit = true
```

安装命令：`uv sync`（uv 会自动选择 cu128 版本的 PyTorch wheel）

### 6.2 torch.compile 兼容性

```python
model = torch.compile(model, dynamic=False)
```

- `dynamic=False`：固定 batch size 形状，允许编译器做更激进的优化
- 在 Windows 上，`torch.compile` 依赖 Triton（通过 `torch.inductor`），RTX 5090 支持需要 PyTorch >= 2.5
- 首次运行会有 30~60 秒编译时间，但此时间不计入 `TIME_BUDGET`（由 `step > 5` 保护）

### 6.3 显存估算

```
RTX 5090 VRAM：32 GB

baseline 显存占用：
  模型参数：   ~1 MB（214K × 4 bytes）
  梯度：       ~1 MB
  优化器状态：  ~2 MB（AdamW 需 2× 参数量）
  激活值：     ~batch_size × 64 × 4 × n_layers ≈ 几 MB
  数据缓冲：    pin_memory ~100 MB

预估 peak_vram：< 200 MB（极充裕）
```

即使将模型扩展到 100M 参数（如 8×2048 MLP），显存仍远低于 32 GB 上限。Agent 无需担心 OOM，可以大胆探索大模型。

### 6.4 Windows 路径注意事项

数据缓存目录：`~/.cache/autoresearch-tl/data/`
在 Windows 上展开为：`C:\Users\<用户名>\.cache\autoresearch-tl\data\`

`os.path.expanduser("~")` 在 Windows 上行为正常，无需特殊处理。

### 6.5 不支持的特性

- `torch.compile` 在 Windows 上不支持 Triton 的部分 kernel（如 fused Adam），退回到 eager 模式
- `pin_memory` 在 Windows 上可用但效率略低于 Linux
- 多进程 DataLoader（`num_workers > 0`）在 Windows 上需要 `if __name__ == '__main__':` 保护，但本项目使用单线程生成器，无此问题
