# 实验记录

## 记录说明

本文件记录每次实验的详细数据，包括配置、训练过程曲线和结论分析。

**实验分支**：`autoresearch-tl/mar23`（2026-03-23 起）

---

## 关键决策记录

### 决策 001：选择 MLP 作为 Baseline 架构
**日期**：2026-03-23 | 以 4×256 MLP 作为起点，简单可靠。

### 决策 002：使用 Huber Loss 替代 MSE
**日期**：2026-03-23 | delta=5.0 dB，对异常值更鲁棒。

### 决策 003：采用分支编码器架构
**日期**：2026-03-23 | 按物理语义拆分输入（geo/SSP/bathy），用 1/5 参数达到更好结果。

### 决策 004：跨分支注意力
**日期**：2026-03-23 | 让 3 个分支互相"看到"对方，学习跨特征组交互，带来最大改善（-16.4%）。

---

## Phase 1 实验详细记录

### Exp0: Baseline — commit 1252596

**配置**：4 层 MLP，hidden=256，GELU，Dropout=0.1，MSE loss，batch=1024，lr=1e-3
**结果**：val_rmse=2.085 dB | 214K params | 31MB VRAM | 125,450 steps | **keep (baseline)**

---

### Exp1: 残差 MLP + Huber — commit 624297f

**变更**：ResidualBlock + LayerNorm + Huber(δ=5)，6×512，batch=2048，lr=3e-3
**结果**：val_rmse=2.069 dB (-0.016) | 6,342K params | 272MB | 56,805 steps | **keep**
**分析**：参数 30 倍增长只换来 -0.016 dB，容量不是瓶颈。

---

### Exp2: 暴力加宽 8×1024 — commit 6065139

**变更**：8×1024，batch=4096，lr=5e-3
**结果**：val_rmse=2.044 dB (-0.025) | 33,665K params | 1,340MB | 14,993 steps | **keep**
**分析**：5 倍参数只换 0.025 dB 改善，瓶颈在架构设计而非模型规模。

---

### Exp3: 分支编码器 — commit c9556af

**变更**：输入按物理语义拆分为 geo(8)+SSP(20)+bathy(20) 三个分支编码器，再融合
**结果**：val_rmse=2.034 dB (-0.010) | 7,113K params | 334MB | 39,523 steps | **keep**
**分析**：1/5 参数、1/4 显存，但 RMSE 更好。领域知识 > 暴力堆参数。

---

### Exp4: 跨分支注意力 — commit 7996430

**变更**：新增 CrossBranchAttention（2 层×4 头），分支间多头自注意力

**训练过程**：

| Step | Progress | Loss | ~RMSE | LR | Epoch |
|------|----------|------|-------|-----|-------|
| 0 | 0% | 364.0 | 19.08 | 0.000 | 1 |
| 2000 | 7% | 2.27 | 1.51 | 0.003 | 21 |
| 5000 | 17% | 1.86 | 1.36 | 0.003 | 52 |
| 10000 | 33% | 1.91 | 1.38 | 0.003 | 104 |
| 15000 | 49% | 1.66 | 1.29 | 0.003 | 155 |
| 20000 | 66% | 1.53 | 1.24 | 0.003 | 207 |
| 25000 | 82% | 1.35 | 1.16 | 0.002 | 258 |
| 30000 | 98% | 1.14 | 1.07 | 0.000 | 310 |

**结果**：val_rmse=**1.701 dB (-0.333, -16.4%)** | 7,640K params | 413MB | 30,669 steps | **keep (重大突破)**
**分析**：仅增加 7% 参数获得 16.4% RMSE 改善。训练/验证差距（1.07 vs 1.70）暗示轻微过拟合。

---

### Exp5: 物理导出特征 — commit 474ffef (discarded)

**变更**：forward 中计算 10 个物理特征（log-freq/range、Thorp、spreading、深度比等），geo_dim 8→18，dropout 0.02→0.05

**训练过程**：

| Step | Progress | Loss | ~RMSE | LR | Epoch |
|------|----------|------|-------|-----|-------|
| 0 | 0% | 364.7 | 19.10 | 0.000 | 1 |
| 5000 | 16% | 2.16 | 1.47 | 0.003 | 52 |
| 15000 | 49% | 1.65 | 1.29 | 0.003 | 155 |
| 25000 | 82% | 1.20 | 1.10 | 0.002 | 258 |
| 30000 | 99% | 0.95 | 0.98 | 0.000 | 310 |

**结果**：val_rmse=1.783 dB (+0.082, 退步) | **discard**
**分析**：训练 RMSE 更低(0.98)但验证更差 → 过拟合加剧。物理特征对当前合成数据冗余。

---

### Exp6: Transformer 风格跨分支 — commit 25e2fcb

**变更**：CrossBranchAttention → CrossBranchBlock（attention + FFN），2→3 层，batch 2048→4096，新增梯度裁剪(max_norm=1.0)

**训练过程**：

| Step | Progress | Loss | ~RMSE | LR | Epoch |
|------|----------|------|-------|-----|-------|
| 0 | 0% | 368.1 | 19.18 | 0.000 | 1 |
| 1000 | 9% | 3.10 | 1.76 | 0.003 | 21 |
| 3000 | 27% | 2.02 | 1.42 | 0.003 | 63 |
| 5000 | 44% | 1.83 | 1.35 | 0.003 | 105 |
| 7000 | 62% | 1.77 | 1.33 | 0.003 | 146 |
| 9000 | 80% | 1.61 | 1.27 | 0.002 | 188 |
| 10500 | 94% | 1.33 | 1.15 | 0.000 | 219 |

**结果**：val_rmse=**1.680 dB (-0.021)** | 8,694K params | 1,038MB | 11,211 steps | **keep**
**分析**：FFN 层和更多注意力层有帮助，但改善幅度收窄。训练/验证差距缩小（1.13 vs 1.68）。

---

### Exp7: 加宽模型 — commit 7635c2f (discarded)

**变更**：branch_dim 256→384，fusion_dim 512→768，n_heads 4→6
**结果**：val_rmse=1.709 dB (+0.029, 退步) | 19,530K params | 1,626MB | **discard**
**分析**：参数翻倍、显存翻倍但 RMSE 更差。模型容量不是当前瓶颈。

---

### Exp8: 降低学习率 + 长 warmdown — commit 70b8738 (discarded)

**动机**：Exp4-6 显示训练/验证差距较大（~0.55 dB），尝试降低 LR 和延长 warmdown 让模型收敛更平稳
**变更**：LR 3e-3→2e-3，warmdown_ratio 0.3→0.4，final_lr_frac 0.01→0.01

**训练过程**：

| Step | Progress | Loss | ~RMSE | LR | Epoch |
|------|----------|------|-------|-----|-------|
| 0 | 0% | 368.1 | 19.18 | 0.000 | 1 |
| 5000 | 25% | 1.69 | 1.30 | 0.002 | 105 |
| 10000 | 50% | 1.40 | 1.18 | 0.002 | 209 |
| 15000 | 75% | 1.05 | 1.03 | 0.001 | 313 |
| 20000 | 100% | 0.67 | 0.82 | 0.000 | 417 |

**结果**：val_rmse=1.908 dB (+0.228) | **discard**
**教训**：训练 RMSE 降至 0.82 但 val 1.91 → **严重过拟合**。低 LR + 长 warmdown 给了模型太多时间在低学习率下记忆训练集。train/val 差距 1.09 dB 是所有实验中最大的。

---

### Exp9: 强正则化 — commit 88dff25 (discarded)

**动机**：针对 exp8 暴露的过拟合问题，加强 weight_decay 和缩短 warmdown
**变更**：weight_decay 1e-4→5e-4，warmdown_ratio 0.3→0.2，final_lr_frac 0.01→0.05

**训练过程**：

| Step | Progress | Loss | ~RMSE | LR | Epoch |
|------|----------|------|-------|-----|-------|
| 0 | 0% | 368.1 | 19.18 | 0.000 | 1 |
| 5000 | 25% | 1.69 | 1.30 | 0.003 | 105 |
| 10000 | 50% | 1.40 | 1.18 | 0.003 | 209 |
| 15000 | 75% | 1.05 | 1.03 | 0.002 | 313 |
| 19500 | 99% | 0.95 | 0.98 | 0.000 | 407 |

**结果**：val_rmse=1.787 dB (+0.107) | **discard**
**教训**：更强的 weight_decay 限制了模型有效容量，train/val 差距仍然较大。weight_decay 不是解决过拟合的有效手段。

---

### Exp10: 高斯噪声增强 — (discarded)

**动机**：通过向输入添加随机噪声实现数据增强
**变更**：batch 4096→2048，训练时给输入加高斯噪声 std=0.05

**结果**：val_rmse=2.163 dB (+0.483) | **discard**
**教训**：噪声 std=0.05 对归一化后的输入来说太强了，严重干扰了模型学习。数据增强的强度需要非常谨慎。

---

### Exp11: Mixup 数据增强 — (discarded)

**动机**：Mixup 是经过验证的正则化方法，通过插值样本创建虚拟训练数据
**变更**：Beta(0.4, 0.4) mixup + fusion 层 6→4

**结果**：val_rmse=2.260 dB (+0.580) | **discard**
**教训**：**Mixup 不适合 TL 回归**。混合不同海洋环境的 TL 值产生物理上不存在的训练目标（比如混合深水和浅水场景的 TL 是无意义的）。Mixup 更适合分类或同质数据的回归。

---

### Exp12: 缩小模型 — (discarded)

**动机**：8.7M 参数 vs 200K 训练样本 = 43:1 参数/样本比，过拟合是必然的。大幅减小模型。
**变更**：branch_dim 256→128，fusion_dim 512→256，branch_layers 3→2，fusion_layers 6→4，batch 4096→2048

**训练过程**：

| Step | Progress | Loss | ~RMSE | LR | Epoch |
|------|----------|------|-------|-----|-------|
| 0 | 0% | 362.1 | 19.03 | 0.000 | 1 |
| 5000 | 18% | 1.90 | 1.38 | 0.003 | 52 |
| 15000 | 55% | 1.55 | 1.24 | 0.003 | 155 |
| 25000 | 90% | 1.11 | 1.05 | 0.001 | 258 |
| 27500 | 99% | 1.01 | 1.01 | 0.000 | 284 |

**结果**：val_rmse=1.784 dB (+0.104) | 1,607K params | 236MB | **discard**
**教训**：过拟合差距缩小（train 1.01 vs val 1.78 = 0.77 dB gap，vs exp6 的 0.55 dB），但模型容量不足。参数/样本比 8:1 更健康但欠拟合。最优点在中间。

---

### Exp13: Conv1D SSP/Bathymetry 编码器 — (discarded)

**动机**：SSP(20维) 和 bathymetry(20维) 是有序序列，Conv1D 应该能捕获局部模式（如声道、地形坡度）
**变更**：SSP 和 bathy 分支的 MLP 编码器替换为 3 层 Conv1D + BatchNorm + 全局池化

**结果**：val_rmse=1.685 dB (+0.005) | 8,522K params | 1,256MB | **discard（极其接近但没改善）**
**教训**：Conv1D 与 MLP 效果几乎相同。说明 prepare.py 的归一化已经把序列特征处理得很好，局部空间关系在 MLP 中也能学到。

---

### Exp14: 随机深度 — (discarded)

**动机**：Stochastic Depth 是深度网络的经典正则化方法
**变更**：fusion blocks 加入 drop_path（从第1层的0线性增加到最后一层的0.15）

**结果**：val_rmse=1.901 dB (+0.221) | **discard**
**教训**：drop_path 破坏了 fusion trunk 的信息流，对于只有 6 层的网络来说太激进了。Stochastic depth 更适合 50+ 层的深度网络。

---

### Exp15: 更多注意力 + MSE loss — (discarded)

**动机**：把计算预算从 MLP fusion 转移到注意力机制，同时尝试 MSE loss（合成数据无异常值）
**变更**：cross-branch blocks 3→5，fusion layers 6→3，Huber→MSE

**结果**：val_rmse=1.705 dB (+0.025) | 6,595K params | 1,160MB | **discard**
**教训**：更多注意力层没有带来改善。3 层 transformer block 已经足够捕获分支间交互。MSE 和 Huber 差异不大。

---

### Exp16: 物理残差学习 — (discarded)

**动机**：让网络预测 TL 与球面扩展估算（20·log10(range_m)）的差值，降低学习难度
**变更**：forward 中计算 baseline = 20·log10(range_km × 1000)，网络预测 correction，pred = baseline + correction

**结果**：val_rmse=1.929 dB (+0.249) | **discard**
**教训**：归一化后的 range 值无法直接映射到 dB 域。物理残差需要在未归一化的原始特征上计算，但我们的输入已经被 prepare.py 归一化了。这个方法在原始数据域可能有效，但在当前框架下不适用。

---

### Exp17: Transformer blocks + batch 2048 — (discarded)

**动机**：exp4(batch 2048)=1.701 vs exp6(batch 4096)=1.680，尝试 transformer blocks + batch 2048 的组合
**变更**：仅 batch 4096→2048，其余和 exp6 相同

**结果**：val_rmse=1.706 dB (+0.026) | 8,694K params | 580MB | **discard**
**教训**：batch 2048 给了更多步数（23K vs 11K）但没改善 val_rmse。batch 4096 的梯度更稳定，对当前模型更有利。

---

### Exp18: 多头预测集成 — (discarded)

**动机**：3 个独立输出头分别预测 TL，训练时都贡献 loss，推理时取平均（隐式集成）
**变更**：out_head 替换为 3 个独立的 Linear(512, 1)，loss = 平均(各头 Huber loss)

**结果**：val_rmse=1.888 dB (+0.208) | **discard**
**教训**：3 个线性头共享相同的 fusion trunk 输出，多样性不足，无法形成有效集成。真正的集成需要不同的模型结构或不同的训练初始化。

### Exp19: EMA 权重 — (discarded)

**动机**：用指数移动平均 (EMA) 维护模型权重副本，推理时用 EMA 权重，通常能获得更平滑的泛化
**变更**：EMA decay=0.999，训练完成后用 EMA 权重评估

**结果**：val_rmse=1.811 dB (+0.131) | **discard**
**教训**：EMA 权重滞后于最优权重。在 5 分钟训练预算内，模型仍在快速改善，EMA 的平滑效果反而拖慢了收敛。EMA 更适合长时间训练（训练充分后权重波动小时）。

### Exp20: LR 5e-3 + Cosine Warm Restarts — (discarded)

**动机**：SGDR（cosine annealing with warm restarts）周期性重启学习率，帮助跳出局部最优
**变更**：LR 5e-3，3 次 cosine warm restart 周期

**结果**：val_rmse=1.731 dB (+0.051) | **discard**
**教训**：Warm restarts 接近但未超过 best。3 个周期意味着每个周期只有 ~100s 训练时间，重启时丢失部分优化进度。在短预算下周期太多反而浪费。

### Exp21: Adam beta2=0.95 + LR 4e-3 — (discarded)

**动机**：降低 Adam beta2（从 0.999 到 0.95），让优化器对近期梯度更敏感，配合略高 LR
**变更**：ADAM_BETAS=(0.9, 0.95)，LR=4e-3

**结果**：val_rmse=1.707 dB (+0.027) | **discard**
**教训**：beta2=0.95 让二阶矩估计更"短视"，理论上对非平稳目标更好。但在此任务中数据分布稳定，标准 beta2=0.999 已足够。略有改善但不显著。

### Exp22: 辅助分支损失 (Deep Supervision) — (discarded)

**动机**：在每个分支编码器输出上添加独立回归头，强制每个分支独立学习有用表示，减少分支间信息冗余
**变更**：3 个 Linear(256,1) 辅助头，aux_loss_weight=0.1，总损失 = main_loss + 0.1 * avg(aux_losses)

**结果**：val_rmse=1.707 dB (+0.027) | **discard**
**教训**：辅助损失要求每个分支独立预测 TL，但单一分支（如仅几何信息或仅 SSP）信息不足以准确预测 TL。强制单分支预测可能引入了误导性的梯度信号。

### Exp23: SwiGLU 激活函数 — (discarded)

**动机**：SwiGLU（SiLU-gated linear unit）在 LLaMA/PaLM 等大模型中表现优异，替代 GELU
**变更**：ResidualBlock 和 CrossBranchBlock FFN 全部替换为 SwiGLU。参数从 8.7M 增至 12.2M (+40%)

**结果**：val_rmse=2.012 dB (+0.332) | **discard**
**教训**：SwiGLU 增加了 40% 参数（额外的门控投影），加剧过拟合。且每步更慢（16K vs 19K steps），在固定时间预算下训练不充分。SwiGLU 的优势在大数据场景下才能体现。

### Exp24: 周期验证 + 最佳检查点 — (discarded)

**动机**：测试是否存在过拟合拐点——模型在训练中期验证性能最佳，之后下降
**变更**：每 3000 步评估验证集，保存最佳权重

**验证曲线**：
| Step | val_rmse |
|------|----------|
| 3000 | 2.058 |
| 6000 | 1.993 |
| 9000 | 1.815 |
| 12000 | 1.846 |
| 15000 | 1.804 |
| 18000 | 1.784 |

**结果**：val_rmse=1.784 dB (+0.104) | **discard**
**教训**：验证损失持续下降到训练结束，**没有过拟合拐点**。step 12000 有小幅波动但总趋势下降。这说明模型在 5 分钟内仍在改善，过拟合不是通过早停可以解决的问题。周期性评估占用训练时间导致总 val_rmse 比 exp6 差。

### Exp25: 输入特征随机遮蔽 — (discarded)

**动机**：随机零化输入特征维度（denoising autoencoder 思路），与 exp10 高斯噪声不同
**变更**：INPUT_DROP_RATE=0.15，带反向缩放保持均值

**结果**：val_rmse=6.223 dB (+4.543) | **discard**
**教训**：灾难性结果。15% 的特征遮蔽率对物理特征过于激进——例如遮蔽"距离"或"频率"等关键特征后，剩余信息根本无法预测 TL。物理回归问题中输入特征之间存在强耦合，随机遮蔽破坏了这些关系。

---

## Phase 1 总结

**共 25 个实验**（exp0-25），耗时约 2 小时。

**架构演进路径**：MLP → 残差 MLP → 分支编码器 → 跨分支注意力 → Transformer blocks

**最终架构**（exp6, val_rmse=1.680 dB）：
- 3 个分支编码器：geo(8→256), ssp(20→256), bathy(20→256)，各 3 层 MLP
- 3 层 CrossBranchBlock（4 头注意力 + FFN，pre-norm 残差）
- 融合投影 768→512 + 6 层 ResidualBlock fusion trunk
- 输出 Linear(512→1)，Huber loss (delta=5.0)
- 参数量：8.7M

**关键发现**：
1. 分支编码器 + 跨分支 transformer 是有效架构，按物理语义拆分输入带来最大收益
2. 200K 合成数据是不可逾越的瓶颈（8.7M:200K = 43:1 参数/样本比）
3. 所有正则化方法（19 种尝试）均未突破此瓶颈
4. 验证损失在 5 分钟训练内持续下降，无过拟合拐点
5. Phase 2 需要更大规模、更真实的数据来突破 1.5 dB

---

## 进展汇总

| # | val_rmse | Δ vs best | Status | 关键变更 |
|---|----------|-----------|--------|---------|
| 0 | 2.085 dB | — | keep | baseline MLP 4×256 |
| 1 | 2.069 dB | -0.016 | keep | 残差MLP 6×512 + Huber |
| 2 | 2.044 dB | -0.025 | keep | 残差MLP 8×1024（暴力加宽） |
| 3 | 2.034 dB | -0.010 | keep | 分支编码器(geo+ssp+bathy) |
| **4** | **1.701 dB** | **-0.333** | **keep** | **跨分支注意力(2层×4头)** |
| 5 | 1.783 dB | +0.082 | discard | 物理导出特征 + dropout 0.05 |
| **6** | **1.680 dB** | **-0.021** | **keep** | **Transformer blocks(3层) + batch 4096 + grad clip** |
| 7 | 1.709 dB | +0.029 | discard | 加宽模型(branch 384, fusion 768) |
| 8 | 1.908 dB | +0.228 | discard | 低 LR 2e-3 + 长 warmdown 0.4 |
| 9 | 1.787 dB | +0.107 | discard | 强 weight_decay 5e-4 |
| 10 | 2.163 dB | +0.483 | discard | 高斯噪声 std=0.05 |
| 11 | 2.260 dB | +0.580 | discard | Mixup beta=0.4 |
| 12 | 1.784 dB | +0.104 | discard | 小模型(1.6M params) |
| 13 | 1.685 dB | +0.005 | discard | Conv1D SSP/bathy 编码器 |
| 14 | 1.901 dB | +0.221 | discard | 随机深度 drop_path 0→0.15 |
| 15 | 1.705 dB | +0.025 | discard | 5 注意力块 + 3 fusion + MSE |
| 16 | 1.929 dB | +0.249 | discard | 物理残差学习 |
| 17 | 1.706 dB | +0.026 | discard | Transformer + batch 2048 |
| 18 | 1.888 dB | +0.208 | discard | 多头预测集成(3头) |
| 19 | 1.811 dB | +0.131 | discard | EMA 权重(decay=0.999) |
| 20 | 1.731 dB | +0.051 | discard | LR 5e-3 + cosine warm restarts(3周期) |
| 21 | 1.707 dB | +0.027 | discard | Adam beta2=0.95 + LR 4e-3 |
| 22 | 1.707 dB | +0.027 | discard | 辅助分支损失(deep supervision) |
| 23 | 2.012 dB | +0.332 | discard | SwiGLU激活(参数+40%,更慢) |
| 24 | 1.784 dB | +0.104 | discard | 周期验证+最佳检查点(无过拟合拐点) |
| 25 | 6.223 dB | +4.543 | discard | 输入特征遮蔽 rate=0.15(破坏特征关系) |

**Phase 1 最终最佳**：1.680 dB（实验 6）| 相对 baseline：-0.405 dB（19.4%↓）
**Phase 1 结论**：25 个实验，6 次 keep，19 次 discard。数据瓶颈已确认，进入 Phase 2。

---

## 经验教训总结

### 有效的改进方向
1. **跨分支注意力**（exp4, -16.4%）：让特征组之间互相"看到"对方，是最大的单次改进
2. **分支编码器**（exp3）：按物理语义拆分输入比暴力加宽更高效
3. **Transformer blocks**（exp6）：attention + FFN 比纯 attention 更好
4. **梯度裁剪**（exp6）：稳定训练
5. **Huber loss**（exp1）：比 MSE 稍好

### 无效或有害的方向
1. **暴力加大模型**（exp2, exp7）：参数量翻倍 RMSE 几乎不变，收益严重递减
2. **各种正则化**（exp8-11, exp14）：weight_decay、dropout、噪声、mixup、stochastic depth 全部失败
3. **特征工程**（exp5, exp16）：在归一化后的数据上添加物理特征反而过拟合
4. **架构变体**（exp12, exp13, exp15, exp18）：Conv1D、多头、模型缩放都无显著改善
5. **优化器调参**（exp19-21）：EMA、warm restarts、beta2 调整都未突破瓶颈
6. **深度监督**（exp22）：辅助分支损失没有帮助，单分支信息不足
7. **更强激活函数**（exp23）：SwiGLU 参数+40%，加剧过拟合
8. **早停/检查点**（exp24）：无过拟合拐点，不需要早停
9. **输入遮蔽**（exp25）：破坏物理特征关系，灾难性结果

### 关键认识
- **200K 合成数据是当前瓶颈**：模型已经能很好地拟合训练集（~1.0 dB），但泛化到验证集只能到 ~1.68 dB
- **Phase 2 更大规模/更真实的数据**是突破 1.5 dB 的关键路径
- 当前架构（分支编码器 + 跨分支 transformer + fusion trunk）是一个良好的基础

---

## 里程碑

| 里程碑 | 目标 | 达成日期 | Commit |
|--------|------|---------|--------|
| Baseline | 任意有效值 | 2026-03-23 | 1252596 |
| val_rmse < 5 dB | 初步 | 2026-03-23 | 1252596 |
| val_rmse < 3 dB | 良好 | 2026-03-23 | 624297f |
| val_rmse < 2 dB | 目标 | 2026-03-23 | 7996430 |
| val_rmse < 1.5 dB | 优秀 | 待达成 | — |

---

## Phase 2 实验详细记录

### Phase 2 概述

**目标**：用 PyRAM 物理模型（抛物方程法）生成更真实的训练数据，突破 Phase 1 合成数据的 1.68 dB 瓶颈。

**数据生成**：
- PyRAM 计算全深度×距离 TL 网格，每次 run 从网格中采样多个点
- 参数约束：freq 20-2000 Hz，grid_budget < 5e8（保证单次 run < 2s）
- SSP 深度扩展到覆盖最大水深 + 1m（修复 pyram 深度校验错误）

**数据规模演进**：
| 阶段 | 训练样本 | 验证样本 | pyram runs | TL 范围 |
|------|---------|---------|-----------|---------|
| 初始 | 1,086,507 | 44,088 | 2,500 | 29.9~200.0 dB |
| 扩展1 | 2,176,377 | 44,088 | 5,000 | 29.9~200.0 dB |
| 扩展2 | ~4,400,000 | 44,088 | 10,000 | 29.9~200.0 dB |

---

### P2-Base: Phase 2 基线 — 1M pyram 数据

**配置**：Phase 1 exp6 架构（branch 256, fusion 512, 4 heads, 3 cross-branch, 6 fusion）直接在 1M pyram 数据上训练
**结果**：val_rmse=**10.314 dB** | 8,694K params | **keep (Phase 2 baseline)**
**分析**：pyram 数据远比解析合成数据复杂（Phase 1 在合成数据上 1.68 dB）。10.3 dB 距离工程精度（<5 dB）还有较大差距。

---

### P2-Exp1: 加宽模型 (384/768) — 1M 数据

**变更**：branch_dim 256→384, fusion_dim 512→768, 19.5M params
**结果**：val_rmse=**9.793 dB (-0.521)** | **keep**
**分析**：更大模型 + pyram 数据有效，改善 5%。说明 pyram 数据的复杂度能支撑更大模型。

---

### P2-Exp2: LR 5e-3 — (discarded)

**变更**：LR 3e-3→5e-3
**结果**：val_rmse=10.220 dB (+0.427) | **discard**
**教训**：LR 5e-3 对宽模型太激进，优化不稳定。

---

### P2-Exp3: 更深模型 — (discarded)

**变更**：branch_dim 256, fusion_dim 512, branch_layers 5, fusion_layers 10
**结果**：val_rmse=10.984 dB (+1.191) | **discard**
**教训**：深而窄远不如宽而浅。层数增加导致梯度传播困难，且步数更少。

---

### P2-Exp4: Batch 2048 — (discarded)

**变更**：batch 4096→2048
**结果**：val_rmse=10.139 dB (+0.346) | **discard**
**教训**：更小 batch 给更多步数但梯度更嘈杂，在大数据集上不如 batch 4096。

---

### P2-Exp5: 5 层 Cross-Branch — (discarded)

**变更**：cross-branch layers 3→5（22M params，仅 9.8K steps）
**结果**：val_rmse=10.043 dB (+0.250) | **discard**
**教训**：更多参数 = 每步更慢 = 5 分钟内步数不够。参数量和训练步数的权衡是 Phase 2 核心张力。

---

### P2-Exp6: Huber delta=10 — (discarded)

**变更**：Huber delta 5.0→10.0
**结果**：val_rmse=10.066 dB (+0.273) | **discard**
**教训**：delta=10 接近纯 MSE，失去 Huber 对大误差的鲁棒性。

---

### P2-Exp7: MSE Loss — (discarded)

**变更**：Huber loss → MSE loss
**结果**：val_rmse=9.910 dB (+0.117) | **discard**
**教训**：MSE 对 pyram 数据中的大 TL 值（200 dB 截断点附近）过于敏感。Huber delta=5.0 是好选择。

---

### P2-Exp8: 8 注意力头 — **新最佳**

**变更**：cross-branch attention heads 4→8
**结果**：val_rmse=**9.383 dB (-0.410, -4.4%)** | 19,530K params | 12,600 steps | **keep**
**分析**：更多注意力头 = 更精细的分支间交互模式。head_dim 从 96→48，但更多独立注意力模式弥补了维度的减小。pyram 数据中环境-几何交互比合成数据更复杂，需要更细粒度的注意力。

---

### P2-Exp9: LR 4e-3 — (discarded)

**变更**：LR 3e-3→4e-3（8 heads 模型上）
**结果**：val_rmse=9.560 dB (+0.177) | **discard**
**教训**：LR 4e-3 仍然偏高。3e-3 是当前架构的甜蜜点。

---

### P2-Exp10: 训练 10 分钟 — (discarded)

**变更**：TIME_BUDGET 300s→600s（25K steps vs 12.6K）
**结果**：val_rmse=10.217 dB (+0.834) | **discard**
**教训**：**1M 数据上训练太久会过拟合**。更多步数让模型更深地记忆训练集。这与 Phase 1 exp8 的教训一致。

---

### P2-Exp11: Batch 8192 — (discarded)

**变更**：batch 4096→8192
**结果**：val_rmse=9.690 dB (+0.307) | 6,613 steps | 3,006MB VRAM | **discard**
**教训**：步数减半（12.6K→6.6K），训练不充分。大 batch 需要更长训练时间才有优势。

---

### P2-Exp12: LR 2e-3 — (discarded)

**变更**：LR 3e-3→2e-3
**结果**：val_rmse=10.069 dB (+0.686) | 12,575 steps | **discard**
**教训**：LR 太低，模型收敛太慢。在 5 分钟预算下无法到达好的损失区域。

---

### P2-Exp13: 中等模型 (256/512) — (discarded)

**变更**：branch_dim 384→256, fusion_dim 768→512（8.7M params, 17.8K steps）
**结果**：val_rmse=10.115 dB (+0.732) | **discard**
**教训**：尽管步数更多，中等模型的容量不足以捕获 pyram 数据的复杂度。384/768 模型的表达能力优势大于步数劣势。

---

### P2-Exp14: 4 层 Cross-Branch — (discarded)

**变更**：cross-branch layers 3→4（20.7M params, 10.8K steps）
**结果**：val_rmse=9.636 dB (+0.253) | **discard**
**教训**：额外的 cross-branch 层增加了 1.2M 参数并减少了 1.8K 步，净效果为负。3 层已足够。

---

### P2-Exp15: Huber delta=2.0 — (discarded)

**变更**：Huber delta 5.0→2.0
**结果**：val_rmse=10.061 dB (+0.678) | **discard**
**教训**：delta=2.0 把太多误差处理为"异常值"（线性惩罚），梯度信号变弱，学习效率下降。

---

### P2-Exp16: 无 Weight Decay — (discarded)

**变更**：weight_decay 1e-4→0
**结果**：val_rmse=9.685 dB (+0.302) | **discard**
**教训**：weight_decay=1e-4 的正则化是有帮助的。即使在欠拟合状态，适度正则化也能改善泛化。

---

### P2-Exp17: 无 LR Warmdown — (discarded)

**变更**：warmdown_ratio 0.3→0.0, final_lr_frac 0.01→1.0（恒定 LR）
**结果**：val_rmse=10.187 dB (+0.804) | **discard**
**教训**：LR warmdown 对最终收敛至关重要。在训练末期降低 LR 让模型精细调整权重。

---

### P2-Exp18: 2.2M 数据 — **数据扩展突破**

**变更**：训练数据从 1M→2.2M（5000 pyram runs × 500 samples/run）
**结果**：val_rmse=**8.115 dB (-1.268, -13.5%)** | **keep**
**分析**：数据翻倍带来 1.27 dB 的巨大改善，而之前 9 个超参/架构实验累计只改善了 0.4 dB。**确认数据量是 Phase 2 的主要瓶颈**。数据/参数比从 ~55:1 提升到 ~112:1。

---

### P2-Exp19: 更宽模型 on 2.2M — (discarded)

**变更**：branch 384→512, fusion 768→1024（34.7M params, 8.2K steps）
**结果**：val_rmse=8.288 dB (+0.173) | **discard**
**教训**：即使有 2.2M 数据，更宽模型在 5 分钟内仍然步数不足。时间预算是宽模型的天花板。

---

### P2-Exp20: LR 3.5e-3 + beta2=0.95 — (discarded)

**变更**：LR 3e-3→3.5e-3, Adam beta2 0.999→0.95
**结果**：val_rmse=8.413 dB (+0.298) | **discard**
**教训**：在 2.2M 数据上，更高 LR 和更短视的 Adam 二阶矩都没有帮助。

---

### P2-Exp21: Geo 跳跃连接 — (discarded)

**变更**：原始 geo 特征跳跃连接到 fusion trunk 输入
**结果**：val_rmse=8.191 dB (+0.076) | **discard**
**教训**：仅边际差距，cross-branch attention 已经充分提取了 geo 信息。额外的跳跃连接是冗余的。

---

### P2-Exp22: FiLM 调制 — (discarded)

**变更**：geo 编码器输出通过 FiLM（Feature-wise Linear Modulation）调制 SSP 和 bathy 分支
**结果**：val_rmse=9.549 dB (+1.434) | **discard**
**教训**：严重退步。FiLM 在 cross-branch attention 之前应用，可能干扰了后续注意力层的学习。geo→SSP/bathy 的单向调制不如对称的 cross-attention 有效。

---

### P2-Exp23: 4.4M 数据 — **继续扩展**

**变更**：训练数据从 2.2M→4.4M（10000 pyram runs × 500 samples/run）
**结果**：val_rmse=**7.677 dB (-0.438, -5.4%)** | **keep**
**分析**：数据继续扩展有效但边际递减：
- 1M→2.2M: -1.27 dB（每翻倍 -1.27）
- 2.2M→4.4M: -0.44 dB（每翻倍 -0.44）
数据/参数比提升到 ~225:1，但提升幅度已明显缩小。

---

### P2-Exp24: 更宽模型 on 4.4M — (discarded)

**变更**：branch 384→512, fusion 768→1024（34.7M params, 8.6K steps）
**结果**：val_rmse=7.944 dB (+0.267) | **discard**
**教训**：4.4M 数据仍无法弥补宽模型的步数劣势。在 5 分钟固定预算下，384/768 (19.5M params) 是最优模型规模。

---

### P2-Exp25: 600s 训练 on 4.4M — **边际改善**

**变更**：TIME_BUDGET 300s→600s，保持 384/768 架构
**结果**：val_rmse=**7.658 dB (-0.019, -0.2%)** | **keep**
**分析**：延长训练时间带来微小改善。384/768 在 600s 内跑了 ~26K 步，模型已接近收敛。

---

### P2-Exp26: 更宽模型 + 600s on 4.4M — **新最佳**

**变更**：branch 384→512, fusion 768→1024 + TIME_BUDGET=600s（34.7M params, ~17K steps）
**结果**：val_rmse=**7.425 dB (-0.233, -3.0%)** | **keep**
**分析**：关键发现！之前 512/1024 在 300s 只有 8.6K 步所以输给 384/768，但 600s 给了 17K 步，足以发挥大模型优势。这改变了最优模型规模的结论。

---

### P2-Exp27: 16 注意力头 — (discarded)

**变更**：cross-branch attention 8→16 heads（head_dim=32）
**结果**：val_rmse=7.535 dB (+0.110) | **discard**
**教训**：head_dim=32 太小，注意力表达力不足。8 heads (head_dim=64) 是更优配置。

---

### P2-Exp28: 8 层 Fusion — (discarded)

**变更**：n_fusion_layers 6→8（43M params, 14.9K steps）
**结果**：val_rmse=7.507 dB (+0.082) | **discard**
**教训**：更多参数但更少步数。43M params 在 600s 内步数不足。

---

### P2-Exp29: LR 2e-3 — (discarded)

**变更**：LEARNING_RATE 3e-3→2e-3
**结果**：val_rmse=7.507 dB (+0.082) | **discard**
**教训**：LR 偏低导致欠拟合，3e-3 仍是最优。

---

### P2-Exp30: Warmdown 0.4 — (discarded)

**变更**：WARMDOWN_RATIO 0.3→0.4（更长冷却期）
**结果**：val_rmse=7.462 dB (+0.037) | **discard**
**教训**：略微更差，0.3 的冷却比例已最优。

---

### P2-Exp31: Dropout=0 — (discarded)

**变更**：DROPOUT 0.02→0（无正则化）
**结果**：val_rmse=7.744 dB (+0.319) | **discard**
**教训**：即使小到 0.02 的 dropout 也有显著正则化效果。

---

### P2-Exp32: 900s 训练 on 4.4M — (discarded)

**变更**：TIME_BUDGET 600s→900s
**结果**：val_rmse=7.586 dB (+0.161) | **discard**
**分析**：训练损失 3.41 vs 验证 7.59，明显过拟合。4.4M 数据不足以支撑 900s 训练。

---

### P2-Exp33: 9M 数据 — **数据扩展再次突破**

**变更**：训练数据 4.4M→~9M（20K pyram runs），保持 512/1024 + 600s
**结果**：val_rmse=**7.176 dB (-0.249, -3.4%)** | **keep**
**分析**：数据扩展继续有效！
- 1M→2.2M: -1.27 dB/倍
- 2.2M→4.4M: -0.44 dB/倍
- 4.4M→9M: -0.25 dB/倍
边际递减明显但仍是最可靠的改进路径。训练 ~18K 步，13 个 epoch。

---

### P2-Exp34: 900s on 9M — (discarded)

**变更**：TIME_BUDGET 600s→900s on 9M 数据
**结果**：val_rmse=7.281 dB (+0.105) | **discard**
**教训**：即使 9M 数据也撑不住 900s 训练。600s 仍是最优时间预算。

---

### P2-Exp35: 384/768 on 9M — (discarded)

**变更**：branch 512→384, fusion 1024→768 on 9M 数据（19.5M params, 26.5K steps）
**结果**：val_rmse=7.297 dB (+0.121) | **discard**
**教训**：虽然步数更多（26.5K vs 18K），但模型容量不足以充分利用 9M 数据。数据量越大，越需要大模型。

---

### P2-Exp36: Batch 2048 on 9M — (discarded)

**变更**：BATCH_SIZE 4096→2048 on 9M 数据（31.9K steps, 65.3M total samples）
**结果**：val_rmse=7.420 dB (+0.244) | **discard**
**教训**：虽然步数多，但每步只看 2048 样本，总共看的样本量反而更少（65M vs 108M）。大 batch 在固定时间内更高效。

---

### P2-Exp37: LR 4e-3 on 9M — (discarded)

**变更**：LEARNING_RATE 3e-3→4e-3 on 9M 数据
**结果**：val_rmse=7.614 dB (+0.438) | **discard**
**教训**：LR 4e-3 对 9M 数据也太高，显著退步。LR 3e-3 在任何数据量级都是最优。

---

### P2-Exp38: AMP bfloat16 — (discarded)

**变更**：启用 bfloat16 混合精度训练
**结果**：val_rmse=9.053 dB (+1.877) | **discard**
**教训**：bf16 精度损失导致严重泛化退化。回归任务对数值精度敏感，fp32 必须。

---

### P2-Exp39: 4 层 Branch Encoder — (discarded)

**变更**：n_branch_layers 3→4（35.5M params, 16.5K steps）
**结果**：val_rmse=7.962 dB (+0.786) | **discard**
**教训**：更深分支增加参数但减少步数，净效果为负。

---

### P2-Exp40: 4 Cross-Branch + 4 Fusion — (discarded)

**变更**：cross-branch 3→4, fusion 6→4（28.4M params, 16.7K steps）
**结果**：val_rmse=7.353 dB (+0.177) | **discard**
**教训**：更多交互层无法弥补 fusion 深度的减少。

---

### P2-Exp41: 自适应 Huber Delta — (discarded)

**变更**：Huber delta 从 10.0 退火到 3.0（初始像 MSE，后期更鲁棒）
**结果**：val_rmse=7.480 dB (+0.304) | **discard**
**教训**：初始大 delta 导致 loss 先升后降，浪费训练早期的宝贵步数。

---

### P2-Exp42: Weight Decay 5e-5 — **新最佳**

**变更**：weight_decay 1e-4→5e-5（更轻正则化）
**结果**：val_rmse=**7.174 dB (-0.002)** | **keep**
**分析**：9M 数据量下，轻正则化微微有益。改善极小但稳定。

---

### P2-Exp43: MoE 4 Expert (小) — (discarded)

**变更**：Mixture of Experts 架构：4 专家 Top-2, 384/512×4层, 23.7M params
**结果**：val_rmse=7.301 dB (+0.127) | **discard**
**教训**：小专家容量不足，路由开销抵消了专业化优势。

---

### P2-Exp44: MoE 4 Expert (大) — (discarded)

**变更**：MoE 架构放大：512/768×4层, 50.5M params, 11.7K steps
**结果**：val_rmse=7.182 dB (+0.008) | **discard**
**分析**：接近最佳但太重（50.5M 只跑 11.7K 步）。MoE 在固定时间预算下不划算，除非能大幅加速推理。

---

## Phase 2 进展汇总

| # | val_rmse | Δ vs best | 数据量 | Status | 关键变更 |
|---|----------|-----------|--------|--------|---------|
| base | 10.314 dB | — | 1M | keep | Phase 1 exp6 架构直接迁移 |
| 1 | 9.793 dB | -0.521 | 1M | keep | 加宽 384/768 |
| 2 | 10.220 dB | +0.427 | 1M | discard | LR 5e-3 太激进 |
| 3 | 10.984 dB | +1.191 | 1M | discard | 深而窄模型 |
| 4 | 10.139 dB | +0.346 | 1M | discard | batch 2048 |
| 5 | 10.043 dB | +0.250 | 1M | discard | 5 层 cross-branch |
| 6 | 10.066 dB | +0.273 | 1M | discard | Huber delta=10 |
| 7 | 9.910 dB | +0.117 | 1M | discard | MSE loss |
| **8** | **9.383 dB** | **-0.410** | **1M** | **keep** | **8 注意力头** |
| 9 | 9.560 dB | +0.177 | 1M | discard | LR 4e-3 |
| 10 | 10.217 dB | +0.834 | 1M | discard | 600s 训练（过拟合） |
| 11 | 9.690 dB | +0.307 | 1M | discard | batch 8192（步数不足） |
| 12 | 10.069 dB | +0.686 | 1M | discard | LR 2e-3（欠拟合） |
| 13 | 10.115 dB | +0.732 | 1M | discard | 中模型 256/512 |
| 14 | 9.636 dB | +0.253 | 1M | discard | 4 层 cross-branch |
| 15 | 10.061 dB | +0.678 | 1M | discard | Huber delta=2.0 |
| 16 | 9.685 dB | +0.302 | 1M | discard | weight_decay=0 |
| 17 | 10.187 dB | +0.804 | 1M | discard | 无 LR warmdown |
| **18** | **8.115 dB** | **-1.268** | **2.2M** | **keep** | **数据翻倍** |
| 19 | 8.288 dB | +0.173 | 2.2M | discard | 宽模型 512/1024 |
| 20 | 8.413 dB | +0.298 | 2.2M | discard | LR 3.5e-3 + beta2=0.95 |
| 21 | 8.191 dB | +0.076 | 2.2M | discard | geo 跳跃连接 |
| 22 | 9.549 dB | +1.434 | 2.2M | discard | FiLM 调制 |
| **23** | **7.677 dB** | **-0.438** | **4.4M** | **keep** | **数据再翻倍** |
| 24 | 7.944 dB | +0.267 | 4.4M | discard | 宽模型 512/1024 (300s) |
| **25** | **7.658 dB** | **-0.019** | **4.4M** | **keep** | **600s 训练** |
| **26** | **7.425 dB** | **-0.233** | **4.4M** | **keep** | **512/1024 + 600s** |
| 27 | 7.535 dB | +0.110 | 4.4M | discard | 16 注意力头 |
| 28 | 7.507 dB | +0.082 | 4.4M | discard | 8 层 fusion |
| 29 | 7.507 dB | +0.082 | 4.4M | discard | LR 2e-3 |
| 30 | 7.462 dB | +0.037 | 4.4M | discard | warmdown 0.4 |
| 31 | 7.744 dB | +0.319 | 4.4M | discard | dropout=0 |
| 32 | 7.586 dB | +0.161 | 4.4M | discard | 900s（过拟合） |
| **33** | **7.176 dB** | **-0.249** | **~9M** | **keep** | **数据再翻倍** |
| 34 | 7.281 dB | +0.105 | ~9M | discard | 900s 训练 |
| 35 | 7.297 dB | +0.121 | ~9M | discard | 384/768 小模型 |
| 36 | 7.420 dB | +0.244 | ~9M | discard | batch 2048 |
| 37 | 7.614 dB | +0.438 | ~9M | discard | LR 4e-3 |
| 38 | 9.053 dB | +1.877 | ~9M | discard | AMP bfloat16（精度损失） |
| 39 | 7.962 dB | +0.786 | ~9M | discard | 4 层 branch encoder |
| 40 | 7.353 dB | +0.177 | ~9M | discard | 4 cross-branch + 4 fusion |
| 41 | 7.480 dB | +0.304 | ~9M | discard | 自适应 Huber delta |
| **42** | **7.174 dB** | **-0.002** | **~9M** | **keep** | **weight_decay 5e-5** |
| 43 | 7.301 dB | +0.127 | ~9M | discard | MoE 4专家（小） |
| 44 | 7.182 dB | +0.008 | ~9M | discard | MoE 4专家（大） |

**Phase 2 旧数据最佳**：7.174 dB（P2-Exp42）| ~9M pyram 数据 | 512/1024 + 600s + WD=5e-5

---

### P2-Exp45~48: 16M CZ 数据系列（新验证集）

**背景**：P2-Exp42 之前的数据最大距离约 100km，缺少深海会聚区（CZ，每 ~55km 一个周期）覆盖。
扩展 prepare.py 添加 `deep_longrange` 体制（20-50Hz 低频，1-3km 深海，最远 300km），
生成 40K pyram runs → 16M 训练样本 + 新验证集（含 CZ 场景）。
这使得问题显著变难，但评估更真实。

| 实验 | val_rmse | vs 前一个 | 数据 | 状态 | 描述 |
|------|----------|----------|------|------|------|
| 45 | 9.009 dB | 基线 | 16M CZ | keep | 新数据基线（512/1024 + 600s） |
| 46 | 8.574 dB | -0.435 | 16M CZ | keep | 900s 训练（7 epochs，无过拟合） |
| 47 | 8.493 dB | -0.081 | 16M CZ | keep | 1200s 训练（9 epochs，仍在改善） |
| **48** | **8.425 dB** | **-0.068** | **16M CZ** | **keep** | **1800s 训练（13 epochs，泛化差距 4.8dB）** |
| 49 | 训练中... | ? | 16M CZ | running | 更大模型 768/1536（1800s） |

#### P2-Exp45: 16M CZ 数据基线 — 512/1024 + 600s

**变更**：使用新生成的 16M CZ 数据和新验证集
**配置**：512/1024, 3 branch + 6 fusion, 8 heads, batch 4096, LR 3e-3, 600s
**结果**：val_rmse=9.009 dB | 34.7M params | 2.3GB | ~16K steps
**分析**：新数据包含 300km 远距离和 CZ 效应，问题比之前（7.17 dB）显著更难。
验证集也重新生成，包含 CZ 场景，因此与旧结果不可直接比较。

#### P2-Exp46: 900s 训练

**变更**：TIME_BUDGET 600→900s
**结果**：val_rmse=8.574 dB (-0.435) | 7 epochs | 无过拟合迹象
**分析**：与旧 9M 数据不同（900s 就过拟合），16M 数据量足以支撑更长训练。
这意味着数据量终于"够了"，瓶颈从数据不足转向了模型能力。

#### P2-Exp47: 1200s 训练

**变更**：TIME_BUDGET 900→1200s
**结果**：val_rmse=8.493 dB (-0.081) | 9 epochs | 仍在改善但回报递减
**分析**：每增加 300s 只换来 0.08 dB，接近 512/1024 模型在 16M 数据上的拟合极限。

#### P2-Exp48: 1800s 训练（当前 CZ 数据最佳）

**变更**：TIME_BUDGET 1200→1800s
**结果**：val_rmse=8.425 dB (-0.068) | 13 epochs | train 3.60 vs val 8.42
**分析**：泛化差距 4.8 dB 很大，说明模型可能容量不足以记住 CZ 的复杂干涉模式。
训练损失已经很低（3.6 dB RMSE），但无法泛化到新场景。
这motivate了 P2-Exp49：尝试更大模型（768/1536）。

#### P2-Exp49: 更大模型 768/1536（进行中）

**变更**：BRANCH_DIM 512→768, FUSION_DIM 1024→1536, TIME_BUDGET=1800s
**预估参数**：~117M（比 34.7M 增大 3.4 倍）
**进度**：57% (step ~13500)，训练 RMSE ~4.6 dB
**假设**：更大模型能缩小 4.8 dB 的泛化差距

**Phase 2 CZ 数据当前最佳**：8.425 dB（P2-Exp48）| 16M CZ 数据 | 512/1024 + 1800s

---

## Phase 2 关键发现

1. **数据量是最大杠杆**：1M→2.2M→4.4M→9M→16M 持续带来改善，远超任何架构/超参调整
2. **时间预算解锁模型规模**：600s 预算让 512/1024 (34.7M params) 有足够步数，超越 384/768
3. **最优配置**：512/1024 + 8 heads + 3 cross-branch + 6 fusion + batch 4096
4. **Huber loss (delta=5.0) 经验证最优**：MSE 和不同 delta 值都更差
5. **LR 3e-3 是甜蜜点**：4e-3 太高（在 1M/4.4M/9M 数据上都一致），2e-3 欠拟合
6. **正则化配置已平衡**：dropout=0.02 + weight_decay=5e-5（数据量大时可减轻 WD）
7. **16M 数据终结了过拟合问题**：900s/1200s/1800s 训练都无过拟合（之前 9M 数据 900s 就过拟合）
8. **CZ 数据使问题变难但更真实**：扩展到 300km 后 val_rmse 从 7.17 跳到 9.01（新验证集）
9. **大泛化差距（4.8 dB）**：train 3.6 vs val 8.4，提示模型容量可能不足
10. **MoE 架构有潜力但受时间预算限制**：50.5M MoE 几乎持平最佳，但步数太少
11. **bf16 不适合此任务**：回归精度对数值精度敏感，必须 fp32

---

## 数据扩展曲线（Phase 2）

### 旧验证集（≤100km 距离）
| 数据量 | val_rmse | 模型 | 数据/参数比 | 每翻倍改善 |
|--------|----------|------|-----------|-----------|
| 1M | 9.383 dB | 384/768 (19.5M) | 55:1 | — |
| 2.2M | 8.115 dB | 384/768 (19.5M) | 112:1 | -1.27 dB |
| 4.4M | 7.425 dB | 512/1024 (34.7M) | 127:1 | -0.69 dB |
| ~9M | 7.176 dB | 512/1024 (34.7M) | 259:1 | -0.25 dB |

### 新验证集（含 CZ，≤300km 距离）
| 数据量 | val_rmse | 模型 | 训练时间 | 备注 |
|--------|----------|------|---------|------|
| 16M | 9.009 dB | 512/1024 (34.7M) | 600s | CZ 基线 |
| 16M | 8.574 dB | 512/1024 (34.7M) | 900s | 更长训练 |
| 16M | 8.493 dB | 512/1024 (34.7M) | 1200s | 边际递减 |
| 16M | 8.425 dB | 512/1024 (34.7M) | 1800s | 当前最佳 |
| 16M | 9.031 dB | 768/1536 (78M) | 1800s | 过大，步数不足 |

注意：新旧验证集不可直接比较（新集包含 CZ 远距离场景，问题更难）。

### 架构探索（CZ 数据）

#### P2-Exp49: 更大模型 768/1536
- **配置**: BRANCH_DIM=768, FUSION_DIM=1536, 78M参数, 1800s
- **结果**: val_rmse=9.031 dB (**discard**)
- **分析**: 模型太大(78M vs 34.7M)，相同时间内步数不足(23.5K vs 55K)，反而更差

#### P2-Exp50: 中等模型 640/1280
- **配置**: BRANCH_DIM=640, FUSION_DIM=1280, ~55M参数, 1800s
- **结果**: GPU崩溃，未完成
- **分析**: VRAM不足导致训练中断

### 决策 008：经验公式 + 残差学习
**日期**：2026-03-26 | 引入物理经验公式作为基线，NN只学习残差修正。

经验公式包含6个物理效应：
1. 几何扩展（球面→柱面过渡，基于水深）
2. Thorp频率吸收
3. Lloyd镜像效应（浅源干涉）
4. 海底反射损失（基于声阻抗）
5. 声道效应（SOFAR深海声道检测）
6. 表面波导效应（正梯度SSP）

最终预测 = 经验TL + NN残差

#### P2-Exp52: 经验公式(6效应) + NN残差, 512/1024
- **配置**: 完整环境感知经验公式 + NN残差, BRANCH_DIM=512, FUSION_DIM=1024, 34.7M参数, 1800s
- **状态**: 训练中 (train ~rmse 3.78 dB @ 41%)
- **初步观察**: 收敛快速，经验公式提供了好的物理基线

### 决策 009：引入全球真实海洋数据
**日期**：2026-03-26 | 从随机SSP转向WOA23真实温盐数据 + GEBCO地形。

- WOA23: 全球0.25°网格, 102层深度, 温度+盐度 → Mackenzie公式推导SSP
- 覆盖24个典型海域（深海、热带、极地、浅海、海峡）
- 后续: 加入Bellhop(射线追踪) + KRAKEN(简正波) 多模型数据

---

## 物理误差参考

| 误差 | 含义 | ASW 影响 |
|------|------|---------|
| > 10 dB | 不可用 | 声呐方程误差极大 |
| 5~10 dB | 粗略估计 | 探测距离误差 2~3 倍 |
| 2~5 dB | 工程精度 | 探测距离误差 1.3~1.8 倍 |
| 1~2 dB | 良好精度 | 可替代低精度物理模型 |
| < 1 dB | 优秀 | 接近 RAM/KRAKEN 精度 |
