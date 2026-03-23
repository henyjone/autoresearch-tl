# 数据设计

## 1. 64 维特征向量完整设计

所有样本使用统一的 64 维浮点特征向量（`MAX_FEATURES = 64`），布局固定，由 `prepare.py` 定义，**不得更改**。

```
索引      字段名                  单位        物理含义
----------------------------------------------------------------------
[0]       frequency_hz           Hz          声源频率（对数均匀采样：10 Hz ~ 10 kHz）
[1]       source_depth_m         m           声源深度
[2]       receiver_depth_m       m           接收器深度
[3]       range_km               km          声源到接收器水平距离
[4]       water_depth_source_m   m           声源位置处水深
[5]       water_depth_receiver_m m           接收器位置处水深
[6]       bottom_soundspeed_mps  m/s         海底沉积层声速
[7]       bottom_density_gcc     g/cm³       海底沉积层密度
[8:28]    ssp_values[0..19]      m/s         20个标准深度处的声速值
[28:48]   bathymetry_profile[0..19] m        沿传播路径20个等距点的水深
[48:64]   reserved               —           保留（当前填零；供特征工程使用）
```

### 常量定义（prepare.py）

```python
IDX_FREQ           = 0
IDX_SRC_DEPTH      = 1
IDX_RCV_DEPTH      = 2
IDX_RANGE          = 3
IDX_WATER_DEPTH_SRC = 4
IDX_WATER_DEPTH_RCV = 5
IDX_BOTTOM_SS      = 6
IDX_BOTTOM_DENSITY = 7
IDX_SSP_START      = 8
IDX_SSP_END        = 28   # 不含
IDX_BATHY_START    = 28
IDX_BATHY_END      = 48   # 不含
IDX_RESERVED_START = 48
IDX_RESERVED_END   = 64   # 不含
```

---

## 2. SSP 采样策略：20 个标准深度

声速剖面（Sound Speed Profile，SSP）在以下 20 个标准深度上采样，覆盖从海面到深海全深度范围：

```python
SSP_DEPTHS = np.array([
    0,    25,   50,   75,   100,  150,  200,  300,  500,  700,
    1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500
], dtype=np.float64)  # 单位：米
```

**设计原则**：

- 浅层（0 ~ 300 m）密集采样：声速变化最剧烈，温跃层、混合层均在此范围
- 中层（300 ~ 2000 m）中等间距：SOFAR 声道轴一般位于 700 ~ 1300 m
- 深层（2000 ~ 5500 m）稀疏采样：声速随压力线性增加，变化平缓

若水深小于某标准深度，该深度及以下的 SSP 值填充为最后有效深度的声速值。

---

## 3. 地形剖面编码：20 个等距点

沿声源到接收器的传播路径，等距取 20 个点记录水深（米），存储于特征向量 `[28:48]`。

```python
t = np.linspace(0, 1, 20)  # 20 个归一化位置
```

这 20 个点捕捉路径上的水深变化，是底部反射次数估算的关键输入。`t=0` 对应声源位置，`t=1` 对应接收器位置。

---

## 4. 五种 SSP 剖面类型

Phase 1 数据按以下概率混合生成（`rng.choice` 加权采样）：

### 4.1 等温层（isovelocity，15%）

```
声速约 1490 m/s，全深度几乎恒定，加小幅高斯噪声
典型场景：极深海、冬季高纬度海区
```

声速范围：1480 ~ 1510 m/s

### 4.2 温跃层（thermocline，30%）

```
表层混合层（高声速）→ 急剧下降（温跃层）→ 深层等温
温跃层深度：50 ~ 200 m，过渡宽度：50 ~ 150 m
典型场景：夏季中低纬度海区
```

关键参数：
- 表层声速：~1520 m/s
- 深层声速：~1490 m/s
- 温跃层以下：压力效应使声速以 0.017 m/s/m 缓慢增加

### 4.3 深海声道（deep_channel，25%）

```
SOFAR 声道（Sound Fixing and Ranging Channel）
声速极小值位于 700 ~ 1300 m 深度
典型场景：大西洋、太平洋深海
```

SOFAR 声道对长距离传播影响最大：声源和接收器均在声道轴附近时，TL 显著降低（声线被声道捕获，减少底部/表面散射）。

### 4.4 北极剖面（arctic，15%）

```
表层声速最低（冰冷海水），随深度单调增加
典型场景：北冰洋、高纬度海区
```

表层声速：~1440 m/s，梯度：+0.016 m/s/m

### 4.5 浅海剖面（shallow，15%）

```
线性梯度，正负随机（模拟浅水区季节变化）
典型场景：大陆架、近岸浅水区
```

表层声速：~1510 m/s，梯度：-0.05 ~ +0.05 m/s/m

---

## 5. 五种地形类型

地形类型决定海底起伏特征，按以下概率混合：

### 5.1 平坦海底（flat，25%）

```
全程水深近似恒定（加 2% 小幅噪声）
典型场景：深海平原、中央海盆
```

### 5.2 斜坡（slope，25%）

```
水深从声源到接收器线性变化，斜率由 water_depth_src 和 water_depth_rcv 决定
典型场景：大陆坡、海底滑坡区
```

### 5.3 陆架断裂（shelf_break，20%）

```
浅陆架（100 ~ 300 m）→ 陡降断裂带 → 深海（2000 ~ 4000 m）
断裂位置：路径 20% ~ 50% 处，宽度 5% ~ 15%
典型场景：大陆架边缘，ASW 的重要战术区域
```

### 5.4 海山（seamount，15%）

```
路径中段有一处隆起（海底山），高度为水深的 30% ~ 70%
典型场景：中大西洋海岭、太平洋火山链
```

### 5.5 粗糙底（rough，15%）

```
随机游走叠加线性趋势，模拟不规则海底起伏
典型场景：断裂带、地质活动区
```

---

## 6. 五种海底类型：Hamilton-Bachman 参数

海底沉积物声学参数采用 Hamilton-Bachman 经验值（水下声学标准参考），加高斯噪声扰动：

```python
BOTTOM_TYPES = {
    "sand":   {"soundspeed": 1650.0, "density": 1.9},  # 砂
    "silt":   {"soundspeed": 1550.0, "density": 1.7},  # 淤泥
    "clay":   {"soundspeed": 1500.0, "density": 1.5},  # 黏土
    "gravel": {"soundspeed": 1800.0, "density": 2.0},  # 砾石
    "rock":   {"soundspeed": 2500.0, "density": 2.5},  # 岩石
}
# 加扰动：声速 ±20 m/s，密度 ±0.1 g/cm³（高斯）
```

| 海底类型 | 声速（m/s） | 密度（g/cm³） | 声学特性 |
|----------|------------|--------------|----------|
| 砂（sand） | 1650 | 1.9 | 中等反射，ASW 最常见底质 |
| 淤泥（silt） | 1550 | 1.7 | 较软，反射损耗较大 |
| 黏土（clay） | 1500 | 1.5 | 最软，声速接近水中，临界角大 |
| 砾石（gravel） | 1800 | 2.0 | 较硬，反射损耗较小 |
| 岩石（rock） | 2500 | 2.5 | 最硬，近乎全反射 |

**临界角**（Critical Angle）判据：当海底声速 > 水中声速时，存在临界角 θ_c = arcsin(c_water / c_bottom)。掠射角小于 θ_c 时底部几乎全反射（损耗小）；大于 θ_c 时损耗快速增加。

---

## 7. TL 解析计算公式说明

`compute_tl_analytical()` 将以下 6 项物理效应叠加：

### 7.1 几何扩展损耗（Geometric Spreading）

```
近场（R ≤ 水深）：球面扩展  TL_spread = 20·log10(R)
远场（R > 水深）：向柱面扩展 TL_spread = 20·log10(H) + 10·log10(R/H)
其中 H = (水深_src + 水深_rcv) / 2
```

这一过渡捕捉了浅水波导从球面扩展向柱面扩展的物理转变。

### 7.2 Thorp 吸收（Thorp Absorption）

```
α = 0.11·f² / (1 + f²) + 44·f² / (4100 + f²) + 2.75×10⁻⁴·f² + 0.003   [dB/km]
（f 单位：kHz）
TL_abs = α · R_km
```

Thorp 公式适用于 100 Hz ~ 10 kHz，低频时取低频近似。高频时吸收损耗主导 TL。

### 7.3 Lloyd 镜面效应（Lloyd Mirror）

```
当声源深度 < 100 m 且 R > 100 m 时：
路径差 Δ = 2·z_src·z_rcv / R
若 Δ < λ（波长）：处于干涉暗区，额外损耗 ≤ 6 dB
```

该效应描述直达波与海面反射波之间的相消干涉，对浅源/浅接收器影响显著。

### 7.4 海底反射损耗（Bottom Reflection Loss）

```
掠射角 θ = arctan(max(z_src, z_rcv) / R)
底部弹跳次数 N ≈ R / (2H / tan(θ))，上限 2·R_km 次
若 θ < θ_c：每次弹跳损耗 ~0.5 ~ 1 dB
若 θ > θ_c：损耗 = 2 + 3·(1 - 1/Z_ratio)  dB/次，上限 8 dB/次
总底部损耗 = N · 损耗/次，上限 30 dB
```

### 7.5 声道增益（Sound Channel Ducting Bonus）

```
若 SSP 存在极小值（声道轴）且声源与接收器均在声道内：
增益 ≈ strength · 5·log10(R/H)  （减少扩展损耗）
strength = min(1.0, Δc/30)，Δc 为 SSP 变化幅度
上限：15 dB
```

SOFAR 声道效应使声线被捕获，等效扩展维度介于球面（3D）和柱面（2D）之间。

### 7.6 表面波导效应（Surface Duct）

```
当 SSP 在浅层具有正梯度（声速随深度增加）且声源/接收器均 < 50 m 时：
波导增益 = min(5, 梯度×100) · log10(R_km)
上限：10 dB
```

### 7.7 合并公式

```
TL = TL_spread + TL_abs + TL_lloyd + TL_bottom - TL_duct - TL_surface_duct
   + N(0, 1.5)  [随机扰动，模拟未建模效应]

物理约束：TL ∈ [10, 200] dB
```

---

## 8. 数据归一化策略：FeatureStats 类

```python
@dataclass
class FeatureStats:
    mean: torch.Tensor    # shape (64,) — 训练集各特征均值
    std:  torch.Tensor    # shape (64,) — 训练集各特征标准差
    target_mean: float    # 训练集 TL 均值（dB）
    target_std:  float    # 训练集 TL 标准差（dB）
```

**归一化规则**：

- 输入特征：`x_norm = (x - mean) / std`（零均值、单位方差）
- 对于标准差 < 1e-8 的常量特征（如保留位），`std` 设为 1.0 避免除零
- **目标值（TL）不归一化**：模型直接预测原始 dB 值，避免解释混乱

**注意**：归一化统计量仅从训练集计算，验证集和测试集使用相同的统计量（防止数据泄漏）。

```python
# 加载统计量
stats = FeatureStats.from_data()

# 归一化特征
x_norm = stats.normalize(x)  # x: (..., 64)

# 反归一化目标（如需要）
tl_db = stats.denormalize_target(y_norm)
```

---

## 9. Phase 1 → 2 → 3 数据演进计划

| 阶段 | 数据来源 | 样本量 | 精度 | 优点 | 局限 |
|------|---------|--------|------|------|------|
| **Phase 1** | 解析公式（prepare.py） | 20万 + 5万 | 低（近似物理） | 秒级生成，可无限扩展 | 不含完整波动效应 |
| **Phase 2** | pyram（Python RAM） | 待定（~10万+） | 中高（完整抛物方程） | 物理精确，含多路径干涉 | 每样本需秒级计算 |
| **Phase 3** | 真实海试数据 | 受限（<1万） | 最高（真实环境） | 泛化到真实条件 | 数据稀少，环境参数不完整 |

**迁移策略**：

```
Phase 1 预训练 → Phase 2 微调（冻结底层特征提取，仅更新顶层）
                → Phase 3 少样本微调（结合数据增强）
```

Phase 2 迁移时需注意：pyram 输出可能在干涉极小值处有尖锐特征，需要调整损失函数（Huber loss 比 MSE 更鲁棒）。

---

## 10. 数据文件存储

```
~/.cache/autoresearch-tl/data/
├── train.npz   # 训练集：features (200000, 64) float32 + targets (200000,) float32
├── val.npz     # 验证集：features (50000, 64) float32 + targets (50000,) float32
└── stats.npz   # 归一化统计：mean (64,), std (64,), target_mean, target_std
```

数据以 `float32` 存储（精度足够，节省内存），加载后在 CPU 归一化，通过 pinned memory 异步传输到 GPU。
