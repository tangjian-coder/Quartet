# SDF 约束梯度链式求导（GN+ALM 方案）

本文描述当前实现中 SDF 边界约束的梯度链式法则，以及 GN J 块的计算方式。

## 1. GN 增量参数化

优化状态为四元数 `q_i`，GN 每轮在切空间增量更新：

```
δ_i ∈ R³,   q_i = Exp(δ_i) * q_i
```

所有雅可比均为 `∂残差/∂δ`（3×3 块）。

## 2. GN J 块：速度残差

定义速度残差（左差约定）：

```
r_i = Log(q_{i-1}^{-1} * q_i) ∈ R³         i ≥ 1
```

增量 `δ_i` 下的导数（右雅可比逆）：

```
∂r_i/∂δ_i = Jr^{-1}(r_i) * R_i^T
∂r_i/∂δ_{i-1} = -Jr^{-1}(r_i) * R_i^T
```

（左扰动 `q_i <- Exp(δ_i) q_i` 时，共轭作用来自被扰动的当前帧 `R_i^T`，而非前一帧。）

其中 `Jr^{-1}(φ)` 为 SO(3) **右**雅可比逆：

```
Jr^{-1}(φ) = I + 0.5·[φ]_× + c·[φ]_×²
c = 1/||φ||² - (1 + cos||φ||) / (2·||φ||·sin||φ||)
```

GN 贡献（权重 `w`，残差 `r`，J 块 `J_a, J_b` 对应帧 `a, b`）：

```
H(a,a) += w · J_a^T J_a
H(a,b) += w · J_a^T J_b
b(a)   += w · J_a^T r
```

## 3. 加速度/Jerk J 块（速度 J 块线性组合）

因 `acc_i = r_i - r_{i-1}`（R³ 减法），J 块为速度 J 块的线性组合：

```
∂acc_i/∂δ_i     = ∂r_i/∂δ_i
∂acc_i/∂δ_{i-1} = ∂r_i/∂δ_{i-1} - ∂r_{i-1}/∂δ_{i-1}
∂acc_i/∂δ_{i-2} = -∂r_{i-1}/∂δ_{i-2}
```

`jerk_i = r_i - 2 r_{i-1} + r_{i-2}` 同理扩展为 4 帧组合。

## 4. SDF 约束：图像空间链式法则

约束函数：每个裁剪角一个约束 `g_j = SDF(corner[j], warped_polygon) ≤ 0`（分别存乘子 `λ_j`、分别累加 penalty，非 max 聚合）。

### 4.1 前向计算

```
# evaluateSDF（优化侧，反向变换裁剪角到固定边界）
R_fs = R_raw * R_state^{-1}              # 相对旋转
y = R_fs * ray_j                         # 角点方向向量
qp = (f·y_x/y_z, f·y_y/y_z)              # 投影到像素
g_j = SDF(qp, fixed_boundary)            # 与固定边界比（像素级）

# verifyConstraints（验证侧，正向变换边界）
dq = q_state * q_raw^{-1}                # 反向相对旋转
warped[k] = K * R_dq * K^{-1} * rcs_grid[boundary[k]]
g_j = SDF(corner[j], {warped})           # 与固定角点比
```

### 4.2 梯度链

从图像空间 SDF 梯度到优化变量 `δ` 的完整链（evaluateSDF 实现）：

```
J_cached = dg_dp^T * Jproj * [y]_× * R_fs
dg_dp    = sign * (P - closest_point) / distance      # sign：内部(-) 外部(+)
Jproj    = ∂qp/∂y                                      # 透视投影雅可比
y        = R_fs * ray                                   # 角点方向向量
∂y/∂δ    = [y]_× * R_fs                                 # 左扰动 q_state
```

### 4.3 ALM penalty 项

在 GN 内循环中对每帧每角添加：

```
μ_eff = max(0, λ + ρ * g)                   # 有效乘子
b += μ_eff * J^T                             # 惩罚梯度（正号）
H += ρ * J^T * J                             # 惩罚 Hessian
```

外循环更新：

```
λ ← max(0, λ + ρ * g)
ρ ← min(ρ * γ, ρ_max)
```

## 5. 坐标约定

- **中心原点像素坐标**：画面范围 `[±(W-1)/2, ±(H-1)/2]`
- RCS 网格缩放：`pix = rcs_val * (W-1)/2`（DataLoader 实现）
- 裁剪角点：`±(W-1)/2 * (1-crop)`（中心原点）
- K 矩阵：`diag(f, f, 1)`，主点 (0,0)

## 6. SO(3) 工具函数

### 右雅可比逆（GN 用）

```
Jr^{-1}(φ) = I + 0.5·[φ]_× + c·[φ]_×²
c = 1/θ² - (1+cosθ)/(2θ·sinθ)               θ = ||φ||
小角度展开: I + 0.5·[φ]_× + (1/12)·[φ]_×²
```

### 左雅可比（SDF 梯度用）

```
Jℓ(φ) = I - (1-cosθ)/θ²·[φ]_× + (θ-sinθ)/θ³·[φ]_×²
Jℓ^T 用于 SDF 梯度链的最终步骤
```

### Exp/Log

```
Exp(w): θ=||w||, q=(cos(θ/2), sin(θ/2)·w/θ)
Log(q): θ=2·acos(q_w), r=(2θ/sin(θ))·q_xyz
```
