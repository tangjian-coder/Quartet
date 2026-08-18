# 四元数优化（滑动窗口）— Gauss-Newton + ALM

本文档描述当前 GN+ALM 优化实现。

## 1. 数据与记号

- `q_orig[i]`：第 `i` 帧原始姿态四元数
- `q_state[i]`：优化过程中的当前状态四元数（GN 增量更新）
- 窗口大小 `N`，前缀固定帧数 `P`

增量更新：`q_state[i] = Exp(δ_i) * q_state[i]`，其中 `δ_i ∈ R³`。

四元数约定：`q_a * q_b` 先应用 `b` 再应用 `a`（Eigen 约定）。

## 2. 滑动窗口策略

- `overlap_frames`：窗口前缀重叠帧数（固定为上一窗口输出）
- `emit_frames`：每窗口输出帧数
- `future_frames`：未来帧数（仅用于优化，不输出）
- `total = overlap + emit + future`

前缀帧通过直接设置 `q_state[i] = q_target[i]`（不参与 GN 优化）来固定。

## 3. 优化框架

### 3.1 Gauss-Newton 内循环（无约束目标）

每轮 GN 迭代构建 Hessian `H` 和梯度 `b`，求解 `dx = -H⁻¹b`，状态更新 `q += Exp(dx) * q`。

**速度残差**（一阶平滑）：

```
r_i = Log(q_{i-1}^{-1} * q_i)              i ≥ 1
∂r_i/∂δ_i = Jr^{-1}(r_i) * R_i^T
∂r_i/∂δ_{i-1} = -Jr^{-1}(r_i) * R_i^T
```

目标项：`λ₁ * ||r_i||²`，GN 贡献 `H += λ₁·J^T J`，`b += λ₁·J^T r`。

**加速度残差**（二阶平滑，R³ 近似）：

```
acc_i = r_i - r_{i-1}                      i ≥ 2
∂acc_i/∂δ_i = ∂r_i/∂δ_i
∂acc_i/∂δ_{i-1} = ∂r_i/∂δ_{i-1} - ∂r_{i-1}/∂δ_{i-1}
∂acc_i/∂δ_{i-2} = -∂r_{i-1}/∂δ_{i-2}
```

**Jerk 残差**（三阶平滑，R³ 近似）：

```
jerk_i = r_i - 2 r_{i-1} + r_{i-2}        i ≥ 3
∂jerk_i/∂δ_i = ∂r_i/∂δ_i
∂jerk_i/∂δ_{i-1} = ∂r_i/∂δ_{i-1} - 2·∂r_{i-1}/∂δ_{i-1}
∂jerk_i/∂δ_{i-2} = 2·∂r_{i-1}/∂δ_{i-2} + ∂r_{i-2}/∂δ_{i-2}
∂jerk_i/∂δ_{i-3} = -∂r_{i-2}/∂δ_{i-3}
```

**Fidelity 残差**（保真项）：

```
r_f_i = Log(q_i^{-1} * q_orig_i)
J_f_i = -Jr^{-1}(r_f_i) * R_orig_i^T
```

目标项：`μ * ||r_f_i||²`。

### 3.2 ALM 外循环（SDF 边界约束）

约束不等式 `g ≤ 0`，其中 `g` 为 SDF（正 = 边界外）。

**约束计算**（evaluateSDF，反向变换裁剪角到固定边界）：

1. 相对旋转：`R_fs = R_q_raw * R_q_state^{-1}`
2. 对每个裁剪角：方向向量 `y = R_fs * ray_j`，投影到像素 `qp = (f·y_x/y_z, f·y_y/y_z)`
3. 约束值 `g_j = SDF(qp, fixed_boundary)`（像素级，使用时乘 `1/f` 归一化）
4. 梯度链（左扰动 q_state）：`∂g/∂δ = dg_dp^T * Jproj * [y]_× * R_fs`

（verifyConstraints 验证侧方向相反：`dq = q_state * q_raw^{-1}`，`Ht = K * R_dq * K^{-1}`，正向变换 RCS 边界后与固定角点比 SDF。）

**ALM penalty**（在 GN 内循环中每帧每角添加）：

```
μ_eff = max(0, λ + ρ * g)                  λ: 乘子, ρ: 惩罚参数
b += μ_eff * J^T                           惩罚梯度（正号，J^T 指向增 g 方向）
H += ρ * J^T * J                           惩罚 Hessian
```

**乘子更新**（每次外循环结束）：

```
λ ← max(0, λ + ρ * g)                      g > 0 时 λ 增加
ρ ← min(ρ * γ, ρ_max)                     惩罚递增
```

收敛条件（外层停止，三 KKT 误差全达标）：

```
e_feas < 1e-6     （可行性：max[g]+）
e_stat < 1.0      （平稳性：||J_r^T r + J_g^T λ||）
e_comp < 1e-2     （互补松弛：||λ ⊙ g||）
```

另有 `max_outer` 轮次上限兜底。

## 4. 坐标系

**中心原点像素坐标**：

- RCS 网格：`x_pix = x_rsc * (W-1)/2`，`y_pix = y_rsc * (H-1)/2`
- 裁剪角点：`±(W-1)/2 * (1-cropH)`，`±(H-1)/2 * (1-cropV)`
- K 矩阵：`diag(f, f, 1)`，主点在 (0,0)

故 RCS 边界和裁剪角点在同一坐标系，直接计算 SDF。

## 5. SO(3) 辅助函数

| 函数 | 用途 |
|---|---|
| `Exp(w)` | R³ → SO(3)：`θ = ||w||`，`q = (cos(θ/2), sin(θ/2)*w/θ)` |
| `Log(q)` | SO(3) → R³：`θ = 2·acos(q_w)`，`r = 2θ/sin(θ) * q_xyz` |
| `Jr^{-1}(φ)` | 右雅可比逆：`I + 0.5·[φ] + c·[φ]²`，其中 `c = 1/θ² - (1+cosθ)/(2θ·sinθ)` |
| `hat(v)` | 3→3×3 反对称矩阵 `[v]_×` |
| `skewInv(M)` | 3×3 反对称矩阵 → 3 维向量 |

## 6. 调参

| 参数 | 默认 | 作用 |
|---|---|---|
| λ₁ | 1.0 | 速度平滑 |
| λ₂ | 2.0 | 加速度平滑 |
| λ₃ | 8.0 | Jerk 平滑 |
| μ | 0.02 | 保真度（越小越平滑，越接近边界）|
| ρ_init | 20.0 | ALM 初始惩罚 |
| ρ_max | 500.0 | ALM 最大惩罚 |
| γ | 1.5 | 惩罚增长率 |
| max_outer | 5 | 最大外循环次数 |
| max_inner | 8 | 最大内循环次数 |

## 7. 关键源文件

| 文件 | 核心内容 |
|---|---|
| `src/Optimizer.cpp` | `solveWindowGN()` 主求解器，`evaluateSDF()` 约束计算 |
| `src/DataLoader.cpp` | `loadRcsGridData()` RCS 网格坐标缩放 |
| `src/GeometryTools.cpp` | `signedDistanceFromPointToPolygon()` SDF + 最近边 |
| `src/main.cpp` | 滑动窗口循环，权重参数 |
