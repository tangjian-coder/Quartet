# 四元数优化(滑动窗口) - 问题定义与梯度计算

本文档描述本仓库当前实现的优化问题：

- 输入：逐帧相机姿态 `q_orig[i]`（四元数）
- 输出：更平滑的优化姿态 `q_filt[i]`（四元数）
- 执行方式：滑动窗口优化，窗口之间用重叠帧对齐

本文内容与当前 C++ 实现保持一致（以代码为准）：

- `src/Optimizer.cpp`
- `include/Optimizer.h`
- `src/main.cpp`

## 1. 数据与记号

全序列长度为 `N` 帧。

- `q_orig[i]`：第 `i` 帧原始姿态四元数
- `q_filt[i]`：第 `i` 帧优化后(滤波后)姿态四元数

每次优化在一个长度为 `T` 的窗口内进行。

四元数表示 `SO(3)` 旋转，使用前会做归一化。

### 四元数运算约定

- 复合：`q_a * q_b` 表示先应用旋转 `b`，再应用旋转 `a`（Eigen 的约定）。
- 逆：`q.inverse()` 表示逆旋转。
- 相对旋转：`q_rel = q_prev.inverse() * q_next`。

## 2. 滑动窗口策略

代码以窗口方式优化序列，用于模拟“手机防抖有一定未来数据”的在线处理。

窗口参数：

- `overlap_frames`（`O`）：窗口前缀重叠帧数，只用于与上一窗口对齐；这些帧会被固定为上一窗口已输出的结果。
- `core_frames`（`C`）：重叠之后的核心帧数，包含本次输出帧 + 未来仅用于优化的帧。
- `emit_frames`（`E`）：每个窗口真正写入全局输出的帧数。
- `future_frames = core_frames - emit_frames`。
- `window_size T = overlap_frames + core_frames = O + C`。

`src/main.cpp` 默认值：

- `O=3`，`C=100`，`E=70`，`future=30`，所以 `T=103`。

### 2.1 窗口索引

对窗口 `k`，定义：

- `next_emit_start`：下一段要输出的全局起始帧号。
- `window_start = next_emit_start - O`。

窗口输入帧通过“夹紧(clamp)”索引到 `[0, N-1]` 构造：

- 若 `window_start + local_idx < 0`，使用帧 `0`
- 若 `window_start + local_idx >= N`，使用帧 `N-1`

这等价于：

- 前缀补齐：第一个窗口隐含有 `O` 帧与第 1 帧相同的“虚拟帧”
- 尾部补齐：最后一个窗口不够时，重复最后一帧直到窗口长度为 `T`

### 2.2 对齐(固定重叠帧)

对窗口 `k > 0`，窗口前 `O` 帧要与全局已输出结果严格一致。

固定的全局帧范围：

- `fixed_global_start = next_emit_start - O`
- 固定帧为 `[fixed_global_start .. fixed_global_start + O - 1]`

优化器需要满足：

- `q_filt_window[j] == q_filt_global[fixed_global_start + j]`，其中 `j=0..O-1`

当前实现通过“硬边界”固定决策变量实现（`lb == ub`）。

每个窗口只输出窗口内 `j = O .. O + E - 1` 这段（即第 `O+1` 到 `O+E` 帧）。

## 3. 决策变量(旋转向量 / 轴角向量)

对窗口内每一帧 `i`（局部编号 `0..T-1`），引入一个小的姿态修正 `dq[i]`：

- `q_filt[i] = dq[i] * q_orig[i]`

不直接优化 `dq[i]`，而是优化 3 维旋转向量 `x_i`（轴角向量）：

- `x_i in R^3`
- `dq[i] = Exp(x_i)`（旋转向量的指数映射到四元数）

优化变量展开为一维向量：

- `x = [x_0, x_1, ..., x_{T-1}]`，维度 `D = 3*T`

在代码里：

- `axisAngleToQuaternion(x_i)` 用 `AngleAxis(angle, axis)` 实现 `Exp(x_i)`，其中 `angle = ||x_i||`。

## 4. 优化目标函数

当前 `Optimizer::computeObjectiveValue` 实现的目标为：

- `J(x) = J_smooth(x) + J_fidelity(x)`

其中 `lambda1`、`lambda2`、`lambda3`、`mu` 为权重。

### 4.1 相对旋转序列

先定义滤波后的姿态：

- `q_filt[i] = Exp(x_i) * q_orig[i]`

相邻帧相对旋转：

- `q_rel[i] = q_filt[i-1]^{-1} * q_filt[i]`（`i >= 1`）

相对旋转向量(对数映射)：

- `r[i] = Log(q_rel[i]) in R^3`（`i >= 1`）
- `r[0] = 0`

代码中：

- `quaternionToAxisAngle(q)` 通过 `Eigen::AngleAxisd(q)` 得到 `angle * axis`。

备注：

- `Log(q)` 在全局上并非处处光滑（`SO(3)` 的拓扑 + 主值角度选择导致），尤其在接近 `pi` 旋转时可能出现不光滑/跳变。
- 但对“逐帧小角度变化”的防抖场景，通常是可接受的。

### 4.2 平滑项(速度 + 加速度 + 加加速度/jerk)

速度项：

- `J_vel = sum_{i=1..T-1} lambda1 * || r[i] ||^2`

加速度项（二阶差分）：

- `acc[i] = r[i] - r[i-1]`（`i >= 2`）
- `J_acc = sum_{i=2..T-1} lambda2 * || acc[i] ||^2`

jerk 项（三阶平滑，使用 r 的二阶差分）：

- `jerk[i] = r[i+1] - 2*r[i] + r[i-1]`（`2 <= i <= T-2`）
- `J_jerk = sum_{i=2..T-2} lambda3 * || jerk[i] ||^2`

总平滑项：

- `J_smooth = J_vel + J_acc + J_jerk`

### 4.3 保真项(限制修正幅度)

惩罚修正旋转向量 `x_i` 的大小，使 `dq` 尽量接近单位旋转：

- `J_fidelity = sum_{i=0..T-1} mu * || x_i ||^2`

### 4.4 总目标

- `J(x) = J_smooth(x) + J_fidelity(x)`

## 5. 重叠帧固定(通过硬边界实现)

对重叠帧，我们有目标滤波四元数 `q_target[i]`（来自上一窗口输出）。

希望满足：

- `q_filt[i] == q_target[i]`

又因为：

- `q_filt[i] = dq[i] * q_orig[i]`

所以需要的修正为：

- `dq_fixed[i] = q_target[i] * q_orig[i]^{-1}`

将其转换到决策变量：

- `x_fixed[i] = Log(dq_fixed[i])`

用硬边界固定变量：

- 对每个分量 `c in {0,1,2}`：
  - `lb[3*i+c] = ub[3*i+c] = x_fixed[i][c]`

这就是当前 `Optimizer::optimize(fixed_prefix_filtered)` 用来做窗口对齐的机制。

## 6. 梯度计算

NLopt 当前算法使用 `LD_MMA`（梯度法），需要一阶导数：

- `grad[j] = dJ/dx_j`，其中 `j = 0..D-1`

### 6.1 数值梯度(有限差分)

当前备用实现：

- `dJ/dx_j ~= (J(x + h e_j) - J(x - h e_j)) / (2h)`

每次梯度代价：

- 额外 `2*D` 次目标函数评估（当 `D=3*T` 比较大时非常慢，例如 `T=103 => D=309`）。

### 6.2 “解析梯度”(建议实现方式)

想得到正确且高效的一阶梯度，常见两条路：

1. 手推并实现 `dJ/dx`（工作量大且易错，尤其涉及 Exp/Log 与四元数链式法则）
2. 自动微分(AutoDiff)，一次评估得到精确一阶导数（工程上更稳）

由于目标函数包含四元数的 Exp/Log 以及旋转复合，AutoDiff 往往是最快拿到正确梯度的方式。

#### 推荐：Eigen AutoDiffScalar

Eigen 的 AutoDiff 在：

- `unsupported/Eigen/AutoDiff`

实现思路：

- 将目标函数改写为模板版本，能同时跑 `double` 与 AutoDiff 标量类型。
- 将每个变量 `x_j` 构造为 `AutoDiffScalar<VectorXd>`，其导数向量长度为 `D`（对应单位基）。
- 只计算一次 `J(x)`，然后把 `J.derivatives()` 拿出来填 `grad`。

关键要求：

- 需要实现能用于 AutoDiff 的 `Exp(rotvec)->quat` 和 `Log(quat)->rotvec`。
- 必须处理小角度时的数值稳定（避免除以 `||x||` 或 `sin(theta)` 接近 0）。

### 6.3 是否需要二阶/三阶导数？

对当前实现：

- `LD_MMA` 只用一阶梯度。
- 不需要 Hessian（二阶）或三阶导数。

如果未来换成二阶方法(Newton/trust-region)：

- 二阶导数(Hessian)可算但代价更高：
  - 完整 Hessian 是 `D x D`（例如 `D=309` 大约 9.5 万个元素）
  - 用嵌套 AutoDiff 或反复求梯度可以实现，但计算成本较大
- 三阶导数一般不现实。

若确实需要二阶信息，优先考虑 Hessian-vector product(HVP)，避免构造完整 Hessian。

## 7. 当前实现状态

- 目标函数 `J(x)` 与上文一致。
- 滑动窗口 + 重叠帧固定已实现：通过传入固定前缀目标四元数，并用 `lb==ub` 固定变量。
- `computeAnalyticGradient()` 目前仍是占位：内部调用了数值梯度。后续应替换为 AutoDiff 梯度(推荐)或手推解析梯度实现。
