# Left-Increment 链路与解析梯度推导（$SO(3)$）

本文固定当前代码中的定义（left-multiplication system）：

$$
x_i\in\mathbb{R}^3,\quad dq_i=\mathrm{Exp}(x_i)\in SO(3),\quad
q_i=dq_i\otimes q_{\text{ori},i},
$$
$$
q_{r,i}=q_i\otimes q_{i-1}^{-1},\quad
r_i=\mathrm{Log}(q_{r,i})\in\mathbb{R}^3.
$$

其中目标函数平滑项为（后向差分）：

$$
L_{\text{smooth}}
= \lambda_1\sum_{i\ge 1}\lVert r_i\rVert^2
+ \lambda_2\sum_{i\ge 2}\lVert r_i-r_{i-1}\rVert^2
+ \lambda_3\sum_{i\ge 3}\lVert r_i-2r_{i-1}+r_{i-2}\rVert^2.
$$

fidelity 项为：

$$
L_{\text{fid}}=\mu\sum_i \lVert x_i\rVert^2.
$$

---

## 1. 记号与基本算子

- $\hat{(\cdot)}:\mathbb{R}^3\to so(3)$ 为反对称矩阵映射。
- $\mathrm{Exp}(\cdot), \mathrm{Log}(\cdot)$ 为 $SO(3)$ 指数/对数映射。
- 左雅可比与其逆：$J_\ell(\phi)$、$J_\ell^{-1}(\phi)$。
- 旋转矩阵表示下，$Ad_R v = Rv$。

采用矩阵形式更容易看链式法则。令

$$
R_i = \mathrm{Exp}(\hat{x}_i)R_{\text{ori},i},\quad
R_{\text{rel},i}=R_iR_{i-1}^T,\quad
r_i=\mathrm{Log}(R_{\text{rel},i}).
$$

---

## 2. 单个 $r_i$ 的回传（核心链路）

先写一个通用上游向量 $g_i:=\frac{\partial L}{\partial r_i}\in\mathbb{R}^3$，则

$$
\delta L_i = g_i^T\,\delta r_i.
$$

左扰动线性化：

$$
\delta r_i = J_\ell^{-1}(r_i)\,\delta\theta.
$$

### 2.1 对 $x_i$ 的通道

由

$$
R_i=\mathrm{Exp}(\hat{x}_i)R_{\text{ori},i},
$$

可得

$$
\delta\theta_i = J_\ell(x_i)\,\delta x_i,
$$

并且在

$$
R_{\text{rel},i}=R_iR_{i-1}^T
$$

中，固定 $R_{i-1}$ 时：

$$
\delta r_i
= J_\ell^{-1}(r_i)\,J_\ell(x_i)\,\delta x_i.
$$

所以

$$
\frac{\partial L}{\partial x_i}\Big|_{\text{from }r_i}
= \left(J_\ell^{-1}(r_i)J_\ell(x_i)\right)^T g_i.
$$

### 2.2 对 $x_{i-1}$ 的通道

固定 $R_i$，由 $R_{i-1}$ 变化得到：

$$
\delta r_i
= -J_\ell^{-1}(r_i)\,R_{\text{rel},i}\,J_\ell(x_{i-1})\,\delta x_{i-1}.
$$

所以

$$
\frac{\partial L}{\partial x_{i-1}}\Big|_{\text{from }r_i}
= \left(-J_\ell^{-1}(r_i)R_{\text{rel},i}J_\ell(x_{i-1})\right)^T g_i.
$$

> 这就是代码里“从每个 $r_i$ 同时回传到 $x_i$ 与 $x_{i-1}$”的统一模板。

---

## 3. 一阶项（速度项）回传

定义

$$
L_1=\lambda_1\sum_{i\ge 1}\lVert r_i\rVert^2.
$$

则每个 $i\ge 1$：

$$
g_i^{(1)}=\frac{\partial L_1}{\partial r_i}=2\lambda_1 r_i.
$$

之后把 $g_i^{(1)}$ 代入第 2 节模板即可。

---

## 4. 二阶与三阶项回传（你关心的补充）

关键点：二阶/三阶项与一阶项相比，**只是先多一步把各项贡献累加到 $g_i=\partial L/\partial r_i$**；累加完后，回传到 $x$ 的链路与第 2 节完全相同。

### 4.1 二阶项（加速度）

$$
L_2=\lambda_2\sum_{i\ge 2}\lVert r_i-r_{i-1}\rVert^2.
$$

记

$$
a_i:=r_i-r_{i-1}.
$$

则单项 $\lambda_2\lVert a_i\rVert^2$ 对两个变量的贡献为：

$$
\frac{\partial}{\partial r_i}=2\lambda_2 a_i,\qquad
\frac{\partial}{\partial r_{i-1}}=-2\lambda_2 a_i.
$$

因此遍历 $i\ge 2$ 累加：

$$
g_i \mathrel{+}= 2\lambda_2 a_i,\qquad
g_{i-1} \mathrel{-}= 2\lambda_2 a_i.
$$

### 4.2 三阶项（jerk）

$$
L_3=\lambda_3\sum_{i\ge 3}\lVert r_i-2r_{i-1}+r_{i-2}\rVert^2.
$$

记

$$
j_i:=r_i-2r_{i-1}+r_{i-2}.
$$

单项 $\lambda_3\lVert j_i\rVert^2$ 对三个变量贡献：

$$
\frac{\partial}{\partial r_i}=2\lambda_3 j_i,\quad
\frac{\partial}{\partial r_{i-1}}=-4\lambda_3 j_i,\quad
\frac{\partial}{\partial r_{i-2}}=2\lambda_3 j_i.
$$

因此遍历 $i\ge 3$ 累加：

$$
g_i \mathrel{+}= 2\lambda_3 j_i,\quad
g_{i-1} \mathrel{-}= 4\lambda_3 j_i,\quad
g_{i-2} \mathrel{+}= 2\lambda_3 j_i.
$$

---

## 5. 总体反向传播流程（实现顺序）

1. **前向缓存**：计算并缓存 $x_i,\ q_i,\ q_{r,i},\ r_i,\ J_\ell(x_i),\ J_\ell^{-1}(r_i),\ R_{\text{rel},i}$。  
2. **先聚合 $g_i=\partial L/\partial r_i$**：
   - 加入一阶项贡献 $2\lambda_1 r_i$；
   - 加入二阶项贡献（按 4.1）；
   - 加入三阶项贡献（按 4.2）。
3. **按第 2 节模板回传到 $x$**：
   - 从每个 $r_i$ 回传到 $x_i$ 与 $x_{i-1}$ 两个位置。
4. **加入 fidelity 梯度**：

$$
\frac{\partial L_{\text{fid}}}{\partial x_i}=2\mu x_i.
$$

5. 得到总梯度向量 `grad`。

---

## 6. $SO(3)$ 左雅可比公式（实现用）

令 $\phi\in\mathbb{R}^3,\ \theta=\lVert\phi\rVert,\ A=\hat{\phi}$。

$$
J_\ell(\phi)=I-\frac{1-\cos\theta}{\theta^2}A+\frac{\theta-\sin\theta}{\theta^3}A^2.
$$

$$
J_\ell^{-1}(\phi)=I+\frac{1}{2}A+\left(\frac{1}{\theta^2}-\frac{1+\cos\theta}{2\theta\sin\theta}\right)A^2.
$$

小角度稳定展开（$\theta\to 0$）：

$$
J_\ell(\phi)\approx I-\frac{1}{2}A+\frac{1}{6}A^2,\qquad
J_\ell^{-1}(\phi)\approx I+\frac{1}{2}A+\frac{1}{12}A^2.
$$

---

## 7. 工程注意事项

1. $q$ 与 $-q$ 表示同一旋转，若不统一分支（如约束 $w\ge 0$），`Log` 可能跳变。  
2. `quaternionToAxisAngle()` 作为工程 `Log` 时，forward/backward 必须使用同一分支约定。  
3. 对 $r_i$ 做差分（acc/jerk）严格来说是跨切空间运算；当前做法属于小旋转近似。

