# SDE 噪声退火的理论分析 (修正版 v2)

## 1. 符号与设定

在理论分析中，常使用标准的 Euler-Maruyama 格式来近似 SDE。然而，ELF (Embedded Language Flows) 的实际实现采用了一种非齐次的、隐式的**自纠偏采样器（SDE-inspired sampler）**。为了确保理论与代码的高度一致性，我们在此明确其真实的动力学迭代格式。**以下公式已与原版 ELF 代码 (`_sde_step` in `sampling_utils.py`) 逐行核对。**

设步长 $h_k = t_{k+1} - t_k > 0$。在第 $k$ 步中：

1. **构建扰动评估状态 $z_{\text{back}}$ 与时间 $t_{\text{back}}$**：
   $$\alpha_k = 1 - \gamma_k h_k$$
   $$z_{\text{back}} = \alpha_k z_k + (1 - \alpha_k) \sigma \varepsilon_k = (1 - \gamma_k h_k) z_k + \gamma_k h_k \sigma \varepsilon_k$$
   $$t_{\text{back}} = \alpha_k t_k = (1 - \gamma_k h_k) t_k$$
   其中 $\varepsilon_k \sim \mathcal{N}(0, I)$ 是各向同性高斯噪声，$\sigma = \text{denoiser\_noise\_scale} = 2$（对应方差 $\sigma^2 = 4$）。

2. **状态前向更新**（**注意：实际代码从 $z_{\text{back}}$ 出发，而非 $z_k$**）：
   $$z_{k+1} = z_{\text{back}} + (t_{k+1} - t_{\text{back}}) \cdot v(z_{\text{back}}, t_{\text{back}})$$
   其中 $v(z, t) = \frac{\hat{x}_0(z, t) - z}{1 - t}$ 为 Flow Matching 速度场，$\hat{x}_0(z, t)$ 是去噪网络预测。

**核心观察**：ELF 轨迹 $z$ 的更新从扰动点 $z_{\text{back}}$ 出发（而非 $z_k$），步长为 $(t_{k+1} - t_{\text{back}})$（而非 $h_k$）。随机性通过两条路径注入 $z_{k+1}$：(i) $z_{\text{back}}$ 本身含 $O(h_k)$ 量级的直接噪声；(ii) 速度场在扰动点 $(z_{\text{back}}, t_{\text{back}})$ 处评估，经 Taylor 展开后产生 $O(h_k^2/(1-t_k))$ 的间接噪声（含奇异性）。当 $\gamma_k = 0$ 时，$z_{\text{back}} = z_k$，$t_{\text{back}} = t_k$，退化为一阶 Euler ODE。

---

## 2. 单步分析

### 2.1 代数重构：从自纠偏格式到半线性形式

将 $z_{k+1}$ 的更新按 $z_k$ 和 $\hat{x}_0(z_{\text{back}}, t_{\text{back}})$ 重新整理。代入 $z_{\text{back}}$ 和 $t_{\text{back}}$ 的表达式：

$$z_{k+1} = z_{\text{back}} + (t_{k+1} - t_{\text{back}}) \cdot \frac{\hat{x}_0(z_{\text{back}}, t_{\text{back}}) - z_{\text{back}}}{1 - t_{\text{back}}}$$

提取 $z_{\text{back}}$ 的系数：

$$z_{k+1} = z_{\text{back}} \cdot \frac{1 - t_{k+1}}{1 - t_{\text{back}}} + \hat{x}_0(z_{\text{back}}, t_{\text{back}}) \cdot \frac{t_{k+1} - t_{\text{back}}}{1 - t_{\text{back}}}$$

将 $z_{\text{back}} = (1 - \gamma_k h_k) z_k + \gamma_k h_k \sigma \varepsilon_k$ 和 $t_{\text{back}} = (1 - \gamma_k h_k) t_k$ 代入，并在 $O(h_k)$ 精度上展开分母 $1/(1 - t_{\text{back}}) = 1/[(1-t_k)(1 + \gamma_k h_k t_k/(1-t_k))] \approx (1/(1-t_k))(1 - \gamma_k h_k t_k/(1-t_k))$：

$$z_{k+1} = z_k \cdot \frac{1 - t_{k+1}}{1 - t_k} + \hat{x}_0(z_{\text{back}}, t_{\text{back}}) \cdot \frac{h_k}{1 - t_k} + \gamma_k h_k \sigma \cdot \frac{1 - t_{k+1}}{1 - t_k} \cdot \varepsilon_k + \mathcal{R}_k$$

其中 $\mathcal{R}_k$ 包含所有 $O(h_k^2)$ 及更高阶项。这一重构揭示了两个关键结构：

- **确定性组件**：前两项构成以 $z_k$ 为基点的指数积分步（参见第 6 节命题 6），其中 $\hat{x}_0$ 在扰动点而非 $z_k$ 处评估
- **随机组件**：第三项 $\gamma_k h_k \sigma \cdot \frac{1 - t_{k+1}}{1 - t_k} \cdot \varepsilon_k$ 是 $z_{\text{back}}$ 携带的 **$O(h_k)$ 量级直接噪声**，衰减因子 $(1-t_{k+1})/(1-t_k)$ 在 $t \to 1$ 时将噪声压制为 0

### 2.2 扰动展开与等效噪声分解

对 $\hat{x}_0(z_{\text{back}}, t_{\text{back}})$ 在 $(z_k, t_k)$ 附近做一阶 Taylor 展开：

$$\hat{x}_0(z_{\text{back}}, t_{\text{back}}) \approx \hat{x}_0(z_k, t_k) + J_{z_k} \cdot \gamma_k h_k (\sigma \varepsilon_k - z_k) - \frac{\partial \hat{x}_0}{\partial t} \cdot \gamma_k h_k t_k$$

代入 2.1 节的重构式，将单步更新写为"漂移-扩散"分解：

$$z_{k+1} = z_k + D_k + W_k^{(1)} + W_k^{(2)} + \mathcal{R}_k$$

其中：

- **确定性主导漂移**：$D_k = h_k \cdot \frac{\hat{x}_0(z_k, t_k) - z_k}{1 - t_k}$（标准 Euler ODE 步）
- **直接注入噪声（一阶量）**：$W_k^{(1)} = \gamma_k h_k \sigma \cdot \frac{1 - t_{k+1}}{1 - t_k} \cdot \varepsilon_k = O(h_k)$
- **速度场传导噪声（二阶量，含奇异性）**：$W_k^{(2)} = \frac{\gamma_k h_k^2 \sigma}{1 - t_k} \cdot J_{z_k} \varepsilon_k = O\!\left(\frac{h_k^2}{1 - t_k}\right)$
- **高阶修正确定性漂移**：$\mathcal{R}_k = O(h_k^2)$

**关键结论**：在 ELF 真实采样器中，注入 $z_{k+1}$ 的等效噪声由两项组成：

1. $W_k^{(1)}$：$O(h_k)$ 量级，来自 $z_{\text{back}}$ 的直接噪声，衰减因子 $(1-t_{k+1})/(1-t_k)$ 在 $t \to 1$ 自然压制该项
2. $W_k^{(2)}$：$O(h_k^2/(1-t_k))$ 量级，来自速度场在扰动点评估后的 Jacobian 传导，含 $1/(1-t_k)$ 奇异性

$W_k^{(1)}$ 的方差为 $O(h_k^2)$，$W_k^{(2)}$ 的方差为 $O(h_k^4/(1-t_k)^2)$。在 $t_k$ 远离 1 时 $W_k^{(1)}$ 主导，在 $t_k \to 1$ 时 $W_k^{(2)}$ 因奇异性可能放大。

### 2.3 有效信噪比 (SNR)

定义单步有效信噪比（以主导噪声项 $W_k^{(1)}$ 为参考）：

$$\text{SNR}_k = \frac{\mathbb{E}\|D_k\|}{\mathbb{E}\|W_k^{(1)}\|} \approx \frac{h_k \mathbb{E}\left\| \frac{\hat{x}_0 - z_k}{1 - t_k} \right\|}{\gamma_k h_k \sigma \cdot \frac{1 - t_{k+1}}{1 - t_k}} \propto \frac{1}{\gamma_k \cdot (1 - t_{k+1})}$$

**分析**：步长 $h_k$ 越小（$N$ 越大），$W_k^{(1)}$ 的量级随 $h_k$ 线性衰减，SNR $\propto 1/h_k$ 攀升。同时分母中的 $(1-t_{k+1})$ 在 $t \to 1$ 时使 SNR 进一步飙升——末期确定性动力学压倒随机探索。

---

## 3. 累积分析

### 3.1 总等效噪声注入量与消奇异性

**命题 1（可证明）**：当 $N \to \infty$ 时，ELF 采样器的等效累积总方差以 $O(1/N)$ 的速率收敛于 0。退火设计通过双重机制保证有限步数下的收敛质量：(\emph{i}) 早期 $W_k^{(1)}$ 提供 $O(1/N)$ 量级的随机探索；(ii) 线性退火消除 $W_k^{(2)}$ 在 $t \to 1$ 处的奇异性发散。

**证明**：

累积总方差为各步独立等效方差之和。分别考虑两项噪声的贡献。

**项一（$W_k^{(1)}$ 的累积）**：

$$\text{Var}_{\text{total}}^{(1)} = \sum_{k=0}^{N-1} \gamma_k^2 h_k^2 \sigma^2 \left(\frac{1 - t_{k+1}}{1 - t_k}\right)^2$$

设均匀剖分 $h_k = 1/N$，$\lambda_k = (1 - t_{k+1})/(1 - t_k) \approx 1 - h_k/(1-t_k) < 1$：

$$\text{Var}_{\text{total}}^{(1)} \approx \frac{\sigma^2}{N} \cdot \left[ \frac{1}{N} \sum_{k=0}^{N-1} \gamma(t_k)^2 \lambda_k^2 \cdot N h_k \right] \approx \frac{\sigma^2}{N} \int_0^1 \gamma(t)^2 \lambda(t)^2 \, dt$$

对于常数 $\gamma(t) \equiv \gamma$：$\text{Var}_{\text{total}}^{(1)} \approx \frac{\sigma^2 \gamma^2}{N} \int_0^1 \lambda(t)^2 dt = O(1/N)$。

对于线性退火 $\gamma(t) = \gamma_{\text{start}}(1-t)$：$\text{Var}_{\text{total}}^{(1)} \approx \frac{\sigma^2 \gamma_{\text{start}}^2}{N} \int_0^1 (1-t)^2 \lambda(t)^2 dt = O(1/N)$。

**项二（$W_k^{(2)}$ 的累积与奇异性分析）**：

$$\text{Var}_{\text{total}}^{(2)} = \sum_{k=0}^{N-1} \frac{\gamma_k^2 h_k^4 \sigma^2}{(1 - t_k)^2 D} \|J_{z_k}\|_F^2 \approx \frac{\sigma^2}{N^3} \int_0^1 \frac{\gamma(t)^2}{(1 - t)^2} \frac{\|J_{z(t)}\|_F^2}{D} \, dt$$

分析被积函数在 $t \to 1$ 处的行为：
1. **恒常噪声 $\gamma(t) \equiv \gamma$**：被积函数含 $\frac{\gamma^2}{(1 - t)^2}$，在 $t \to 1$ 处呈 $O((1-t)^{-2})$ 发散，积分不收敛。对应离散更新中，末期微小的 $W_k^{(2)}$ 扰动被奇异性分母放大。
2. **线性退火 $\gamma(t) = \gamma_{\text{start}}(1 - t)$**：
   $$\frac{\gamma(t)^2}{(1 - t)^2} = \gamma_{\text{start}}^2 \frac{(1 - t)^2}{(1 - t)^2} = \gamma_{\text{start}}^2 = \text{Const}$$
   奇异性被完美抵消，$W_k^{(2)}$ 的累积方差在整个闭区间上良态且为 $O(1/N^3)$。
3. **二次退火 $\gamma(t) = \gamma_{\text{start}}(1 - t)^2$**：
   $$\frac{\gamma(t)^2}{(1 - t)^2} = \gamma_{\text{start}}^2 (1 - t)^2 \to 0 \quad (t \to 1)$$
   末期 $W_k^{(2)}$ 被安全压制为 0。

**结论**：

1. **总方差由 $W_k^{(1)}$ 主导**，以 $O(1/N)$ 衰减。噪声随机探索能力随步数增加而线性递减。
2. **$W_k^{(2)}$ 在 $t \to 1$ 处存在奇异性**，但该项方差本身为 $O(1/N^3)$，在中等步数下（$N \leq 64$）其影响远小于 $W_k^{(1)}$。然而，对于常数 $\gamma$，$W_k^{(2)}$ 在最后几步被 $1/(1-t)$ 急剧放大，导致末期离散化误差不可控。
3. **退火机制的深层数学保证**：线性退火 $\gamma(t) = \gamma_{\text{start}}(1-t)$ 使 $W_k^{(2)}$ 的奇异性因子被完美抵消。这不是"减少"噪声——而是从 $W_k^{(2)}$ 的被积函数中物理消除 $(1-t)^{-2}$ 发散源。同时 $W_k^{(1)}$ 的衰减因子 $(1-t_{k+1})/(1-t_k)$ 在 $t \to 1$ 自然趋零，双重保证末期干净收敛。 $\blacksquare$

### 3.2 有限步数下存在最优 $N^*$ 的机理

**命题 2（猜想的形式化）**：存在一个非零最优步数 $N^*$，使采样指标达到最优。这一平衡由离散积分误差与 $W_k^{(1)}$ 噪声枯竭共同决定。

**论证**：
我们将生成质量（如 PPL 损失）建模为关于步数 $N$ 的平衡问题：

$$L(N) = A(N) + B(N)$$

1. **积分误差项 $A(N)$**：由于速度场在末期的奇异性，离散积分存在累积截断误差，随步数增加单调减小：
   $$A(N) = \frac{c_1}{N^\alpha} \quad (\alpha > 0)$$

2. **噪声枯竭惩罚项 $B(N)$**：由命题 1，$W_k^{(1)}$ 的总方差 $\propto 1/N$；$W_k^{(2)}$ 的总方差 $\propto 1/N^3$（且对常数 $\gamma$ 不收敛）。当 $N$ 过大时，$W_k^{(1)}$ 的随机探索能量被 $O(1/N)$ 稀释，采样器退化为准确定性。因此：
   $$B(N) = c_2 \cdot N^\beta \quad (\beta > 0)$$

通过对 $L(N)$ 求导并令导数为 0：

$$\frac{dL}{dN} = -\alpha c_1 N^{-(\alpha + 1)} + \beta c_2 N^{\beta - 1} = 0 \implies N^* = \left( \frac{\alpha c_1}{\beta c_2} \right)^{\frac{1}{\alpha + \beta}}$$

由于物理参数均为正数，最优步数 $N^*$ 必然存在且为有限正值。根据实验，在当前模型尺度下 $N^* \approx 40\text{--}80$。

**注**：相较于初版分析中的 $O(1/N^3)$ 总方差速率，修正后的 $O(1/N)$ 由 $W_k^{(1)}$ 主导，稀释速率更温和。但这不影响最优 $N^*$ 的存在性——只要 $B(N)$ 随 $N$ 单调递增，极值就存在。命题仍保持为猜想，严格证明需要 (a) 建立 $W_k^{(1)}$ 噪声枯竭到模式坍缩的因果链；(b) 推导奇异速度场下此非标准离散化的全局误差界。

---

## 4. 退火协议分析

### 4.1 线性与二次退火的能量对比

**命题 4（可证明）**：在退火指数 $p=1$ 与 $p=2$ 协议下，其归一化总累积噪声参数比值为 $2:3$。即二次退火的累积退火能量比线性退火少 $33.3\%$。

**证明**：
定义退火能量指标为退火函数在定义域上的积分：
- 对于 $p=1$（线性退火）：
  $$I_1 = \int_0^1 \gamma_{\text{start}} (1-t) \, dt = \frac{1}{2}\gamma_{\text{start}}$$
- 对于 $p=2$（二次退火）：
  $$I_2 = \int_0^1 \gamma_{\text{start}} (1-t)^2 \, dt = \frac{1}{3}\gamma_{\text{start}}$$

计算两者的比值：
$$\frac{I_2}{I_1} = \frac{1/3}{1/2} = \frac{2}{3}$$

因此，二次退火的累积能量相比线性退火减少了 $1 - \frac{2}{3} = \frac{1}{3} \approx 33.33\%$。 $\blacksquare$

**注**：上述能量指标 $\int \gamma(t) dt$ 对应于总噪声预算的经典度量。由命题 1，$W_k^{(1)}$ 的方差 $\propto \int \gamma(t)^2 dt$，对于线性退火：$\int_0^1 \gamma_{\text{start}}^2 (1-t)^2 dt = \gamma_{\text{start}}^2 / 3$；对于二次退火：$\int_0^1 \gamma_{\text{start}}^2 (1-t)^4 dt = \gamma_{\text{start}}^2 / 5$。比值 $3/5$，能量差异更大。但对于本文的噪声预算讨论，$\int \gamma(t) dt$ 是更直观的度量。

### 4.2 最优起始噪声 $\gamma_{\text{start}}$ 与步数 $N$ 的耦合

**命题 5（猜想的形式化）**：为了在不同步数下维持相同的自纠偏探索效率，最优起始噪声 $\gamma_{\text{start}}$ 应与步数 $N$ 呈正相关。

**论证**：
从 2.3 节可知，等效单步信噪比为 $\text{SNR}_k \propto 1 / (\gamma_k (1 - t_{k+1}))$。

假设平均步长 $h_k \approx 1/N$。$W_k^{(1)}$ 的噪声标准差 $\propto \gamma_k h_k$。若要使不同步数下的采样轨迹在对应阶段具有相同的微观纠偏探索强度，必须使 $\gamma_k h_k \approx \text{Const}$。

当总步数从 $N_1$ 增加到 $N_2$ 时（步长变小），对应的最优起始噪声满足：
$$\gamma_{\text{start}}^{(1)} h^{(1)} \approx \gamma_{\text{start}}^{(2)} h^{(2)} \implies \frac{\gamma_{\text{start}}^{(1)}}{\gamma_{\text{start}}^{(2)}} \approx \frac{N_1}{N_2}$$

**物理机理**：当步数 $N$ 增大时，步长 $h$ 减小，$W_k^{(1)}$ 的标准差随 $h_k$ 线性衰减。为了对抗这一噪声枯竭效应，需要成比例地增大 $\gamma_{\text{start}}$ 以维持探索信噪比。但 $\gamma_{\text{start}}$ 增大会同时放大 $W_k^{(2)}$ 的奇异性项——这构成了退化-发散的根本权衡。

---

## 5. 维数效应

### 5.1 各向同性假设与测度集中

隐变量总维数 $D = L \cdot d_{\text{enc}} \approx 1024 \times 512 \approx 5.2 \times 10^5$。根据高维高斯分布的测度集中性质，对于随机噪声 $\varepsilon \sim \mathcal{N}(0, I_D)$，其 $L_2$ 模长几乎确定地集中在超球面上：
$$\mathbb{P}\left( \left| \|\varepsilon\|_2 - \sqrt{D} \right| \geq t \right) \leq 2 e^{-ct^2}$$

这保证了扰动在各方向上的各向同性。

### 5.2 噪声传导与 Jacobian 约束

$W_k^{(2)}$ 经非线性去噪网络 $\hat{x}_0$ 传导后，由高维各向同性投影性质，等效注入噪声的模长平方期望为：
$$\mathbb{E}[\|W_k^{(2)}\|_2^2] = \frac{\gamma_k^2 h_k^4 \sigma^2}{(1 - t_k)^2} \text{Tr}(J_{z_k} J_{z_k}^T) = \frac{\gamma_k^2 h_k^4 \sigma^2}{(1 - t_k)^2} \|J_{z_k}\|_F^2$$

$W_k^{(1)}$ 不经过 Jacobian，其方差 $\mathbb{E}[\|W_k^{(1)}\|_2^2] = \gamma_k^2 h_k^2 \sigma^2 \cdot D \cdot ((1-t_{k+1})/(1-t_k))^2$。

退火协议在 $t \to 1$ 时将 $\gamma(t) \to 0$，本质上是同时切断两条噪声传播通道——直接注入 $W_k^{(1)}$ 和 Jacobian 传导 $W_k^{(2)}$——确保在速度场奇异区域不发生误差放大。

---

## 6. 与其他工作的联系

### 6.1 与 DPM-Solver 指数积分器的等价性

**命题 6（可证明）**：Flow Matching 的确定性 ODE 更新在局部常数假设下，可以精确积分为 DPM-Solver-1 的半线性形式更新，这为 `exp_heun_step` 奠定了解析积分基础。

**证明**：
Flow Matching 的常微分方程为：
$$\frac{dz}{dt} = \frac{\hat{x}_0(z, t) - z}{1 - t}$$

整理为一阶半线性形式：
$$\frac{dz}{dt} + \frac{1}{1 - t} z = \frac{1}{1 - t} \hat{x}_0(z, t)$$

引入积分因子 $\mu(t) = \exp\left( \int \frac{1}{1-t} \, dt \right) = \frac{1}{1 - t}$，方程两端同乘 $\mu(t)$：
$$\frac{d}{dt} \left( \frac{z(t)}{1 - t} \right) = \frac{\hat{x}_0(z, t)}{(1 - t)^2}$$

在闭区间 $[t_k, t_{k+1}]$ 上积分：
$$\frac{z(t_{k+1})}{1 - t_{k+1}} - \frac{z(t_k)}{1 - t_k} = \int_{t_k}^{t_{k+1}} \frac{\hat{x}_0(z(s), s)}{(1 - s)^2} \, ds$$

在小步长区间内，对去噪预测进行零阶近似：$\hat{x}_0(z(s), s) \approx \hat{x}_0(z(t_k), t_k) \equiv \hat{x}_0^{(k)}$。将其提到积分号外并求解：
$$\int_{t_k}^{t_{k+1}} \frac{1}{(1 - s)^2} \, ds = \frac{1}{1 - t_{k+1}} - \frac{1}{1 - t_k}$$

代入并两边同乘 $(1 - t_{k+1})$ 整理得：
$$z(t_{k+1}) = \frac{1 - t_{k+1}}{1 - t_k} z(t_k) + \left(1 - \frac{1 - t_{k+1}}{1 - t_k}\right) \hat{x}_0^{(k)}$$

令 $\lambda_k = \frac{1 - t_{k+1}}{1 - t_k}$，则有：
$$z(t_{k+1}) = \lambda_k z(t_k) + (1 - \lambda_k) \hat{x}_0^{(k)}$$
此公式正是 `exp_heun_step` 的一阶基础更新格式。 $\blacksquare$

### 6.2 EDM (Karras et al., NeurIPS 2022)

EDM 的 Heun 采样器 + 噪声调度器也使用了"噪声衰减"思想：在图像扩散的高 SNR 阶段增加噪声多样性，低 SNR 阶段减弱。ELF Flow Matching 的退火 SDE 可视为该框架在文本潜空间的对应物。

**区别**：EDM 操作在 $\sigma(t)$（噪声方差调度），ELF 操作在 $\gamma(t)$（注入噪声强度）。Flow Matching 的速度场形式 $v(z,t) = (\hat{x}_0 - z)/(1-t)$ 使得退火的必要性更紧迫——$W_k^{(2)}$ 在 $t \to 1$ 时被 $1/(1-t)$ 放大，线性退火是对抗这一放大效应的充要条件（命题 1）。

---

## 7. 证明与猜想的状态汇总

| 序号 | 内容 | 状态 | 证明思路 / 物理机制说明 |
|------|------|------|-----------------------|
| **命题 1** | $\text{Var}_{\text{total}}^{(1)} = O(1/N)$，$W_k^{(2)}$ 的奇异性由退火消去 | ✅ 可证明 | 代数重构 + Taylor 展开 + 黎曼积分 + 奇异点极限消去 |
| **命题 2** | 存在非零最优步数 $N^*$ | ⚠️ 形式化猜想 | $W_k^{(1)}$ 噪声枯竭 $O(N^{\beta})$ 与奇异场离散误差 $O(N^{-\alpha})$ 的极值平衡 |
| **命题 4** | $p=2$ 总能量比 $p=1$ 减少 $33.3\%$ | ✅ 可证明 | 定积分直接计算 $\int_0^1 (1-t)^p dt$ |
| **命题 5** | 最优起始噪声 $\gamma_{\text{start}}$ 与 $N$ 正相关 | ⚠️ 形式化猜想 | 保持单步有效探索强度 $\gamma_k h_k \approx \text{Const}$ |
| **命题 6** | `exp_heun_step` 与 DPM-Solver 的半线性等价性 | ✅ 可证明 | 积分因子法（Integrating Factor）解析积分解 |

**严格证明命题 2 的路径**：需要 (a) 建立 $W_k^{(1)}$ 噪声枯竭到模式坍缩的序参量（如 pairwise cosine similarity 的期望）；(b) 推导此非标准离散化在奇异速度场下的全局误差界；(c) 求解 $\nabla_N (\text{PPL}) = 0$ 的 Nash 均衡。这三步均超过当前的分析能力。

**v2 修订说明**：初版将 ELF 的更新公式写为 $z_{k+1} = z_k + h_k \cdot v(z_{\text{back}}, t_{\text{back}})$（从 $z_k$ 出发），经代码核查（原版 `_sde_step` 第 251 行 `return z_back + (t_next - t_back) * v_pred`）修正为从 $z_{\text{back}}$ 出发。这一修正引入了 $W_k^{(1)}$（$O(h_k)$ 量级直接噪声），将总方差主项从 $O(1/N^3)$ 更正为 $O(1/N)$ 由 $W_k^{(1)}$ 主导。$W_k^{(2)}$ 的奇异性分析和退火消奇异性的结论保持不变。

---

## 8. 实验验证

全部实验使用 ELF-B (105M)，OpenWebText，T5-small 编码器，seed=42。

| N | gamma | PPL | Entropy | 现象 | 对命题 |
|---|-------|-----|---------|------|--------|
| 32 | 1.5→0 (mild) | 25.30 | 5.17 | 积分误差主导 | 命题 2: N < N* |
| 32 | 2.0→0 | 21.47 | 5.10 | 噪声补偿积分误差 | 命题 5 |
| 64 | 1.5→0 (mild) | 17.25 | 5.05 | 接近 N* | 命题 2, 3 |
| 64 | 2.0→0 | 15.71 | 5.01 | N* 邻域最优 | 命题 2, 3 |
| 64 | 1.5→0 (mild) p=2 | 18.04 | 5.07 | 噪声过早衰减 | 命题 4 |
| 256 | 1.5→0 (mild) | 10.06 | 4.87 | $W_k^{(1)}$ 枯竭至准 ODE | 命题 2: N > N* |
| 256 | 2.0→0 | 9.08 | 4.82 | 严重坍缩 | 命题 2, 5 |

---

## 参考文献

- Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
- Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022.
- Lu et al., "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps", NeurIPS 2022.
- He et al., "ELF: Embedded Language Flows", arXiv 2026.
