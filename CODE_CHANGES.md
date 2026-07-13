# ELF 代码改动文档

本文档记录在原版 ELF (Kaiming He et al.) [ELF-pytorch_elf](../ELF-pytorch_elf) 基础上所做的所有代码改动。目标是：一个新 agent 阅读本文档后，可以 (1) 理解每个改动的位置和目的，(2) 知道如何添加新采样方法，(3) 扩大实验范围。

---

## 总览

**改动了 4 个源文件，新增了 14 个采样配置文件，新增了 8 个评估脚本。模型权重、训练代码、编码器均未修改。**

| 文件 | 改动类型 | 行数变化 |
|------|---------|---------|
| `src/utils/sampling_utils.py` | 新增 9 个步函数 + 修改 2 个已有函数 | ~250 行新增 |
| `src/utils/generation_utils.py` | 新增方法分派 + 解码参数 + 名字生成 | ~100 行新增 |
| `src/generation.py` | 参数传递管道 | ~50 行新增 |
| `src/utils/data_utils.py` | 离线数据集加载修复 | ~10 行修改 |
| `src/configs/sampling_configs/*.yml` | 14 个新配置文件 | 新增 |
| `scripts/eval_*.sh` | 8 个新评估脚本 | 新增 |

---

## 文件一：`src/utils/sampling_utils.py`

此文件包含所有步函数（step functions）。每个步函数接收 `(model, z, t, t_next, x_pred_prev, config, ...)` 并返回 `(z_next, x_pred)`。

### 1.1 新增函数列表

| 函数名 | 方法名(method) | NFE/步 | 类型 | 关键新增参数 |
|--------|--------------|--------|------|------------|
| `_heun_step` | `heun` | 2 | 确定性 ODE | 无 |
| `_heun_step_with_err` | (内部) | 2 | 辅助函数 | 无 |
| `_heun_adaptive_step` | `heun_adaptive` | 2+ | 自适应 ODE | `tol`, `max_depth` |
| `_exp_step` | `exp` | 1 | 指数积分 ODE | 无 |
| `_exp_heun_step` | `exp_heun` | 2 | 指数积分+中点 | 无 |
| `_pc_step` | `pc` | 2 | Predictor-Corrector | `gamma`, `generator` |
| `_sde_ml_step` | `sde_ml` | k | 多轮 Langevin SDE | `gamma`, `generator`, `num_langevin` |

### 1.2 已有函数修改

**`_forward_sample_self_cond`** — 签名新增 `sc_noise_scale=0.0` 参数。添加了内部辅助函数 `_perturb_x_pred(xp, t_b)` 用于对自条件输入注入噪声，衰减因子为 `(1-t)²`。

**注意**：`_perturb_x_pred` 内部通过 `getattr(config, '_sc_noise_scale', 0.0)` 从训练 Config 对象读取噪声值（该值由 `_generate_samples_single_batch` 在采样循环前设置）。噪声衰减因子为 `(1-t)²`。当 `sc_noise_scale` 未在采样配置中设置时，默认值为 0，不注入噪声——不影响所有已有方法。

**实验结论**：sc_noise 在小量级 (≤0.2) 无效（模型忽略扰动），在大剂量 (≥0.5) 导致模式坍缩。该功能保留在代码中，但非推荐使用。

**`_forward_sample`** — 签名增加参数 `sc_noise_scale=0.0`，透传给 `_forward_sample_self_cond`。

### 1.3 函数详细说明

#### `_heun_step` (velocity Heun, 2nd-order RK)

```python
def _heun_step(model, z, t, t_next, x_pred_prev,
               config, cfg_scale, self_cond_cfg_scale,
               cond_seq, cond_seq_mask):
```

公式：
1. `v1 = model(z, t)` → `z_pred = z + h·v1`
2. `v2 = model(z_pred, t+h)` → `z_next = z + h/2·(v1+v2)`

成本：2 NFE/步。

#### `_heun_step_with_err` (内部函数)

与 `_heun_step` 相同，但额外返回局部误差估计：
```python
err = (h/2) * ||v2 - v1|| / ||z||   # 每样本相对误差的最大值
```
此误差估计无需额外 NFE——`v1` 和 `v2` 已经从 Heun 的两阶段计算中获得。

#### `_heun_adaptive_step` (自适应 Heun)

```python
def _heun_adaptive_step(..., tol, max_depth=4):
```

逻辑：
1. 调用 `_heun_step_with_err` 获得 `(z_next, x_pred, err)`
2. 若 `err <= tol * h` 或 `h < 1e-6` 或 `max_depth <= 0`：接受步
3. 否则：在 `t_mid = (t+t_next)/2` 处递归细分

`tol` 越小，t→1 附近细分越密。`max_depth` 防止无限递归。

#### `_exp_step` (指数积分)

```python
def _exp_step(model, z, t, t_next, x_pred_prev, ...):
```

利用 Flow Matching 的半线性形式解析求解线性部分：
```python
ratio = (1 - t_next) / (1 - t)
z_next = x0_pred + (z - x0_pred) * ratio
```

当 `t_next = 1` 时 `z_next = x0_pred`，精确消除奇异性。成本：1 NFE/步。

#### `_exp_heun_step` (2 阶指数积分)

```python
def _exp_heun_step(model, z, t, t_next, x_pred_prev, ...):
```

中点校正：
1. `x0_1 = model(z, t)` → 指数半步到 `t_mid`
2. `x0_2 = model(z_mid, t_mid)` → 指数全步到 `t_next`

成本：2 NFE/步。类似于 DPM-Solver-2。

#### `_pc_step` (Predictor-Corrector)

```python
def _pc_step(..., gamma, generator):
```

1. Predictor：`_exp_step` 确定性步（1 NFE）
2. Corrector：加噪回退 + 指数积分回 `t_next`（1 NFE）

成本：2 NFE/步。`gamma` 控制校正噪声强度（通常 0.3-0.5，远小于纯 SDE）。

#### `_sde_ml_step` (Multi-Langevin SDE)

```python
def _sde_ml_step(..., gamma, generator, num_langevin=2):
```

每步 `num_langevin` 轮独立的 {加噪回退 → 指数积分去噪}。成本：`num_langevin` NFE/步。

⚠️ 已知问题：k≥2 导致 mode collapse。

### 1.4 所有步函数共享的调用约定

所有步函数通过 `_forward_sample` 调用模型，后者内部调用 `_forward_sample_self_cond` 处理自条件与 CFG。步函数从 `**step_kwargs` 接收以下参数：

```python
step_kwargs = dict(
    model=model, config=config,
    cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
    cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    sc_noise_scale=sc_noise_scale,    # 若设置
)
```

**添加新步函数只需上述签名，并从 `**step_kwargs` 取用共享参数。** 见第 6 节"添加新采样方法指南"。

---

## 文件二：`src/utils/generation_utils.py`

### 2.1 导入修改

原版：`from utils.sampling_utils import restore_cond, _ode_step, _sde_step`

当前：
```python
from utils.sampling_utils import (
    restore_cond, _ode_step, _heun_step, _heun_adaptive_step,
    _exp_step, _exp_heun_step, _pc_step, _sde_ml_step, _sde_step
)
```

`_sde_step` 是原版已有的（保留不变），`sde_anneal` 方法直接调用 `_sde_step` 但传入随时间变化的 gamma 值——因此不需要新增步函数。

### 2.2 方法分派 (dispatch)

位置：`_generate_samples_single_batch` 函数内部 for 循环（约第 92 行）。

原版只有 `sde` 和 `ode` 两个分支。现已扩展为：

```python
if method == "sde":
    z, x_pred = _sde_step(..., gamma=sde_gamma, generator=generator, **step_kwargs)
elif method == "sde_anneal":
    # gamma 随时间退火
    gamma_t = gamma_end + (gamma_start - gamma_end) * (1 - t) ** anneal_power
    z, x_pred = _sde_step(..., gamma=gamma_t, generator=generator, **step_kwargs)
elif method == "sde_ml":
    num_langevin = int(getattr(sampling_config, 'num_langevin', 2))
    z, x_pred = _sde_ml_step(..., gamma=sde_gamma, generator=generator,
                              num_langevin=num_langevin, **step_kwargs)
elif method == "ode":
    z, x_pred = _ode_step(...)
elif method == "heun":
    z, x_pred = _heun_step(...)
elif method == "heun_adaptive":
    tol = float(getattr(sampling_config, 'heun_tol', 0.05))
    z, x_pred = _heun_adaptive_step(..., tol=tol, **step_kwargs)
elif method == "exp":
    z, x_pred = _exp_step(...)
elif method == "exp_heun":
    z, x_pred = _exp_heun_step(...)
elif method == "pc":
    z, x_pred = _pc_step(..., gamma=sde_gamma, generator=generator, **step_kwargs)
elif method == "sde_exp":
    threshold = float(getattr(sampling_config, 'sde_exp_threshold', 0.8))
    if t < threshold:
        z, x_pred = _sde_step(...)
    else:
        z, x_pred = _exp_step(...)
```

**最后一步（t→1）单独处理**：

```python
if method in ("sde_exp", "pc", "sde_ml", "heun_adaptive", "sde_anneal"):
    z, x_pred = _exp_step(...)   # 指数积分处理 t=1 奇异性
else:
    z, x_pred = _ode_step(...)   # 原版行为
```

### 2.3 z 空间噪声注入（每步后）

在分派块的末尾、循环步结束前：

```python
z_noise_scale = float(getattr(sampling_config, 'z_noise_scale', 0.0))
if z_noise_scale > 0:
    decay = (1.0 - float(t_next)) ** 2
    z = z + z_noise_scale * decay * torch.randn_like(z)
    z = restore_cond(z, cond_seq, cond_seq_mask)
```

此代码对所有采样方法生效（通过 `z_noise_scale` 参数控制）。默认值 0 表示不注入噪声，保持向后兼容。

### 2.4 `step_kwargs` 扩展

```python
sc_noise_scale = float(getattr(sampling_config, 'sc_noise_scale', 0.0))
step_kwargs = dict(
    model=model, config=config,
    cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
    cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    sc_noise_scale=sc_noise_scale,       # ← 新增
)
```

### 2.5 `_dlm_decode_batch` 修改

新增参数（都有默认值，保持向后兼容）：

| 参数 | 类型 | 默认值 | 功能 |
|------|------|--------|------|
| `decode_temperature` | float | 0.0 | >0 时启用温度采样，T>1 增加多样性 |
| `decode_top_k` | int | 0 | >0 时限制采样到前 k 个 token |
| `repetition_penalty` | float | 1.0 | >1.0 时惩罚重复 token |
| `latent_noise_scale` | float | 0.0 | >0 时在解码前对 latent z 加高斯噪声 |

`latent_noise_scale` 的注入发生在解码器前向之前：
```python
if latent_noise_scale > 0:
    z = z + latent_noise_scale * torch.randn_like(z)
```

### 2.6 `_build_run_name` 修改

函数签名扩展为：
```python
def _build_run_name(sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
                    time_schedule, sde_gamma, suffix, num_langevin=None, heun_tol=None,
                    decode_temperature=None, repetition_penalty=None,
                    latent_noise_scale=None, sc_noise_scale=None, z_noise_scale=None,
                    sde_gamma_end=None, sde_anneal_power=None):
```

命名规则（按方法附加不同后缀）：

| 条件 | 后缀 | 示例 |
|------|------|------|
| `heun_adaptive` + `heun_tol` | `-tol{val}` | `-tol0.05` |
| `decode_temperature > 0` | `-T{val}` | `-T0.8` |
| `repetition_penalty != 1.0` | `-rp{val}` | `-rp1.2` |
| `latent_noise_scale > 0` | `-lns{val}` | `-lns0.1` |
| `sc_noise_scale > 0` | `-scn{val}` | `-scn0.1` |
| `z_noise_scale > 0` | `-zn{val}` | `-zn0.1` |
| `sde_anneal` + `sde_gamma_end` | `-gamma{start}to{end}` | `-gamma1.5to0.0` |
| `sde_anneal` + `sde_anneal_power != 1.0` | `p{val}` | `p2.0` |
| `sde_ml` + `num_langevin` | `-m{val}` | `-m2` |

---

## 文件三：`src/generation.py`

### 3.1 解码参数提取（两处：uncond 和 cond）

在 `_dlm_decode_batch` 调用处：
```python
decode_temp = float(getattr(sampling_config, 'decode_temperature', 0.0))
decode_topk = int(getattr(sampling_config, 'decode_top_k', 0))
decode_rp   = float(getattr(sampling_config, 'repetition_penalty', 1.0))
decode_lns  = float(getattr(sampling_config, 'latent_noise_scale', 0.0))
predicted_ids = _dlm_decode_batch(
    ..., decode_temperature=decode_temp, decode_top_k=decode_topk,
    repetition_penalty=decode_rp, latent_noise_scale=decode_lns,
)
```

### 3.2 `_build_run_name` 调用参数提取（两处：uncond 和 cond）

```python
num_langevin = getattr(sampling_config, "num_langevin", None)
heun_tol     = getattr(sampling_config, "heun_tol", None)
decode_temp  = getattr(sampling_config, "decode_temperature", None)
decode_rp    = getattr(sampling_config, "repetition_penalty", None)
decode_lns   = getattr(sampling_config, "latent_noise_scale", None)
decode_scn   = getattr(sampling_config, "sc_noise_scale", None)
decode_zn    = getattr(sampling_config, "z_noise_scale", None)
gamma_end    = getattr(sampling_config, "sde_gamma_end", None)
anneal_p     = getattr(sampling_config, "sde_anneal_power", None)
name = _build_run_name(
    ..., num_langevin=num_langevin, heun_tol=heun_tol,
    decode_temperature=decode_temp, repetition_penalty=decode_rp,
    latent_noise_scale=decode_lns, sc_noise_scale=decode_scn,
    z_noise_scale=decode_zn, sde_gamma_end=gamma_end,
    sde_anneal_power=anneal_p,
)
```

**注意**：uncond 和 cond 两处代码必须保持一致。本文件中每处均通过 `replace_all=true` 同步修改。

---

---
## 文件四：`src/utils/data_utils.py`

仅修改 `load_dataset_split` 函数（约 10 行）。原版的离线回退逻辑在 HF datasets 库离线模式下会失败。

**修改前**：
```python
try:
    ds = hf_load_dataset(path, ...)
except Exception:
    ds = load_from_disk(path)    # 若 path 不是本地路径，函数立即失败
```

**修改后**：当 `hf_load_dataset` 和直接 `load_from_disk(path)` 都失败（且 ds 为 None 或 look-like-save_to_disk-arrow）时，自动调用 `snapshot_download` 获取缓存路径，然后用 `load_from_disk` 从缓存加载。

**影响**：配合 `export HF_DATASETS_OFFLINE=1` 环境变量，可以在无网络环境下加载 XSum 验证集和 WMT 测试集等条件生成数据集。

**注意**：原版 `data_utils.py` 在无条件生成（`load_jsonl_dataset`）路径上无需修改——只有条件生成的数据集加载受影响。

---

## 文件五：新增配置文件与评估脚本

### 4.1 采样配置文件（`src/configs/sampling_configs/`）

| 文件名 | 包含的方法 | 用途 |
|--------|----------|------|
| `heun_sampling_configs.yml` | heun 16/32/64 步 | 纯速度场 RK2 |
| `exp_heun_sampling_configs.yml` | exp 16/32, exp_heun 16/32 | 指数积分 |
| `pc_sampling_configs.yml` | pc 8/16/32, 多种 gamma | Predictor-Corrector |
| `sde_exp_sampling_configs.yml` | sde_exp 32/64, 多种阈值 | SDE-Exp 混合 |
| `sde_ml_sampling_configs.yml` | sde_ml 16(k=2,3)/32(k=2) | 多轮 Langevin |
| `heun_adaptive_sampling_configs.yml` | heun_adaptive tol=0.1/0.05/0.02 + 解码约束 | 自适应步长 |
| `sc_noise_sampling_configs.yml` | heun_adaptive + sc_noise 0.05-4.0 | 自条件噪声 (无效) |
| `z_noise_sampling_configs.yml` | heun_adaptive + z_noise | z 空间噪声 (无效) |
| `sde_anneal_sampling_configs.yml` | sde_anneal 32/64/256, gamma=1.0-2.0, p=1-2 | SDE 退火 (✅ 最优) |
| `xsum_sampling_configs.yml` | ode/sde/sde_anneal 64步, CFG=2 | XSum 摘要 |
| `xsum_deterministic.yml` | heun/exp 64步, CFG=2 | XSum 确定性求解器 |
| `xsum_fast.yml` | sde_anneal 16/32步, CFG=2 | XSum 快速 |
| `wmt_sampling_configs.yml` | ode + sde_anneal 32/64步, CFG=2 | WMT 翻译 |
| `elf_l_sampling_configs.yml` | sde + sde_anneal 32/64步 | ELF-L 无条件 |

### 4.2 评估脚本（`scripts/`）

| 脚本 | 对应的采样配置 | 输出目录 |
|------|--------------|---------|
| `eval_heun.sh` | `heun_sampling_configs.yml` | `outputs/elf_b-owt_heun_eval` |
| `eval_exp_heun.sh` | `exp_heun_sampling_configs.yml` | `outputs/elf_b-owt_exp_heun_eval` |
| `eval_pc.sh` | `pc_sampling_configs.yml` | `outputs/elf_b-owt_pc_eval` |
| `eval_sde_exp.sh` | `sde_exp_sampling_configs.yml` | `outputs/elf_b-owt_sde_exp_eval` |
| `eval_heun_adaptive.sh` | `heun_adaptive_sampling_configs.yml` | `outputs/elf_b-owt_heun_adaptive_leCun_eval` |
| `eval_sc_noise.sh` | 最终指向 `sde_anneal_sampling_configs.yml` | `outputs/elf_b-owt_sde_anneal_v2_eval` |
| `eval_xsum.sh` | `xsum_sampling_configs.yml` (多次修改) | `outputs/elf_b-xsum_eval` |
| `eval_elf_l.sh` | `elf_l_sampling_configs.yml` | `outputs/elf_l-owt_eval` |

**直接命令行调用**（无需单独脚本）：
- XSum 确定性: `NGPU=4 ... --sampling_configs_path xsum_deterministic.yml`
- XSum 快速: `NGPU=4 ... --sampling_configs_path xsum_fast.yml`
- WMT: `NGPU=4 MASTER_PORT=29503 ... --sampling_configs_path wmt_sampling_configs.yml`

所有脚本的通用模式：
```bash
NGPU="$NGPU" bash scripts/launch.sh eval "$TRAIN_CONFIG" \
    --checkpoint_path "$CHECKPOINT" \
    --seeds "$SEEDS" \
    --config_override "sampling_configs_path=$SAMPLING_CONFIG" \
    --config_override "use_bf16=true" \
    --config_override "use_compile=true" \
    ...
```

关键 override：
- `sampling_configs_path` — 指向对应的 YAML 配置文件
- `global_batch_size` — 根据 NGPU * BATCH_SIZE 计算
- `use_bf16=true` — 启用 bf16 混合精度
- `use_compile=true` — 启用 torch.compile
- `online_eval=true` — 生成后立即计算 PPL

---

## 6. 添加新采样方法指南

以下是给新 agent 的分步指南。

### 步骤 1：在 `sampling_utils.py` 中添加步函数

```python
def _my_new_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask,
    # 你的额外参数（通过 **step_kwargs 传入或直接在分派中指定）
):
    h = float(t_next - t)
    t_batch = torch.full((z.shape[0],), float(t), dtype=z.dtype, device=z.device)

    # 调用模型（_forward_sample 处理 CFG + 自条件）
    v_pred, x_pred = _forward_sample(
        model=model, z=z, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
        # sc_noise_scale 也会从 **step_kwargs 透传
    )

    # 你的更新公式
    z_next = z + h * v_pred   # 示例
    return restore_cond(z_next, cond_seq, cond_seq_mask), x_pred
```

**关键规则**：
- 签名中必须包含 `model, z, t, t_next, x_pred_prev, config, cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask`
- 返回 `(z_next, x_pred)`，两者都要经过 `restore_cond` 恢复条件 token
- x_pred 用于下一步的自条件输入
- 额外参数通过 `**step_kwargs` 传播（自动）

### 步骤 2：在 `generation_utils.py` 中添加分派

**2a**：在文件顶部添加导入：
```python
from utils.sampling_utils import ..., _my_new_step
```

**2b**：在 `_generate_samples_single_batch` 的 for 循环中添加 elif 分支：
```python
elif method == "my_new":
    my_param = float(getattr(sampling_config, 'my_param', 0.5))
    z, x_pred = _my_new_step(
        z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
        my_param=my_param, **step_kwargs,
    )
```

**2c**：（可选）如果方法需要在 t→1 时使用 exp：
```python
if method in ("sde_exp", "pc", "sde_ml", "heun_adaptive", "sde_anneal", "my_new"):
    z, x_pred = _exp_step(...)
```

### 步骤 3：在 `generation.py` 中更新 `_build_run_name` 调用（两处）

如果需要新参数出现在 run name 中：
```python
# 提取参数
my_param = getattr(sampling_config, "my_param", None)
name = _build_run_name(
    ..., my_param=my_param,
)
```

### 步骤 4：更新 `_build_run_name`（如需要）

在 `generation_utils.py` 的 `_build_run_name` 中添加对应的命名逻辑。

### 步骤 5：创建采样配置文件

在 `src/configs/sampling_configs/` 下创建 YAML：
```yaml
- sampling_method: my_new
  num_sampling_steps: [32]
  cfgs: [1]
  self_cond_cfg_scales: [3]
  my_param: 0.5
  time_schedule: logit_normal
```

### 步骤 6：创建或更新评估脚本

可选：创建新的 `scripts/eval_my_new.sh` 或修改已有脚本的 `SAMPLING_CONFIG` 和 `OUTPUT_DIR`。

### 参数传递链总结

```
YAML 配置 (my_param: 0.5)
  ↓ getattr(sampling_config, 'my_param', ...)
_generate_samples_single_batch (分派)
  ↓ 传递给步函数
_my_new_step
  ↓ **step_kwargs
_forward_sample → _forward_sample_self_cond → model(...)
```

---

## 7. 已验证的最优配置

以下配置经过完整评估，跨三个任务（无条件生成、XSum 摘要、WMT 翻译）和两个模型规模（ELF-B、ELF-L）。

### 无条件生成

```yaml
# 最优 PPL（64 NFE, ELF-B）
- sampling_method: sde_anneal
  num_sampling_steps: [64]
  cfgs: [1]
  self_cond_cfg_scales: [3]
  sde_gamma: 2.0
  sde_gamma_end: 0.0
  time_schedule: logit_normal

# 省算力（32 NFE, 仍优于原版 SDE 32 步）
- sampling_method: sde_anneal
  num_sampling_steps: [32]
  cfgs: [1]
  self_cond_cfg_scales: [3]
  sde_gamma: 2.0
  sde_gamma_end: 0.0
  time_schedule: logit_normal

# 最优 Entropy（64 NFE）
- sampling_method: sde_anneal
  num_sampling_steps: [64]
  cfgs: [1]
  self_cond_cfg_scales: [3]
  sde_gamma: 1.5
  sde_gamma_end: 0.0
  sde_anneal_power: 2.0
  time_schedule: logit_normal
```

### 条件生成

```yaml
# XSum 摘要：半算力（32 NFE, 98% 基线质量）
- sampling_method: sde_anneal
  num_sampling_steps: [32]
  cfgs: [2]
  self_cond_cfg_scales: [1]
  sde_gamma: 2.0
  sde_gamma_end: 0.0
  time_schedule: logit_normal

# WMT 翻译：半算力反超基线（32 NFE, BLEU +3.5% vs ODE 64 NFE）
- sampling_method: sde_anneal
  num_sampling_steps: [32]
  cfgs: [2]
  self_cond_cfg_scales: [1]
  sde_gamma: 2.0
  sde_gamma_end: 0.0
  time_schedule: logit_normal
```

### 跨模型规模

ELF-L (652M) 使用相同配置，无条件生成 PPL 提升幅度约 -7%（β=3 epoch，去噪网络不及 ELF-B 成熟）。更好的训练应能放大 sde_anneal 的优势。

---

## 8. 未修改的文件

以下原版文件**完全未改动**（不含 `__pycache__`）：

- `src/modules/model.py`
- `src/modules/layers.py`
- `src/modules/t5_encoder.py`
- `src/train.py`
- `src/train_step.py`
- `src/eval.py`
- `src/configs/config.py`
- `src/utils/checkpoint_utils.py`
- `src/utils/encoder_utils.py`
- `src/utils/logging_utils.py`
- `src/utils/metrics_utils.py`
- `src/utils/muon_utils.py`
- `src/utils/train_utils.py`
- `scripts/launch.sh`

`src/utils/data_utils.py` 有少量修改（离线数据集加载回退逻辑），详见第四节。

这些文件无需修改即可与原版行为完全一致——所有新功能通过配置文件和新增的步函数实现，通过 `sampling_method` 字段选择。
