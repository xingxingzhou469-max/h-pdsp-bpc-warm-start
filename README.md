# H-PDSP 精确上下界求解器与独立证明复验仓库（v2.1 多车 + v1.1 单车）

本仓库提供人道主义取送问题（Humanitarian Pickup and Distribution Problem, H-PDSP；Eisenhandler & Tzur, *Operations Research* 2019 / *Transportation Science* 2019）的**完整可复现求解代码、修正论文对照基线、已封板实验结果表、全部端点可行见证与严格上界证书**，以及零第三方依赖的独立精确复验器。

所有下界 $L$ 与严格上界 $U$ 均以 Python `fractions.Fraction` 零容差核验；纯标准库复验器（`python3 -B -S`）在不加载 HiGHS、NumPy、SciPy 或 C++ 求解器的前提下，可从原始输入 JSON 独立重放全部严格证明并确认 $L \le U$。

---

## 目录导航

- **多车 H-PDSP `v2.1`（本版本新增主线）**
  - [多车核心结果汇总](#一多车-h-pdsp-v21-核心实验结果) · [多车快速复验与求解](#二多车快速开始复验--求解) · [论文基线对照](#三同条件论文方法对照25-例正式面板)
  - [数学模型、四类严格上界证明与学术边界](multi_vehicle/docs/MODEL_AND_PROOFS.md) · [论文对照基线实现说明](multi_vehicle/baselines/README.md) · [多车源码指纹清单](multi_vehicle/SOURCE.json)
- **单车 H-PDSP `V8` 与 BPC 初解（完整保留 `v1.1`）**
  - [单车 60 例精确上下界、V8 求解与 BPC 接入](#四单车-h-pdsp-v8-与-bpc-初解保留-v11)

---

## 一、多车 H-PDSP `v2.1` 核心实验结果

### 1. Stage B 独立盲测（96 实例，Apple M4，单线程 60 秒冷启动）

覆盖 4 组独立盲测规模（$n \in \{19, 40, 60, 100\}$）$\times$ 3 种车队规模（$K \in \{2, 3, 4\}$），每单元 8 例共 96 例。四臂（`polygon_v1`、`joint_v2.1`、`bare_v2.1`、`upgrade_v2.1`）各自从空车队冷启动 60 秒。

- **严格 $\le 5\%$ 认证（同实例共同 $L$）**：**`v1` 57/96 $\rightarrow$ `v2.1` 78/96**（新增 **+21**、丢失 **0**；仅用 `upgrade` 下界同样为 **78/96**）。
- **严格证优（$L = U$，精确零差距）**：**25/96**（其中 $n=19$ 全组 **24/24 全部严格闭合**，$n=40, K=2$ 闭合 1/8）。
- **上界单调性（`joint_v2.1` vs `polygon_v1`）**：**68 胜 / 28 平 / 0 负**（最差退化 `0.000%`）。
- **找解对比（`upgrade_v2.1` vs `bare_v2.1`）**：**95/96 胜或平**（唯一微负 `-0.0055%`，仍处于 `0.19%` 差距内）。
- **Stage B 预登记门槛 G1–G5 全部通过**，全部 96 例严格区间满足 $L \le U$，0 冲突。

| 站点数 $n$ | 车辆数 $K$ | 例数 | $U$ 降 / 平 / 升 | $U$ 降幅中位数 | 严格 $\le 5\%$（`v1` $\to$ `v2.1`） | `v2.1` 严格证优 |
| ---: | ---: | ---: | :---: | ---: | :---: | ---: |
| 19 | 2 | 8 | 8 / 0 / 0 | 12.219% | 0/8 $\to$ **8/8** | **8/8** |
| 19 | 3 | 8 | 8 / 0 / 0 | 8.462% | 2/8 $\to$ **8/8** | **8/8** |
| 19 | 4 | 8 | 8 / 0 / 0 | 1.312% | 5/8 $\to$ **8/8** | **8/8** |
| 40 | 2 | 8 | 8 / 0 / 0 | 1.166% | 6/8 $\to$ **8/8** | 1/8 |
| 40 | 3 | 8 | 7 / 1 / 0 | 0.246% | 3/8 $\to$ **4/8** | 0/8 |
| 40 | 4 | 8 | 0 / 8 / 0 | 0.000% | 7/8 $\to$ **7/8** | 0/8 |
| 60 | 2 | 8 | 8 / 0 / 0 | 0.653% | 7/8 $\to$ **7/8** | 0/8 |
| 60 | 3 | 8 | 4 / 4 / 0 | 0.019% | 8/8 $\to$ **8/8** | 0/8 |
| 60 | 4 | 8 | 1 / 7 / 0 | 0.000% | 8/8 $\to$ **8/8** | 0/8 |
| 100 | 2 | 8 | 8 / 0 / 0 | 1.750% | 4/8 $\to$ **4/8** | 0/8 |
| 100 | 3 | 8 | 5 / 3 / 0 | 1.249% | 3/8 $\to$ **4/8** | 0/8 |
| 100 | 4 | 8 | 3 / 5 / 0 | 0.000% | 4/8 $\to$ **4/8** | 0/8 |
| **合计** | — | **96** | **68 / 28 / 0** | — | **57/96 $\to$ 78/96** | **25/96** |

逐例精确分数与证书索引：[stage_b_96_blind.csv](multi_vehicle/results/stage_b_96_blind.csv)。

### 2. 真实 Latet 案例（20 配置，Apple M4，单线程 60 秒冷启动）

10 份以色列 Latet 原始工作簿（逐例核对 SHA-256 与论文印刷供给 10/10 一致）$\times$ $K \in \{2, 3\}$，$T=540$：

- **$K=2$（原论文双车口径）**：严格 $\le 5\%$ 认证 **10/10**，最差精确差距 **3.639%**。
- **$K=3$（本研究多车扩展）**：严格 $\le 5\%$ 认证 **9/10**（`v1` 为 6/10），最差精确差距 **5.537%**（`latet-2209-k3`；在 Kaggle CPU 3600 秒长预算下收敛至 `5.255%`）。
- **合计**：60 秒单次冷启动下 **19/20 严格 $\le 5\%$**。逐例表：[stage_b_20_latet.csv](multi_vehicle/results/stage_b_20_latet.csv)。

### 3. 跨硬件补充实验（Kaggle CPU，单列汇报）

为避免跨硬件混报，云端补充结果单独归档于 [cloud_supplementary.csv](multi_vehicle/results/cloud_supplementary.csv)：
- **0.5–2 小时难例压界（9 臂）**：4/9 达到严格 $\le 5\%$（包括 `r4-lgA-100-k2` 3.624%、`r2-lgA-100-k4` 4.856%）。
- **多种子稳定性（6 例 $\times$ 10 种子 = 60 臂，60 秒）**：严格上界 $U$ 跨种子方差极小，60 秒差距波动主要来自找解下界 $L$。
- **3600 秒长预算基线对照（12 臂）**：在长预算下 `v2.1` 仍保持完整严格上下界；该早期云端批中 `mv_lns` 使用修正前抽样，已在表中明确标注。

---

## 二、多车快速开始（复验 & 求解）

### 1. 零依赖独立复验已归档的全部多车证明（仅需 Python 3 标准库）

无需安装 `numpy`、`scipy`、`highspy`，也无需编译 C++：

```bash
# 快速抽样复验（覆盖 19/40/60/100 站、Latet、4 类严格上界机制与多法交叉检查，实测约 30 秒）
python3 -B -S verify_multi.py --sample

# 全量复验：Stage B 96 盲图 + 20 Latet + 25 例正式基线面板（Apple M4 实测约 6.8 分钟）
# 共 141 个严格区间、216 份可行见证 L、141 份严格上界证书 U，强制校验 0 个求解器模块加载
python3 -B -S verify_multi.py
```

### 2. 编译多车原生扩展并在新实例上求解

首次运行求解器需安装依赖（Python 3.13+ 及 C++17 编译器）并编译两份 C++ 扩展（`rebuild.cpp` 与 `price.cpp`）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 编译多车 v2.1 原生扩展（输出至 multi_vehicle/build/）
python3 multi_vehicle/build_native.py

# 使用 v2.1 求解（默认前半预算找解 L，后半预算求严格上界 U）并导出独立证明
python3 solve_multi.py multi_vehicle/data/formal_panel/strB-19-k2.json \
  --seconds 10 --output my-mv-solution.json --proof-output my-mv-proof.json

# 用纯标准库进程独立复验刚生成的解与严格上界证书
python3 -B -S verify_multi.py --single multi_vehicle/data/formal_panel/strB-19-k2.json \
  --solution my-mv-solution.json --proof my-mv-proof.json

# 同入口运行论文对照基线（site / mv_lns / segment）
python3 solve_multi.py multi_vehicle/data/formal_panel/strB-19-k2.json \
  --method mv_lns --seconds 10 --output my-mv-lns.json
```

---

## 三、同条件论文方法对照（25 例正式面板）

在 Apple M4 单线程、种子 7、每臂 60 秒冷启动下，将冻结 `v2.1` 与三组论文方法在 25 个实例（13 个既有难例含 4 个 Latet + 12 个冻结后新生成的 100 站独立盲图）上同台比较。**注意预算分配差异**：`v2.1` 用 30 秒找解 $L$ + 30 秒求严格上界 $U$，三个对手全部 60 秒仅用于找解 / MILP。

| 对手方法（60 秒全找解 / MILP） | `v2.1` 可行值胜 / 平 / 负 | 旧组（13 例） | 新 100 站盲图（12 例） | Latet（4 例） | 相对提升中位数 | 严格上界能力 |
| --- | :---: | :---: | :---: | :---: | ---: | --- |
| `Site-MILP`（HiGHS 三索引） | **25 / 0 / 0** | 13 / 0 / 0 | 12 / 0 / 0 | 4 / 0 / 0 | +10,227%（17/25 硬超时） | 仅浮点数值对偶 `numeric_U` |
| 修正 `MV-LNS`（角色归一化） | **18 / 1 / 6** | 10 / 1 / 2 | 8 / 0 / 4 | **4 / 0 / 0** | +1.531% | 无 |
| 修正 `TS Super-Segment`（60 列受限复现） | **18 / 1 / 6** | 10 / 1 / 2 | 8 / 0 / 4 | **4 / 0 / 0** | +1.536%（3/25 硬超时） | 无（受限池非全局界） |

- `v2.1` 在该 25 例面板上以自身 30 秒下界实现 **13/25 严格 $\le 5\%$**（旧组 7/13、新 100 站组 6/12；`strB-19-k2` 严格证优 `490699/689`），且四个方法的全部 100 份可行下界均满足 $L \le U_{\text{v2.1}}$。
- **诚实边界**：在 6/25 例（尤其是高供给 100 站图）上，使用完整 60 秒找解的修正 `MV-LNS` 取得了略高于 `v2.1`（30 秒找解切片）的可行值；100 站紧时间难例（如 `new-2-01`、`new-3-01`）的 60 秒严格差距仍在 9%–18%，是后续改进方向。正式批中有 61/125 臂与外部系统 CPU 负载样点（$\ge 40\%$）相交，洁净对与相交对的 5% 达标结论一致。
- 逐例 100 臂精确数据与证书：[formal_25_baselines.csv](multi_vehicle/results/formal_25_baselines.csv) · 基线细节：[multi_vehicle/baselines/README.md](multi_vehicle/baselines/README.md)。

---

## 四、单车 H-PDSP `V8` 与 BPC 初解（保留 `v1.1`）

本仓库完整保留单车 `v1.1.0` 的全部数据、源码、独立证明与 BPC 暖启动接入，原有命令与测试均保持兼容：

- **60 个单车实例**（48 个 Eisenhandler & Tzur 公开算例 + 12 个盲测算例）：纯标准库全量复验得到 **52 例严格最优、8 例开放且附可重放整数对偶证书**。[60 例索引](index.csv) · [单车源码指纹](single_vehicle/SOURCE.json) · [8 例开放证书](proofs/open_bounds/README.md)
- **核心命令**：

```bash
# 1. 纯标准库验证单车已有初解并检查车队目标（K=2，单车路线作为初解）
python3 warm_start.py instances/random-20-1.json solutions/random-20-1.json --vehicles 2

# 2. 纯标准库独立复验全部 60 例单车严格上下界（需 C++20 编译器用于部分闭合检查）
python3 -B -S verify_bounds.py

# 3. 编译单车 V8 原生组件并在新实例上求解与导出独立证明
python3 single_vehicle/build_native.py
python3 solve_single.py instances/random-20-1.json --seconds 60 --output my-solution.json --proof-output my-proof.json
python3 -B -S solve_single.py instances/random-20-1.json --verify-proof my-proof.json

# 4. 将单车初解送入多车 BPC（需安装 requirements.txt）
python3 warm_start.py instances/random-20-1.json solutions/random-20-1.json --vehicles 2 --run-seconds 10
```

---

## 五、全套自动化回归测试

完成依赖安装及双引擎原生构建后，可运行全部单车 + 多车自动化回归（覆盖初解验算、HiGHS MIP 初解接入、单车证明导出、多车纯标准库 `-B -S` 重放、篡改拒收、实时 `v2.1` 证优闭环及三组基线冒烟）：

```bash
python3 single_vehicle/build_native.py
python3 multi_vehicle/build_native.py
python3 -m unittest discover -s tests -v
```

## 参考文献

1. Eisenhandler, O., & Tzur, M. (2019). The humanitarian pickup and distribution problem. *Operations Research*, 67(1), 10–32. https://doi.org/10.1287/opre.2018.1751
2. Eisenhandler, O., & Tzur, M. (2019). A segment-based formulation and a matheuristic for the humanitarian pickup and distribution problem. *Transportation Science*, 53(5), 1389–1408. https://doi.org/10.1287/trsc.2019.0916
