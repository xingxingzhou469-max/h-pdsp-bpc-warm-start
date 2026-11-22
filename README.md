# 单车 H-PDSP V8 与 BPC 初始解

在同一张取送网络上，把已有单车路线交给一辆车，其余车辆暂时闲置，就得到多车 BPC 可以直接使用的第一个可行解。本仓库提供 **60 个实例各一份最终路线与逐站取送量**、生成新路线的 V8 源码，以及把路线加入列池并向 HiGHS 整数主问题提交初解的示例。

## 直接使用已有结果

在仓库根目录运行下面的命令。读取和验证现成结果只需 Python 标准库。

```bash
python3 warm_start.py instances/random-20-1.json solutions/random-20-1.json --vehicles 2
```

例中第 1 辆车的路线是 `0 → 2 → 15 → 4 → 16 → 14 → 0`，第 2 辆车闲置。逐站取送量依次为 `182, 12282/107, 85, 9078/107, 7209/107`。含服务及回仓的总时间为 **66**，车队初始目标为 **28569/458**。输出 JSON 中的 `routes` 和 `amounts` 每行对应一辆实际使用的车，省略首尾仓库。整数和分数字符串保留原始精度。

程序核对输入文件的 SHA-256 和实例 ID，再逐站复算有向路时、服务时间、取送量、供给和车载库存，最后按全体机构人口核对车队目标。传入生成该解时的原始 JSON 即可。车数通过 `--vehicles` 单独指定。如果换了路网、车辆容量或时限，可用下节的 V8 命令为目标实例生成新的初解。

## 交给 BPC 继续求解

以下命令使用 Python 3.13。首次运行 BPC 时安装依赖。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python warm_start.py instances/random-20-1.json solutions/random-20-1.json --vehicles 2 --run-seconds 10
```

`initial` 是读入的单车可行解，`final` 是结束时重新验算的车队解。BPC 将给定路线登记为初始列并保留其目标值。搜索或下探找到更好的车队时，整数主问题会收到更新后的整套初解，其中包含选列变量、机构配送量和公平性差额。`mip_start.routes` 记录实际提交的路线，`mip_start.accepted` 记录 HiGHS 的可行性判定。若时间在进入整数主问题前用尽，这一项为 `null`。

接自己的 BPC 时，可以直接取出已验路线和数量。以下示例调用本仓库的定价与下探入口。

```python
from warm_start import load_start
from mvp.bpc import BpcSolver

instance, start = load_start("instances/random-20-1.json",
                             "solutions/random-20-1.json", vehicles=2)
routes, value, record = BpcSolver(instance, 2).solve(
    initial_routes=start["routes"], seconds=10)
print(record["seed"]["column_indices"], record["best_value_exact"])
```

示例采用同仓同质车、至多使用 K 辆、每车有独立时限、全车队每站最多访问一次的模型。增加路线后，要把各机构收到的配送量合并，再计算全车队公平目标。例如两家等人口机构各收到 1 单位时，两条路线分别计分各为 1/2，合并后的车队目标为 2。单车最优性记录只适用于该单车问题。示例返回已验可行下界，受限主问题的求解状态只对应当前列池。

## 在新实例上运行 V8

单车入口使用研究阶段最终封存的 **V8**。首次运行需具备支持 C++20 的编译器，构建一次即可。预存的 60 份初解可以直接导入。

```bash
python single_vehicle/build_native.py
python solve_single.py instances/random-20-1.json --seconds 60 --output my-solution.json
python warm_start.py instances/random-20-1.json my-solution.json --vehicles 2
```

将两条命令的实例路径替换为自己的规范 JSON。`travel[i][j]` 是有向弧时，`service`、`supply`、`population` 使用同一节点编号，仓库为 0。`capacity` 为车容，`limit` 为单条路线含服务及回仓的时限。可参考随包的任一实例。输出 `feasible` 表示在预算内找到且验过一条路线。`no_witness` 表示未取得可核验的非空初解。搜索按墙钟给定预算，初始化或一次原生调用可能使程序稍晚退出。[封存来源及交付源码指纹](single_vehicle/SOURCE.json)

## 独立复验单车上界与最优性

在仓库根目录运行下列命令。第一条检查全部 60 例，第二条只检查一个已闭合和一个未闭合的实例。复验使用 Python 3.13 标准库和支持 C++20 的编译器，部分算例会在临时目录编译独立的精确检查程序。整个流程直接读随包实例、解和上界材料，无需安装 HiGHS，也无需启动 V8。

```bash
python3 -B -S verify_bounds.py
python3 -B -S verify_bounds.py blind-01 random-60-6
```

每行 JSON 都给出精确分数 `lower_exact` 和 `upper_exact`。下界来自对给定路线的零容差复算，上界来自该例对应的独立证明，两者相等才输出 `strict_optimal`。例如 `blind-01` 的两端均为 `63875/139`；`random-60-6` 的区间为 `[249553/819, 252650/819]`。全量复验得到 **52 例相等、8 例仍有正差距**。[逐例上界及验证入口](index.csv)

可用的上界证据有不同形式。整数对偶证书保存了分支叶、割和乘子，验证器从原始模型逐叶重建并检查树的完整性。部分已闭合的实例使用供给上界或时间树即可证明，其余使用独立的访问集合、供点分区或反向路线检查程序重新覆盖所有可能改善的分支。这些计算需要一定时间。8 个开放实例均附有可重放的整数证书。[归档对偶证书及来源](proofs/dual/sources.json) · [8 个开放实例的证书](proofs/open_bounds/README.md)

对于新实例，可以在求解时另存本次运行的上界材料，随后在独立进程复核。

```bash
python solve_single.py instances/random-20-1.json --seconds 10 --output route.json --proof-output proof.json
python -B -S solve_single.py instances/random-20-1.json --verify-proof proof.json
```

`replayable` 记录能由文件独立验出的上界，包括解析先验及本次取得的整数对偶证书。`trusted_search` 单列精确搜索内核报告的全局界。搜索内核可能已经闭合，而本次导出的独立证据仍有间隙；复验命令只按可重放部分判定。多车 BPC 示例的结果仍按可行解解读。

## 数据和验收

48 个公开算例来自 Eisenhandler 与 Tzur 的 H-PDSP 数据，12 个是本项目的历史盲测算例。每例只收录一份已验证的路线。52 份达到历史严格最优值，另外 8 份保留已核验的最好可行值。记录来自多次独立运行，其中 55 份选自 V8、4 份选自独立研究路线、1 份选自 C1 长时运行。索引记录实例文件、解文件、输入和解的指纹，以及研究归档中的原件位置。公开算例使用随包规范 JSON 的数值口径。[60 例索引](index.csv) · [作者数据页](https://ohadeis.wixsite.com/ohadeis/research) · [论文](https://doi.org/10.1287/opre.2018.1751)

| 检查 | 结果 |
| --- | --- |
| 60 份最终路线 | 单车及车队验算器以零容差得到相同目标。固定路线重新配量后，60 份逐站数量也完全一致。 |
| 60 次整数主问题初解 | 逐例设 K=2，仅放入原路线。HiGHS 对 60 个完整起点均返回可行，回读路线和目标与输入一致。 |
| 单车小图穷举 | 38 个原有小图和 120 个固定种子新图共枚举 1,713 条可行路线，逐路线独立线性规划的精确上下界一致。V8 在 108 个有解图上得到穷举最优值，50 个无解图均返回无初解。新图种子为 20261122。 |
| 接入回归 | 覆盖超时路线、站点冲突、超出车数、目标值篡改、精确小数时限、定价对偶符号、HiGHS 接收完整初解，以及双车改进。 |

启用原生搜索组件后，原单车工程的 288 项测试全部通过。随包回归覆盖初解、整数主问题接入和单车上界导出。

本次修正了接入示例对非法初解的处理，并把显式整数起点送入 HiGHS。修正了旧构造定价从单个供点出发时无法生成首列的问题，以及二进制浮点舍入会漏掉恰好按时回仓路线的问题。后一个回归使用三段各 0.1、时限 0.3 的有向路线。原生构建与完整接入流程已在 Apple Silicon 的 macOS 上实测。构建脚本另含 Linux 分支。

安装上述依赖并完成原生构建后，可重跑随包验收。其中一项会把 60 个初解逐个提交给真实的 HiGHS 主问题。

```bash
python -m unittest discover -s tests -v
```
