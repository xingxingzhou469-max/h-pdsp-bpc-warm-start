# Latet 单车上界

运行 `python3 proofs/latet/replay.py` 可以核对本目录的 9 例。`certificates.json` 记录输入指纹、有理数门槛和各自的核验方法。验证器从公开实例重建时间、供给和人口，按供点访问集合与首个供点覆盖每条可能的路线。完整的上下界配对使用仓库根目录的 `verify_bounds.py`。

- 七个算例的总供给不超过车容。`nonbinding.cpp` 用整数尾程人口上界排除分支，再检查仍有机会超过门槛的路线前缀。
- 在 `latet-2508` 的两供点模型中，车容可能限制分配。`two_supplier.cpp` 还按第二次取货前后的配送组计算可达的公平分数，计入沿途载重限制。
- 对于 `latet-2808`，证书保留了原运行的整数价格和跟踪站点。`priced.cpp` 重建尾程表，并对各供点组合覆盖余下的前缀。只要有一个终点的上界仍可能超过门槛，检查即报失败。
- 第十例 `latet-2708` 使用单列的[五供点验证器](../latet2708/README.md)。

两个通用检查程序取自研究归档的 `runs/overnight-20261119/latet-multiple-independent/` 及其 `v3-additions/`。2808 的价格取自 `runs/next-stage-20261119/upper-r01/from-raw.json`，检查程序来自同轮的 `latet-audit-r01/check.cpp`。上界的适用对象是随包规范 JSON 的原始字节。
