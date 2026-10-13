# 多数据集拓扑交互式评估报告

## 方法

本报告使用当前 TCIG checkpoint 在 IXI、Lausanne 和 CROWN 三个外部数据集上进行交互式评估。评估协议采用 largest-error click simulator：每一轮根据当前预测与二值标注之间的最大错误连通区域生成前景或背景点击，然后重新前向推理。每个病例记录 0-click 到最大点击预算内的完整曲线。

实验设置为每个数据集使用全部 20 个本地病例，最大交互预算为 20 次点击。checkpoint、manifest、data root、method 等配置保存在各数据集子目录的 `connectivity_summary.json` 中。

对照方法包括：`tcig` 为完整方法；`prompt_only` 使用同样的点击高斯 prompt，但禁用 IIRE/EPICA/空间散射图上下文；`no_interaction` 完全忽略点击，等价于同一模型的 zero-click 非交互预测。这样可以分别观察交互 prompt 的作用和图式交互建模的增益。

拓扑相关指标包括 Dice、clDice、Betti-0 误差、预测/标注连通分量数、连通分量 precision/recall/F1，以及达到 Dice 或 clDice 阈值所需点击数 NoC。Betti-0 exact NoC 表示第一次达到连通分量数量完全匹配所需的点击数；若未达到，则记为最大点击数加一并在 coverage 中反映失败比例。

## 结果文件

- 跨数据集曲线图：`multidataset_connectivity_curves.png`
- 最终指标柱状图：`multidataset_final_metrics.png`
- TeX 表格：`multidataset_topology_table.tex`
- TeX 图片片段：`multidataset_figures.tex`
- 汇总 CSV：`multidataset_summary.csv`
- 病例级 CSV：`multidataset_cases.csv`

## 汇总结果

| 数据集 | 方法 | 病例数 | Dice | clDice | B0误差 | 分量F1 | Dice NoC@0.90 | clDice NoC@0.90 | B0 exact NoC | B0 exact coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IXI | TCIG | 20 | 0.955 ± 0.012 | 0.976 ± 0.024 | 2.600 ± 2.853 | 0.769 | 1.950 | 4.000 | 16.300 | 0.450 |
| IXI | Prompt-only | 20 | 0.887 ± 0.063 | 0.845 ± 0.100 | 6.950 ± 4.610 | 0.526 | 10.150 | 15.850 | 18.700 | 0.150 |
| IXI | No-interaction | 20 | 0.795 ± 0.179 | 0.686 ± 0.227 | 22.200 ± 24.091 | 0.318 | 12.600 | 19.950 | 21.000 | 0.000 |
| Lausanne | TCIG | 20 | 0.954 ± 0.018 | 0.978 ± 0.034 | 1.400 ± 1.772 | 0.873 | 2.250 | 3.400 | 13.850 | 0.650 |
| Lausanne | Prompt-only | 20 | 0.886 ± 0.128 | 0.886 ± 0.168 | 6.100 ± 9.396 | 0.611 | 7.800 | 7.650 | 16.650 | 0.350 |
| Lausanne | No-interaction | 20 | 0.799 ± 0.220 | 0.741 ± 0.257 | 19.300 ± 26.329 | 0.391 | 11.550 | 13.650 | 19.950 | 0.050 |
| CROWN | TCIG | 20 | 0.960 ± 0.018 | 0.975 ± 0.038 | 1.100 ± 1.261 | 0.818 | 3.100 | 4.150 | 15.200 | 0.550 |
| CROWN | Prompt-only | 20 | 0.883 ± 0.095 | 0.880 ± 0.103 | 4.400 ± 3.760 | 0.583 | 10.700 | 10.900 | 19.000 | 0.150 |
| CROWN | No-interaction | 20 | 0.782 ± 0.180 | 0.668 ± 0.247 | 26.350 ± 28.436 | 0.269 | 12.600 | 17.850 | 21.000 | 0.000 |

## 结果分析

从重叠精度看，`CROWN / TCIG` 的最终 Dice 最高。clDice 最高的设置为 `Lausanne / TCIG`，它更直接反映中心线和管状结构一致性，因此可作为血管拓扑质量的主要观察点。

从连通性看，`CROWN / No-interaction` 的平均 Betti-0 误差最高，说明该设置下仍存在较明显的断裂、粘连或分量数量偏差。`Lausanne / TCIG` 的 B0 exact coverage 最高，表示交互点击更容易把连通分量数量校正到与标注一致。

TCIG 相对 prompt-only 的平均变化为：Dice +0.071，clDice +0.106，B0 误差下降 +4.117，分量 F1 +0.247。

prompt-only 相对 no-interaction 的平均变化为：Dice +0.094，clDice +0.172，B0 误差下降 +16.800，分量 F1 +0.247。

需要注意的是，Dice/clDice 的 NoC 与 B0 exact NoC 衡量的是不同目标：前者关注达到重叠或中心线质量阈值所需点击数，后者关注连通分量数量是否完全匹配。若 Dice 较高但 B0 exact coverage 较低，说明模型能恢复大部分血管体素，但仍可能保留局部断裂或错误连接；这正是拓扑交互式指标相对单纯 Dice 更有价值的地方。

总体上，当前评估框架已经能同时保存病例级曲线、数据集级统计、论文表格和跨数据集图。后续如果加入真实交互式 UI，可复用相同指标协议，只需要把 largest-error simulator 生成的点击替换为真实用户点击记录。
