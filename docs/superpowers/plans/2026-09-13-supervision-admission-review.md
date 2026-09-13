# 9-13-3 执行计划

依据：`docs/9-13-3.md`。从已推送 main 的 T9 两帧联调记录开始，完成后停止。

全局限制：不扩大真实采集、不训练实际 value policy、不调用 GoT/MTR、不改 receiver 算法/预算、不修改旧 T9 结果、不引入数据库/认证/新模型。只允许契约测试的合成夹具及必要的小型 CPU 测试；B 使用已有本地 tokenizer 做文本计数，不加载生成模型。旧 v1 回归与监督公式保持。

1. A1/A2（root）：追踪 `query_value.py`、`query_data.py` 和 `bundle_data.py` 原始索引/分支/标签加载；先新增并运行 role 污染及 loss/target 联动污染 RED，做共享入口最小修复，GREEN 后回归。验证拒绝发生在规范化/优化器前；合法 train、失败终态、STOP=0 和 incremental costs 维持。
2. B（admission agent）：只新增只读诊断脚本、必要脚本测试和报告，读取 main 既有 `outputs/t9_real_smoke_2026_09_13_v1`，不覆盖它。逐 unit 复算 provider rank/购买出处/ledger/排序/token 增量/累计/admission/drop/提示位置，分别记录单独增量和贪心尝试增量。仅模拟共同 diversity admission，不生成驾驶轨迹。写新输出组及完整 token 表，提出 2–3 个所有方法共用的候选与公平性风险。
3. 验证整合：root 交叉检查 B；独立 reviewer 审查完整改动；运行完整轻量与针对性 CPU 测试；更新当前状态和最终报告，同步本地 main，不自动推送/训练/生成。

依赖和职责核对：

| 工作项 | 共享文件/接口 | 结论 |
|---|---|---|
| A1/A2 | `query_value.py`、归档验证和训练入口 | 由 root 顺序实现，避免并发修改；可复用原有 branch/label 验证 |
| A/B | B 只读 receiver/context，与 A 的监督加载无共享写入 | 可以并行；B 不改 `src/` |
| A 自检 | 用户要求 RED→GREEN，与最小边界修复一致 | 不运行真实训练，负例测试进入公开 fitter 后必须在 `_fit` 前拒绝 |
| B 自检 | counterfactual admission 不等于模型效果 | 单独注明模拟，保持实际 Z/轨迹归档不变 |

裁定：本地集成属于本轮授权的可逆实现工作；既有临时模型结果保持不可变。仅 B 使用子 agent，紧耦合 A1/A2 留给 root；结束另做完整代码审查。

执行裁定：公开 bundle fitter 收紧为 measured v2 原始归档路径，以落实“任何拟合输入均能恢复原始 role”；私有合成夹具仍用于 CPU 优化器契约。旧 alternating targets 缺标签 sidecar 时明确拒绝，迁移仅在日后需要使用这些 targets 时做离线重导出，本批不批量重建。

B 的阶段语义已区分实际外层 request event 与 first visible plan；最终复审发现初始/Ego 阶段本车 dropped refs 漏导出，以移动既有导出循环补齐，未改变任何 admission 行为或预算。
