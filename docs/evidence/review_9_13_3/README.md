# 9-13-3 验证证据

本目录只保存本批契约测试、只读复核与集成记录，没有真实研究训练或新模型输出。

- `baseline_light.log`：修改前 290 项轻量回归。
- `red.log`：原始 role 与 source/terminal/first loss 联动污染的 RED。
- `red_bundle_entry.log`：公开 bundle 入口 dict / synthetic v1 降级的 RED。
- `green.log`：同组四个测试方法通过，包括合法 train 与污染拒绝。
- `dataset_tests.log`：六个监督数据契约测试。
- `cpu_contracts.log`：十个禁用 GPU 的临时小型合成拟合、重载与恢复检查。
- `full_light.log`：最终完整轻量回归；不导入 PyTorch / Transformers。
- `a_independent_review.md`：A1/A2 独立只读审查。
- `final_independent_review.md`：全分支独立终审与 482 行 dropped refs 修复后的复核。
- `final_admission_tests.log`：最终补齐 dropped 导出后的 4 项定向测试。
- `integration_verification.json`：主 agent 对 CSV、原始任务、请求阶段、入模 refs/位置和 token 算术的交叉检查。

日志内容来自本次真实命令输出，存档仅规范化行末空格；RED 日志中的断言失败是修复前的预期问题复现，不能误读为当前回归失败。

B 的完整八个数据文件见相邻 `../t9_admission_audit_2026_09_13/`，其 token 总数来自原 tokenizer 的离线计数，counterfactual 只模拟装包，不含新 GoT 驾驶结果。
