# ToolV2X 协议审查与修正结果

日期：2026-09-10。结论：**数据接口修正与验证完成；正式四配置训练 HOLD。**
本轮未训练模型、未启动 MTR 推理、未修改 V2V-GoT 原始数据或 CMP 模型。
实验规范见 [experiment_protocol.md](experiment_protocol.md)。本报告不构成方法有效性结论。

## 核实发现

| 事项 | 证据 | 处理 |
|---|---|---|
| 整段未来信息影响历史输入 | tracks_to_cmp 原始 raw 也调用双向插值；GT 模式使用全序列身份众数 | raw 改为不匹配 GT、不插值；独立导出 time-t 因果窗口 |
| 训练/测试记录重叠 | 全部 7105/1993 帧位姿检查，147 对重复 | 按完整记录隔离 15 个源 train 片段；隔离后位姿重复为 0 |
| 移动坐标系与时间编码 | 旧轨迹是每帧 ego 系；CMP dataset 时间公式为 0.01*i | 新窗口所有历史在 ego(t)，时间间隔 0.1s；未混用旧权重 |
| F 不输出远端独有目标 | CMP MotionAggregatorTransformer 只遍历 ego ID；assert 输出数量与 ego 一致 | 标为原方案语义阻断；未把现有 F 当作新增目标服务 |
| 合作车被排除出评价 | oracle 原先排除 ego/CAV1 原点附近 GT ID | 改为只排 ego；后出现的 ego 标注也排除；合作车保留 |
| 缺帧/坏预测掩盖失败 | 四配置取交集；NaN 模态可逃过 proximity 检查 | 独立 manifest 一致性检查；非有限/短时域/非法概率报错 |
| 通信收益与最优占比误用 | 固定 P=1,F=0.3；argmin 对并列偏置；10–15% 停止线 | 删除未经实测的通信节省及硬阈值；部分并列不再写成全体打平 |
| 原训练入口可能继续旧协议 | 默认 GT 适配、直接训练 | prepare 改为因果窗口；train/test 默认 HOLD；历史诊断需显式标识 |

## 已生成和已验证

- 原始六份跟踪文件都有完整场景/帧键，状态形状和数值有限性通过检查。
- 新划分：15 train、2 validation、9 test 片段。独立评价帧分别为 3027、108、1453。
- 原始物理目录 train 中的 17 个窗口片段包含 train 和 validation，消费端必须按 manifest 分开。
- 新窗口：78 个场景/来源文件，17664 个窗口，231351 个当前目标记录，约 90 MB。
- 全部窗口形状、mask、有限数值检查通过；234 个真实窗口通过截断未来后的不变性复核；31780 个状态坐标用独立矩阵公式复算通过。
- 9 个回归用例通过，覆盖历史插值泄漏、移动系重表达、0.1s 时间、远端独有目标、合作车排除、后出现 ego 标注、缺动作帧、坏预测、歧义结果目录和误导性报告等行为。
- shell 语法检查通过；默认 train 命令实测在启动 trainer 之前返回 2，带 HOLD 原因。
- 旧脚本备份在 `outputs/protocol_audit/backups/`；没有删除原跟踪、旧适配结果或权重。

证据文件：

- `outputs/protocol_audit/audit.json`：全部位姿重复帧对、跟踪统计、门槛原因。
- `outputs/protocol_audit/split_manifest.json`：逐角色场景名单和隔离名单。
- `outputs/protocol_audit/{train,validation,test}_frames.json`：独立决策帧。
- `outputs/protocol_audit/artifact_validation.json`：实际窗口检查计数。
- `outputs/protocol_audit/training_entry_check.json`：默认训练入口检查。
- `outputs/protocol_audit/environment.json`：当前 CMP Python 模块发现结果（不是 GPU/完整 import 检查）。

## 尚未通过的条件与下一步

1. **预测接入**：原 CMP dataset 仍按 GT 身份交集与未来 validity 选目标，且模型接口尚未消费新的因果窗口。需要独立监督匹配和推理目标集合，不能仅替换数据路径。
2. **远端目标接收**：导出独立 CAV1 预测，并在简单接收器中保留 peer-only 目标；同目标去重仅使用在线几何/历史。
3. **感知训练来源**：新隔离名单不能消除旧检测器的训练历史。现有缓存只能作为固定感知诊断；正式结果需核实权重训练数据或按隔离划分重训。
4. **通信语义**：现有 P 是 CoBEVT 历史缓存，不是已测量的线上消息；需补实际响应字段、历史获取/缓存成本、计算时延 ledger。没有此 ledger 不报告真实通信 Pareto。
5. **环境**：cmp Python 中发现 torch/numpy，但未发现 yaml、spconv、torch_geometric、easydict、scipy。现存四份 CMP best_model.pth 的存在已确认，未载入/评估，不认定适配新协议。

这些是明确的实验/接入待办，不是方向 NO-GO，也不是资源不足结论。

**推荐接续任务：因果窗口 → 独立 ego/CAV1 预测输出 → 保留远端独有目标的简单接收器，一个场景先验证。**
四配置旧 YAML 不应直接启动 30 epoch。修正后的历史 oracle 仍是 GT 参考路径条件速度诊断，不能宣传在线规划、几何碰撞率或闭环安全。
