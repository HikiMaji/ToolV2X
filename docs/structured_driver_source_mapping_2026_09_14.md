# 结构化驾驶端：参考源码下载与逐函数映射

日期：2026-09-14。源码调研与实现依据，不是原框架复现结果。用户已授权直接下载小于 100 MB 的资源。

## 下载与验证

| 资源 | 官方源码地址 | 压缩包字节 | 解压字节 | 文件数 | 本地目录 |
|---|---|---:|---:|---:|---|
| UniV2X main | https://github.com/AIR-THU/UniV2X | 1,598,011 | 4,070,704 | 159 | `/root/autodl-tmp/UniV2X` |
| VAD main | https://github.com/hustvl/VAD | 542,442 | 1,602,485 | 185 | `/root/autodl-tmp/VAD` |

两个官方 API/HEAD 请求可访问，但 archive 响应不提供 Content-Length。下载时按 100,000,000 字节硬上限流式接收，解压前检查成员路径、展开总量并通过 zip CRC；没有下载模型、数据集或执行外部仓库代码。原包和读取记录保存在 `/root/autodl-tmp/toolv2x_reference_sources_2026_09_14/`。两个仓库提供 Apache 2.0 LICENSE。

`main` 是读取时分支，不是永久不变版本。已下载的文件作为本轮固定参考；后续重新下载不能默认为相同源码。本项目明确记录直接引用与重实现的区别，不将下载即视为复现。

## UniV2X：实际实现与论文概述的差异

文件：`projects/mmdet3d_plugin/univ2x/fusion_modules/agent_fusion.py`。

- `_query_matching`：根据三维参考点距离构造成本；`_dis_filt` 按车辆预测尺寸逐轴门控，再执行 SciPy Hungarian。不是一个预训练的全局实体身份识别器。
- `_query_fusion`：接受匹配后，向车辆 query 的特征部分加上 `cross_agent_fusion(inf.query)`；当前层是 Linear。不能直接把论文概述中的多层拼接融合说成该文件逐行实现。
- `_query_complementation`：将未匹配 infrastructure query 拼接到车辆实例；这是保留远端独有目标的直接参考。
- `forward`：从各自归一化参考点还原坐标，使用相对位姿变换，并将旋转矩阵拼入 query 特征投影。当前 ToolV2X 框已在 ego 系，不得照搬再变换；P/F 也不携带这些原始 query hidden features。
- 自车处理采用固定矩形，并删除首先命中的 query。首版 ToolV2X 使用有历史佐证的角色判定，缺乏佐证则保留未确定目标；不照搬原代码的固定尺寸与首项删除策略。

可借鉴：门控一对一关联、匹配特征融合、未匹配项补入。需要适配：源内 track handles、因果历史、显式歧义、receipt 与父字段闭包。不可假称原样复用：其 Instances、学习 query 权重和车路感知全栈。

## VAD：规划与监督的具体连接

文件：`projects/mmdet3d_plugin/VAD/VAD_head.py`。

- 约 709 行的 planning 分支：`motion_hs` 经 `agent_fus_mlp` 整理成 agent query；`select_and_pad_query` 按检测分数选择和填充，再让 ego query 与 agent query 交互。
- 随后的 ego-map decoder 明确需要 lane/map queries 和位置，不能用当前检测头 classification/regression 图冒充该地图分支。
- `ego_fut_decoder` 是数值 MLP，输出配置数量的未来方案。其 `ego_fut_mode` 默认 3，训练的 `ego_fut_cmd` 决定对应导航模式的监督；当前 ToolV2X 没有这个合法导航输入，不复制三路模式选择并用未来 GT 填命令。
- `loss_planning`：数值轨迹监督使用 `ego_fut_cmd × ego_fut_masks`；另有道路边界、agent 冲突和方向损失。ToolV2X 首版只采用明确可用标签的数值轨迹监督，不能声称同时迁移了这些几何约束。

可借鉴：带位置的 ego-agent 交互、数值轨迹头、有效位监督。新增适配：P 历史和 F 多模式显式数值入口、与实际上一方案共用的修订接口、原 GoT 检测场景图的独立数值分支。不是直接将 F 文本换成 `motion_hs`，也不是完整 VAD 复现。

## 当前本地代码的复用

| 本地组件 | 处理 |
|---|---|
| `prediction/causal_windows.py` 与统一位姿变换 | 复用时间/坐标契约，不重建跟踪缓存 |
| `tools/task_spec.py`、`tools/vehicle.py` | 保留真实 P/F 执行、完整预测上下文、任务排序、receipt 与字节控制 |
| `planning/evidence.py` 的 remote_units | 复用已购/派生验证及同信息 forecast 合并；本车 history 采用新版本可选加入 |
| `vendor/cmp_mtr/.../utils/polyline_encoder.py` | 参考带 mask 的 PointNet 折线编码；新 driver 改用适合小集合的 LayerNorm，原冻结 MTR 不变 |
| `planning/inputs.py`、原 `llava_arch.py` shallow 分支 | 复用本车分类/回归图布局与 5×4 分块；新网络不依赖语言 tokenizer、CLIP 或原 7B |
| `planning/method_episode.py` 及控制/评价 | 通过数值 driver 分派复用共同执行链，旧 Q9 标记与费用保持原义 |

具体代码与测试完成状态以本轮实现报告为准。这里说明来源和边界，不将设计中的组件标为已经完成。

## 原 GoT 误差公式核对（09-15）

实际本地源码 `V2V-GoT/LLaVA/scripts/eval_v2v4real_3d_grounding.py` 的 `evaluate_future_trajectory`（736–743 行）先计算六个点的欧氏误差，再分别平均前 2、4、6 个点作为 1/2/3 秒指标。原 ToolV2X 的 `L2_1/L2_2` 是时刻端点误差，定义不同。新评价将增加明确命名的前缀指标，保留旧指标原义。公式对齐仍不代表数据划分、标签精度、输入条件和任务相同，因此不能据此直接宣称超过 GoT 论文结果。这里只读源代码，没有运行原评价器或生成新轨迹。
