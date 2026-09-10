# 原框架来源与本地适配

本次整理日期：2026-09-10。来源均为原工作站已有文件，没有重新下载上游仓库或权重。复制文件逐字比较；不将本地存在性视为上游发布版本、权重训练来源或 CUDA 一致性已验证。

`docs/evidence/v2v_got_inference_source.py` 是早期文献/因果审查引用的原源码副本，与本次 `vendor/v2vgot_opencood/inference.py` 逐字一致；保留它是为了让历史报告中的源码引用仍可复查，来源与许可记录同下表。

## 随仓库提供的代码和配置

| 仓库内位置 | 本地来源 | 附带原文件 |
| --- | --- | --- |
| `vendor/cmp_mtr/mtr/` | `/root/autodl-tmp/CMP/MTR/mtr/` | [原 MTR LICENSE](../vendor/cmp_mtr/LICENSE)、[源文件对应表](../vendor/cmp_mtr/source_manifest.json) |
| `vendor/cmp_mtr/easydict/` | `/root/autodl-tmp/conda-envs/dmstrack/lib/python3.7/site-packages/easydict/`，1.9 | [原 LICENSE](../vendor/cmp_mtr/easydict/LICENSE)、[原包 METADATA](../vendor/cmp_mtr/easydict/METADATA)；源码逐字一致 |
| `vendor/v2vgot_llava/llava/` | `/root/autodl-tmp/V2V-GoT/LLaVA/llava/` | [原 LICENSE](../vendor/v2vgot_llava/LICENSE)、原 README 和 pyproject.toml；包内非缓存文件逐字一致 |
| `vendor/v2vgot_opencood/inference.py` | `/root/autodl-tmp/V2V-GoT/DMSTrack/V2V4Real/opencood/tools/inference.py` | [原目录 LICENSE](../vendor/v2vgot_opencood/LICENSE)、原 setup.py 与 PKG-INFO；文件逐字一致 |

MTR 和 LLaVA 目录的 LICENSE 文件是 Apache 2.0 文本。EasyDict 1.9 附带 GNU LGPL v3 文本，其 METADATA 中的 `LPGL` 拼写原样保留。未为这些第三方文件重新授予统一的项目许可证。

V2V4Real 目录存在许可元数据冲突：LICENSE 是 Apache 2.0 文本，setup.py 顶部注释为 `TDG-Attribution-NonCommercial-NoDistrib`，setup.py 与 PKG-INFO 的许可字段又写 MIT。这里同时保留 [upstream_setup.py](../vendor/v2vgot_opencood/upstream_setup.py) 和 [upstream_PKG-INFO](../vendor/v2vgot_opencood/upstream_PKG-INFO)，不以其中某一个字段消除该冲突。这两个元数据文件不参与运行。

相关根仓库的原 README 在本地未给出统一许可条款；上游源码仍保留作者和文件头信息。原 LLaVA README 对数据集、基础模型和 checkpoint 的许可有独立说明。本仓库不发布这些权重或完整数据。

## CMP MTR 适配内容

[source_manifest.json](../vendor/cmp_mtr/source_manifest.json) 是最初复制时的路径对应记录，不是一份完整改动清单。当前与本地 CMP 源文件不同的文件为：

- `context_encoder/mtr_encoder.py`：移除未使用的导入，把输入张量移到模型实际设备。
- `motion_decoder/mtr_decoder.py`：intention points 支持指定设备，当前连接使用 CPU。
- `ops/attention/__init__.py`：使用新增的 `attention_torch.py` 作为 portable attention 实现。
- `ops/knn/knn_utils.py`：保留原入口并提供 torch KNN，按 batch 隔离，缺邻居填 -1；排序与原 CUDA 的堆顺序、边界同距选择可能不同。
- `utils/common_utils.py`：batch offsets 使用输入张量的设备。

原 C++/CUDA 源码保留供审查，不代表当前执行了这些算子，也没有测定数值或性能一致性。`src/prediction/cmp_adapter.py` 构造保留当前源上下文的因果 22 通道历史，将时间间隔显式设为 0.1 秒；在线推理不调用原 GT 驱动 dataset。

原配置和辅助资源逐字复制自：

- `CMP/MTR/tools/cfgs/v2v4real/v2v4real_multiego_no_coop.yaml` → `vendor/cmp_mtr/configs/v2v4real_multiego_no_coop.yaml`。
- `CMP/preprocessed_data/v2v4real/v2v4real_cluster_64_center_dict.pkl` → `vendor/cmp_mtr/preprocessed_data/v2v4real/v2v4real_cluster_64_center_dict.pkl`（1205 字节）。这是原 intention points 辅助文件，其拟合数据来源仍未验证。

`vendor/cmp_mtr/reference/v2v4real_multiego_dataset.py` 来自当前本地 CMP 工作树，仅供静态参考和提取原特征构造 helper 的测试。该上游工作树在本次整理前已有改动，其相对本地 Git 基线的补丁保存在 `reference/local_dataset_changes.patch`；本轮没有修改原 CMP 仓库。不要将此文件当作新实现的因果训练 dataset。

## V2V-GoT 执行边界

`src/planning/v2vgot.py` 默认从随仓库提供的原 LLaVA 包加载 projector、点云 token 构造、对话模板、tokenizer 接口和原模型加载器。包内源码保持复制时的内容。ToolV2X 的变化位于 `src/`：Ego 输入隔离、提示/证据构造、显式预算、原始回答解析，以及原 Q8 到 Q9 的实际生成链。

`src/planning/adaptation_data.py` 只从原 `inference.py` AST 中提取 `get_suggested_speed_steering` 与 `get_future_trajectory_str` 两个纯函数；不导入整个 OpenCOOD CLI/GT QA 图。该模块只用于离线准备，在线工具和规划器不导入它。原文件中其他 GT 逻辑随源码保留供检查，不等于它们进入了当前推理。

## 外部资源

原 AB3DMOT/OpenCOOD 完整预处理栈、模型 checkpoint、tokenizer、基础 LLaVA、CLIP、检测缓存和完整数据仍由本机提供。源代码可审查与完整实验可重放是不同范围；环境和路径说明见 [README](../README.md)。
