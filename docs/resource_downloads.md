# ToolV2X 资源缺项清单

现场核查时间：2026-09-10 18:44 HKT（UTC+08:00）。

后续主 agent 已在同日使用这些本地文件成功加载完整 LLaVA、CLIP 与原 V2V-GoT LoRA，四个固定动作均完成实际 Q8/Q9 生成，见 [planning_connection.md](planning_connection.md)。下文“没有运行模型”描述的是资源盘点子任务本身；文件存在性现已有真实离线加载验证。主 agent 收到用户域名分工要求后已停止先前的下载。

本清单只核对原 V2V-GoT 规划端、CMP 原 MTR 和 P/F 串联当前需要的模型文件。没有下载或跟随 GitHub、GitHubusercontent、GitHubassets、Hugging Face 的下载跳转，也没有运行模型。当前主线的两个可读目录固定为以下路径；它们已由主 agent 从本机完整历史 cache 链接就位，当前没有必须让用户下载的资源：

- `/root/autodl-tmp/ToolV2X/models/llava-v1.5-7b`
- `/root/autodl-tmp/ToolV2X/models/clip-vit-large-patch14-336`

## 当前立即需要就位的最小文件

### LLaVA v1.5 7B 基础模型

代码入口：`src/planning/v2vgot.py`。其预检查从 `pytorch_model.bin.index.json` 的 `weight_map` 读取两片权重，并额外要求 `config.json` 与 `tokenizer.model`。下列文件现在均已在目标目录可用；它们来自本机完整历史 cache 的链接，不需要重复下载。

| 文件名 | 目标目录当前状态 | 实际大小（字节） | 官方下载 URL（仅作后备） | 当前必需 |
|---|---|---|---|---|
| `pytorch_model.bin.index.json` | 已有本地链接 | 27068 | `https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/main/pytorch_model.bin.index.json` | 是 |
| `pytorch_model-00001-of-00002.bin` | 已有本地链接 | 9976634558 | `https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/main/pytorch_model-00001-of-00002.bin` | 是 |
| `pytorch_model-00002-of-00002.bin` | 已有本地链接 | 3542276251 | `https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/main/pytorch_model-00002-of-00002.bin` | 是 |
| `tokenizer.model` | 已有本地链接 | 499723 | `https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/main/tokenizer.model` | 是 |
| `config.json` | 已有本地链接 | 1161 | `https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/main/config.json` | 是 |
| `generation_config.json` | 已有本地链接 | 124 | `https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/main/generation_config.json` | 否；保留即可 |

当前目标目录：`/root/autodl-tmp/ToolV2X/models/llava-v1.5-7b/`。对应的本地 cache 目录见“历史 cache 核查”一节。

`tokenizer_config.json`、`special_tokens_map.json` 也已从本地 cache 链接到目标目录（分别为 749、438 字节），但当前 `v2vgot.py` 的硬性文件检查没有要求它们；不需要另行下载。

### CLIP ViT-L/14 336

V2V-GoT 的 checkpoint 配置把视觉塔声明为 `openai/clip-vit-large-patch14-336`。原 `CLIPVisionTower.load_model()` 会调用 `CLIPImageProcessor` 和 `CLIPVisionModel`，即使驾驶端最终使用点云特征，也需要本地 CLIP 配置、预处理配置和视觉权重。

| 文件名 | 目标目录当前状态 | 实际大小（字节） | 官方下载 URL（仅作后备） | 当前必需 |
|---|---|---|---|---|
| `config.json` | 已有本地链接 | 4757 | `https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/main/config.json` | 是 |
| `preprocessor_config.json` | 已有本地链接 | 316 | `https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/main/preprocessor_config.json` | 是 |
| `pytorch_model.bin` | 已有本地链接 | 1711974081 | `https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/main/pytorch_model.bin` | 是 |

当前目标目录：`/root/autodl-tmp/ToolV2X/models/clip-vit-large-patch14-336/`。对应的本地 cache 目录见“历史 cache 核查”一节。

## 历史 cache 核查与当前状态

完整文件位于以下本机 cache 目录，目标目录中的链接映射记录在 `models/local_resources.json`：

| 模型 | 可由代码定位的 cache 目录 | 目标目录 | 用户是否需要下载 |
|---|---|---|---|
| LLaVA v1.5 7B | `/root/autodl-tmp/huggingface/transformers/models--liuhaotian--llava-v1.5-7b` | `/root/autodl-tmp/ToolV2X/models/llava-v1.5-7b` | 否 |
| CLIP ViT-L/14 336 | `/root/autodl-tmp/huggingface/transformers/models--openai--clip-vit-large-patch14-336` | `/root/autodl-tmp/ToolV2X/models/clip-vit-large-patch14-336` | 否 |

代码可在不写死 cache 内部版本目录的情况下定位唯一 snapshot：

```python
from pathlib import Path

def local_snapshot(cache_repo: str) -> Path:
    snapshots = sorted(
        p for p in (Path(cache_repo) / "snapshots").iterdir() if p.is_dir()
    )
    if len(snapshots) != 1:
        raise RuntimeError(f"expected one local snapshot, got {len(snapshots)}")
    return snapshots[0]
```

文件级核查确认这些 snapshot 链接均有真实目标，JSON 配置可解析，两个 LLaVA 分片和 CLIP 权重均为非空 PyTorch zip archive；上表字节数是实际文件大小。没有进行模型加载或大文件内容扫描，因此这里确认的是本地文件完整可链接，原模型加载仍由主 agent 的离线验证负责。

## 已在机器上确认的现有资源

- V2V-GoT 发布 checkpoint：
  `/root/autodl-tmp/V2V-GoT/LLaVA/checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2/checkpoint-4330/`。
  其中 `adapter_model.safetensors`、`adapter_config.json`、`config.json`、`non_lora_trainables.bin` 和 tokenizer 文件均在；`non_lora_trainables.bin` 提供原任务 projector，当前不需要另下官方 `mm_projector.bin`。
- CMP MTR 权重均在：
  - `/root/autodl-tmp/CMP/MTR/output/v2v4real_multiego_cobevt_c256/ckpt/best_model.pth`
  - `/root/autodl-tmp/CMP/MTR/output/v2v4real_multiego_cobevt_c256_no_agg/ckpt/best_model.pth`
  - `/root/autodl-tmp/CMP/MTR/output/v2v4real_multiego_no_coop/ckpt/best_model.pth`
  - `/root/autodl-tmp/CMP/MTR/output/v2v4real_multiego_v2vnet/ckpt/best_model.pth`
- CMP V2V4Real GT 轨迹为 `gt_multiego_speedless/{train,test}`（64 个 train、18 个 test 轨迹文件）；现有 tracking pickle、detection result 缓存和三个 MTR dataset cache 也在 `preprocessed_data/v2v4real/`。
- CMP 可用感知权重为 `point_pillar_cobevt_multiego_256x/{config.yaml,net_epoch75.pth}`、`point_pillar_sinbevt/{config.yaml,net_epoch60.pth}` 和 `point_pillar_v2vnet_multiego/{config.yaml,net_epoch75.pth}`。`point_pillar_cobevt_multiego_1x` 只有 `config.yaml`，同名权重文件是已知损坏副本，不计为可用资源；当前主线不需要它。
- 当前因果窗口和 V2V-GoT shallow Ego 特征在 ToolV2X 输出及 `/root/autodl-tmp/V2V-GoT` 已有，足以继续主线接口串联。

## 机器上的其他缓存说明

`/root/autodl-tmp/huggingface/transformers` 下的历史 LLaVA base cache 和 CLIP cache 已确认完整，并已由主 agent 链接到主线指定的两个可读目录；不要因为目标目录曾经缺文件而重复下载。官方下载 URL 只用于本地链接失效时的人工后备来源，agent 不会访问或下载这些 URL。`/root/autodl-tmp/ToolV2X/models/hf_cache` 中遗留的临时大文件仍按未完成下载处理，不参与加载。

原始 V2V4Real 点云/yaml 和 CMP 完整协作聚合器使用的 `fused_feature_<scene>_<cav>_<t_idx>.pkl.npy` 不属于当前真实组件串联的立即下载项；它们只在后续要做原始 CMP CoBEVT 感知重跑或完整聚合器复现时再单独盘点。
