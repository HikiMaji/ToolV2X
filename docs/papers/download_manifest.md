# ToolV2X 论文下载与正文提取清单

下载日期：2026-09-07。此文件只记录文件处理结果，不包含论文分析或精读结论。

提取方式：`pdftotext -layout`。`pages` 来自 `pdfinfo`；`text bytes` 是提取文本文件的字节数。PDF 与 TXT 均保存在本目录及其 `text/` 子目录中。

|编号|论文|来源|本地 PDF|页数|本地正文 TXT|text bytes|状态|
|---|---|---|---|---:|---|---:|---|
|01|V2V-GoT|[arXiv 2509.18053](https://arxiv.org/pdf/2509.18053.pdf)|`01_v2v_got.pdf`|8|`text/01_v2v_got.txt`|69212|已提取|
|02|Select2Drive|[arXiv 2501.12040](https://arxiv.org/pdf/2501.12040.pdf)|`02_select2drive.pdf`|15|`text/02_select2drive.txt`|158441|已提取|
|03|EDDI|[arXiv 1809.11142](https://arxiv.org/pdf/1809.11142.pdf)|`03_eddi_arxiv.pdf`|22|`text/03_eddi.txt`|119797|已提取；PMLR 文件另存为 `03_eddi.pdf`，该版本无可用文本层|
|04|Defer to Plan|[arXiv 2607.19774](https://arxiv.org/pdf/2607.19774.pdf)|`04_defer_to_plan.pdf`|6|`text/04_defer_to_plan.txt`|43881|已提取|
|05|Co-MTP|[arXiv 2502.16589](https://arxiv.org/pdf/2502.16589.pdf)|`05_co_mtp.pdf`|7|`text/05_co_mtp.txt`|57898|已提取|
|06|V2XPnP|[CVF ICCV 2025 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_V2XPnP_Vehicle-to-Everything_Spatio-Temporal_Fusion_for_Multi-Agent_Perception_and_Prediction_ICCV_2025_paper.pdf)|`06_v2xpnp.pdf`|11|`text/06_v2xpnp.txt`|81601|已提取；工具输出 PDF 字符串格式警告|
|07|V2XVerse|[arXiv 2404.09496](https://arxiv.org/pdf/2404.09496.pdf)|`07_v2xverse.pdf`|27|`text/07_v2xverse.txt`|198467|已提取|
|08|COOPERNAUT|[arXiv 2205.02222](https://arxiv.org/pdf/2205.02222.pdf)|`08_coopernaut.pdf`|11|`text/08_coopernaut.txt`|79019|已提取；工具输出单个数字格式警告|
|09|DiFA|[AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/download/25934/25706)|`09_difa.pdf`|9|`text/09_difa.txt`|64291|已提取|
|10|Learning to Acquire Information|[arXiv 1704.06131](https://arxiv.org/pdf/1704.06131.pdf)|`10_learning_to_acquire_information.pdf`|10|`text/10_learning_to_acquire_information.txt`|55974|已提取|
|11|When2com|[arXiv 2006.00176](https://arxiv.org/pdf/2006.00176.pdf)|`11_when2com.pdf`|10|`text/11_when2com.txt`|67783|已提取|
|12|Who2com|[arXiv 2003.09575](https://arxiv.org/pdf/2003.09575.pdf)|`12_who2com.pdf`|8|`text/12_who2com.txt`|54241|已提取|
|13|Where2comm|[arXiv 2209.12836](https://arxiv.org/pdf/2209.12836.pdf)|`13_where2comm.pdf`|24|`text/13_where2comm.txt`|94310|已提取|
|14|How2comm|[NeurIPS 2023 PDF](https://papers.neurips.cc/paper_files/paper/2023/file/4f31327e046913c7238d5b671f5d820e-Paper-Conference.pdf)|`14_how2comm.pdf`|14|`text/14_how2comm.txt`|81522|已提取|
|15|INSTINCT|[CVF ICCV 2025 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_INSTINCT_Instance-Level_Interaction_Architecture_for_Query-Based_Collaborative_Perception_ICCV_2025_paper.pdf)|`15_instinct.pdf`|10|`text/15_instinct.txt`|68218|已提取|
|16|TOCOM-V2I|[arXiv 2407.20748](https://arxiv.org/pdf/2407.20748.pdf)|`16_tocom_v2i.pdf`|6|`text/16_tocom_v2i.txt`|41748|已提取|
|17|V2V4Real|[arXiv 2303.07601](https://arxiv.org/pdf/2303.07601.pdf)|`17_v2v4real.pdf`|18|`text/17_v2v4real.txt`|88241|已提取|
|18|DriveAgent-R1|[arXiv 2507.20879](https://arxiv.org/pdf/2507.20879.pdf)|`18_driveagent_r1.pdf`|35|`text/18_driveagent_r1.txt`|163569|已提取|

检查结果：18 个编号均有非空正文文本；没有开始方法比较、创新性判断或实验设计。
