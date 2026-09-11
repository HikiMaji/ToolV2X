# 9 月 10 日审查修复计划

> 执行方式：当前会话按项完成测试、修改和验证。用户已授权修复。

**Goal:** 修复本机已复现的五项实现缺陷，保留失败记录和历史实验产物。

**Architecture:** 沿用现有运行器、P/F 服务和原模型适配层；在共同入口修正路径、预算与计量，不引入新框架。报文字节检查独立于模型加载。

**Tech Stack:** Python 3.8、unittest、NumPy/SciPy；模型边界检查使用本机原 tokenizer 和 PyTorch。

**Spec:** `docs/review_9_10_verification.md`。

## 约束

- 不改原 CMP/V2V-GoT 源目录、历史输出、用户审查文件或现有缓存。
- Q9 保留实际生成的完整 Q8；预算溢出不得静默截断或用 GT 回填。
- 文件比较直接读取字节；缺失文件明确未检查。
- 测试区分真实原模块和模型边界替身；不把通过测试视为质量收益。

## 执行项

- [x] `run_connection.prepare`：用临时数据目录执行真实特征/运动读取，先复现绕过配置，再统一采用 `M.V2VGOT_ROOT`。
- [x] `v2vgot.fit_evidence` / `plan`：真实 tokenizer 的长 Q8 测试先失败，再加入父回答余量；实际 Q9 超限返回结构化失败，并验证 `infer` 保存该动作、继续下一动作。
- [x] `compare_evidence`：增加同长但不同内容、缺失文件、归档文件相同的测试；逐字节比较请求/响应并把结果写入报告。模型依赖移入执行函数，供轻量检查复用。
- [x] `local_adapter`：测试相对路径、展开用户目录、绝对路径和冲突目标；规范化来源路径并检查链接目标存在。
- [x] `VehicleTools.query`：测试混合/全回退、ROI 返回数、缓存命中及零计时；按新计算的 `model_used` 计数，并分列回退/返回目标。
- [x] 补充 ROI 完整上下文对照范围、跟踪输出有效位语义；运行完整集成和 NumPy/SciPy 轻量检查，记录本轮结果。

## 验证命令

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/conda-envs/llava/bin/python -m unittest discover -s tests -p 'test_*.py'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python scripts/check_review.py
```

轻量检查使用仅安装 `requirements-review.txt` 的现成隔离环境。每项先确认新测试因对应缺陷失败，再进行源码修改；最终记录新增测试、通过项数以及未执行的模型训练/质量评价。
