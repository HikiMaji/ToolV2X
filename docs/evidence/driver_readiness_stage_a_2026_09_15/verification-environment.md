# Verification environment
Light: /root/miniconda3/bin/python scripts/check_review.py
Model: /root/autodl-tmp/conda-envs/llava/bin/python -m unittest discover -s tests -p 'test_*.py'
Use OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1.
TOOLV2X_V2VGOT_ROOT=/root/autodl-tmp/V2V-GoT
TOOLV2X_CMP_ROOT=/root/autodl-tmp/CMP
TOOLV2X_LLAVA_BASE=/root/autodl-tmp/ToolV2X/models/llava-v1.5-7b
TOOLV2X_CLIP_ROOT=/root/autodl-tmp/ToolV2X/models/clip-vit-large-patch14-336
Three ignored train_g0719 JSON fixtures copied unchanged from main; all other prior archived fixtures tracked.
Clear PYTHONPATH for final model discovery; per-task targeted tests may use PYTHONPATH=src:tests.
No real dataset or original-model generation. Torch/optimizer tests use temporary synthetic CPU data only.
