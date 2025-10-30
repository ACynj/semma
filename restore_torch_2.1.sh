#!/bin/bash
# 恢复 PyTorch 2.1.0 和兼容的依赖版本

echo "=============================================="
echo "恢复 PyTorch 2.1.0 环境"
echo "=============================================="

echo ""
echo "1. 卸载当前的 PyTorch 相关包..."
pip uninstall -y torch torchvision torch-scatter torch-geometric

echo ""
echo "2. 安装 PyTorch 2.1.0 (CUDA 12.1)..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "3. 安装兼容的 torch-scatter..."
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

echo ""
echo "4. 安装 torch-geometric 2.4.0 (锁定版本)..."
pip install torch-geometric==2.4.0

echo ""
echo "=============================================="
echo "✅ 环境恢复完成！"
echo "=============================================="

echo ""
echo "验证版本:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

echo ""
echo "如果看到 PyTorch: 2.1.0+cu121, 说明恢复成功！"

