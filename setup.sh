#!/bin/bash

# 使用北外镜像源加速安装
PYPI_MIRROR="https://mirrors.bfsu.edu.cn/pypi/web/simple"

# Install prerequisites
echo "Installing requirements..."
# It is highly recommended to install PyTorch first, matching your CUDA version.
# The requirements.txt file specifies a version compatible with CUDA 11.8.
# Example: pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 使用国内镜像源安装torch（如果官方源慢，可以尝试镜像源）
# 先安装torch，确保版本锁定为2.1.0
pip install torch==2.1.0 -i ${PYPI_MIRROR}
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html # repace with correct torch and cuda versions

# 单独安装torch-geometric，使用--no-deps避免升级torch
# 注意：torch-geometric的依赖（如torch-scatter）已经在上面安装
pip install torch-geometric==2.4.0 --no-deps -i ${PYPI_MIRROR}

# 创建临时requirements文件，排除torch相关包（因为它们已经单独安装）
TEMP_REQ=$(mktemp)
grep -v "^torch" requirements.txt | grep -v "^#.*torch" > ${TEMP_REQ}

# 安装其他requirements
pip install -r ${TEMP_REQ} -i ${PYPI_MIRROR}

# 清理临时文件
rm ${TEMP_REQ}

echo ""
echo "IMPORTANT: The 'flash-attn' package can have complex installation dependencies related to your CUDA toolkit version."
echo "If you encounter issues with 'flash-attn', please refer to its official documentation for installation instructions specific to your environment."
echo ""

# Download and prepare fb_mid2name.tsv
echo "Downloading fb_mid2name.tsv..."
wget -O fb_mid2name.tsv 'https://drive.google.com/uc?id=0B52yRXcdpG6MaHA5ZW9CZ21MbVk'

echo "Unzipping fb_mid2name.tsv"

echo "Setup complete."
echo "Note: If fb_mid2name.tsv was downloaded as a zip archive, you might need to manually unzip it or adjust this script." 