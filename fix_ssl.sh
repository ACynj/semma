#!/bin/bash
# 解决SSL连接问题的脚本

echo "🔧 设置SSL环境变量以解决连接问题..."

# 设置Python SSL环境变量
export PYTHONHTTPSVERIFY=0
export SSL_VERIFY=False

# 设置urllib3的SSL警告
export PYTHONWARNINGS="ignore:Unverified HTTPS request"

echo "✅ SSL环境变量已设置"
echo "   - PYTHONHTTPSVERIFY=0"
echo "   - SSL_VERIFY=False" 
echo "   - PYTHONWARNINGS=ignore:Unverified HTTPS request"

echo ""
echo "🚀 现在可以运行您的命令了:"
# Get checkpoint path from flags.yaml
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_PATH=$(python3 -c "import yaml, os; config = yaml.safe_load(open(os.path.join('$SCRIPT_DIR', 'flags.yaml'))); print(os.path.join(config['ckpt_path'], 'semma.pth'))")
echo "python script/run.py -c config/inductive/inference.yaml --dataset WN18RRInductive --version v1 --ckpt $CKPT_PATH --gpus [0] --bpe null --epochs 0"
