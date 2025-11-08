#!/bin/bash

# semma预训练多次运行脚本
# 使用方法: bash script/pretrain_many.sh
# 或者: chmod +x script/pretrain_many.sh && ./script/pretrain_many.sh

CONFIG="config/transductive/pretrain_semma.yaml"
GPUS="[0]"
SEEDS=(1024 42 1337 512 256)
N_RUNS=5  # 运行次数，可以根据需要调整

echo "=========================================="
echo "开始多次预训练运行 (semma)"
echo "配置文件: $CONFIG"
echo "运行次数: $N_RUNS"
echo "=========================================="

# 检查flags.yaml中的run参数
FLAGS_FILE="flags.yaml"
if [ -f "$FLAGS_FILE" ]; then
    RUN_VALUE=$(grep "^run:" "$FLAGS_FILE" | head -1 | awk '{print $2}')
    if [ "$RUN_VALUE" != "semma" ]; then
        echo ""
        echo "⚠️  警告: flags.yaml 中的 run 参数当前为 '$RUN_VALUE'"
        echo "   建议设置为 'semma' 以确保使用正确的模型"
        echo "   继续运行..."
        echo ""
    fi
fi

for i in $(seq 1 $N_RUNS); do
    seed=${SEEDS[$((i-1))]}
    echo ""
    echo "=========================================="
    echo "运行 $i/$N_RUNS - 随机种子: $seed"
    echo "=========================================="
    
    python script/pretrain.py \
        -c $CONFIG \
        --gpus $GPUS \
        --seed $seed
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ 运行 $i 失败，停止后续运行"
        exit 1
    else
        echo ""
        echo "✅ 运行 $i 完成"
    fi
done

echo ""
echo "=========================================="
echo "所有运行完成！"
echo "=========================================="
echo ""
echo "结果保存在: output/Ultra/JointDataset/"
echo "每次运行会生成独立的时间戳目录"

