#!/bin/bash
# 对比有无KG-ICL Prompt Enhancement的性能测试

echo "=============================================="
echo "KG-ICL Prompt Enhancement 性能对比测试"
echo "=============================================="

DATASET="FB15k237_10"
CONFIG="config/transductive/inference-fb.yaml"
CHECKPOINT="ckpts/semma.pth"

# 确保在semma环境
source ~/anaconda3/bin/activate
conda activate semma

echo ""
echo "1. 测试基线模型（禁用KG-ICL enhancement）"
echo "----------------------------------------------"

# 临时修改flags.yaml
cp flags.yaml flags.yaml.backup
sed -i 's/use_kg_icl_prompt: True/use_kg_icl_prompt: False/' flags.yaml

# 运行评估
python script/run.py \
    -c $CONFIG \
    --dataset $DATASET \
    --epochs 0 \
    --bpe null \
    --ckpt $CHECKPOINT \
    --gpus [0] \
    > baseline_results.log 2>&1

echo "基线测试完成，结果保存到 baseline_results.log"

echo ""
echo "2. 测试增强模型（启用KG-ICL enhancement）"
echo "----------------------------------------------"

# 恢复配置
mv flags.yaml.backup flags.yaml

# 运行评估
python script/run.py \
    -c $CONFIG \
    --dataset $DATASET \
    --epochs 0 \
    --bpe null \
    --ckpt $CHECKPOINT \
    --gpus [0] \
    > enhanced_results.log 2>&1

echo "增强测试完成，结果保存到 enhanced_results.log"

echo ""
echo "=============================================="
echo "性能对比"
echo "=============================================="

# 提取并对比指标
echo ""
echo "基线性能:"
grep -E "(MRR|Hits@)" baseline_results.log | head -n 5 || echo "请查看 baseline_results.log"

echo ""
echo "增强性能:"
grep -E "(MRR|Hits@)" enhanced_results.log | head -n 5 || echo "请查看 enhanced_results.log"

echo ""
echo "详细结果请查看:"
echo "  - baseline_results.log"
echo "  - enhanced_results.log"






