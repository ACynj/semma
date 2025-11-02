#!/bin/bash
# 相似度增强功能测试脚本
# 对比基线SEMMA和EnhancedUltra的性能

set -e  # 遇到错误立即退出

echo "=============================================="
echo "相似度增强功能测试 - 性能对比"
echo "=============================================="

# 配置
DATASET="${1:-FB15k237}"  # 默认使用FB15k237
CONFIG="config/transductive/inference.yaml"
CHECKPOINT="${2:-ckpts/semma.pth}"  # 默认checkpoint
GPUS="${3:-[0]}"  # 默认GPU

echo ""
echo "配置信息:"
echo "  数据集: $DATASET"
echo "  配置文件: $CONFIG"
echo "  Checkpoint: $CHECKPOINT"
echo "  GPUs: $GPUS"
echo ""

# 确保在项目根目录
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)

# 备份flags.yaml
if [ -f "flags.yaml" ]; then
    cp flags.yaml flags.yaml.backup
    echo "✅ 已备份 flags.yaml"
else
    echo "❌ 错误: 找不到 flags.yaml"
    exit 1
fi

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "⚠️  警告: Checkpoint文件不存在: $CHECKPOINT"
    echo "   请先训练模型或指定正确的checkpoint路径"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        mv flags.yaml.backup flags.yaml
        exit 1
    fi
fi

# 创建结果目录
RESULTS_DIR="test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo "✅ 结果将保存到: $RESULTS_DIR"

# ==========================================
# 测试1: 基线SEMMA（无增强）
# ==========================================
echo ""
echo "=============================================="
echo "测试1: 基线SEMMA（无增强）"
echo "=============================================="

# 设置flags.yaml为基线
sed -i 's/run: EnhancedUltra/run: semma/' flags.yaml || sed -i 's/run:.*/run: semma/' flags.yaml

# 运行测试
echo "运行基线测试..."
python script/run.py \
    -c "$CONFIG" \
    --dataset "$DATASET" \
    --epochs 0 \
    --bpe null \
    --ckpt "$CHECKPOINT" \
    --gpus "$GPUS" \
    > "$RESULTS_DIR/baseline_${DATASET}.log" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ 基线测试完成"
else
    echo "❌ 基线测试失败，请查看日志: $RESULTS_DIR/baseline_${DATASET}.log"
    mv flags.yaml.backup flags.yaml
    exit 1
fi

# 提取关键指标
echo ""
echo "基线性能指标:"
grep -E "(mrr|hits@|mr:)" "$RESULTS_DIR/baseline_${DATASET}.log" | tail -5 || echo "未找到指标"

# ==========================================
# 测试2: EnhancedUltra（有增强）
# ==========================================
echo ""
echo "=============================================="
echo "测试2: EnhancedUltra（有增强）"
echo "=============================================="

# 设置flags.yaml为EnhancedUltra
sed -i 's/run: semma/run: EnhancedUltra/' flags.yaml

# 运行测试
echo "运行增强测试..."
python script/run.py \
    -c "$CONFIG" \
    --dataset "$DATASET" \
    --epochs 0 \
    --bpe null \
    --ckpt "$CHECKPOINT" \
    --gpus "$GPUS" \
    > "$RESULTS_DIR/enhanced_${DATASET}.log" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ 增强测试完成"
else
    echo "❌ 增强测试失败，请查看日志: $RESULTS_DIR/enhanced_${DATASET}.log"
    mv flags.yaml.backup flags.yaml
    exit 1
fi

# 提取关键指标
echo ""
echo "增强性能指标:"
grep -E "(mrr|hits@|mr:)" "$RESULTS_DIR/enhanced_${DATASET}.log" | tail -5 || echo "未找到指标"

# ==========================================
# 对比结果
# ==========================================
echo ""
echo "=============================================="
echo "性能对比"
echo "=============================================="

# 提取并对比MRR
BASELINE_MRR=$(grep -oP 'mrr:\s+\K[0-9.]+' "$RESULTS_DIR/baseline_${DATASET}.log" | tail -1 || echo "0")
ENHANCED_MRR=$(grep -oP 'mrr:\s+\K[0-9.]+' "$RESULTS_DIR/enhanced_${DATASET}.log" | tail -1 || echo "0")

if [ "$BASELINE_MRR" != "0" ] && [ "$ENHANCED_MRR" != "0" ]; then
    echo ""
    echo "MRR对比:"
    echo "  基线:   $BASELINE_MRR"
    echo "  增强:   $ENHANCED_MRR"
    
    # 计算提升（使用bc进行浮点数计算）
    if command -v bc &> /dev/null; then
        IMPROVEMENT=$(echo "scale=4; ($ENHANCED_MRR - $BASELINE_MRR) / $BASELINE_MRR * 100" | bc)
        ABSOLUTE_IMPROVEMENT=$(echo "scale=4; $ENHANCED_MRR - $BASELINE_MRR" | bc)
        echo "  绝对提升: +$ABSOLUTE_IMPROVEMENT"
        echo "  相对提升: +$IMPROVEMENT%"
        
        if (( $(echo "$IMPROVEMENT > 0" | bc -l) )); then
            echo "  ✅ 性能提升!"
        elif (( $(echo "$IMPROVEMENT == 0" | bc -l) )); then
            echo "  ➖ 性能持平"
        else
            echo "  ⚠️  性能下降，可能需要调整参数"
        fi
    else
        echo "  (需要bc命令来计算提升百分比)"
    fi
fi

# 提取并对比Hits@1
BASELINE_H1=$(grep -oP 'hits@1:\s+\K[0-9.]+' "$RESULTS_DIR/baseline_${DATASET}.log" | tail -1 || echo "0")
ENHANCED_H1=$(grep -oP 'hits@1:\s+\K[0-9.]+' "$RESULTS_DIR/enhanced_${DATASET}.log" | tail -1 || echo "0")

if [ "$BASELINE_H1" != "0" ] && [ "$ENHANCED_H1" != "0" ]; then
    echo ""
    echo "Hits@1对比:"
    echo "  基线:   $BASELINE_H1"
    echo "  增强:   $ENHANCED_H1"
    
    if command -v bc &> /dev/null; then
        IMPROVEMENT=$(echo "scale=4; ($ENHANCED_H1 - $BASELINE_H1) / $BASELINE_H1 * 100" | bc)
        ABSOLUTE_IMPROVEMENT=$(echo "scale=4; $ENHANCED_H1 - $BASELINE_H1" | bc)
        echo "  绝对提升: +$ABSOLUTE_IMPROVEMENT"
        echo "  相对提升: +$IMPROVEMENT%"
    fi
fi

# 提取并对比Hits@10
BASELINE_H10=$(grep -oP 'hits@10:\s+\K[0-9.]+' "$RESULTS_DIR/baseline_${DATASET}.log" | tail -1 || echo "0")
ENHANCED_H10=$(grep -oP 'hits@10:\s+\K[0-9.]+' "$RESULTS_DIR/enhanced_${DATASET}.log" | tail -1 || echo "0")

if [ "$BASELINE_H10" != "0" ] && [ "$ENHANCED_H10" != "0" ]; then
    echo ""
    echo "Hits@10对比:"
    echo "  基线:   $BASELINE_H10"
    echo "  增强:   $ENHANCED_H10"
    
    if command -v bc &> /dev/null; then
        IMPROVEMENT=$(echo "scale=4; ($ENHANCED_H10 - $BASELINE_H10) / $BASELINE_H10 * 100" | bc)
        ABSOLUTE_IMPROVEMENT=$(echo "scale=4; $ENHANCED_H10 - $BASELINE_H10" | bc)
        echo "  绝对提升: +$ABSOLUTE_IMPROVEMENT"
        echo "  相对提升: +$IMPROVEMENT%"
    fi
fi

# ==========================================
# 清理和总结
# ==========================================

# 恢复flags.yaml
mv flags.yaml.backup flags.yaml
echo ""
echo "✅ 已恢复 flags.yaml"

# 总结
echo ""
echo "=============================================="
echo "测试完成！"
echo "=============================================="
echo ""
echo "结果文件:"
echo "  基线日志: $RESULTS_DIR/baseline_${DATASET}.log"
echo "  增强日志: $RESULTS_DIR/enhanced_${DATASET}.log"
echo ""
echo "查看详细结果:"
echo "  cat $RESULTS_DIR/baseline_${DATASET}.log | grep -E '(mrr|hits@)'"
echo "  cat $RESULTS_DIR/enhanced_${DATASET}.log | grep -E '(mrr|hits@)'"
echo ""


