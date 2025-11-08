#!/bin/bash
# 在GPU空闲时运行Trial 2的补充评估
# 使用方法: bash optuna_tune/run_trial2_completion.sh

cd /T20030104/ynj/semma

echo "=========================================="
echo "Trial 2补充评估脚本"
echo "=========================================="
echo ""
echo "此脚本将评估缺失的2个数据集（FB15k237和WN18RR）"
echo "并更新Trial 2的完整结果"
echo ""

# 检查GPU是否可用
check_gpu() {
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | while IFS=',' read -r used total; do
        used=$(echo $used | xargs)
        total=$(echo $total | xargs)
        free=$((total - used))
        free_gb=$((free / 1024))
        
        if [ $free_gb -lt 10 ]; then
            echo "⚠ GPU内存不足: 仅剩 ${free_gb}GB 可用（需要至少10GB）"
            echo "   当前有其他进程在使用GPU"
            return 1
        else
            echo "✓ GPU可用: ${free_gb}GB 空闲"
            return 0
        fi
    done
}

# 等待GPU可用
wait_for_gpu() {
    echo "等待GPU空闲..."
    while ! check_gpu; do
        echo "  $(date '+%Y-%m-%d %H:%M:%S') - GPU仍在使用中，等待30秒..."
        sleep 30
    done
    echo "✓ GPU已空闲，开始评估..."
}

# 主函数
main() {
    if [ "$1" == "--wait" ]; then
        wait_for_gpu
    else
        if ! check_gpu; then
            echo ""
            echo "请选择："
            echo "  1. 等待GPU空闲后自动运行（推荐）"
            echo "  2. 稍后手动运行: conda activate semma && python optuna_tune/complete_trial2_evaluation.py"
            echo ""
            read -p "输入选择 (1/2): " choice
            if [ "$choice" == "1" ]; then
                wait_for_gpu
            else
                echo "稍后请运行: conda activate semma && python optuna_tune/complete_trial2_evaluation.py"
                exit 0
            fi
        fi
    fi
    
    # 运行评估
    echo ""
    echo "开始评估..."
    conda run -n semma python optuna_tune/complete_trial2_evaluation.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 评估完成！Trial 2的结果已更新"
    else
        echo ""
        echo "❌ 评估失败，请检查错误信息"
    fi
}

main "$@"

