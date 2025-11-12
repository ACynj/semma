#!/bin/bash
# 清理重复或不太重要的图表
# 保留最重要的分析图表

cd /T20030104/ynj/semma/analyze/figures

# 建议删除的图表（可以根据需要调整）
# 这些图表可能与其他图表有重复或功能相似

# 删除建议：
# - 图表3: 3_dataset_type_comparison.png (与图表7-9有重复)
# - 图表6: 6_performance_statistics.png (信息已在其他图表中体现)
# - 图表10: 10_key_datasets_features_radar.png (与图表5有重复，且图表15更详细)
# - 图表11: 11_performance_vs_base.png (信息已在图表2中体现)

echo "建议删除以下重复图表："
echo "  3_dataset_type_comparison.png (与图表7-9重复)"
echo "  6_performance_statistics.png (信息已体现在其他图表中)"
echo "  10_key_datasets_features_radar.png (与图表5和15重复)"
echo "  11_performance_vs_base.png (信息已在图表2中体现)"
echo ""
read -p "是否删除这些图表? (y/n): " confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    rm -f 3_dataset_type_comparison.png
    rm -f 6_performance_statistics.png
    rm -f 10_key_datasets_features_radar.png
    rm -f 11_performance_vs_base.png
    echo "✅ 已删除重复图表"
else
    echo "❌ 取消删除"
fi

