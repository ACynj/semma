#!/usr/bin/env python
"""
刷新Optuna Dashboard的可视化图表
修复历史曲线显示问题
"""

import os
import sys
import optuna
import optuna.visualization as vis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def refresh_visualizations():
    """重新生成所有可视化图表"""
    study_db = "/T20030104/ynj/semma/optuna_tune/trials/study.db"
    viz_dir = "/T20030104/ynj/semma/optuna_tune/trials/visualizations"
    
    if not os.path.exists(study_db):
        print(f"❌ Optuna数据库不存在: {study_db}")
        return False
    
    print("="*70)
    print("🔄 刷新Optuna可视化图表")
    print("="*70)
    
    # 加载study
    try:
        study = optuna.load_study(
            study_name="enhancement_params_tuning",
            storage=f"sqlite:///{study_db}"
        )
    except Exception as e:
        print(f"❌ 加载study失败: {e}")
        return False
    
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"\n📊 当前trials状态:")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"  完成trials: {len(completed)}/{len(study.trials)}")
    for t in completed:
        value = -t.value if t.value != float('inf') else 'inf'
        if isinstance(value, float):
            print(f"    Trial {t.number}: value={value:.4f}")
        else:
            print(f"    Trial {t.number}: value={value}")
    
    # 重新生成所有图表
    print(f"\n📈 重新生成可视化图表...")
    
    try:
        # 1. 优化历史
        print("  1. 优化历史曲线...")
        fig = vis.plot_optimization_history(study)
        fig.write_html(os.path.join(viz_dir, "optimization_history.html"))
        print(f"     ✓ 已保存: optimization_history.html")
    except Exception as e:
        print(f"     ✗ 失败: {e}")
    
    # 2. 参数重要性
    if len(completed) > 3:
        try:
            print("  2. 参数重要性...")
            fig = vis.plot_param_importances(study)
            fig.write_html(os.path.join(viz_dir, "param_importances.html"))
            print(f"     ✓ 已保存: param_importances.html")
        except Exception as e:
            print(f"     ✗ 失败: {e}")
    
    # 3. 等高线图
    if len(completed) > 5:
        try:
            print("  3. 参数等高线图...")
            fig = vis.plot_contour(
                study, 
                params=['similarity_threshold_init', 'enhancement_strength_init']
            )
            fig.write_html(os.path.join(viz_dir, "contour_plot.html"))
            print(f"     ✓ 已保存: contour_plot.html")
        except Exception as e:
            print(f"     ✗ 失败: {e}")
    
    # 4. 平行坐标图
    if len(completed) > 5:
        try:
            print("  4. 平行坐标图...")
            fig = vis.plot_parallel_coordinate(study)
            fig.write_html(os.path.join(viz_dir, "parallel_coordinate.html"))
            print(f"     ✓ 已保存: parallel_coordinate.html")
        except Exception as e:
            print(f"     ✗ 失败: {e}")
    
    # 5. 参数关系图
    if len(completed) > 2:
        try:
            print("  5. 参数关系图...")
            fig = vis.plot_slice(study)
            fig.write_html(os.path.join(viz_dir, "slice_plot.html"))
            print(f"     ✓ 已保存: slice_plot.html")
        except Exception as e:
            print(f"     ✗ 失败: {e}")
    
    print(f"\n✅ 可视化图表已刷新")
    print(f"   图表目录: {viz_dir}")
    print(f"\n💡 提示:")
    print(f"   - 如果Optuna Dashboard仍显示旧数据，请刷新浏览器页面（Ctrl+F5）")
    print(f"   - 或者重启Optuna Dashboard: optuna-dashboard sqlite:///{study_db}")
    
    return True

if __name__ == "__main__":
    success = refresh_visualizations()
    sys.exit(0 if success else 1)

