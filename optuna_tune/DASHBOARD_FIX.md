# Optuna Dashboard 历史曲线修复指南

## 问题描述

Optuna Dashboard的历史曲线显示可能有问题，即使数据已经正确更新。

## 当前数据状态

✅ **所有trials数据已正确更新：**

| Trial | 实验次数 | 参数 | 分数 | 状态 |
|-------|---------|------|------|------|
| 0 | 第一次 | (0.65, 0.15) | 0.6706 | 完成 |
| 1 | 第二次 | (0.85, 0.09) | 0.6786 | 完成 |
| 2 | 第三次 | (0.55, 0.03) | **0.7023** | 完成（最佳） |
| 3 | 第四次 | (0.5, 0.13) | 0.6708 | 完成 |
| 4 | 第五次 | (0.8, 0.11) | 0.6748 | 完成 |
| 5 | 第六次 | (0.5, 0.15) | - | 运行中 |

**最佳值历史（应该单调递增）：**
- Trial 0: 0.6706
- Trial 1: 0.6786 ↑
- Trial 2: 0.7023 ↑ (当前最佳)
- Trial 3: 0.6708 ↓ (但最佳仍为0.7023)
- Trial 4: 0.6748 ↑ (但最佳仍为0.7023)

## 解决方案

### 方法1：刷新浏览器（最简单）

1. **强制刷新浏览器页面**：
   - Windows/Linux: `Ctrl + F5` 或 `Ctrl + Shift + R`
   - Mac: `Cmd + Shift + R`

2. **清除浏览器缓存**：
   - 打开浏览器开发者工具（F12）
   - 右键点击刷新按钮
   - 选择"清空缓存并硬性重新加载"

### 方法2：重启Optuna Dashboard

```bash
# 1. 停止当前的Dashboard（在运行Dashboard的终端按Ctrl+C）

# 2. 重新启动Dashboard
conda activate semma
optuna-dashboard sqlite:///optuna_tune/trials/study.db

# 3. 在浏览器中打开 http://localhost:8080
```

### 方法3：查看本地HTML图表（推荐）

如果Dashboard仍有问题，可以直接查看本地生成的HTML图表：

```bash
# 打开优化历史图表
firefox optuna_tune/trials/visualizations/optimization_history.html

# 或者使用其他浏览器
google-chrome optuna_tune/trials/visualizations/optimization_history.html
```

这些HTML图表已经重新生成，包含最新的数据。

### 方法4：重新生成可视化图表

如果图表仍然有问题，可以手动重新生成：

```bash
conda activate semma
python optuna_tune/refresh_dashboard.py
```

## 验证数据正确性

运行以下命令验证数据是否正确：

```bash
conda activate semma
python optuna_tune/check_status.py
```

或者直接查看数据库：

```bash
conda activate semma
python -c "
import optuna
study = optuna.load_study(
    study_name='enhancement_params_tuning',
    storage='sqlite:///optuna_tune/trials/study.db'
)
print('所有trials:')
for i, trial in enumerate(study.trials):
    if trial.state == optuna.trial.TrialState.COMPLETE:
        value = -trial.value if trial.value != float('inf') else None
        print(f'  Trial {i}: {value:.4f if value else \"inf\"}, params={trial.params}')
print(f'最佳trial: Trial {study.best_trial.number}, 分数={-study.best_trial.value:.4f}')
"
```

## 常见问题

### Q: 历史曲线显示的值不对？

**A:** 可能是浏览器缓存问题，尝试方法1（强制刷新）。

### Q: 曲线显示为空白或错误？

**A:** 尝试方法2（重启Dashboard）或方法3（查看本地HTML）。

### Q: 最佳值曲线没有单调递增？

**A:** 这是正常的！最佳值曲线应该显示"到当前为止的最佳值"，所以：
- 如果新trial的值更好，曲线会上升
- 如果新trial的值更差，曲线会保持水平（不下降）

### Q: 时间轴显示有问题？

**A:** 检查数据库中的时间戳是否正确：
```bash
sqlite3 optuna_tune/trials/study.db "SELECT number, datetime_start, datetime_complete FROM trials ORDER BY number"
```

## 已重新生成的图表

以下图表已经重新生成，包含最新的5个完成的trials：

1. **optimization_history.html** - 优化历史曲线
2. **param_importances.html** - 参数重要性
3. **slice_plot.html** - 参数关系图

这些图表保存在：`optuna_tune/trials/visualizations/`

## 联系

如果问题仍然存在，请检查：
1. Optuna Dashboard的版本是否最新
2. 浏览器控制台是否有错误信息
3. 数据库文件是否可读

