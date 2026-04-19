# 实验清单（Experiment Index）

本文档仅记录当前仓库中已存在的实验及其对应文件夹位置，不包含结果分析。

## 实验总览

当前在 `results/` 下发现以下实验组（共 7 组）：

1. `AudioSet`
2. `bell`
3. `pink_wind`
4. `pure_pink`
5. `speech_baseline`
6. `test`
7. `wind_sub`

## 实验与目录映射

### 1) AudioSet
- 实验目录：`results/AudioSet/`
- 训练目录：`results/AudioSet/training/`
- 配置快照：`results/AudioSet/training/multirun.yaml`

### 2) bell
- 实验目录：`results/bell/`
- 训练目录：`results/bell/training/`
- 配置快照：`results/bell/training/multirun.yaml`

### 3) pink_wind
- 实验目录：`results/pink_wind/`
- 训练目录：`results/pink_wind/training/`
- 配置快照：`results/pink_wind/training/multirun.yaml`

### 4) pure_pink
- 实验目录：`results/pure_pink/`
- 训练目录：`results/pure_pink/training/`
- 配置快照：`results/pure_pink/training/multirun.yaml`

### 5) speech_baseline
- 实验目录：`results/speech_baseline/`
- 训练目录：`results/speech_baseline/training/`
- 配置快照：`results/speech_baseline/training/multirun.yaml`

### 6) test
- 实验目录：`results/test/`
- 训练目录：`results/test/training/`
- 配置快照：`results/test/training/multirun.yaml`

### 7) wind_sub
- 实验目录：`results/wind_sub/`
- 训练目录：`results/wind_sub/training/`
- 配置快照：`results/wind_sub/training/multirun.yaml`

## 相关实验脚本（索引）

以下脚本与实验配置生成/运行相关，供快速定位：

- `exp.py`
- `exp_audioset.py`
- `exp_audioset_baseline.py`
- `exp_speechcommand_baseline.py`
- `audioset_wind_test.py`
- `speechcommand_wind_test.py`
- `wind_sub_col_test.py`
- `pink_test.py`
