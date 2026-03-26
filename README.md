# 非线性阻尼单摆 TCN Sim2Real

基于 TCN 的单变量角度序列预测项目，目标是先在 RK4 合成数据上预训练，再在真实实验数据上微调，形成一条可复现实验链路。

## 环境

```bash
pip install -r requirements.txt
```

## 数据格式

- CSV 按“每行一条完整序列”组织。
- 默认第一列可以是 `sample_id`，由配置项 `data.has_sample_id_column` 控制是否丢弃。
- 其余列按时间顺序排列，单位为 `rad`。
- 每行至少需要满足 `input_steps + pred_steps` 个时间点。

示例：

```text
sample_id,t_00000,t_00001,...,t_21599
sim_00000,-0.2617,-0.2610,...,0.0132
sim_00001,-0.2617,-0.2609,...,0.0128
```

## 运行流程

### 1. 生成仿真数据

```bash
python scripts/01_generate_sim.py --config configs/pretrain_config.yaml
```

### 2. 在仿真数据上做 Optuna 搜参

```bash
python scripts/02_tune_optuna.py --config configs/pretrain_config.yaml
```

### 3. 预训练基座模型

```bash
python scripts/03_pretrain.py --config configs/pretrain_config.yaml
```

输出包括独立 run 目录，以及 `checkpoints/pretrained_model_latest.pt` 最新别名。

### 4. 用预训练模型直接评估真实数据

```bash
python scripts/04_eval_baseline.py --config configs/finetune_config.yaml
```

### 5. 在真实数据上微调

```bash
python scripts/05_finetune.py --config configs/finetune_config.yaml
```

输出包括独立 run 目录，以及 `checkpoints/finetuned_model_latest.pt` 最新别名。

### 6. 评估微调后模型

```bash
python scripts/06_eval_final.py --config configs/finetune_config.yaml
```

## 两条推荐路径

### 单阶段 demo

```bash
python scripts/01_generate_sim.py --config configs/pretrain_config.yaml
python scripts/02_tune_optuna.py --config configs/pretrain_config.yaml
python scripts/03_pretrain.py --config configs/pretrain_config.yaml
```

### 正式 Sim2Real

```bash
python scripts/01_generate_sim.py --config configs/pretrain_config.yaml
python scripts/02_tune_optuna.py --config configs/pretrain_config.yaml
python scripts/03_pretrain.py --config configs/pretrain_config.yaml
python scripts/04_eval_baseline.py --config configs/finetune_config.yaml
python scripts/05_finetune.py --config configs/finetune_config.yaml
python scripts/06_eval_final.py --config configs/finetune_config.yaml
```

## 目录说明

- `configs/`: 预训练和微调的 YAML 配置
- `data/`: 合成数据和真实数据
- `src/`: 数据集、模型、训练器和工具模块
- `scripts/`: 六步实验入口
- `artifacts/runs/`: 每次运行的独立结果目录
- `checkpoints/`: 最新模型别名，不是唯一产物来源

## 配置约定

- `data.fps`, `data.input_seconds`, `data.pred_seconds` 决定时间窗口
- `data.scale_factor` 控制统一缩放，训练和评估全链路复用
- `data.split.persist_path` 用于固定 split，真实数据的 `04/05/06` 必须共用
- `artifacts.output_root` 指定 run 目录根路径
- `artifacts.alias_dir` 指定最新 checkpoint 别名目录

## 真实数据准备

默认真实数据路径是 `data/real/pendulum_real_sequences.csv`。如果路径或 CSV 格式不同，只需要修改 `configs/finetune_config.yaml`。
