# TCN 时序预测项目：非线性阻尼单摆的 Sim2Real 轨迹预测

## 1. 🌟 项目背景与简介 (Project Context)
本项目针对**非线性阻尼单摆的运动学轨迹预测**问题而设计（适用于物理实验竞赛等场景）。单摆在空气中摆动时，由于受到与速度成正比的线性阻力（Stokes阻力）以及与速度平方成正比的非线性阻力（空气动力学阻力）的共同作用，其微分方程极其复杂，难以求得解析解。

本项目抛弃传统的纯方程拟合思路，采用**纯数据驱动的深度学习方法**。通过构建**时间卷积网络 (Temporal Convolutional Network, TCN)**，让 AI 模型直接从历史观测序列中自动提取相位、周期、阻尼衰减等物理特征，并对未来的运动轨迹进行高精度预测。

## 2. 🎯 核心目标与最终效果 (Objectives & Expected Outcomes)
### 核心挑战
* **长时序依赖**：需要通过长达 3 分钟的历史数据，精准预测未来 1 分钟的轨迹，极其考验模型的“长程记忆”能力。
* **Domain Gap (域偏差)**：仿真数据（理想情况+高斯噪声）与真实实验数据（存在传感器误差、系统误差、未知摩擦力）之间存在天然的分布差异。

### 我们要实现的效果
1. **构建基座模型 (Pre-trained Model)**：利用 RK4 数值积分生成的数千组仿真数据，训练出一个能完美掌握单摆运动周期的 TCN 基座模型（测试集 MAE 控制在 `0.001 rad` 级别）。
2. **打通 Sim2Real 链路**：引入真实实验采集的数据，对基座模型进行极低学习率的微调（Fine-tuning）。
3. **可视化产出**：自动生成极其直观的 4 分钟完整轨迹对比图（前 3 分钟灰色历史 + 后 1 分钟预测与真实值的红绿重合曲线），直观展示微调带来的精度跃升。

## 3. 🧠 技术路线与架构解析 (Technical Architecture)
### 3.1 核心网络：残差膨胀 TCN
* **放弃 RNN/LSTM**：时序过长（数万帧），RNN 会面临严重的梯度消失和计算低效问题。
* **膨胀因果卷积 (Dilated Causal Conv)**：确保预测不泄露未来信息（因果），并通过指数级增长的膨胀系数（$d=1, 2, 4...$）在不丢失时间分辨率的前提下，将感受野扩大到足以覆盖 3 分钟历史。
* **残差连接 (Residual Links)**：保障 9 层深度网络的梯度稳定回传。

### 3.2 优化引擎：Optuna 贝叶斯寻优
* 采用 TPE (Tree-structured Parzen Estimator) 算法，自动在验证集上搜索最优的 `Kernel Size`、`Layers` 和 `Learning Rate`，并使用 Median Pruner 机制提前剪枝劣质试验，节约算力。

## 4. 📊 数据规格约定 (Data Specification)
* **采样率 (FPS)**：`90 Hz`
* **输入长度 (Input Sequence)**：前 `3 分钟` $\rightarrow$ $3 \times 60 \times 90 = 16200$ 帧
* **输出长度 (Prediction Horizon)**：未来 `1 分钟` $\rightarrow$ $1 \times 60 \times 90 = 5400$ 帧
* **特征维度**：单变量时间序列，仅输入角度 $\theta$ (单位：rad)。

---

## 5. 📂 目录结构规范 (Directory Structure)
本项目采用高度模块化的工程规范，严格分离**数据、配置与源码**。

```text
tcn_project/
├── .gitignore             # 忽略大型数据文件和缓存
├── README.md              # 项目白皮书
├── requirements.txt       # 环境依赖 (torch, optuna, pandas, matplotlib)
│
├── configs/               # ⚙️ 配置文件目录
│   ├── pretrain_config.yaml  # 仿真数据预训练参数 (例如 lr: 1e-3, batch_size: 16)
│   └── finetune_config.yaml  # 真实数据微调参数 (例如 lr: 1e-5, batch_size: 4)
│
├── data/                  # 🗂️ 数据流转中心 (严格禁止 git commit)
│   ├── synthetic/         # RK4 合成数据 (大规模)
│   └── real/              # 实验采集数据 (小规模)
│
├── src/                   # 🧠 核心算法层 (高度内聚，可被外部自由调用)
│   ├── __init__.py
│   ├── dataset.py         # 封装 Dataset/DataLoader，处理截断与归一化
│   ├── models/
│   │   └── tcn.py         # 纯净的 TCN 网络拓扑 (Chomp1d, TemporalBlock, TCN)
│   ├── utils/
│   │   ├── metrics.py     # 计算 MAE, RMSE
│   │   └── plot.py        # 封装绘图逻辑 (生成红绿对比图、误差直方图)
│   └── trainer.py         # 封装 Epoch 循环、AMP 混合精度、权重保存机制
│
├── checkpoints/           # 💾 模型权重归档
│   ├── pretrained_model.pth  
│   └── finetuned_model.pth   
│
└── scripts/               # 🚀 执行层 (只做胶水逻辑，调度 src 模块)
    ├── 01_generate_sim.py    # RK4 生成仿真数据
    ├── 02_tune_optuna.py     # 贝叶斯搜索最优架构
    ├── 03_pretrain.py        # 训练基座模型 -> pretrained_model.pth
    ├── 04_eval_baseline.py   # 用 pretrained_model 测试真实数据，确立误差基线
    ├── 05_finetune.py        # 加载真实数据微调 -> finetuned_model.pth
    └── 06_eval_final.py      # 最终效果测试与全面出图
```

---

## 6. 🛠️ 阶段性开发任务书 (Development Roadmap)

### Phase 1: 基础设施构建 (Infrastructure) - `估计耗时: 1 天`
* **任务 1.1**：初始化 Git，编写 `.gitignore` 过滤 `data/` 和 `checkpoints/`。
* **任务 1.2**：实现 `src/models/tcn.py`，确保网络层不耦合任何外部业务逻辑，仅做张量的前向传播。
* **任务 1.3**：实现 `src/dataset.py`，支持读取 CSV 矩阵并自动切片为 `(1, 16200)` 和 `(1, 5400)` 的 Tensor。
* **任务 1.4**：实现 `src/trainer.py`，封装出一个 `Trainer` 类，支持 `train()` 和 `validate()` 方法，支持加载现成权重。

### Phase 2: 仿真数据闭环 (Sim Pipeline) - `估计耗时: 1-2 天`
* **任务 2.1**：将 RK4 生成脚本迁移至 `scripts/01_generate_sim.py`，设定生成 1000 组数据并落盘至 `data/synthetic/`。
* **任务 2.2**：打通 `scripts/02_tune_optuna.py`，在合成数据上跑通 TPE 搜索，确定 `kernel_size`, `layers` 的最佳组合。
* **任务 2.3**：完善 `configs/pretrain_config.yaml`，填入最优超参。编写 `scripts/03_pretrain.py` 跑完全量训练，验证训练集的收敛性。

### Phase 3: 真实数据微调验证 (Sim2Real Pipeline) - `估计耗时: 2 天`
* **任务 3.1**：清理真实实验获取的 CSV 数据，格式对齐后放入 `data/real/`。
* **任务 3.2**：编写 `scripts/04_eval_baseline.py`，直接把 `pretrained_model.pth` 扔到真实测试集上跑，利用 `src/utils/metrics.py` 记录下未经微调的 MAE（必定会存在一定的误差放大）。
* **任务 3.3**：配置 `finetune_config.yaml`。编写 `scripts/05_finetune.py`，实例化模型后先 `model.load_state_dict()`，然后在真实数据上用极小的 `lr` 跑少量 Epoch。

### Phase 4: 结果展示与报告生成 (Evaluation & Plotting) - `估计耗时: 1 天`
* **任务 4.1**：编写 `scripts/06_eval_final.py`，对微调后的模型进行最终考核。
* **任务 4.2**：完善 `src/utils/plot.py`，自动渲染两类核心图表：
  * **轨迹对比图**：生成包含 3 分钟灰色输入与 1 分钟红绿预测的精美大图，要求支持局部放大 (Zoom-in) 显示交界处的相位差。
  * **误差分布直方图**：统计测试集所有样本的 MAE 分布。

---

## 7. 🤝 编码与协作规范 (Coding Standards)
1. **零硬编码原则**：代码中绝对不允许出现孤立的数字（如 `16200`, `90`）。所有物理常数、网络参数必须从 YAML 文件读取。
2. **面向对象封装**：`scripts/` 下的代码应该像伪代码一样干净，具体的循环逻辑和绘图 API 必须隐藏在 `src/` 中。
3. **环境隔离**：新开发者必须先运行 `pip install -r requirements.txt`。
