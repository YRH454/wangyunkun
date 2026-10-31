# 9. 独立训练器：功能、边界与运行方法

本项目**没有原作者代码**。下面是依据本项目协议独立实现的学习性方案，只用虚构数据做过功能测试，不证明与论文公式、工程细节或历史结果一致。正式结果必须由有权使用的数据、事先冻结的设置和真实运行产生；不允许将新实验写成原作者的旧结果。

## 9.1 目前代码覆盖什么

| 方法 / 功能 | 命令参数 / 实现范围 |
|---|---|
| 同骨干监督 | `--condition ce`：字符级 ESIM 风格 BiLSTM + CE；`ce_scl` 加监督对比损失 |
| 两阶段 | `A3`：CE+SCL 后，恢复第一阶段验证集最优权重、重置优化器，再做 CE+UL |
| A0—A8 | `--condition A0` … `A8`，与[消融矩阵](03_experiment_matrix.md)对应；A7 用 `--u-fraction 0.25/0.5/1.0` |
| 自训练 | `self_train`：第一阶段 CE；最优权重一次性推断 U，置信度默认至少 0.95 才接收；第二阶段 `CE(L) + 0.5×CE(伪标签)`，记录接收数量及类别。零接收时第二阶段只有 CE，须报告这一事实 |
| 一致性 | `consistency`：第一阶段 CE；第二阶段用停止梯度的原文预测和相同术语保护掩蔽视图，置信度默认至少 0.95，强视图伪标签 CE 默认权重 0.5；是本项目 FixMatch-style 对照，不是官方实现 |
| 预测与指标 | 保存本地逐样本概率、预测、真值；输出二类混淆矩阵、各类 P/R/F1、macro-F1、accuracy、balanced accuracy |
| 同条件汇总 | 在检查数据划分、模型配置、实现哈希和种子后，计算各指标均值和样本标准差；明确列出缺少的预设种子 |
| CPU 测量 | 验证集上 batch=1，分别计时模型本体和含字符编码的端到端推理；默认单线程、预热100次、计时1000次，记录中位数和p95。也记录参数量和checkpoint文件大小 |

两类对比损失都只使用**当前微批**作为负例范围；梯度累积不会扩大负例集合。SCL 无同类正例时跳过该锚点，整批无有效锚点时为零并计数；目前随机洗牌、微批内不重复样本，但**不强制每类均有两个样本**。UL 将相同规范化对从负例中屏蔽；原文和增强视图相同、或有效视图不足2条时跳过并计数。未知的语义假负例仍存在。

字符词表只由**选中的有标签训练 L** 拟合，所有条件采用相同规则；U 可能含未登录字符。掩蔽在前 `max_length` 个原始字符上使用共同字符类型；术语匹配查看完整原句。A6 的随机对照与术语保护对照按每对样本的**被掩蔽类型数量**匹配，但相同类型数量不保证相同的字符位置比例，必须核查实际日志、报告差异并审查增强是否改变标签含义。A5 使用去标签的 L 自身作为增强视图来源；A7 对已选 U 做固定种子嵌套前缀抽样。

默认单阶段4000步上限，两阶段各2000步，微批L=16，累积2次；U方法另用U微批16。每200步验证、连续4次未显著改善时提前停止；最高验证macro-F1的checkpoint被保存，严格同分保留最早一步。早停计数的显著改善门槛为0.0001。可用 `--fixed-steps` 关闭早停以实施**事先登记**的匹配步数控制，但仍然按验证集选checkpoint；这不自动保证计算量或样本曝光完全匹配。实际步数、L/U曝光、mask位置、跳过次数、伪标签接收数、模型参数、训练时间及可用的GPU显存峰值保存在本地。

**尚未实现：** BERT／轻量预训练模型及其下载和训练、Mean Teacher、领域继续预训练、自动6组调参调度、组级bootstrap及Wilcoxon/Holm检验、绘图／句长和误差分析、FLOPs及CPU峰值进程内存测量、将人工复核决定自动应用于新划分。这些不能因为有方案或配置就称为已完成。

## 9.2 真实训练之前的门槛

1. 分别确认数据及模型的训练使用权限；复核文本来源、标签冲突、近重复、生成U的来源与质量。取得可用于本项目的术语词表，不能把合成测试词表称作农业词表或原作者词表。
2. 根据[划分说明](08_clean_split_workflow.md)完成新划分和人工复核，并运行 `verify_clean_split.py` 生成与当前 `manifest.json` 哈希一致的本地报告。该报告只覆盖其明示的精确及所选表面规则，不证明语义、来源或生成器无泄漏。
3. 生成本地 LDP 子集；同一数据集、比例和采样种子下，所有比较方法使用同一个子集目录和相同 L/U。外部U与剩余训练样本去标签U是不同训练信息，不能混称。
4. 在目标机器确认 Python/PyTorch/CUDA 兼容、运行合成测试及小规模本地 smoke。正式实验前预先冻结调参预算、验证集选模规则、种子和统计计划；不得用测试集调参。

训练器要求显式提供 `--confirm-data-rights --confirm-reviewed-data`。它们是操作者声明，**程序无法核实授权是否真实、复核是否充分或所有泄漏是否排除**。只有这些事情确实完成、剩余限制已记录时才能填写。U输出禁止包含标签字段，训练器校验子集与父训练池的对应关系；父训练池中未被选中的标签不会传给模型。训练阶段只读取L、必要时读取U及验证集，**不会打开测试文件**。训练前的数据审计会读取测试集以检查划分；与模型调参分开。

## 9.3 运行示例

以下是在 **Windows 10 / Quadro RTX 5000 目标服务器的 PowerShell** 中运行的示意命令，路径指服务器上实际存在的位置，并非假定开发电脑与服务器共享磁盘。先完成真实数据授权、人工复核，并在服务器上确认驱动、PyTorch 和 CUDA 兼容；若GPU不可用，先排查环境，不把开发电脑的结果当作服务器结果。`runs/`、`data/`、`reports/` 被 Git 忽略，但仍须在提交前检查，不能依靠忽略规则保证隐私。

```powershell
# 当前 split、LDP 与核验报告均是经审查锁定的本地版本
$split = 'data\agri_clean'
$ldp = 'data\agri_ldp30_seed11'
$verification = 'reports\agri_clean_verification.json'
$terms = 'data\reviewed_agricultural_terms.txt' # UTF-8，一行一个有来源的术语

# 只在授权及人工复核真实完成后填写这两个确认参数
python -B scripts/train_pairmatch.py `
  --split-dir $split --subset-dir $ldp --verification-report $verification `
  --run-dir 'runs\agri_ce_30_seed11' --condition ce --seed 11 `
  --confirm-data-rights --confirm-reviewed-data --device cuda

python -B scripts/train_pairmatch.py `
  --split-dir $split --subset-dir $ldp --verification-report $verification `
  --run-dir 'runs\agri_a3_30_seed11' --condition A3 --seed 11 `
  --term-lexicon $terms --confirm-data-rights --confirm-reviewed-data --device cuda
```

`ce`、`ce_scl`、`A0`、`A1` 和 `self_train` 不需要术语词表；`A2`—`A8` 和 `consistency` 需要。`self_train` 和 `consistency` 均需实际U，但 `A5` 使用L自身的增强视图。训练完成目录包含 `config.json`、`provenance.json`、阶段最优权重、训练／验证日志和 `summary.json`；中途失败则保留 `failure.json`。配置、词表和逐样本产物可能敏感，只留本地。目录已存在时拒绝覆盖，包括失败目录；每次重试使用新目录。

在完成预先确定的调参、模型和分析计划后，才能对最终checkpoint运行测试；确认参数只是人为声明，不是程序替代研究纪律。输出目录同样必须是全新的：

```powershell
python -B scripts/predict_pairmatch.py `
  --checkpoint 'runs\agri_a3_30_seed11\stage_2_best.pt' `
  --split-dir $split --split test --confirm-test-locked `
  --output-dir 'runs\agri_a3_30_seed11_test'
```

脚本校验测试文件哈希和完成状态、仅允许该次训练最后阶段的最优checkpoint参与测试。不要根据测试结果回选阶段、种子、阈值或超参数。生成的 `predictions.jsonl` 含样本ID、标签和概率，不应上传到公开仓库。

完成多个已预设种子后，可将各自的 `metrics.json` 路径交给 `summarize_results.py --metrics <文件1> <文件2> ... --output <新文件>`。该汇总仅覆盖同一条件与同一测试版本，且要求采样种子与训练种子一致；它**不等于**方法之间的配对检验或置信区间。五个种子为11、23、37、47、59。

目标服务器完成训练后，可对验证集运行本地CPU测量：

```powershell
python -B scripts/benchmark_pairmatch.py `
  --checkpoint 'runs\agri_a3_30_seed11\stage_2_best.pt' `
  --split-dir $split --confirm-data-rights `
  --output 'runs\agri_a3_30_seed11_cpu_benchmark.json'
```

程序按文件次序取前32条验证对并循环使用，不是具有代表性长度分层抽样；比较模型时必须使用相同输入、长度、线程和设备负载。模型本体不含编码，端到端含编码与构造张量；两者均在输出logit后结束，不含softmax／预测后处理或磁盘读取。当前不测CPU峰值进程内存或FLOPs。不同设备、驱动或软件的耗时不可混称为目标服务器实测。
