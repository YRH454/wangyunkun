# 9. 独立训练器：功能、边界与运行方法

本仓库代码是依据论文与材料整理的独立实现；功能测试使用合成数据，不能把新运行结果写成历史结果。真实训练使用有权使用的数据和事先确定的设置。

## 9.1 目前代码覆盖什么

| 方法 / 功能 | 命令参数 / 实现范围 |
|---|---|
| 同骨干监督 | `--condition ce`：字符级 ESIM 风格 BiLSTM + CE；`ce_scl` 加监督对比损失 |
| 两阶段 | `A3`：CE+SCL 后，恢复第一阶段验证集最优权重、重置优化器，再做 CE+UL |
| A0—A8 | `--condition A0` … `A8`，与[消融矩阵](03_experiment_matrix.md)对应；A7 用 `--u-fraction 0.25/0.5/1.0` |
| 自训练 | `self_train`：第一阶段 CE；最优权重一次性推断 U，置信度默认至少 0.95 才接收；第二阶段 `CE(L) + 0.5×CE(伪标签)`，记录接收数量及类别。零接收时第二阶段只有 CE，须报告这一事实 |
| 一致性 | `consistency`：第一阶段 CE；第二阶段用停止梯度的原文预测和相同术语保护掩蔽视图，置信度默认至少 0.95，强视图伪标签 CE 默认权重 0.5；是本项目 FixMatch-style 对照，不是官方实现 |
| Mean Teacher | `mean_teacher`：第二阶段教师由阶段一最优模型复制，使用原文教师概率与掩蔽后学生概率的 MSE；每次优化器更新后 EMA，默认0.999 |
| dropout 预训练 | `dropout_pretrain`：阶段一只用U的两个 dropout 视图、固定步数末的权重，不用验证标签挑选；阶段二用L监督训练。不是官方 SimCSE 实现 |
| 预训练监督模型 | 独立的 `train_transformer.py` 支持离线 BERT、中文 RoBERTa 和轻量模型，使用同一L和验证集；其外部预训练信息须单列 |
| BERT领域适配 | `adapt_bert_mlm.py` 固定步数仅用训练U做MLM，再用 `train_transformer.py --model bert_domain_adaptation` 在L上微调；每个采样种子单独适配 |
| 预测与指标 | 保存本地逐样本概率、预测、真值；输出二类混淆矩阵、各类 P/R/F1、macro-F1、accuracy、balanced accuracy |
| 同条件汇总 | 在检查数据划分、模型配置、实现哈希和种子后，计算各指标均值和样本标准差；明确列出缺少的预设种子 |
| CPU 测量 | 验证集上 batch=1，分别计时模型本体和含字符编码的端到端推理；默认单线程、预热100次、计时1000次，记录中位数和p95。也记录参数量和checkpoint文件大小 |

两类对比损失都只使用**当前微批**作为负例范围；梯度累积不会扩大负例集合。SCL 无同类正例时跳过该锚点，整批无有效锚点时为零并计数；目前随机洗牌、微批内不重复样本，但**不强制每类均有两个样本**。UL 将相同规范化对从负例中屏蔽；原文和增强视图相同、或有效视图不足2条时跳过并计数。未知的语义假负例仍存在。

使用U的配置按选中的L和可用U训练文本建立字符词表，不使用验证／测试文本。对本仓库识别出的农业训练划分和完整外部U的A3配置，10%、25%、30%和50%档的字符目标分别为2,906、3,228、3,412和3,758；当前实现按字符频次降序、同频按Unicode编码顺序选择，三个特殊标记另计。掩蔽在前 `max_length` 个原始字符上使用共同字符类型；术语匹配查看完整原句。A6 的随机对照与术语保护对照按每对样本的**被掩蔽类型数量**匹配，但相同类型数量不保证相同的字符位置比例。A5 使用去标签的 L 自身作为增强视图来源；A7 对已选 U 做固定种子嵌套前缀抽样。

默认单阶段最多32轮，两阶段各最多16轮，微批L=16，累积2次；U方法另用U微批16。每轮验证、连续4轮未显著改善时提前停止；保存验证macro-F1最高的checkpoint，严格同分保留最早轮次，早停改善门槛为0.0001。训练器也支持显式步数模式；使用非默认参数时应单独记录。实际步数、L/U曝光、掩蔽位置、模型参数、训练时间及可用的GPU显存峰值保存在本地。

**尚未实现：** 自动下载或核验远程模型来源、可靠覆盖所有算子的 FLOPs 计算、自动识别语义泄漏或生成数据质量。实现与服务器上真实训练的操作见[服务器工作流](10_server_workflow.md)。

## 9.2 真实训练之前的门槛

1. 分别确认数据及模型的训练使用权限；复核文本来源、标签冲突、近重复和生成U的来源与质量。正式训练使用仓库提供的119项农业术语词表，不使用合成测试词表。
2. 根据[划分说明](08_clean_split_workflow.md)完成新划分和人工复核，并运行 `verify_clean_split.py` 生成与当前 `manifest.json` 哈希一致的本地报告。该报告只覆盖其明示的精确及所选表面规则，不证明语义、来源或生成器无泄漏。
3. 生成本地 LDP 子集；农业数据用采样种子11，同一比例下各比较方法和五个训练种子使用同一L子集。外部U与剩余训练样本去标签U是不同训练信息，不能混称。
4. 在目标机器确认 Python/PyTorch/CUDA 兼容、运行合成测试及小规模本地 smoke。正式实验前预先冻结调参预算、验证集选模规则、种子和统计计划；不得用测试集调参。

训练器要求显式提供 `--confirm-data-rights --confirm-reviewed-data`。它们是操作者声明，**程序无法核实授权是否真实、复核是否充分或所有泄漏是否排除**。只有这些事情确实完成、剩余限制已记录时才能填写。U输出禁止包含标签字段，训练器校验子集与父训练池的对应关系；父训练池中未被选中的标签不会传给模型。训练阶段只读取L、必要时读取U及验证集，**不会打开测试文件**。训练前的数据审计会读取测试集以检查划分；与模型调参分开。

## 9.3 运行示例

以下是在 **Windows 10 / Quadro RTX 5000 目标服务器的 PowerShell** 中运行的示意命令，路径指服务器上实际存在的位置，并非假定开发电脑与服务器共享磁盘。先完成真实数据授权、人工复核，并在服务器上确认驱动、PyTorch 和 CUDA 兼容；若GPU不可用，先排查环境，不把开发电脑的结果当作服务器结果。仓库公开农业测试集及术语表；其他本地数据和运行产物按授权条件管理。

```powershell
# 当前 split、LDP 与核验报告均是经审查锁定的本地版本
$split = 'data\agri_clean'
$ldp = 'data\agri_ldp30_seed11'
$verification = 'reports\agri_clean_verification.json'
$terms = 'data\agriculture_sentence_clean_v3\agricultural_terms.txt' # UTF-8，每行一个术语
$a3LearningRate = Read-Host '输入该次 A3 训练的学习率'

# 只在授权及人工复核真实完成后填写这两个确认参数
python -B scripts/train_pairmatch.py `
  --split-dir $split --subset-dir $ldp --verification-report $verification `
  --run-dir 'runs\agri_ce_30_seed11' --condition ce --seed 11 `
  --confirm-data-rights --confirm-reviewed-data --device cuda

python -B scripts/train_pairmatch.py `
  --split-dir $split --subset-dir $ldp --verification-report $verification `
  --run-dir 'runs\agri_a3_30_seed11' --condition A3 --seed 11 `
  --term-lexicon $terms --learning-rate $a3LearningRate --confirm-data-rights --confirm-reviewed-data --device cuda
```

`ce`、`ce_scl`、`A0`、`A1`、`self_train`、`dropout_pretrain` 不需要术语词表；`A2`—`A8`、`consistency`、`mean_teacher` 需要。半监督方法需实际U，但 `A5` 使用L自身的增强视图。训练完成目录包含 `config.json`、`provenance.json`、阶段最优权重、训练／验证日志和 `summary.json`；中途失败则保留 `failure.json`。逐样本运行产物应按授权条件管理；目录已存在时拒绝覆盖，每次重试使用新目录。

在完成预先确定的调参、模型和分析计划后，才能对最终checkpoint运行测试；确认参数只是人为声明，不是程序替代研究纪律。输出目录同样必须是全新的：

```powershell
python -B scripts/predict_pairmatch.py `
  --checkpoint 'runs\agri_a3_30_seed11\stage_2_best.pt' `
  --split-dir $split --split test --confirm-test-locked `
  --output-dir 'runs\agri_a3_30_seed11_test'
```

脚本校验测试文件哈希和完成状态、仅允许该次训练最后阶段的最优checkpoint参与测试。不要根据测试结果回选阶段、种子、阈值或超参数。生成的 `predictions.jsonl` 含样本ID、标签和概率，不应上传到公开仓库。

完成多个已预设训练种子后，可将各自的 `metrics.json` 路径交给 `summarize_results.py --metrics <文件1> <文件2> ... --output <新文件>`。该汇总检查同一条件与同一数据版本，农业数据使用固定采样种子11，而训练种子分别为11、23、37、47、59；它不等于方法之间的配对检验或置信区间。

目标服务器完成训练后，可对验证集运行本地CPU测量：

```powershell
python -B scripts/benchmark_pairmatch.py `
  --checkpoint 'runs\agri_a3_30_seed11\stage_2_best.pt' `
  --split-dir $split --confirm-data-rights `
  --output 'runs\agri_a3_30_seed11_cpu_benchmark.json'
```

程序按文件次序取前32条验证对并循环使用，不是具有代表性长度分层抽样；比较模型时必须使用相同输入、长度、线程和设备负载。模型本体不含编码，端到端含编码与构造张量；两者均在输出logit后结束，不含softmax／预测后处理或磁盘读取。Windows 有 psutil 时记录从进程启动起的进程峰值工作集，包括Python、依赖和权重加载，不是纯模型占用；不计算FLOPs。不同设备、驱动或软件的耗时不可混称为目标服务器实测。
