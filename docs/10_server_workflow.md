# 10. Windows 服务器操作：复核、基线、调参与统计

这些示例面向 **Windows 10、Quadro RTX 5000 服务器**。仓库提供独立实现，代码不代表历史运行记录；确认开关是操作者对数据及模型使用权限的声明。除已公开的农业测试集和术语表外，训练数据、模型文件、预测和中间结果按其授权条件管理。

建议执行顺序：核验服务器环境 → 人工复核并生成新划分 → 锁定数据版本和分析计划 → 准备各LDP → 分方法做验证集调参 → 按五种子训练 → 冻结后测试 → 配对统计、绘图和目标设备测量。每个输出目录必须不存在，失败后使用新目录，保留失败信息。

## 10.1 服务器环境

按[服务器环境说明](05_windows_environment.md)建立 Python 3.10 环境并安装拟验证的 PyTorch CUDA 版本，再安装 `requirements-server.txt`。这些版本只是建议起点，必须以服务器实测为准：

```powershell
python -B scripts/check_environment.py
nvidia-smi
python -B scripts/validate_configs.py
python -B -m unittest discover -s tests -v
```

测试中的样本及小模型为虚构内容，不能当作真实实验结果。训练前检查 PyTorch 能发现 Quadro RTX 5000；本仓库不会更改驱动，不会下载模型，也不保证服务器上现有CUDA组合一定兼容。

## 10.2 应用人工复核决定

输入应是此前生成的父划分、其近重复候选目录，以及**生成父划分时同顺序的原始输入文件**。首先运行 `review_near_candidates.py`；如果其 `summary.json` 的 `truncated` 为 true，增大 `--max-rows` 重做候选导出，不能把截断的部分当作全部已复核。

```powershell
$parent = 'data\agri_clean_v3'
$candidates = 'reports\agri_near_v3'
$review = 'reports\agri_review_round1'
$revised = 'data\agri_clean_reviewed_v1'
# 将这些路径替换为服务器上与父 manifest 记录的哈希和顺序完全一致的原始输入
$source1 = 'D:\path\to\train_50.txt'
$source2 = 'D:\path\to\dev.txt'
$source3 = 'D:\path\to\test.txt'

python -B scripts/apply_review_decisions.py template `
  --split-dir $parent --candidate-dir $candidates --output-dir $review
```

查看候选目录的 `review_candidates.jsonl`（`candidate_index` 从0开始，对应文件顺序）和父划分 `quarantined_conflicts.jsonl`；在 `$review\decisions.jsonl` 中逐行填写 `decision`、`reviewer` 和 `reason`。不得修改候选原文件或模板的 `manifest.json`。近重复选项：

| 决定 | 行为 |
|---|---|
| `GROUP` | 两个句子及其关联的标注对成为同组；与验证／测试组关联的U会排除 |
| `KEEP_SEPARATE` | 人工判断不强制同组；不是证明它们不存在语义泄漏 |
| `EXCLUDE_A` / `EXCLUDE_B` / `EXCLUDE_BOTH` | 将该侧句子关联的全部列出样本对隔离，不参加训练或评估 |

冲突标签选项为 `KEEP_QUARANTINED`、`LABEL_0`、`LABEL_1`。只有有依据时才人工选择标签；程序保留原始观测和裁定，不自行推断标签。每个候选和冲突都必须有一条完整决定，否则停止。确认复核和重划规则是在看测试性能**之前**制定后执行：

```powershell
python -B scripts/apply_review_decisions.py apply `
  --split-dir $parent --candidate-dir $candidates --template-dir $review `
  --inputs $source1 $source2 $source3 --output-dir $revised `
  --confirm-before-test-analysis
```

程序校验候选和原文件哈希、保留原文件、生成全新划分并执行复核。新目录中的 `verification_report.json` 可用于后续训练；`reviewed_edges.jsonl`、人工隔离记录、冲突原始观测及决定均含敏感内容，仅留服务器。重新划分后应再次导出并复核新产生的跨划分候选；下一轮以新划分为父目录，脚本会继承之前的分组、隔离和冲突裁定。已人工决定保持分开的候选可能再次出现，可在后续轮次继续作出有依据的决定。组过大、比例或类别约束不满足时程序会停止，不拆组强行通过。新划分不是官方或原论文测试集。

## 10.3 LDP、权重和验证集自动调参

农业数据的标注子集使用采样种子11分层抽取；同一LDP下，各方法和五个报告训练种子11、23、37、47、59使用同一标注子集。调参训练种子101与采样种子是不同概念；不得拿测试结果选择参数。

```powershell
$split = 'data\agri_clean_reviewed_v1'
$verified = Join-Path $split 'verification_report.json'
python -B scripts/create_ldp_subset.py --split-dir $split `
  --output-dir 'data\agri_30_seed11' --percent 30 --seed 11 --unlabeled-mode external
```

公开基准使用训练池剩余样本去标签作为U，与农业外部生成U分开报告。论文的 AFQMC、LCQMC、PAWS-X 标注子集分别为3,093、20,805、4,251对，各方法使用相同的15,000对U。农业数据的九档子集按固定采样种子嵌套。

独立训练器包含 `ce`、`ce_scl`、`A0`—`A8`、`self_train`、`consistency` 等条件。论文明确列出的验证集候选为CE九组、BERT-base和四层RoBERTa各六组，分别由对应学习率与有效批量组合构成；每个公共数据集使用自己的验证集。具体候选见[训练协议](02_training_protocol.md)，脚本参数见 `python -B scripts/auto_tune.py --help`。

```powershell
$terms = 'data\agriculture_sentence_clean_v3\agricultural_terms.txt'
$a3LearningRate = Read-Host '输入该次 A3 训练的学习率'
python -B scripts/train_pairmatch.py --split-dir $split `
  --subset-dir 'data\agri_30_seed11' --verification-report $verified `
  --run-dir 'runs\agri_a3_30_seed11' --condition A3 --seed 11 `
  --term-lexicon $terms --temperature 0.07 --mask-rate 0.20 --learning-rate $a3LearningRate `
  --device cuda --confirm-data-rights --confirm-reviewed-data
```

其中 `0.20` 来自实验工作簿中A3的验证集选参记录；论文测试集敏感性扫描的最高点不代表验证集选值。运行时输入本次实际采用的学习率，并使用不存在的新输出目录。

预训练模型请先通过合法可信的渠道在服务器准备**与配置中模型ID和完整revision对应**的本地目录，里面需要config、tokenizer和权重。核实模型和训练语料许可，尤其轻量模型的许可目前未知。优先使用 `safetensors`；若只有来自可信来源的 `.bin`，才按需要明确加 `--allow-unsafe-pickle-weights`。脚本仅本地读取并记录文件哈希，`--confirm-model-rights-and-revision` 是人工声明，不会自动证明文件来自指定revision。预训练模型利用了外部知识，应与随机初始化BiLSTM分组解释。

```powershell
$bert = 'data\pretrained\bert_base_chinese_pinned'
python -B scripts/train_transformer.py --model bert_base_zh --checkpoint-dir $bert `
  --split-dir $split --subset-dir 'data\agri_30_seed11' --verification-report $verified `
  --run-dir 'runs\agri_bert_30_seed11' --seed 11 --device cuda `
  --learning-rate 0.00002 `
  --confirm-data-rights --confirm-reviewed-data --confirm-model-rights-and-revision
```

预训练模型ID还支持 `roberta_wwm_zh` 和 `small_zh`；更换ID时必须使用相应的本地权重及独立调参锁。所选模型的 tokenizer 文件保存在运行目录，预测时无需访问网络。

可选 BERT 领域适配先在训练U上做固定步数MLM，再在同一LDP的L上微调，不作为论文已列明的三类基线调参网格。MLM保护特殊token；按15%采样后使用80% `[MASK]`、10%随机token、10%不变的规则。记录实际U曝光、两阶段时间和GPU显存；下例为种子11的独立运行。

```powershell
python -B scripts/adapt_bert_mlm.py --checkpoint-dir $bert --split-dir $split `
  --subset-dir 'data\agri_30_seed11' --verification-report $verified `
  --output-dir 'runs\agri_bert_mlm_30_seed11' --seed 11 --device cuda `
  --confirm-data-rights --confirm-reviewed-data --confirm-model-rights-and-revision

python -B scripts/train_transformer.py --model bert_domain_adaptation `
  --checkpoint-dir 'runs\agri_bert_mlm_30_seed11\adapted_model' `
  --split-dir $split --subset-dir 'data\agri_30_seed11' --verification-report $verified `
  --run-dir 'runs\agri_bert_domain_30_seed11' --seed 11 --device cuda `
  --confirm-data-rights --confirm-reviewed-data --confirm-model-rights-and-revision
```

可选MLM适配默认使用固定步数且不早停；下游微调默认按轮验证并使用早停。MLM计算成本需与微调成本相加，不与仅监督模型称为计算等价。适配checkpoint绑定父划分、LDP、U哈希、种子、基础权重、代码与适配设置；不得拿其他数据集或种子的适配权重冒用。

## 10.4 冻结后预测、误差和统计

按各方法的验证集选定设置及五个训练种子运行训练。预先冻结比较、阈值和分析计划后，再对每个最终checkpoint测试。此处展示两条命令；后续统计需为各自其他种子生成对应结果目录。

```powershell
python -B scripts/predict_pairmatch.py `
  --checkpoint 'runs\agri_a3_30_seed11\stage_2_best.pt' `
  --split-dir $split --split test --confirm-test-locked `
  --output-dir 'runs\eval_a3_30_seed11'
python -B scripts/predict_pairmatch.py `
  --checkpoint 'runs\agri_bert_30_seed11\stage_1_best.pt' `
  --split-dir $split --split test --confirm-test-locked `
  --output-dir 'runs\eval_bert_30_seed11'
```

预测文件只留服务器。错误分析和长度分组要在查看测试结果前确定；输出的 `errors.jsonl` 含样本ID，可供授权人员在本地回查，不自动判断错误原因。

```powershell
python -B scripts/analyze_errors.py --metrics 'runs\eval_a3_30_seed11\metrics.json' `
  --split-dir $split --output-dir 'runs\errors_a3_seed11' --confirm-prespecified-analysis
python -B scripts/build_bootstrap_groups.py --split-dir $split --split test `
  --output-dir 'runs\agri_test_groups'
```

分组采用现有句子、指定表面近重复规则、可用来源和人工确认的分组；缺少来源或语义关联仍不等于独立。配对比较需相同数据版本、评估ID、LDP、L样本和采样种子；两个半监督方法还核对U条件。监督与半监督的信息量不同，结果会标注；若主动比较两种不同U的半监督方法，需事先说明并显式加 `--allow-different-u-information`。

```powershell
# 示例仅列一个seed；正式五seed时分别将其他4个metrics.json加到左右两组中
python -B scripts/paired_statistics.py `
  --left 'runs\eval_a3_30_seed11\metrics.json' `
  --right 'runs\eval_ce_30_seed11\metrics.json' `
  --metric macro_f1 --groups-dir 'runs\agri_test_groups' `
  --output 'runs\paired_a3_vs_ce.json' --confirm-prespecified-analysis

# 多项预设假设构成一个完整家族时，列出全部检验结果再校正
python -B scripts/holm_correction.py `
  --reports 'runs\paired_a3_vs_ce.json' 'runs\paired_other.json' `
  --family-name 'prespecified-primary-family' --output 'runs\holm_primary.json' `
  --confirm-complete-prespecified-family
```

默认按独立组联合重采样2000次（seed 20261030），区间是给定已训练模型的测试样本不确定性，不是跨训练种子的置信区间。Wilcoxon 为双侧精确符号秩分布，零差值丢弃、并列取平均秩；五个非零且无并列的种子最小双侧p为0.0625。只做单样本重采样时必须显式 `--bootstrap-unit samples`，并报告独立样本假设。组不足、缺类重采样数、依赖性及对外部预训练知识的不同接触都应如实说明。

## 10.5 图表、特征和效率

下列图使用真实运行的本地记录生成 SVG、PDF、PNG 预览和 `source_data.json`；加 `--tiff` 可选600dpi TIFF。点是各seed，误差线是样本标准差（仅一个seed时没有SD）；文件包含指标、样本数量和哈希，不要将敏感源数据上传。提交前仍需在实际尺寸下人工检查字体、图例和期刊要求。

```powershell
python -B scripts/plot_results.py training --run-dir 'runs\agri_a3_30_seed11' --output-dir 'runs\fig_training'
python -B scripts/plot_results.py confusion --metrics 'runs\eval_a3_30_seed11\metrics.json' --output-dir 'runs\fig_confusion'
python -B scripts/plot_results.py length --length-summary 'runs\errors_a3_seed11\length_summary.json' --output-dir 'runs\fig_length'
python -B scripts/plot_results.py ablation `
  --metrics 'runs\eval_a3_30_seed11\metrics.json' 'runs\eval_ce_30_seed11\metrics.json' `
  --output-dir 'runs\fig_ablation'
python -B scripts/plot_results.py ldp `
  --metrics 'runs\eval_a3_30_seed11\metrics.json' 'runs\eval_ce_30_seed11\metrics.json' `
  --output-dir 'runs\fig_ldp'
```

单seed示例只验证格式；正式图需提供所有预设比例／种子／条件的结果。长度图展示各组样本数、准确率和macro-F1；论文对最短组正类召回率另报告假定样本独立的精确二项区间。消融图应同时检查实际更新、U曝光及掩蔽强度，不能从不同训练成本直接归因。

特征提取使用固定ID顺序和seed抽取最多1000对，两个模型在同一划分会得到相同样本。t-SNE只是描述性投影，不用于选模型或证明泛化；独立拟合的两个投影坐标轴不可直接比较。农业测试集及其样本ID和标签见仓库的数据目录，模型预测按授权条件管理。

```powershell
python -B scripts/extract_embeddings.py --checkpoint 'runs\agri_a3_30_seed11\stage_2_best.pt' `
  --split-dir $split --split test --output-dir 'runs\embeddings_a3' --device cuda --confirm-prespecified-analysis
python -B scripts/plot_tsne.py --embeddings-dir 'runs\embeddings_a3' --output-dir 'runs\fig_tsne_a3'

python -B scripts/benchmark_pairmatch.py --checkpoint 'runs\agri_bert_30_seed11\stage_1_best.pt' `
  --split-dir $split --output 'runs\bert_cpu_benchmark.json' --confirm-data-rights
```

CPU benchmark 在独立进程中测批量1的模型本体和含编码延迟，默认单线程、预热100次、计时1000次，并尽可能记录 Windows 进程峰值工作集；该值包含Python和权重加载，不是纯模型内存。GPU训练峰值记录在训练 `summary.json`。默认输入只是验证集前32条并循环，不是按32／64／128长度分层测量；需要可靠的长度分层资源结论时应另行实现并运行该方案。当前不计算FLOPs，也不能凭参数量推断CPU速度。
