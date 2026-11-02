# 10. Windows 服务器操作：复核、基线、调参与统计

这些命令在用户的 **Windows 10、Quadro RTX 5000 服务器**执行。仓库只提供独立实现，不是原作者代码，也没有真实结果。数据权限、生成来源、语义近重复、术语词表和预训练权重授权需要人工确认；代码中的确认开关仅是操作者声明。下面的路径均为服务器上的示例；原始数据、模型文件、预测和中间结果不要提交到公开仓库。

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

每个报告种子分别生成对应的L/U子集，并使用同一个子集对比各方法。默认规定：**训练种子与LDP采样种子一致**。调参专用种子为101、比例30%；不得拿测试结果选择参数。

```powershell
$split = 'data\agri_clean_reviewed_v1'
$verified = Join-Path $split 'verification_report.json'
python -B scripts/create_ldp_subset.py --split-dir $split `
  --output-dir 'data\agri_30_seed101' --percent 30 --seed 101 --unlabeled-mode external
python -B scripts/create_ldp_subset.py --split-dir $split `
  --output-dir 'data\agri_30_seed11' --percent 30 --seed 11 --unlabeled-mode external
```

公开基准若使用训练池剩余样本去标签作为U，应选择 `--unlabeled-mode remaining`，并与外部生成U分开报告。其余种子为23、37、47、59。多LDP分别生成对应目录；相同种子下各比例按协议嵌套。

独立训练器包含 `ce`、`ce_scl`、`A0`—`A8`、`self_train`、`consistency`、`mean_teacher`、`dropout_pretrain`。需要术语保护的条件必须提供真实可用的术语词表。自动调参默认每个数据集、方法跑六个候选，可能耗时很长，建议先在虚构数据上验证流程；异常候选不会被悄悄跳过。

```powershell
$terms = 'data\reviewed_agricultural_terms.txt'
python -B scripts/auto_tune.py --method A3 --split-dir $split `
  --subset-dir 'data\agri_30_seed101' --verification-report $verified `
  --output-dir 'runs\tune_agri_a3' --term-lexicon $terms --device cuda `
  --confirm-data-rights --confirm-reviewed-data

python -B scripts/train_pairmatch.py --split-dir $split `
  --subset-dir 'data\agri_30_seed11' --verification-report $verified `
  --run-dir 'runs\agri_a3_30_seed11' --condition A3 --seed 11 `
  --term-lexicon $terms --tuning-lock 'runs\tune_agri_a3\selected_lock.json' `
  --device cuda --confirm-data-rights --confirm-reviewed-data
```

每种方法要单独调参；锁定文件记录选中参数、固定设置、实现和输入哈希。后续运行的架构、批量、步数等要与调参设置一致，参数锁自动注入学习率与weight decay；代码或词表改变后不能沿用旧锁。`--allow-nonstandard-grid` 仅供有理由的另行登记或开发测试，锁会标记非标准；不能把缩小搜索或smoke当作六组正式调参。

预训练模型请先通过合法可信的渠道在服务器准备**与配置中模型ID和完整revision对应**的本地目录，里面需要config、tokenizer和权重。核实模型和训练语料许可，尤其轻量模型的许可目前未知。优先使用 `safetensors`；若只有来自可信来源的 `.bin`，才按需要明确加 `--allow-unsafe-pickle-weights`。脚本仅本地读取并记录文件哈希，`--confirm-model-rights-and-revision` 是人工声明，不会自动证明文件来自指定revision。预训练模型利用了外部知识，应与随机初始化BiLSTM分组解释。

```powershell
$bert = 'data\pretrained\bert_base_chinese_pinned'
python -B scripts/auto_tune.py --method bert_base_zh --split-dir $split `
  --subset-dir 'data\agri_30_seed101' --verification-report $verified `
  --checkpoint-dir $bert --output-dir 'runs\tune_agri_bert' --device cuda `
  --confirm-data-rights --confirm-reviewed-data --confirm-model-rights-and-revision

python -B scripts/train_transformer.py --model bert_base_zh --checkpoint-dir $bert `
  --split-dir $split --subset-dir 'data\agri_30_seed11' --verification-report $verified `
  --run-dir 'runs\agri_bert_30_seed11' --seed 11 --device cuda `
  --tuning-lock 'runs\tune_agri_bert\selected_lock.json' `
  --confirm-data-rights --confirm-reviewed-data --confirm-model-rights-and-revision
```

预训练模型ID还支持 `roberta_wwm_zh` 和 `small_zh`；更换ID时必须使用相应的本地权重及独立调参锁。所选模型的 tokenizer 文件保存在运行目录，预测时无需访问网络。

可选的 BERT 领域适配先在训练U上做固定步数MLM，再在同一LDP的L上微调；不读取测试文件，验证集只供划分和加载检查，不用于选MLM阶段。保护特殊token；按15%采样后使用80% `[MASK]`、10%随机token、10%不变的规则。极小微批若一个token也未选中，会强制选中一个并记录次数。每个LDP／种子必须使用对应U分别适配；记录实际U曝光、两阶段时间和GPU显存。下例是报告种子11的适配与微调；正式调参应先对seed101的同类U做适配，再使用 `auto_tune.py --method bert_domain_adaptation --checkpoint-dir <seed101适配输出的adapted_model>`，得到的参数锁用于相同适配方案的其他种子。

```powershell
python -B scripts/adapt_bert_mlm.py --checkpoint-dir $bert --split-dir $split `
  --subset-dir 'data\agri_30_seed11' --verification-report $verified `
  --output-dir 'runs\agri_bert_mlm_30_seed11' --seed 11 --device cuda `
  --confirm-data-rights --confirm-reviewed-data --confirm-model-rights-and-revision

python -B scripts/train_transformer.py --model bert_domain_adaptation `
  --checkpoint-dir 'runs\agri_bert_mlm_30_seed11\adapted_model' `
  --split-dir $split --subset-dir 'data\agri_30_seed11' --verification-report $verified `
  --run-dir 'runs\agri_bert_domain_30_seed11' --seed 11 --device cuda `
  --tuning-lock 'runs\tune_agri_bert_domain\selected_lock.json' `
  --confirm-data-rights --confirm-reviewed-data --confirm-model-rights-and-revision
```

默认MLM和下游微调各2000次更新上限；MLM阶段不早停，微调阶段验证集早停。MLM计算成本需与微调成本相加，不与仅监督模型称为计算等价。适配checkpoint绑定父划分、LDP、U哈希、种子、基础权重、代码与适配设置；不得拿其他数据集或种子的适配权重冒用。输入checkpoint若缺失MLM头，框架可能新初始化该头，应检查并记录加载日志。

## 10.4 冻结后预测、误差和统计

按各方法的验证集锁及五种子运行训练。预先冻结比较、阈值和分析计划后，再对每个最终checkpoint测试。此处展示两条命令；后续统计需为各自其他种子生成对应结果目录。

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

单seed示例只验证格式；正式图需提供所有预设比例／种子／条件的结果。长度图的Wilson区间假设正类样本独立，有来源相关性时应优先结合分组重采样解释。消融图应同时检查实际更新、U曝光及掩蔽强度，不能从不同训练成本直接归因。

特征提取使用固定ID顺序和seed抽取最多1000对，两个模型在同一划分会得到相同样本。t-SNE只是描述性投影，不用于选模型或证明泛化；独立拟合的两个投影坐标轴不可直接比较。测试标签和ID仍只留本地。

```powershell
python -B scripts/extract_embeddings.py --checkpoint 'runs\agri_a3_30_seed11\stage_2_best.pt' `
  --split-dir $split --split test --output-dir 'runs\embeddings_a3' --device cuda --confirm-prespecified-analysis
python -B scripts/plot_tsne.py --embeddings-dir 'runs\embeddings_a3' --output-dir 'runs\fig_tsne_a3'

python -B scripts/benchmark_pairmatch.py --checkpoint 'runs\agri_bert_30_seed11\stage_1_best.pt' `
  --split-dir $split --output 'runs\bert_cpu_benchmark.json' --confirm-data-rights
```

CPU benchmark 在独立进程中测批量1的模型本体和含编码延迟，默认单线程、预热100次、计时1000次，并尽可能记录 Windows 进程峰值工作集；该值包含Python和权重加载，不是纯模型内存。GPU训练峰值记录在训练 `summary.json`。默认输入只是验证集前32条并循环，不是按32／64／128长度分层测量；需要可靠的长度分层资源结论时应另行实现并运行该方案。当前不计算FLOPs，也不能凭参数量推断CPU速度。
