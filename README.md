# 农业文本匹配：公平、可复现的训练方案

这是一个**独立学习和实验设计项目**。它借鉴文本对匹配的两阶段训练思路以及对实验严谨性的审查关注，目标是为自己的训练建立透明的数据处理、对照与记录规范；它不是原作者的官方仓库，也不是对原论文结果已经完成的复现。

目前仓库包含：

- 基于本地文件结构的**数据配置**，以及实验、模型、硬件和消融的机器可读 JSON 配置。
- 面向 Windows 和单 GPU 的**完整实验方案**，包括分组划分、数据泄漏检查、基线、统计、硬件测量和记录模板。
- 独立编写的数据审计、分组防泄漏划分、标注比例采样工具；原始输入只读，生成的数据仅保存本地。
- 独立实现的字符级 ESIM 风格训练器：监督对照、两阶段对比、A0—A8、自训练、一致性、Mean Teacher 和 dropout 预训练对照。
- 离线 BERT／中文 RoBERTa 监督基线、验证集网格调参、人工复核后重划分、配对统计、图表和效率测量代码。只经过合成数据功能测试；不自动下载模型。

**当前没有原作者的完整训练代码、真实实验结果或已经验证的目标服务器运行记录。** 训练代码是根据本项目协议独立实现的，不保证与原论文完全一致。现有预训练基线是离线加载并微调通用模型，并非原论文 SBERT、CoSENT、PairSup、BERT-ATT、R-Drop BERT 或 DeepSeek 的逐项复现；如需以这些确切名称比较，还须取得可核验实现或另行实现和测试。可靠覆盖所有算子的 FLOPs 及按长度分层的效率测量也尚未实现。`python scripts/validate_configs.py` 成功只说明配置结构一致，不表示实验可以开始或数据已经安全。

## 重要边界

1. 本仓库公开；不包含原论文、审稿意见原文、原作者代码、原始文本、原始样本、模型权重或本地审计报告。
2. 本地数据文件、原始代码和各模型权重的**使用与再分发权限需要分别确认**。把数据放在本地不等于拥有公开分享或训练使用授权。
3. 原论文的数据、训练细节与本地候选文件尚未一一确认。我们只把配置标为“本项目建议”，不把它们冒充为原作者的真实设置。
4. 当前目录未声明仓库代码许可证；如需开放复用，请由仓库所有者另行决定。数据和预训练模型的许可与本仓库代码许可是两回事。
5. `reports/`、`data/`、`datasets/`、权重和常见数据格式默认被 Git 忽略。忽略规则不是安全保证，每次提交前仍需人工检查。

## 快速开始：先检查，再训练

在仓库根目录使用 PowerShell。数据审计检查只依赖 Python 标准库；若环境安装了 PyTorch，单元测试还会在临时合成数据上训练小模型，未安装时这些训练测试会跳过。不会读取本地真实数据或下载依赖；Python 3.10 及以上可用于数据审计。

```powershell
python -B scripts/validate_configs.py
python -B -m unittest discover -s tests -v
python -B scripts/check_environment.py
```

准备好数据访问权限后，将 `$datasetRoot` 改为本机 `data` 目录，即其下能看到 `Agriculture9_S`、`AFQMC_S` 等文件夹的目录：

```powershell
$datasetRoot = 'D:\path\to\local\data'
$agri = Join-Path $datasetRoot 'Agriculture9_S'
python -B scripts/audit_jsonl.py `
  --train (Join-Path $agri 'train_50.txt') `
  --supervised (Join-Path $agri 'train_sup_50.txt') `
  --validation (Join-Path $agri 'dev.txt') `
  --test (Join-Path $agri 'test.txt') `
  --require-sentence-disjoint `
  --output 'reports\agriculture_audit.json'
$LASTEXITCODE
```

`0` 表示该脚本的**精确检查**通过；`2` 表示发现影响所选门槛的风险；`1` 表示文件／编码／输出路径等操作失败。它不输出任何原句，报告不会覆盖已有文件。报告只包含聚合统计及文件哈希，应留在本地。

该工具仅核查规范化后的完整对、倒序对、单句复用、标签和子集对应关系，**不会检查语义近重复、来源级泄漏、词表拟合范围、数据授权或官方版本**。通过检查也不表示论文可以声称没有泄漏。不同任务的句序是否可交换需要人工确认；如果句序有意义，不能把倒序对自动视为同一语义，只能当作待复核的保守候选。

## 建议的阅读顺序

1. [数据集、划分和风险](docs/01_data_and_splits.md)
2. [训练协议与公平性](docs/02_training_protocol.md)
3. [基线、消融和资源预算](docs/03_experiment_matrix.md)
4. [指标、统计和结果记录](docs/04_statistics_reporting.md)
5. [Windows 目标服务器与运行环境](docs/05_windows_environment.md)
6. [检查清单与待确认事项](docs/06_checklist.md)
7. [实验关注点与方案对照](docs/07_experiment_coverage.md)
8. [防泄漏划分与标注比例采样：操作说明](docs/08_clean_split_workflow.md)
9. [独立训练器：功能、边界和运行方法](docs/09_training_implementation.md)
10. [服务器完整操作：人工复核、预训练基线、调参、统计与画图](docs/10_server_workflow.md)

配置清单：[数据集](configs/datasets.json)、[训练协议](configs/protocol.json)、[模型候选](configs/models.json)、[基线参数](configs/baseline_protocols.json)、[实验矩阵](configs/experiment_matrix.json)、[分析配置](configs/analysis_plan.json)、[目标服务器](configs/target_server.json)。所有路径是相对于用户本地数据根目录的说明，不会从互联网自动下载或复制数据。

## 本地生成新划分

确认有权处理数据后，按照[操作说明](docs/08_clean_split_workflow.md)执行 `scripts/prepare_clean_split.py`，再用 `scripts/verify_clean_split.py` 重新读取并复核；需要不同标注比例时使用 `scripts/create_ldp_subset.py`。输出放在被忽略的 `data/` 下，不会覆盖原始文件或已有输出。

这些程序建立的是**新的重划分实验**，不能当成原论文或官方基准的原测试集结果；检查仅覆盖确切重复和定义的字符近重复，不保证语义、来源或模型预训练层面的完全独立。`scripts/review_near_candidates.py` 可导出相似度较低的本地候选供人工复核，不自动决定或删除样本。
