# Semi-TM：农业文本对匹配

本仓库提供按照论文方法整理的 Semi-TM 实现和农业文本对测试集。任务是判断两个中文句子是否表达相同的问题。模型以字符为输入，使用 ESIM 风格的双向 LSTM 交互编码器，并通过两阶段训练利用有标注和无标注句对。

论文：Yunkun Wang、Yunsheng Song、Xinyue Zhao、Ruihui Yao，*Agricultural Text-Pair Matching with Limited Labels: A Lightweight Two-Stage Semi-Supervised Model*。

本仓库代码是依据现有论文与材料整理的独立实现；仓库代码本身不是对历史训练运行记录的证明。

## 方法

模型对句子逐字符编码，将两个句子的上下文表示进行交互、对齐和池化，再预测句对是否匹配。字符嵌入为 300 维，按 N(0, 0.1²) 初始化；使用无标注数据的配置以所选标注句对和所用无标注句对共同建立训练词表，不使用验证或测试文本建立词表。对比学习使用当前微批中的句对表示：

在本仓库对应的农业数据划分和 A3 配置中，10%、25%、30% 和 50% 标注比例的字符词表目标分别为 2,906、3,228、3,412 和 3,758。当前实现按训练文本中的字符出现频次从高到低选择；频次相同时按 Unicode 编码顺序选择，未选字符映射为 `<unk>`。这是一条为当前实现确定的筛选规则；字符数不包含另外保留的 `<pad>`、`<unk>` 和 `<mask>` 三个特殊标记。当训练文本中的不同字符少于对应目标时，程序会停止并给出提示，不会添加训练文本中未出现的字符。

- 第一阶段使用有标注句对，目标为交叉熵损失与有监督对比损失。
- 第二阶段继续使用有标注句对的交叉熵损失，同时对无标注句对及其掩蔽视图使用无监督对比损失。掩蔽针对两个句子共有的字符类型，并保护所提供农业术语中出现的字符。
- 两阶段配置中，第二阶段从第一阶段验证集选出的模型继续训练；优化器和学习率调度器在阶段切换时重新初始化。

论文中的两阶段对比温度为 0.07。两阶段各分配 16 轮；ESIM 风格模型每个有标注微批为 16 对，梯度累积两次，第二阶段无标注微批与有标注微批一一对应。当前实现每轮重新打乱标注数据，不足一次完整更新的尾部不参与该轮训练，以保持每次更新的批量一致。每轮依据验证集 macro-F1 检查模型，使用 AdamW、线性预热与衰减。A3 默认掩蔽概率为 0.20，取自实验记录中验证集选定值；论文所示的测试集敏感性扫描单独报告各掩蔽概率的结果。

## 目录

```text
pairmatch/        模型、数据读取、训练目标和评价指标
scripts/          训练、预测、数据处理与分析脚本
configs/          数据与实验配置
tests/            功能测试
data/agriculture_sentence_clean_v3/
  test.jsonl      农业测试集
  agricultural_terms.txt  农业术语表
  manifest.json   划分信息和文件 SHA-256
```

## 环境

获取仓库并进入项目目录：

```powershell
git clone https://github.com/YRH454/wangyunkun.git
cd wangyunkun
```

建议使用 Python 3.10 或更新版本，并按所用 CPU 或 NVIDIA GPU 安装适配的 PyTorch。其余依赖可按需安装：

```powershell
pip install torch
pip install -r requirements-server.txt
```

在仓库根目录运行功能测试：

```powershell
python -B -m unittest discover -s tests -v
```

## 农业测试集

测试集为 UTF-8 编码的 JSON Lines 文件，每行是一条句对。共 1,632 条，其中匹配 946 条、不匹配 686 条。

| 字段 | 含义 |
| --- | --- |
| `sample_id` | 根据规范化句对生成的样本标识 |
| `sentence1` | 第一个中文句子 |
| `sentence2` | 第二个中文句子 |
| `gold_label` | 匹配标签：`1` 为匹配，`0` 为不匹配 |

测试文件 SHA-256：

```text
3b50b0f63448f564e9d6783838a836a4ad51050a0357cb88cf13149bdf5ed672
```

使用 PowerShell 可核对文件：

```powershell
Get-FileHash data/agriculture_sentence_clean_v3/test.jsonl -Algorithm SHA256
```

测试集中的样本编号与标签和论文 30% 标注比例、随机种子 11 的逐句对句长分析记录对应。`manifest.json` 保存划分信息和文件摘要，不包含训练、验证或无标签样本。

## 训练

训练使用本地准备的训练集、验证集、无标注数据、标注比例子集及划分核验文件。`agricultural_terms.txt` 包含 119 个农业术语，分为作物、病害、虫害、农事操作和投入品，UTF-8 编码，每行一个术语。

```powershell
python -B scripts/train_pairmatch.py --help
```

两阶段模型使用 `--condition A3`；参数 `--split-dir`、`--subset-dir`、`--verification-report`、`--term-lexicon` 和 `--run-dir` 分别指定划分、标注子集、核验文件、术语表和新的输出目录。`--seed` 控制训练随机种子；相同标注比例可使用同一子集分别运行种子 `11`、`23`、`37`、`47`、`59`。`--device cuda` 使用 GPU，`--device cpu` 使用 CPU。训练前应确认输入数据的使用权限及相应复核情况，再提供程序要求的确认参数。

例如，下列命令会提示输入事先准备好的本地路径和该次运行的学习率：

```powershell
$splitDir = Read-Host '输入本地数据划分目录'
$subsetDir = Read-Host '输入标注子集目录'
$verificationReport = Read-Host '输入划分核验文件路径'
$runDir = Read-Host '输入新的训练输出目录'
$learningRate = Read-Host '输入该次运行的学习率'
$maskRate = 0.20
$termLexicon = 'data/agriculture_sentence_clean_v3/agricultural_terms.txt'
python -B scripts/train_pairmatch.py `
  --split-dir $splitDir --subset-dir $subsetDir `
  --verification-report $verificationReport `
  --term-lexicon $termLexicon --run-dir $runDir `
  --condition A3 --seed 11 --device cuda `
  --temperature 0.07 --mask-rate $maskRate --learning-rate $learningRate `
  --confirm-data-rights --confirm-reviewed-data
```

训练输出包括配置、数据摘要、各阶段选出的模型、训练与验证记录以及汇总结果。

## 测试与指标

使用训练所得最终阶段模型及其对应的划分目录执行测试：

```powershell
$checkpoint = Read-Host '输入最终阶段模型文件路径'
$evaluationDir = Read-Host '输入新的评估输出目录'
python -B scripts/predict_pairmatch.py `
  --checkpoint $checkpoint --split-dir $splitDir --split test `
  --confirm-test-locked --output-dir $evaluationDir
```

输出包括逐样本预测、匹配概率、准确率、分类别精确率、召回率、F1 和 macro-F1。macro-F1 为匹配与不匹配两个类别 F1 的算术平均值。

## 引用

如使用本仓库代码或测试集，请引用上文列出的论文标题及作者，并注明所使用的仓库版本。
