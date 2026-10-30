# 4. 指标、统计与结果记录

本文件规定**如何记录将来的真实实验**。仓库中没有预测、指标、显著性结果或已有训练结论。

## 4.1 统一指标与标签口径

对每个预设种子分别根据逐样本预测和真实标签计算混淆矩阵。正类为匹配 `1`，负类为不匹配 `0`；概率阈值0.5在测试前固定。验证集 macro-F1 用于早停和候选选择；测试至少报告：

- Accuracy，正类 Precision、Recall、F1。
- 各类别的 Precision、Recall、F1 和 support；固定对 `{0,1}` 两类计算 macro-F1，缺类或无预测时使用预设的 `zero_division=0` 并明确标注。
- TN、FP、FN、TP及各类样本数；对类别不平衡的数据额外报告 balanced accuracy。
- 将相同的输入版本、标签映射、预测阈值和指标实现用于全部模型。

`F1 = 2PR/(P+R)` 对**同一次、同一类别、相同样本集**的P和R成立。多次运行分别算出的F1再求均值，一般不等于把平均P和平均R代入公式。macro-F1也不能用某个单类P/R替代。不要用四舍五入表格数字反推或修改缺失结果；应从逐次预测重新计算。

## 4.2 独立重复与不确定性

每个数据集、LDP、方法，计划报告五次预先固定种子各自的结果、mean和样本标准差（`ddof=1`）。同一比较配对意味着同一数据版本、相同训练／验证／测试划分、同一LDP和同一抽样种子对应的L/U样本ID；不能只因两次运行编号相同就假定完全配对。

固定同一测试集时，五种子主要体现**给定划分下**的抽样、初始化及训练波动，不是跨数据集泛化能力。九个LDP通常嵌套且相关，不应当作九次独立运行或用它们代表运行间方差。

预设主要比较为每个数据集和指定LDP上“两阶段方法 vs 同骨干CE”；在配对成立时报告逐种子差值、平均差值、标准差和合理的不确定性区间。五对非零差值且无并列秩时，双侧精确Wilcoxon最小p为0.0625，所以五种子下不能期待必然得到p<0.05。若有预先定义的多个假设检验，记录检验范围并使用 Holm 校正；无法支持有意义的显著性结论时，如实报告差值和不确定性，不为追求显著性反复增加种子或切换检验。

可补充基于测试样本的配对 bootstrap：采用预先固定的2,000次重采样和seed 20261030，对同一个测试样本／组同时重采样两种方法的预测，计算指标差值的百分位区间。若样本通过相同句子或来源形成相关组，按独立组而不是单个文本对重采样；如有效独立组太少或某次重采样缺少类别，应记录并说明区间不稳定。bootstrap衡量的是测试样本抽样不确定性，不能替代跨训练种子的方差。

## 4.3 本地结果记录格式

每个运行在被忽略的 `runs/` 下保存配置、版本、日志与逐样本预测；不将原文、样本ID、未经授权的标签或本地路径提交到公开仓库。建议的聚合记录字段：

```text
dataset_id, dataset_version, input_hashes, split_id, split_policy,
model_id, checkpoint_revision, condition, ldp, labeled_count,
unlabeled_source, unlabeled_count, sampling_seed, training_seed,
optimizer, learning_rate, weight_decay, batch_size, accumulation,
max_length, truncation_rate, stage1_steps, stage2_steps,
best_validation_step, selection_metric, threshold,
accuracy, precision_positive, recall_positive, f1_positive,
macro_f1, balanced_accuracy, TN, FP, FN, TP,
training_seconds, peak_gpu_allocated_mb, peak_gpu_reserved_mb,
framework_version, cuda_build, gpu_model, driver_version, code_commit
```

逐样本结果另有匿名ID、预测概率、预测标签和真实标签，只存本地；U的隐藏标签不允许出现在训练记录中。每次更新或重跑必须生成新运行ID，保留失败记录及失败原因；不得覆盖不理想结果。

## 4.4 图表应包含的最小信息

1. LDP曲线：每档准确分母、均值及跨种子SD／原始点；图例说明数据集、版本和协议。不能将跨LDP的箱线图解释为种子稳定性。
2. 消融表：固定LDP、训练条件、每项mean±SD、L/U来源、实际步数；单独识别只增加数据、只增加目标和改变顺序的对照。
3. 训练过程：每阶段训练／验证loss与macro-F1，标示切换、最佳checkpoint和停止点。
4. 句长分析：事先定义按字符或token的区间，给总样本数及正类数；当某组recall为1时也应给区间，不能因为点估计为1而宣称完美。
5. t-SNE：预设最多1,000对相同样本、分类前pair表示、perplexity 30、learning rate 200、PCA初始化、1,000次迭代及seed 42；实际样本数必须大于perplexity，并记录预处理、库版本和模型标签。它只作为描述性图，不以可视化选择最终模型。
6. 资源表：给硬件、批量、长度、运行线程、是否包括tokenizer、预热与重复次数、统计单位；不能把参数少等同于CPU一定更快。

要制作最终图表，先确保有真实原始结果和可复核的绘图脚本；本方案不会替你填充任何结果数字。

掩蔽概率、句长分组、t-SNE、延迟和bootstrap的建议起点可查看 [analysis_plan.json](../configs/analysis_plan.json)。句长预设按两句**截断前Unicode字符数的均值**分组，区间为 `[0,5), [5,10), [10,15), [15,20), [20,25), [25,∞)`；若样本过少，保留数量并提示不确定性，不因结果好坏重划分。
