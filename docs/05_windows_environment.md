# 5. Windows 目标服务器与环境

目标配置来自用户提供的信息，**不是已在本仓库训练验证的环境**：Lenovo ThinkSystem ST558、Intel Xeon Silver 4208（8核16线程、2.10 GHz）、NVIDIA Quadro RTX 5000 16 GB、驱动536.25、内存192 GB（6×32 GB DDR4 ECC，模组标称2933 MT/s，当前配置2400 MT/s）、Windows 10专业工作站版64位 Build 19045。用户还报告RAID 530-8i控制器和BIOS O0E172F-3.00；电源模块详情未知。设备完整信息和当前实际进程状态以正式运行机为准。

项目准备期间的命令执行环境与上述目标设备可能不同。进入真正用于训练的机器后，先在项目根目录执行：

```powershell
python -B scripts/check_environment.py
nvidia-smi
```

把输出保存到本地运行记录，核对GPU型号、可用显存、驱动、Python、PyTorch及其CUDA构建。`nvidia-smi` 显示的 CUDA Version 是驱动支持的最高CUDA级别，不等同于当前Python所用的CUDA构建，也不证明本方案已兼容。

## 5.1 建议的独立训练环境

以下为**拟验证**的起点，不会由本仓库自动安装；先确保目标机有合适的 Python 3.10、磁盘空间和软件／模型使用权限。兼容性以安装和运行测试为准。审计工具只用标准库；本项目的独立训练器需要 PyTorch，具体见[实现说明](09_training_implementation.md)。

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-server.txt
python -B scripts/check_environment.py
```

如果 PowerShell 禁止激活脚本，可直接用 `.\.venv\Scripts\python.exe` 执行，不必修改全局执行策略。不要盲目更新驱动或全局CUDA；先在目标机验证驱动、PyTorch wheel 和GPU能否配合工作。CUDA wheel 的运行时与本机安装的 CUDA Toolkit 是不同概念；除非需要自行编译扩展，一般无需仅为运行 wheel 而改动系统 Toolkit。版本实际安装结果必须记录，不能根据此计划宣称已经验证通过。

训练时建议单GPU独占一个进程、FP32、`num_workers=0`。正式运行前固定所有随机种子，记录 Python、NumPy、PyTorch、Transformers、操作系统、driver、GPU及代码提交；按框架文档设置确定性算法并记录不支持的操作。对CUDA相关的确定性设置须在导入相关库前设置，例如 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；仍不保证不同驱动或设备逐位相同。

## 5.2 CPU推理延迟和内存的预注册方案

比较时使用相同的软件环境、固定的原始输入集合、`batch_size=1`、`model.eval()` 和推理模式。模型本体与包含tokenizer的端到端时间分开测量。CPU测试先使用1个计算线程（含 BLAS／PyTorch 线程设置，在计算开始前固定），记录具体CPU、系统电源计划、进程并发、模型、输入长度和是否包含磁盘读取；再按需要添加多线程实验，不能混报。

每个配置预热100次，随后计时1,000次，使用高分辨率单调时钟；记录中位数、p95、最小值和重复设置。输入长度可预设32、64、128三个级别；不同tokenizer的token单位不同，需记录实际截断和字符覆盖。不要从FLOPs或参数数推断实测延迟的方向。若机器负载或线程设置不能稳定控制，说明限制并重测。

参数量分别记录总参数、可训练参数、嵌入是否包含；模型磁盘文件大小说明保存格式与精度。FLOPs记录统计工具、输入长度、批量和算子覆盖。CPU内存可在新进程内使用可报告 Windows 峰值工作集的工具（例如核实适用版本的 psutil）测量，称为“峰值进程内存”，不要误称为纯模型权重占用。GPU分别记录 `max_memory_allocated` 与 `max_memory_reserved`，包括重置峰值的时机。

目标服务器功耗上限、散热状态与当前显存占用会随时间改变；把正式测量时的状态作为运行记录，不把先前一次读数当作不变配置。没有目标设备的测量就不能宣称已经验证边缘部署。
