# 单卡A800工程验收

2026-08-28，`/root/workspace/task5_reproduction`。结论：**小规模四阶段流程通过，可迁移到正式服务器做启动检查**。
这不是正式实验结果，也没有启动168组全量训练。

> 本文记录迁移前的完整四阶段模型验收。2026-08-29的YAML配置与Shell入口迁移另见
> [yaml_shell_migration.md](yaml_shell_migration.md)；新入口67项服务器回归全部通过。

## 环境与输入

Python 3.12.3；torch 2.11.0+cu126；Transformers 4.57.6；datasets 3.6.0；NumPy 2.4.3；
PyArrow 25.0.1；Matplotlib 3.11.1。单卡A800-SXM4-80GB。使用旧虚拟环境解释器直接运行新源码，未降级或安装旧环境依赖。
用户批准并实测后，新库固定 `pyarrow==25.0.1`；主要包版本约束在 `configs/local/server-tested.txt`。

输入复制到新库 `inputs/`，共32个文件，约528MiB；不是符号链接。复制前后及smoke结束后均核验SHA256，原始文件内容未改变。

| 输入 | SST-2 | MNLI |
|---|---|---|
| dense来源 | `sst2__seed0__20260819T224020/checkpoint-best` | `mnli__seed0__20260819T231255/checkpoint-best` |
| split | `replicate0/parameter_seed0` | `replicate0/parameter_seed0` |
| train条数 | 67,349 | 392,702 |
| validation条数 | 872 | 9,815（matched） |

任务、权重与split关联、12层labels、每expert32神经元、标签名称/编号顺序均已核验。
拆分labels摘要与preposition报告一致；完整来源路径和逐文件摘要见 `inputs/provenance.json`。
复制的split manifest保留原路径作溯源；加载实际使用新配置路径和内容摘要，不依赖旧目录。

## 修复及回归

1. datasets四进程给64字符fingerprint追加后缀，超过长度上限。内部fingerprint改为32字符；完整SHA256目录和声明不变。
2. 第三方logger保留已结束任务的stderr代理。代理退出时解除旧文件关联，后续转发到当前流；覆盖连续任务、异常退出、嵌套及Linux fork。
3. 新增日志测试在Windows显式用UTF-8读取，避免默认GBK误解码。

前两项均先用新增测试复现，再修复。服务器总计**60通过、0跳过、0失败**；Windows回归44通过、16跳过、0失败。
R3继续使用RMS+1e-6；未修改路由、训练超参、任务模板或实验矩阵。

## 四阶段smoke：server_smoke02

每任务取前512条train、前32条validation；k13、seed0、batch256、10个小epoch（每次训练20个更新）。
含R4、G1、G2三档、G3、G4共7个可训练变体；另含R1/R2/R3/G0和dense。

| 检查/产物 | 实测 |
|---|---:|
| Phase0 | 每任务44个条件（11变体×4档k）；最大logits误差0 |
| 小训练run | 14 |
| 训练checkpoint | 154（14×11） |
| A：逐样本预测 | 164份（154+8静态稀疏+2dense） |
| B：逐层负载 | 34份（best=final时复用） |
| C：probe选中集合 | 162份 |
| 其中同时含D：完整expert激活和 | 52份 |
| E：dense共激活 | 2任务×12层=24份矩阵及计数 |
| 六组metric及汇总 | 33,448条normalized记录及汇总记录 |
| 图表 | 60 PNG + 60 PDF；5个CSV/Markdown表格文件 |

训练全部完成后才采集；A完成后选best，再采集B/C/D及E。离线metric、汇总、制表、绘图在
`CUDA_VISIBLE_DEVICES=''` 的独立进程执行；现有离线导入测试也确认不加载模型库。
smoke的E基于这32条评测子集，不是正式完整validation的E；不能混用于主实验。

所有checkpoint/capture完整性标记与文件摘要通过检查，全部层/token覆盖由采集和指标阶段校验。
60对图文件完成格式检查；另人工查看了主图、churn轨迹、分层热图各1张，文字/图例无裁切。
单budget/单seed的smoke图不用于判断方法优劣，也不提供三seed误差条结论。

## 真实CUDA恢复：server_resume02

MNLI的R4/G3/G4分别在epoch5退出，再由新进程从step10恢复到epoch10。
对照server_smoke02的连续训练，在step10和final两处：**router、optimizer、scheduler及全部RNG状态逐项精确相同**。
这额外产生3个小训练run，不属于正式实验矩阵。

## 证据与交付范围

小型机器可读报告、两任务Phase0和3张图样例放在 [server_evidence/](server_evidence/)。
完整smoke checkpoint/capture/指标/图表保留在小服务器 `tmp/smoke/`，不纳入本地交付的正式输出目录。
本地回同步包含代码、配置、核验后的输入副本与验收证据；旧本地源码备份在项目 `tmp/task5_server_transfer/`。

尚未验证：正式全量训练、完整validation/probe采集规模、三seed/四budget全矩阵、最终服务器的多卡独立分片执行、
64/128/256吞吐比较与全负载显存峰值。换正式服务器后先检查环境、preflight、smoke及batch/chunk短测。
不承诺训练已收敛、方法优势已成立或dense-mask带来实际加速。
