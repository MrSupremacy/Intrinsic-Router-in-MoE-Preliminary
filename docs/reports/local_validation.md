# 本地验证状态

> 历史验收记录，不是运行手册。当前所有可执行步骤只见[仓库全流程手册](../../RUNBOOK.md)。

该报告保留最初本地验证的范围；当时没有运行真实SST-2/MNLI训练或产生研究结果。

本页下文保留最初本地交付记录；后续真实资产测试见 [服务器验证报告](server_validation.md)。
服务器修复后的Windows回归为60项：44通过、16项因缺依赖或不支持Linux fork跳过，0失败/0错误。
对应 `tmp/local-tests/after_remote_fixes.json`；不能将这些skip算成CUDA测试通过。

2026-08-28 本地最终回归：**55项，41项通过，14项缺依赖跳过，0失败/0错误**。
本段是2026-08-27迁移前记录：当时环境为Python 3.12.13、NumPy 2.3.5；64个Python源文件通过语法检查，
当时的TOML/JSON schema均可解析。当前YAML/Shell结果见[yaml_shell_migration.md](yaml_shell_migration.md)。

## 已执行

- 独立源码语法检查与主/smoke配置解析。
- 主矩阵：168次训练、1848个训练状态、48个static稀疏状态、2个dense，共1898状态。
- smoke：14次训练，每次11点；训练512/评测32的配置已展开。
- 纯NumPy指标、best选择、seed/层统计、采样、缓存隔离、数据完整性与I/O保护测试。
- 合成离线全链路：全部候选A→各seed独立best→六组metric→汇总→表格；未创建模型或训练checkpoint。
  该测试模拟表格读取接口，**不是**真实Parquet读写测试。
- R3在配置中保持RMS+1e-6，不恢复旧严格R2/R3集合门槛。

完整测试列表与精确计数以当时的`tmp/local-tests/report.json`为准；当前测试入口见唯一运行手册。

## 已写测试、当前环境未执行

当前Python环境没有torch、transformers、datasets、pyarrow、matplotlib。测试明确skip，不伪装通过：

- torch各router数值、可训练梯度、G2 mask/aux、G3独立噪声、G4 bias位置/更新、hash前缀；
- R3小范数RMS回归；
- 随机初始化微型T5的无权全选、mask/local q、G0生成cache token身份；
- R4/G3/G4保存恢复与不中断训练的对照；
- 实际Parquet schema/dtype/压缩读写。

图表代码已实现PDF/300dpi PNG、分层/轨迹和随机区间展示，但本地未渲染，不能声称视觉验收已通过。

## 服务器验收顺序

1. 安装/对齐模型与离线依赖，重跑测试并检查不再有依赖skip。
2. 核验两套dense、两套固定split、数据集/tokenizer真实文件与环境版本。
3. 真实底座Phase0、正常路由及R2/R3非阻断诊断。
4. 完整smoke与保存/恢复对照，检查A–E数目、字段、sample/token/layer对齐。
5. batch/chunk吞吐、显存、数值检查后统一固定，再启动正式矩阵。

本地测试成功不等于模型训练已收敛、不等于主张成立，也不等于CUDA端到端已验证。
