# Phase A F0 增补代码验收（2026-09-04）

范围：只新增R2-soft/R4-hard，backbone冻结；不启动正式实验、不实现fullFT。

- 服务器原源码与本地Git基线91个配置/源码/测试文件内容一致；63处差异仅CRLF/LF。部署前源码备份为`/mnt/luoyulin_code/fanxuankai/task5_code_backup_20260904_phaseA_f0.tar.gz`，不动inputs和旧输出。
- 本地76项测试：54通过，22项因本地缺少Torch/模型依赖或非Linux而跳过。服务器相同76项全部通过，包含系数ST的精确前向/soft代理梯度、expert直接梯度、上游输入梯度、冻结参数、Tiny-T5保存续训、完整D/churn与独立八臂报告测试。
- 服务器真实两任务的dense/split/数据各取2条validation样本，检查R2-soft与R4-hard的teacher-forcing、生成、12层q/counts；R4-hard做一次临时optimizer更新，12层质心均有非零梯度且更新，全部冻结参数哈希不变，保存重载后logits完全一致。
- 旧v3结果（含R4-R2Init）的normalized SHA256固定为`a25f8c83d467a25377b952aa382c7c8bdf54de73a162366fa3aa77dfac55bcbc`；正式补测入口自动核验旧训练配置/seed、输入、共享dense A/E和结果完整性。新记录用新protocol，旧记录身份不改写。

小模型指标和图表只用于合成测试；真实小样本输出也仅为诊断，**都不是正式补实验结果**。正式`runs/.../R2-soft|R4-hard`尚未启动。当前测试服务器只有一张A800；八卡划分通过逻辑分片测试，不声称做过八卡端到端正式执行。

证据：`docs/reports/server_evidence/phaseA_f0_unit.json`与`phaseA_f0_real_assets.json`。正式运行步骤只见仓库根`RUNBOOK.md` §17。
