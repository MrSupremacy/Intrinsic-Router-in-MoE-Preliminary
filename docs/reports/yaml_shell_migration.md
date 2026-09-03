# YAML与Shell入口迁移验收

2026-08-29完成。实验配置已由TOML迁移为YAML，Linux公开阶段入口已由`run.py`薄封装迁移为`run.sh`；
训练、采集、指标与制图仍由同一套Python核心实现。`pyproject.toml`保留，因为它是Python标准打包元数据，
不属于实验配置。

## 等价性与防护

- 迁移前保存主套件、smoke套件分别与无local/example/server组合后的6份完整解析结果。
- 迁移后逐字段比较值及Python标量类型，6份全部完全一致；主矩阵仍为218条件、168个训练run、1898个逻辑状态。
- R3仍为`frozen_raw_centroid_rms_hard`，`rms_epsilon`仍以浮点数`1.0e-6`解析；不启用R2/R3集合一致门槛。
- 使用PyYAML 6.0.3 SafeLoader，并额外拒绝重复键、非字符串键、非法`extends`、非mapping根节点和Python对象标签。
- Shell入口使用`set -euo pipefail`、完整参数引用和`exec`，统一从仓库根运行，可通过`PYTHON`环境变量选择解释器；不修改用户的CUDA环境变量。
- `.gitattributes`固定Shell/YAML为LF，避免Windows传到Linux后的CRLF问题。

## A800回归

在独立目录`/root/workspace/task5_reproduction_yaml_test`测试，未修改2026-08-28的既有smoke目录及产物：

- 15个Shell文件全部通过`bash -n`。
- 从`/tmp`调用阶段脚本成功自动定位仓库；默认配置与`server.yaml`实物preflight均成功。
- 实物preflight重新核验两任务dense、split与四个Parquet文件，并输出稳定内容摘要。
- 服务器67项测试全部通过，0失败、0错误、0跳过；环境含torch 2.11.0+cu126、PyArrow 25.0.1和PyYAML 6.0.3。
- 机器级证据见`server_evidence/yaml_shell_unit.json`。

本次没有重跑已完成的整套小规模四阶段smoke：配置语义已逐字段/类型证明等价，Python模型路径由新增的
真实依赖集成/回归测试覆盖，而迁移特有部分由真实资产preflight与Linux Shell回归覆盖。2026-08-28的完整
四阶段smoke仍是模型计算验收记录；正式实验应使用新run-id，不能把旧TOML实现摘要下的capture冒充新协议产物。
