# Task 5 实验报告模板

此文件只提供结构，不生成或预写研究结论。正式结果尚未运行。

## 输入和协议

填写dense/split/data身份、实现版本、run-id、环境、各阶段batch与测试通过记录。
明确固定K-Means、router-only训练、R3 RMS、R4-R2Init centroid初始化、best-validation及M19处理。

## 主性能与负载

链接results/tables/main、主性能图、各seed best epoch/step、final对照与配对差。
报告R4相对R2、R4-R2Init相对R4及每个G候选的增量，不按指标分别挑赢家。

## 漂移与选择质量

链接完整11点轨迹、终态epoch9→10 churn、R4与R4-R2Init overlap/coverage、共激活selected/random/excess/ratio。
保留逐层、逐seed及G2三档补充表，区分随机参照区间与seed std。

## 预期、反例与五个问题

1. R2/R3同输入排序与实际性能有何关系？R3使用RMS，不强制严格等价。
2. R4相对R2的增量在哪些预算档？
3. G1–G4的负载、漂移、性能各有何代价或反例？
4. 性能与三个选择质量probe是否一致？
5. 对当前summary-token router能作何结论，哪些参数派生/单因素归因主张尚不受支持？

每项结论附表/图/原始产物位置；没有证据就标未验证，不把零失败或进程退出0当作研究结论。
