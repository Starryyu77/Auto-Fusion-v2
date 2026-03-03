# AutoFusion 2.0 - 实验执行日志

> **执行环境**: NTU GPU43 (gpu43.dynip.ntu.edu.sg)
> **API Key**: 阿里百炼云 (sk-fa81e2c...)
> **执行模式**: 串行 (Serial)
> **GitHub**: https://github.com/Starryyu77/Auto-Fusion-v2

---

## 实验矩阵

| 编号 | 场景 | 模型 | GPU | 状态 | 开始时间 | 完成时间 | 最佳Reward |
|-----|------|------|-----|------|---------|---------|-----------|
| 1 | MMMU | Kimi-K2.5 | 0 | <待开始> | - | - | - |
| 2 | MMMU | GLM-5 | 0 | <待开始> | - | - | - |
| 3 | MMMU | Qwen-Max | 0 | <待开始> | - | - | - |
| 4 | MMMU | DeepSeek-V3 | 0 | <待开始> | - | - | - |
| 5 | VQA-RAD | Kimi-K2.5 | 1 | <待开始> | - | - | - |
| 6 | VQA-RAD | GLM-5 | 1 | <待开始> | - | - | - |
| 7 | VQA-RAD | Qwen-Max | 1 | <待开始> | - | - | - |
| 8 | VQA-RAD | DeepSeek-V3 | 1 | <待开始> | - | - | - |
| 9 | RoboSense | Kimi-K2.5 | 2 | <待开始> | - | - | - |
| 10 | RoboSense | GLM-5 | 2 | <待开始> | - | - | - |
| 11 | RoboSense | Qwen-Max | 2 | <待开始> | - | - | - |
| 12 | RoboSense | DeepSeek-V3 | 2 | <待开始> | - | - | - |

---

## 执行进度

### Phase 1: 基础设施验证 (Week 1)

#### Day 1-2: 单元测试
- [ ] 数据适配器测试
- [ ] 内环沙盒测试
- [ ] 安全沙盒显存隔离测试
- [ ] 奖励函数测试
- [ ] GitHub 提交: `tests: Add unit tests for core components`

#### Day 3-4: Mock LLM 端到端
- [ ] 200 iter Mock 测试
- [ ] 显存泄漏检测
- [ ] 编译成功率验证 (>95%)
- [ ] GitHub 提交: `test: Mock LLM end-to-end validation`

#### Day 5: 真实 API 小规模验证
- [ ] 20 iter × 4 models 测试
- [ ] API 稳定性验证
- [ ] 成本基准测试
- [ ] GitHub 提交: `test: Real API smoke test`

**Phase 1 完成标准**: 所有测试通过，编译成功率 >95%，无显存泄漏

---

### Phase 2: 正式实验 (Week 2-5)

每个实验完成后更新此表格：

#### 实验 #1: MMMU + Kimi-K2.5
```yaml
Status: <进行中/完成/失败>
Start: 2026-XX-XX XX:XX
End: 2026-XX-XX XX:XX
Duration: Xh Xm
Iterations: 200/200
Best Reward: X.XXX
Best Accuracy: XX.X%
Compile Success Rate: XX.X%
API Calls: XXX
Cost: $X.XX
Notes:
  - <任何观察或问题>
GitHub Commit: <hash>
```

---

## 结果汇总

### 场景 A: MMMU (高维图文推理)

| 模型 | Best Reward | Best Accuracy | FLOPs | 编译成功率 | 成本 |
|-----|-------------|---------------|-------|-----------|------|
| Kimi-K2.5 | - | - | - | - | - |
| GLM-5 | - | - | - | - | - |
| Qwen-Max | - | - | - | - | - |
| DeepSeek-V3 | - | - | - | - | - |

### 场景 B: VQA-RAD (医疗 VQA)

| 模型 | Best Reward | Best Accuracy | FLOPs | 编译成功率 | 成本 |
|-----|-------------|---------------|-------|-----------|------|
| Kimi-K2.5 | - | - | - | - | - |
| GLM-5 | - | - | - | - | - |
| Qwen-Max | - | - | - | - | - |
| DeepSeek-V3 | - | - | - | - | - |

### 场景 C: RoboSense (边缘机器人)

| 模型 | Best Reward | Best Accuracy | FLOPs | 编译成功率 | 成本 |
|-----|-------------|---------------|-------|-----------|------|
| Kimi-K2.5 | - | - | - | - | - |
| GLM-5 | - | - | - | - | - |
| Qwen-Max | - | - | - | - | - |
| DeepSeek-V3 | - | - | - | - | - |

---

## 问题与解决

### 问题 #1: <标题>
- **发生时间**:
- **描述**:
- **解决方案**:
- **状态**: <已解决/待解决>

---

## 基线对比

### Baseline: Concat+MLP

| 场景 | 准确率 | FLOPs | 参数量 |
|-----|--------|-------|--------|
| MMMU | XX.X% | X.XXM | X.XXM |
| VQA-RAD | XX.X% | X.XXM | X.XXM |
| RoboSense | XX.X% | X.XXM | X.XXM |

### FiLM 对比 (仅 MMMU)

| 指标 | FiLM (人类) | AutoFusion 2.0 (Kimi) | 对比 |
|-----|-------------|----------------------|------|
| 准确率 | 46% | - | - |
| FLOPs | 6.29M | - | - |

---

## 代码提交历史

| 日期 | Commit | 描述 |
|-----|--------|------|
| 2026-03-03 | 1fedc99 | Initial commit: Core components |
| <更新中> | - | - |

---

## 下一步行动

1. <待完成>

---

*最后更新: 2026-03-03*
*下次更新: 每个实验完成后*
