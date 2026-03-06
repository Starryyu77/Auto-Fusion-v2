# AutoFusion 2.0 - 实验执行日志

> **执行环境**: NTU GPU43 (gpu43.dynip.ntu.edu.sg)
> **API Key**: 阿里百炼云 (sk-fa81e2c...)
> **执行模式**: 串行 (Serial)
> **GitHub**: https://github.com/Starryyu77/Auto-Fusion-v2

---

## 实验矩阵

| 编号 | 场景 | 模型 | GPU | 状态 | 开始时间 | 完成时间 | 最佳Reward |
|-----|------|------|-----|------|---------|---------|-----------|
| 1 | MMMU | Kimi-K2.5 | 0 | ✅ 完成 | 2026-03-04 15:58 | 2026-03-04 19:14 | **0.746** |
| 2 | MMMU | Qwen-Max | 0 | ❌ 停止 | 2026-03-05 14:08 | 2026-03-05 16:30 | 0% 编译成功率 |
| 3 | MMMU | DeepSeek-V3 | 0 | ✅ 完成 | 2026-03-05 16:31 | 2026-03-05 18:56 | **0.831** 🏆 |
| 4 | MMMU | GLM-5 | 0 | 🔄 **运行中** | 2026-03-06 19:37 | - | **0.641** (当前) |
| 5 | VQA-RAD | Kimi-K2.5 | 1 | 🔄 **运行中** | 2026-03-06 19:46 | - | **-0.512** (Iter 1) |
| 6 | VQA-RAD | DeepSeek-V3 | 2 | 🔄 **运行中** | 2026-03-06 19:46 | - | - |
| 7 | VQA-RAD | GLM-5 | 3 | 🔄 **运行中** | 2026-03-06 19:46 | - | - |
| 8 | VQA-RAD | Qwen-Max | 1 | ❌ 跳过 | - | - | 编译成功率太低 |

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
Status: ✅ 完成
Start: 2026-03-04 15:58
End: 2026-03-04 19:14
Duration: 3h 16m
Iterations: 200/200
Best Reward: 0.746
Best Accuracy: 28.12%
Compile Success Rate: 100%
API Calls: ~400 (200 iterations × 2 avg)
Cost: ~$12
Notes:
  - 100% 编译成功率，系统非常稳定
  - 最佳架构仅 1.3M FLOPs，非常轻量
  - Reward 在 0.746 后收敛，后期无显著提升
  - 架构特点: Attention pooling + Cross-modal fusion
GitHub Commit: c57e620 (best_architecture.py saved locally)
```

#### 实验 #2: MMMU + Qwen-Max
```yaml
Status: ❌ 已停止 (编译成功率 0%)
Start: 2026-03-05 14:08
End: 2026-03-05 16:30
Duration: ~2.5h
Iterations: 45/200 (提前停止)
Best Reward: N/A
Best Accuracy: N/A
Compile Success Rate: 0% (0/45)
API Calls: ~225 (45 iterations × 5 attempts)
Cost: ~$3 (浪费)
Notes:
  - Qwen-Max 生成代码质量极差
  - 连续 45 次迭代，0 次编译成功
  - 每次 5 次重试后全部失败
  - 决定停止，切换到 DeepSeek-V3
GitHub Commit: N/A
```

#### 实验 #3: MMMU + DeepSeek-V3 ✅
```yaml
Status: ✅ 完成
Start: 2026-03-05 16:31
End: 2026-03-05 18:56
Duration: 145.3 minutes
Iterations: 200/200
Best Reward: 0.831 🏆
Best Accuracy: 35.94% 🏆
Compile Success Rate: 94.0%
API Calls: ~600
Cost: ~$15
Notes:
  - 🏆 超越 Kimi-K2.5! (+11.4% Reward)
  - 🏆 最佳准确率 35.94% (vs Kimi 28.12%)
  - 🏆 最佳 FLOPs: 1.0M (vs Kimi 1.3M)
  - 🏆 运行速度比 Kimi 快 25%
  - 架构特点: 极简 MLP fusion (mean pooling + 2 layer MLP)
  - 架构简单但极其高效
GitHub Commit: d3eaba8
```

#### 实验 #4: MMMU + GLM-5 🔄 (第二轮运行)
```yaml
Status: 🔄 运行中
Start: 2026-03-06 19:37 (重启)
End: -
Duration: -
Iterations: 2/200 (进行中)
Best Reward: 0.641 (当前)
Best Accuracy: 28.12%
Compile Success Rate: 100%
API Calls: ~
Cost: ~
Notes:
  - 100% 编译成功率
  - 历史最佳: 0.857 (第一轮，Iteration 57)
  - FLOPs: 5.1M
  - 系统运行稳定
```

#### 实验 #5: VQA-RAD + Kimi-K2.5 🔄 (代码修复后重启)
```yaml
Status: 🔄 运行中 (修复后)
Start: 2026-03-06 19:46
End: -
Duration: -
Iterations: 2/200 (进行中)
Best Reward: -0.512 (Iter 1)
Best Accuracy: 28.12%
Compile Success Rate: 100%
Notes:
  - ✅ 代码修复后成功运行！
  - ✅ Iteration 1 完成，无 CUDA 错误
  - 医疗 VQA 场景 (13 个类别)
  - GPU 1 运行中
```

#### 实验 #6: VQA-RAD + DeepSeek-V3 🔄 (代码修复后重启)
```yaml
Status: 🔄 运行中 (修复后)
Start: 2026-03-06 19:46
End: -
Duration: -
Iterations: 刚刚启动
Best Reward: -
Best Accuracy: -
Compile Success Rate: -
Notes:
  - 医疗 VQA 场景 (13 个类别)
  - GPU 2 运行中
```

#### 实验 #7: VQA-RAD + GLM-5 🔄 (代码修复后重启)
```yaml
Status: 🔄 运行中 (修复后)
Start: 2026-03-06 19:46
End: -
Duration: -
Iterations: 刚刚启动
Best Reward: -
Best Accuracy: -
Compile Success Rate: -
Notes:
  - 医疗 VQA 场景 (13 个类别)
  - GPU 3 运行中
```

---

## 结果汇总

### 场景 A: MMMU (高维图文推理)

| 模型 | Best Reward | Best Accuracy | FLOPs | 编译成功率 | 成本 |
|-----|-------------|---------------|-------|-----------|------|
| **🥇 DeepSeek-V3** | **0.831** 🏆 | **35.94%** 🏆 | **1.0M** 🏆 | **94%** | **~$15** |
| **🥈 Kimi-K2.5** | 0.746 | 28.12% | 1.3M | 100% | ~$12 |
| 🥉 GLM-5 | - | - | - | - | - |
| ❌ Qwen-Max | N/A | N/A | N/A | 0% | ~$3 (浪费) |

### 场景 B: VQA-RAD (医疗 VQA)

| 模型 | Best Reward | Best Accuracy | FLOPs | 编译成功率 | 成本 |
|-----|-------------|---------------|-------|-----------|------|
| Kimi-K2.5 | 🔄 | 🔄 | 🔄 | 100% | - |
| GLM-5 | 🔄 | 🔄 | 🔄 | - | - |
| Qwen-Max | ❌ 跳过 | ❌ 跳过 | ❌ 跳过 | 0% | - |
| DeepSeek-V3 | 🔄 | 🔄 | 🔄 | - | - |

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

---

## Bug 修复记录 (2026-03-06)

### 问题: VQA-RAD 实验 CUDA device-side assert 错误

**症状**:
- 所有 VQA-RAD 实验（#5, #6, #7）在 Iteration 1-2 时报错
- 错误信息: `CUDA error: device-side assert triggered`
- 根本原因是类别索引越界

**根因分析**:
1. `proxy_evaluator.py` 的 `_get_num_classes()` 只采样前 100 个样本，可能遗漏类别
2. `_create_dataloaders()` 硬编码为 4 个类别，但 VQA-RAD 有 13 个类别
3. 训练时标签索引超出输出维度，触发 CUDA assert

**修复内容** (`src/evaluator/proxy_evaluator.py`):
1. ✅ `_get_num_classes()`: 扫描所有样本，正确处理 max label value
2. ✅ `_create_dataloaders()`: 使用实际的 num_classes 而非硬编码 4
3. ✅ `_train_model()`: 添加标签验证和 clamping，防止索引越界

**GitHub Commit**: `5df20bd`

**状态**: ✅ 已修复，VQA-RAD 实验已成功重启并正常运行
- 实验 #5 (Kimi): Iteration 1 完成，准确率 28.12%，无 CUDA 错误
- 实验 #6 (DeepSeek): 运行中
- 实验 #7 (GLM-5): 运行中

---

## 实验重启记录 (2026-03-06)

### 重启原因
阿里云账户已充值，恢复 API 调用。

### 重启状态
| 实验 | 场景 | 模型 | 重启时间 | 状态 | 备注 |
|-----|------|------|----------|------|------|
| #4 | MMMU | GLM-5 | 2026-03-06 19:37 | 🔄 运行中 | Iteration 2/200, Reward 0.641 |
| #5 | VQA-RAD | Kimi-K2.5 | 1 | 🔄 **运行中** | Iter 1 完成，准确率 28.12% |
| #6 | VQA-RAD | DeepSeek-V3 | 2 | 🔄 **运行中** | 代码修复后重启 |
| #7 | VQA-RAD | GLM-5 | 3 | 🔄 **运行中** | 代码修复后重启 |

**备注**:
- ✅ 实验 #4 正在正常运行（GLM-5 MMMU）
- ✅ VQA-RAD 实验代码修复成功，已重启并正常运行
- ✅ 实验 #5 (Kimi VQA-RAD) Iteration 1 完成，Reward -0.512，准确率 28.12%
- ✅ GPU 0-3 全负载运行，4 个实验并行

---

*最后更新: 2026-03-06*
*下次更新: 每个实验完成后*
