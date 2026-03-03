# AutoFusion 2.0: 实验执行计划

> **版本**: 1.0
> **状态**: 待确认
> **预计周期**: 4-5 周
> **并行度**: 3 个场景可同时运行

---

## 1. 执行概览

### 1.1 实验矩阵

| 场景 | 数据集 | 目标 | 迭代数 | 预计时间 | GPU |
|-----|--------|------|-------|---------|-----|
| A | MMMU | 验证高维图文推理 | 200 iter | ~8h | GPU 0 |
| B | VQA-RAD | 验证医疗领域适配 | 200 iter | ~10h | GPU 1 |
| C | RoboSense | 验证边缘低延迟 | 150 iter | ~6h | GPU 2 |

### 1.2 关键里程碑

```
Week 1: 基础设施验证
  ├─ Day 1-2: 内环/外环单元测试
  ├─ Day 3-4: Mock LLM 端到端验证
  └─ Day 5: 真实 API 小规模测试 (20 iter)

Week 2-3: 正式实验执行
  ├─ Week 2: 场景 A + B 并行
  └─ Week 3: 场景 C + 补跑失败实验

Week 4: 基线与消融
  ├─ Baseline (Concat+MLP) 全量评估
  ├─ 消融实验 (无内环/无历史)
  └─ FiLM 对比评估

Week 5: 分析与文档
  ├─ 数据分析与可视化
  ├─ 论文图表生成
  └─ 开源代码整理
```

---

## 2. 详细执行步骤

### Phase 1: 基础设施验证 (Week 1)

#### Day 1-2: 核心组件单元测试

**任务清单**:
- [ ] 数据适配器: 测试 3 种数据格式 (MMMU/VQA-RAD/自定义 JSON)
- [ ] 内环沙盒: 验证 100% 编译成功率
- [ ] 安全沙盒: 测试显存隔离与 OOM 恢复
- [ ] 奖励函数: 验证多目标权衡

**验证命令**:
```bash
# 运行完整测试套件
cd autofusion2
python -m pytest tests/ -v --tb=short

# 显存压力测试 (模拟 LLM 生成泄漏代码)
python tests/test_memory_isolation.py --stress-test
```

**成功标准**:
- 所有单元测试通过
- 显存隔离测试: 连续 100 次执行无 OOM
- 内环编译: 连续 50 次 Mock 代码编译成功率 100%

#### Day 3-4: Mock LLM 端到端验证

**任务**: 使用 Mock LLM 跑通完整 200 iter

```bash
# Mock 模式 (不消耗 API)
python src/main.py \
  --scenario high_dim_reasoning \
  --mock-llm \
  --max-iterations 200 \
  --output-dir ./test_mock
```

**Mock LLM 行为**:
- 50% 概率生成正确代码
- 30% 概率生成维度错误 (测试内环修复)
- 20% 概率生成语法错误 (测试错误恢复)

**监控指标**:
```python
{
  "iterations_completed": 200,
  "compile_success_rate": ">=95%",
  "avg_inner_loop_attempts": "<2.0",
  "checkpoint_saved_every": "10 iter",
  "memory_leak_detected": false
}
```

#### Day 5: 真实 API 小规模验证

**任务**: 使用真实 LLM 跑 20 iter 验证 API 稳定性

| 模型 | 目的 | 预算 |
|-----|------|------|
| Kimi-K2.5 | 主实验模型 | ~$5 |
| GLM-5 | 对比模型 | ~$5 |

**风险控制**:
- 设置 `--max-cost 10` 自动停止
- 每 5 iter 保存 checkpoint
- 实时监控 API 错误率

---

### Phase 2: 正式实验执行 (Week 2-3)

#### 场景 A: MMMU (高维图文推理)

**执行命令**:
```bash
export ALIYUN_API_KEY="sk-xxxxx"
export CUDA_VISIBLE_DEVICES=0

python src/main.py \
  --config configs/scenario_a_mmmu.yaml \
  --llm-model kimi-k2.5 \
  --max-iterations 200 \
  --output-dir ./results/scenario_a_kimi
```

**监控面板**:
```bash
# 实时监控 (每 30s 更新)
watch -n 30 'python scripts/monitor.py ./results/scenario_a_kimi'

# 预期输出:
# Iteration: 45/200
# Compile Success: 44/45 (97.8%)
# Best Reward: 1.234
# Est. Time Remaining: 5.2h
# API Cost: $2.35
```

**早停条件**:
- 连续 30 iter 无改进 (reward 提升 < 0.01)
- API 成本超过预算 ($20)
- 编译成功率 < 80% (系统异常)

#### 场景 B: VQA-RAD (医疗 VQA)

**执行命令**:
```bash
export CUDA_VISIBLE_DEVICES=1

python src/main.py \
  --config configs/scenario_b_medical.yaml \
  --llm-model kimi-k2.5 \
  --max-iterations 200 \
  --output-dir ./results/scenario_b_kimi
```

**特殊监控**:
- 医疗场景准确率需 >70%，否则触发告警
- 监控 GPU 内存 (医疗图像分辨率更高)

#### 场景 C: Edge Robotics (并行执行)

**执行命令**:
```bash
export CUDA_VISIBLE_DEVICES=2

python src/main.py \
  --config configs/scenario_c_edge.yaml \
  --llm-model kimi-k2.5 \
  --max-iterations 150 \
  --output-dir ./results/scenario_c_kimi
```

**边缘场景特殊要求**:
- 每 10 iter 导出模型测试推理延迟
- 模拟器验证 (<10ms 目标)

---

### Phase 3: 基线与消融 (Week 4)

#### 3.1 下界基线: Concat+MLP

**实现**:
```python
class BaselineConcatMLP(nn.Module):
    def forward(self, visual, text):
        v = visual.mean(dim=1)  # [B, 1024]
        t = text.mean(dim=1)    # [B, 768]
        fused = torch.cat([v, t], dim=-1)  # [B, 1792]
        return self.mlp(fused)  # [B, num_classes]
```

**评估协议**:
- 每个场景运行 3 次取平均
- 与 AutoFusion 2.0 发现的 Top-3 架构对比

#### 3.2 消融实验

| 消融项 | 配置 | 目的 |
|-------|------|------|
| 无内环 | `use_inner_loop=false` | 证明内环必要性 |
| 无历史 | `use_history=false` | 证明历史记忆必要性 |
| 固定种子 | `seed=42` 重复 3 次 | 验证结果稳定性 |

**预期结果**:
- 无内环: 编译成功率 < 15%，系统无法运转
- 无历史: 收敛速度降低 30-50%

#### 3.3 FiLM 对比

使用 Phase 4 的 FiLM 实现作为上界基线:
```bash
python archive/phase4_optimization/evaluate_baselines.py \
  --dataset mmmu \
  --method film \
  --compare-with ./results/scenario_a_kimi
```

---

### Phase 4: 分析文档 (Week 5)

#### 4.1 数据分析

```python
# 生成对比图表
python scripts/analyze_results.py \
  --results ./results/scenario_{a,b,c}_kimi \
  --baselines ./baselines \
  --output ./analysis
```

**生成图表**:
1. 编译成功率随迭代变化
2. Reward 收敛曲线
3. 架构复杂度 vs 性能散点图
4. 场景间泛化能力雷达图

#### 4.2 开源准备

- [ ] 代码清理 (移除硬编码路径)
- [ ] README 完善 (中英双语)
- [ ] 示例数据准备 (各场景 100 样本)
- [ ] LICENSE 选择 (MIT/Apache)

---

## 3. 资源需求

### 3.1 硬件资源

| 资源 | 数量 | 用途 | 预估成本 |
|-----|------|------|---------|
| RTX A5000 | 3× | 主实验 | 自有 |
| Jetson Nano | 1× | 边缘验证 | $99 (一次性) |
| 存储 | 500GB | 数据集 + 结果 | ~$50/月 |

### 3.2 API 预算

**按场景估算** (200 iter × 平均 2.5 API calls/iter):

| 模型 | 单价 | 单场景成本 | 3 场景总成本 |
|-----|------|-----------|-------------|
| Kimi-K2.5 | $0.003/1K tokens | ~$15 | ~$45 |
| GLM-5 | $0.002/1K tokens | ~$10 | ~$30 |
| **预留缓冲** | - | - | **+$25** |
| **总计** | - | - | **~$100** |

**成本控制措施**:
- 设置 `--max-cost-per-scenario 20` 自动停止
- 使用 Mock LLM 预验证 (零成本)
- 预算告警: 80% 时 Slack 通知

### 3.3 人力投入

| 角色 | 时间 | 任务 |
|-----|------|------|
| 实验工程师 | 3 周 | 执行 + 监控 + 调参 |
| 数据分析 | 1 周 | 可视化 + 统计检验 |
| 论文写作 | 1 周 | 方法论 + 结果描述 |

---

## 4. 风险评估与应对

### 4.1 技术风险

| 风险 | 概率 | 影响 | 应对措施 |
|-----|------|------|---------|
| **API 服务中断** | 中 | 高 | 准备备用模型 (GLM-5 备用); 启用 checkpoint 续跑 |
| **OOM 崩溃** | 低 | 高 | 已实施三层显存防护; 设置 `--max-vram 4096` |
| **LLM 陷入循环** | 中 | 中 | 历史记忆机制; 5 次失败自动 reset prompt |
| **编译成功率低** | 低 | 高 | Mock 测试先行; 若 <90% 暂停排查 Prompt |

### 4.2 数据风险

| 风险 | 应对 |
|-----|------|
| VQA-RAD 获取困难 | 备用: 使用 SLAKE 或 PathVQA |
| 机器人数据集缺失 | 自建: 使用 Habitat Simulator 生成 |
| 数据预处理错误 | 每个数据集人工验证 10 个样本 |

### 4.3 时间风险

**缓冲策略**:
- Week 1 延误 → 压缩 Phase 3 (消融可减少)
- 单场景失败 → 优先保证场景 A (核心验证)
- API 限速 → 切换至 GLM-5 (更快更便宜)

---

## 5. 监控与日志

### 5.1 实时监控脚本

```bash
# 启动监控 (每 60s 更新)
python scripts/monitor.py \
  --watch-dirs ./results/scenario_{a,b,c} \
  --alert-webhook "https://hooks.slack.com/xxx"
```

**监控面板输出**:
```
╔══════════════════════════════════════════╗
║  AutoFusion 2.0 - Live Monitor           ║
╠══════════════════════════════════════════╣
║  Scenario A (Kimi)                       ║
║    Progress: 87/200 (43.5%)              ║
║    Best Reward: 1.247 ↑                  ║
║    Compile Rate: 98.9% ✓                 ║
║    Est. Remaining: 4.2h                  ║
║    API Cost: $6.78                       ║
╠══════════════════════════════════════════╣
║  Scenario B (Kimi)                       ║
║    Progress: 45/200 (22.5%)              ║
║    Best Reward: 0.892 ↑                  ║
║    Compile Rate: 97.8% ✓                 ║
║    Est. Remaining: 7.1h                  ║
║    API Cost: $3.45                       ║
╠══════════════════════════════════════════╣
║  Last Updated: 2026-03-03 14:32:15       ║
╚══════════════════════════════════════════╝
```

### 5.2 日志规范

**日志级别**:
- `INFO`: 迭代完成、checkpoint 保存
- `WARNING`: 编译失败 (但已修复)、API 延迟
- `ERROR`: 连续 5 次编译失败、OOM、数据加载失败

**日志路径**:
```
results/
└── scenario_a_kimi/
    ├── checkpoints/           # 每 10 iter
    ├── logs/
    │   ├── main.log          # 完整日志
    │   ├── errors.log        # 仅错误
    │   └── api_calls.jsonl   # API 调用记录
    └── final_report.json
```

---

## 6. 成功标准

### 6.1 必达目标 (P0)

| 目标 | 衡量标准 | 验证方式 |
|-----|---------|---------|
| 系统可运行 | 完成 200 iter 无崩溃 | 日志文件 |
| 编译成功率 | ≥ 95% | 统计日志 |
| 场景 A 有效 | Reward > 1.0 (超越随机) | 对比 baseline |
| 跨场景泛化 | 同一套代码跑通 3 个场景 | 实验复现 |

### 6.2 期望目标 (P1)

| 目标 | 衡量标准 |
|-----|---------|
| 场景 A 准确率 | ≥ 40% (接近 FiLM 46%) |
| 内环效率 | 平均尝试次数 < 2.0 |
| 边缘延迟 | < 20ms on Jetson Nano |
| 消融验证 | 无内环成功率 < 15% |

### 6.3 超预期目标 (P2)

| 目标 | 衡量标准 |
|-----|---------|
| 发现新架构 | 人类专家未预定义的结构 |
| 超越 FiLM | 场景 A 准确率 > 46% |
| 论文发表 | NeurIPS/ICML/CVPR 投稿 |

---

## 7. 待确认事项

### 7.1 需要用户决策

- [ ] **LLM 模型选择**: Kimi-K2.5 (推荐) / GLM-5 / Qwen-Max?
- [ ] **实验预算上限**: 确认 $100 API 预算是否可接受
- [ ] **边缘设备**: 是否有 Jetson Nano? 或使用模拟器?
- [ ] **VQA-RAD 数据**: 是否已获取? 需要备用方案?
- [ ] **并发策略**: 3 场景并行 (快) 还是串行 (资源省)?

### 7.2 预检查清单

实验启动前请确认:
- [ ] `export ALIYUN_API_KEY="sk-xxxxx"` 已设置
- [ ] GPU 服务器可访问: `ssh gpu43.dynip.ntu.edu.sg`
- [ ] 数据已下载: `/data/{mmmu,vqa_rad,robo_sense}`
- [ ] 磁盘空间 > 100GB
- [ ] Python 环境: `torch>=2.0`, `transformers>=4.30`

---

## 附录

### A. 紧急联系

| 问题类型 | 联系人 | 响应时间 |
|---------|-------|---------|
| 服务器宕机 | NTU IT | 24h |
| API 异常 | Aliyun 客服 | 4h |
| 代码 Bug | AutoFusion Team | 2h |

### B. 相关文档

- [系统设计](DESIGN.md)
- [实验规范](EXPERIMENTS.md)
- [API 文档](API.md) (待创建)
- [故障排查](TROUBLESHOOTING.md) (待创建)

---

*计划版本: 1.0*
*最后更新: 2026-03-03*
*待确认: 需要用户审阅并批准后开始执行*
