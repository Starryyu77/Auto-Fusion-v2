# AutoFusion 2.0: Search Space-Free Multimodal Architecture Search

**基于双层闭环与动态适配的无搜索空间多模态架构搜索**

---

## 核心创新

AutoFusion 2.0 突破了传统 NAS 的模板限制，实现了真正的"Search Space-Free"架构搜索：

| 特性 | AutoFusion 1.0 | AutoFusion 2.0 |
|-----|---------------|----------------|
| **搜索空间** | 预定义 5 种模板 | 完全开放，原生代码 |
| **编译保障** | 模板保证 100% | 内环自愈保证 100% |
| **数据适配** | 硬编码数据加载器 | 动态数据适配器 |
| **跨任务泛化** | 需重写数据接口 | 更换文件夹即可 |

### 双层闭环反馈

```
┌────────────────────────────────────────────────────────────────┐
│                    Dual-Loop Feedback                          │
│                                                                │
│  【Inner Loop】Auto-Debugging                                  │
│  LLM → Raw Code → Sandbox Dry-run → Error Feedback → Retry    │
│  (Guarantees 100% compile success)                             │
│                                                                │
│  【Outer Loop】Performance Evolution                           │
│  Proxy Training → Metric Collection → Reward Update → Evolve  │
│  (Architecture topology optimization)                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
autofusion2/
├── src/
│   ├── adapter/              # 动态数据适配器
│   │   └── data_adapter.py
│   ├── controller/           # 双层闭环控制器
│   │   └── dual_loop_controller.py
│   ├── sandbox/              # 安全沙盒与内环编译
│   │   ├── inner_loop.py
│   │   └── secure_sandbox.py
│   ├── evaluator/            # 代理评估与奖励函数
│   │   ├── proxy_evaluator.py
│   │   └── reward_function.py
│   ├── utils/                # 工具函数
│   │   └── llm_backend.py
│   └── main.py               # 主入口
├── configs/                  # 场景配置文件
│   ├── scenario_a_mmmu.yaml
│   ├── scenario_b_medical.yaml
│   └── scenario_c_edge.yaml
├── docs/                     # 文档
│   ├── DESIGN.md             # 系统设计文档
│   └── EXPERIMENTS.md        # 实验规范
└── README.md                 # 本文件
```

---

## 快速开始

### 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate pyyaml matplotlib numpy pillow
pip install openai  # For LLM API
```

### 配置 API Key

```bash
export ALIYUN_API_KEY="your-api-key"
# 或
export DEEPSEEK_API_KEY="your-api-key"
```

### 运行实验

```bash
cd autofusion2

# 场景 A: 高维图文推理 (MMMU)
python src/main.py \
    --data_dir ./data/mmmu \
    --scenario high_dim_reasoning \
    --max_iterations 200

# 场景 B: 医疗视觉问答 (VQA-RAD)
python src/main.py \
    --data_dir ./data/vqa_rad \
    --scenario medical_vqa \
    --max_iterations 200

# 场景 C: 边缘机器人 (超低延迟)
python src/main.py \
    --data_dir ./data/robo_sense \
    --scenario edge_robotics \
    --max_iterations 150
```

### 使用配置文件

```bash
# 场景 A
python src/main.py --config configs/scenario_a_mmmu.yaml

# 场景 B
python src/main.py --config configs/scenario_b_medical.yaml

# 场景 C
python src/main.py --config configs/scenario_c_edge.yaml
```

---

## 三个验证场景

### 场景 A: 高维图文推理 (MMMU)

- **数据集**: MMMU (多学科多模态理解)
- **输入**: 高维视觉特征 [B, 576, 1024] + 长文本特征 [B, 77, 768]
- **约束**: < 10M FLOPs, < 50M 参数
- **挑战**: 复杂推理，细粒度视觉理解

```yaml
constraints:
  max_flops: 10_000_000
  max_params: 50_000_000
  target_accuracy: 0.45  # 接近 FiLM 46%
```

### 场景 B: 医疗视觉问答 (VQA-RAD)

- **数据集**: VQA-RAD (放射科问答)
- **输入**: 高分辨率医学影像 + 专业问题
- **约束**: < 50M FLOPs (宽松，追求高精度)
- **挑战**: 领域专业性，高精度要求

```yaml
constraints:
  max_flops: 50_000_000
  max_params: 100_000_000
  target_accuracy: 0.80  # 高精度医疗诊断
```

### 场景 C: 边缘机器人 (超低延迟)

- **数据集**: 机器人传感器数据
- **输入**: 低分辨率视觉 + 一维传感器序列
- **约束**: < 2M FLOPs, < 1M 参数，< 10ms 延迟
- **挑战**: 极端算力限制，实时性

```yaml
constraints:
  max_flops: 2_000_000
  max_params: 1_000_000
  target_accuracy: 0.90
  inference_time_ms: 10
```

---

## 核心组件

### 1. 动态数据适配器 (DynamicDataAdapter)

自动嗅探数据维度，生成 API 契约：

```python
from src.adapter import DynamicDataAdapter

adapter = DynamicDataAdapter()
dataset, contract = adapter.ingest_folder("./data/mmmu")

print(contract.to_prompt())
# 【API Interface Contract】
# Input Specifications:
#   - visual: Shape [B, 576, 1024], Dtype float32
#   - text: Shape [B, 77, 768], Dtype float32
# Constraints:
#   - max_flops: 10000000
#   - max_params: 50000000
```

### 2. 双层闭环控制器 (DualLoopController)

```python
from src.controller import DualLoopController

controller = DualLoopController(
    llm_backend=llm,
    api_contract=contract,
    proxy_evaluator=evaluator,
    reward_fn=reward_fn,
    max_iterations=200
)

best_result = controller.search()
```

### 3. 内环自愈编译 (InnerLoopSandbox)

```python
from src.sandbox import InnerLoopSandbox

sandbox = InnerLoopSandbox(llm, contract, max_retries=5)
code, attempts = sandbox.self_healing_compile(prompt)
# Returns: (compiled_code, number_of_attempts)
```

---

## 实验目标与成功标准

### 必达目标 (Must Have)

| 标准 | 阈值 | 验证方式 |
|-----|------|---------|
| 编译成功率 | ≥ 95% | 统计所有迭代 |
| 场景 A 准确率 | ≥ 40% | 超越 Concat+MLP |
| 场景 B 准确率 | ≥ 70% | 超越随机基线 |
| 场景 C 延迟 | ≤ 20ms | 实际设备测试 |
| 跨场景运行 | 是 | 仅更换数据文件夹 |

### 期望目标 (Nice to Have)

- 场景 A 准确率 ≥ 45% (接近 FiLM 46%)
- 发现人类未预定义的新架构范式
- 搜索收敛 < 100 轮

---

## 与 AutoFusion 1.0 对比

| 维度 | 1.0 (Phase 5.5) | 2.0 (当前) |
|-----|----------------|-----------|
| 搜索空间 | 5 种固定模板 | 完全开放 |
| 架构创新 | 有限 | 无限可能 |
| 编译成功率 | 100% (模板保证) | 100% (内环保证) |
| 跨任务泛化 | 需修改代码 | 更换文件夹即可 |
| 数据接口 | 硬编码 | 动态适配 |

---

## 文档

- [系统设计文档](docs/DESIGN.md) - 详细架构设计
- [实验规范](docs/EXPERIMENTS.md) - 三个场景的详细实验设计

---

## Citation

```bibtex
@article{autofusion2,
  title={AutoFusion 2.0: Search Space-Free Multimodal Architecture Search with Dual-Loop Feedback},
  author={AutoFusion Team},
  year={2026}
}
```

---

*Version: 2.0.0*
*Last Updated: 2026-03-03*
