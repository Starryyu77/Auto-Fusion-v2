# AutoFusion 2.0: 泛化性验证实验设计

## 1. 实验总览

为了证明 AutoFusion 2.0 是一个**只需更换输入文件夹**就能自动适配各种场景的"通用兵工厂"，我们设计了三个差异巨大的测试场景，覆盖不同的数据模态、任务类型和算力约束。

### 1.1 实验矩阵

| 维度 | 场景 A | 场景 B | 场景 C |
|-----|--------|--------|--------|
| **任务类型** | 多学科图文推理 | 医疗视觉问答 | 端侧传感器融合 |
| **数据集** | MMMU | VQA-RAD | 开源机器人数据集 |
| **视觉特征** | 高维 (576×1024) | 高分辨病理图 | 低分辨率 (224×224) |
| **文本特征** | 长文本 (77×768) | 极短问题 | 传感器序列 |
| **算力约束** | 中等 (<10M FLOPs) | 宽松 (追求精度) | 极严 (<2M FLOPs) |
| **核心挑战** | 复杂推理 | 领域知识 | 实时推理 |

### 1.2 实验目标

1. **验证无搜索空间范式的有效性**：证明 LLM 可以生成人类未预定义的新架构
2. **验证跨任务泛化能力**：同一系统仅更换输入文件夹即可适配不同任务
3. **验证双层闭环的必要性**：消融实验证明内环对系统运转的关键作用
4. **对比人类设计**：与领域专家设计的专用架构进行公平对比

---

## 2. 场景 A：高维图文推理 (MMMU)

### 2.1 任务描述

MMMU (Massive Multi-discipline Multimodal Understanding) 是一个大学级别的多学科多模态理解基准测试，涵盖艺术、商业、科学、医学等多个领域。每个问题包含一张图片和一段文字描述，要求模型进行复杂的多模态推理。

### 2.2 数据规格

```yaml
# 数据接口契约示例
dataset: MMMU
task_type: multiple_choice_qa

input_structure:
  visual:
    source: image
    backbone: clip-vit-l-14
    shape: [B, 576, 1024]  # [batch, num_patches, hidden_dim]
    dtype: float32
  text:
    source: question + options
    backbone: clip-text-l
    shape: [B, 77, 768]  # [batch, max_seq_len, hidden_dim]
    dtype: float32
  label:
    shape: [B]  # class index
    num_classes: variable_per_subject  # 通常 2-4 个选项

# 物理文件夹结构
folder_structure: |
  /data/mmmu/
  ├── images/
  │   ├── val/
  │   │   ├── Art/
  │   │   ├── Business/
  │   │   ├── Science/
  │   │   └── Medicine/
  │   └── test/...
  └── annotations.json

annotations_format: |
  {
    "id": "Science_1",
    "image": "images/val/Science/1.jpg",
    "question": "What is the main function of...",
    "choices": ["A. Function A", "B. Function B", ...],
    "answer": "B",
    "subject": "Science"
  }
```

### 2.3 约束条件

```python
constraints = {
    "max_flops": 10_000_000,      # 10M FLOPs
    "max_params": 50_000_000,     # 50M 参数
    "target_accuracy": 0.45,      # 目标 45% (接近人类设计的 FiLM)
    "min_accuracy": 0.25,         # 最低可接受 25%
    "inference_time": 50,         # 50ms on RTX A5000
}
```

### 2.4 评估协议

**代理评估 (外环使用)**：
- 子集：每个学科随机抽取 128 个样本
- 训练：Few-shot (16 shots per class)
- 轮数：5 epochs (快速评估)
- 指标：Top-1 Accuracy

**完整评估 (最终测试)**：
- 全集：完整的 validation set (~3,000 样本)
- 训练：标准 supervised learning
- 轮数：30 epochs
- 指标：
  - Overall Accuracy
  - Per-Subject Accuracy (Art, Business, Science, Medicine, etc.)
  - FLOPs & Parameters
  - Inference Latency

### 2.5 预期 LLM 设计挑战

**场景特征带来的挑战**：
1. **长序列建模**：文本 77 tokens，需要有效的长程依赖建模
2. **细粒度视觉理解**：576 个 patch，需要捕捉局部细节
3. **跨模态对齐**：视觉和文本特征维度不同 (1024 vs 768)，需要有效的投影和对齐
4. **多选推理**：需要整合信息做出选择，而非简单分类

**期望的架构创新**：
- 层次化注意力机制（先局部后全局）
- 跨模态门控融合（动态权重）
- 多尺度特征聚合

---

## 3. 场景 B：空间医疗问答 (VQA-RAD)

### 3.1 任务描述

VQA-RAD (Visual Question Answering for Radiology) 是一个放射科视觉问答数据集。医生提出问题关于医学影像（如 CT、MRI、X-ray），模型需要基于图像内容给出准确答案。这是一个领域专业性极强的任务，需要理解医学术语和视觉病理特征。

### 3.2 数据规格

```yaml
# 数据接口契约示例
dataset: VQA-RAD
task_type: medical_vqa

input_structure:
  visual:
    source: medical_image
    backbone: clip-vit-l-14  # 或专门的医学影像 backbone
    shape: [B, 576, 1024]  # 高分辨率病理图特征
    dtype: float32
    special_note: "医学影像细节至关重要，需要高分辨率处理"
  text:
    source: question
    backbone: clip-text-l
    shape: [B, 77, 768]
    dtype: float32
    special_note: "问题通常很短，但涉及专业医学术语"
  label:
    shape: [B]
    num_classes: variable  # Open-ended 或封闭集

# 物理文件夹结构
folder_structure: |
  /data/vqa_rad/
  ├── images/
  │   ├── brain/          # 脑部影像
  │   ├── chest/          # 胸部影像
  │   ├── abdomen/        # 腹部影像
  │   └── ...
  └── annotations.json

annotations_format: |
  {
    "id": "synpic54617",
    "image": "images/chest/synpic54617.jpg",
    "question": "What abnormality is seen in the right lung?",
    "answer": "Pneumothorax",
    "question_type": "what",  # what/where/when/is/are
    "answer_type": "OPEN"     # OPEN or CLOSED
  }
```

### 3.3 约束条件

```python
constraints = {
    "max_flops": 50_000_000,      # 50M FLOPs (宽松，追求高精度)
    "max_params": 100_000_000,    # 100M 参数
    "target_accuracy": 0.80,      # 目标 80% (数据集较难)
    "min_accuracy": 0.60,         # 最低可接受 60%
    "inference_time": 200,        # 200ms (医疗场景可接受)
    "special": "医学影像不容错，需要高置信度预测"
}
```

### 3.4 评估协议

**代理评估**：
- 子集：随机抽取 200 个样本（平衡 question_type）
- 训练：Few-shot (32 shots)
- 轮数：10 epochs
- 指标：
  - Overall Accuracy
  - Open-ended Accuracy
  - Closed-ended Accuracy

**完整评估**：
- 全集：完整的 train/val split
- 训练：标准训练 + 数据增强
- 轮数：50 epochs (医学数据通常需要更充分训练)
- 额外指标：
  - 按身体部位的准确率 (Brain, Chest, Abdomen, etc.)
  - 按问题类型的准确率 (What, Where, Is, etc.)
  - 敏感性/特异性 (Sensitivity/Specificity)

### 3.5 预期 LLM 设计挑战

**场景特征带来的挑战**：
1. **细粒度视觉特征**：医学影像需要识别微小病变
2. **领域知识整合**：需要理解医学术语和解剖结构
3. **高精度要求**：误诊代价高，需要高置信度
4. **多模态对齐**：问题通常指向图像特定区域

**期望的架构创新**：
- 空间注意力机制（关注问题相关区域）
- 多尺度特征金字塔（捕捉不同大小的病变）
- 领域自适应投影（医学术语到视觉特征）

---

## 4. 场景 C：极低延迟端侧 (扫地机器人)

### 4.1 任务描述

使用开源扫地机器人传感器数据集，融合低分辨率摄像头视觉信息和一维传感器数据（IMU、激光雷达距离、碰撞传感器等），进行实时场景理解和障碍物检测。这是典型的边缘计算场景，对延迟和功耗有极致要求。

### 4.2 数据规格

```yaml
# 数据接口契约示例
dataset: RoboSense  # 假设的开源机器人数据集
task_type: sensor_fusion_navigation

input_structure:
  visual:
    source: low_res_camera
    backbone: mobilenet-v3-small  # 轻量化 backbone
    shape: [B, 49, 576]  # [batch, 7x7 patches, channels] 低分辨率
    dtype: float32
    note: "224x224 输入，极度压缩"
  sensor:
    source: imu + lidar + bump
    shape: [B, 1, 64]  # 一维传感器序列
    dtype: float32
    note: "非文本模态，纯数值传感器数据"
  label:
    shape: [B]
    num_classes: 5  # [free, obstacle, cliff, stuck, docking]

# 物理文件夹结构
folder_structure: |
  /data/robo_sense/
  ├── images/           # 低分辨率摄像头帧
  │   ├── frame_001.jpg
  │   └── ...
  ├── sensor_data/      # JSON 格式的传感器读数
  │   ├── frame_001_sensor.json
  │   └── ...
  └── annotations.json

annotations_format: |
  {
    "id": "frame_001",
    "image": "images/frame_001.jpg",
    "sensor": {
      "imu_accel": [x, y, z],
      "imu_gyro": [x, y, z],
      "lidar_distances": [d1, d2, ..., d16],
      "bump_sensors": [b1, b2, b3, b4],
      "battery_level": 0.85
    },
    "label": "obstacle"
  }
```

### 4.3 约束条件

```python
constraints = {
    "max_flops": 2_000_000,        # 2M FLOPs (极严格)
    "max_params": 1_000_000,       # 1M 参数
    "target_accuracy": 0.90,       # 目标 90% (任务相对简单)
    "min_accuracy": 0.80,          # 最低可接受 80%
    "inference_time": 10,          # 10ms on edge device
    "power_budget": 100,           # 100mW
    "special": "实时性要求极高，延迟敏感"
}
```

### 4.4 评估协议

**代理评估**：
- 子集：随机抽取 512 个连续帧
- 训练：Few-shot (64 shots per class)
- 轮数：5 epochs
- 指标：
  - Frame-wise Accuracy
  - F1-score (处理类别不平衡)

**完整评估**：
- 全集：完整序列数据
- 训练：Sequence-aware training (考虑时序连续性)
- 轮数：30 epochs
- 部署测试：
  - 实际 Jetson Nano / Raspberry Pi 推理延迟
  - 功耗测量
  - 连续运行稳定性测试

### 4.5 预期 LLM 设计挑战

**场景特征带来的挑战**：
1. **极端算力限制**：2M FLOPs 限制下完成有效融合
2. **异构模态**：视觉 (2D patch) + 传感器 (1D sequence) 完全不同的表示
3. **实时性**：需要极简架构，无复杂计算
4. **鲁棒性**：传感器噪声和视觉遮挡

**期望的架构创新**：
- 超轻量级融合 (如简单的加权求和或门控)
- 早期融合策略 (减少计算量)
- 知识蒸馏友好的结构

---

## 5. 基线对比 (Baselines)

### 5.1 下界基线 (Lower Bound)

**Simple Concat + MLP**：
```python
class BaselineConcatMLP(nn.Module):
    """证明 LLM 不仅做了拼接，还做了深度设计"""

    def forward(self, visual, text):
        # 全局平均池化
        v = visual.mean(dim=1)  # [B, 1024]
        t = text.mean(dim=1)    # [B, 768]

        # 拼接
        fused = torch.cat([v, t], dim=-1)  # [B, 1792]

        # 简单 MLP
        return self.mlp(fused)  # [B, num_classes]
```

**目的**：如果 AutoFusion 2.0 无法超越此基线，说明搜索失败。

### 5.2 上界基线 (Upper Bound)

| 场景 | 基线方法 | 描述 |
|-----|---------|------|
| A (MMMU) | **FiLM** | 人类设计的特征级调制，46% accuracy |
| B (VQA-RAD) | **MMBERT** | 医学多模态 BERT，领域 SOTA |
| C (Edge) | **MobileNet-SE** | 轻量化注意力，人工优化版本 |

**目的**：证明 AutoFusion 2.0 能达到或接近人类专家水平。

### 5.3 消融基线 (Ablation)

**消融 1：无内环 (No Inner Loop)**
- 直接让 LLM 生成代码并执行，不经过自愈编译
- 预期：编译成功率 < 10%，系统无法运转

**消融 2：无历史反馈 (No History)**
- 每轮独立生成，不利用历史 Reward 信息
- 预期：收敛极慢，效果接近随机搜索

**消融 3：固定随机种子 (Fixed Seed)**
- 证明系统不是依赖运气
- 预期：多次运行结果稳定

---

## 6. 评估指标

### 6.1 主要指标

| 指标 | 符号 | 说明 | 目标 |
|-----|------|------|------|
| 任务准确率 | Acc | 任务特定准确率 | 接近或超越人类基线 |
| 编译成功率 | CSR | Compile Success Rate | **100%** |
| 内环迭代次数 | ILI | 平均自愈尝试次数 | < 3 |
| 搜索收敛轮数 | SC | 达到最佳 Reward 的轮数 | < 150 |
| 计算效率 | FLOPs | 前向传播计算量 | 满足约束 |
| 模型大小 | Params | 可训练参数量 | 满足约束 |

### 6.2 次要指标

| 指标 | 说明 |
|-----|------|
| 架构多样性 | 不同迭代生成的架构类型分布 |
| 代码复杂度 | 生成代码的行数和循环嵌套深度 |
| API 调用次数 | LLM API 调用总次数（成本） |
| 总搜索时间 | 从数据输入到最优架构的时间 |

### 6.3 跨场景泛化指标

```python
# 泛化能力评分
def generalization_score(results_a, results_b, results_c):
    """
    衡量系统在不同场景下的一致表现
    """
    scores = []

    for result in [results_a, results_b, results_c]:
        # 归一化每个场景的表现
        normalized_acc = result['accuracy'] / result['baseline_human']
        normalized_efficiency = result['target_flops'] / result['actual_flops']

        scene_score = 0.6 * normalized_acc + 0.4 * normalized_efficiency
        scores.append(scene_score)

    # 跨场景一致性
    consistency = 1 - np.std(scores)

    # 平均表现
    avg_performance = np.mean(scores)

    return {
        'consistency': consistency,
        'avg_performance': avg_performance,
        'generalization_score': avg_performance * consistency
    }
```

---

## 7. 实验流程

### 7.1 准备阶段 (Week 0)

```bash
# 1. 数据准备
mkdir -p autofusion2/data/{mmmu,vqa_rad,robo_sense}
# 下载并放置数据集到对应目录

# 2. 环境配置
pip install -r autofusion2/requirements.txt

# 3. API 配置
export ALIYUN_API_KEY="your-api-key"
```

### 7.2 执行阶段 (Week 1-4)

```bash
# 场景 A: MMMU
python autofusion2/src/main.py \
    --data_dir ./data/mmmu \
    --scenario high_dim_reasoning \
    --max_iterations 200 \
    --output_dir ./results/scenario_a

# 场景 B: VQA-RAD
python autofusion2/src/main.py \
    --data_dir ./data/vqa_rad \
    --scenario medical_vqa \
    --max_iterations 200 \
    --output_dir ./results/scenario_b

# 场景 C: Edge
python autofusion2/src/main.py \
    --data_dir ./data/robo_sense \
    --scenario edge_robotics \
    --max_iterations 200 \
    --output_dir ./results/scenario_c
```

### 7.3 分析阶段 (Week 5)

```bash
# 生成对比报告
python autofusion2/scripts/analyze_results.py \
    --results ./results/scenario_{a,b,c} \
    --output ./results/final_report.html

# 生成可视化图表
python autofusion2/scripts/visualize.py \
    --results ./results \
    --output ./results/figures/
```

---

## 8. 成功标准

### 8.1 必达目标 (Must Have)

| 标准 | 阈值 | 验证方式 |
|-----|------|---------|
| 编译成功率 | ≥ 95% | 统计所有迭代的编译结果 |
| 场景 A 准确率 | ≥ 40% | 超越 Concat+MLP (约 35%) |
| 场景 B 准确率 | ≥ 70% | 超越随机基线 |
| 场景 C 延迟 | ≤ 20ms | 实际设备测试 |
| 跨场景运行 | 是 | 同一套代码，仅更换数据文件夹 |

### 8.2 期望目标 (Nice to Have)

| 标准 | 阈值 |
|-----|------|
| 场景 A 准确率 | ≥ 45% (接近 FiLM 46%) |
| 场景 B 准确率 | ≥ 75% |
| 场景 C 延迟 | ≤ 10ms |
| 发现新架构范式 | 是 (人类未预定义的结构) |
| 搜索效率 | < 100 轮收敛 |

### 8.3 超预期目标 (Exceptional)

| 标准 | 阈值 |
|-----|------|
| 场景 A 准确率 | > 46% (超越 FiLM) |
| 发表顶会论文 | NeurIPS/ICML/CVPR |
| 开源社区采用 | > 100 stars |

---

## 9. 风险预案

| 风险 | 可能性 | 应对方案 |
|-----|-------|---------|
| 内环无法收敛 | 中 | 增加示例代码 (few-shot)，简化初始 Prompt |
| 外环收敛到局部最优 | 高 | 引入多样性奖励 (diversity bonus)，定期重置 |
| API 成本过高 | 中 | 使用缓存机制，批量评估，限制迭代次数 |
| 数据集无法获取 | 低 | 使用替代数据集 (ScienceQA 替代 MMMU) |
| 边缘设备无法部署 | 中 | 使用模拟器评估，或改用稍强的设备 (Jetson TX2) |

---

## 10. 交付物清单

### 10.1 代码
- [ ] `src/adapter/` - 动态数据适配器
- [ ] `src/controller/` - 双层闭环控制器
- [ ] `src/sandbox/` - 安全沙盒环境
- [ ] `src/evaluator/` - 代理评估器
- [ ] `src/utils/` - 工具函数

### 10.2 文档
- [ ] `docs/DESIGN.md` - 系统设计文档 (已完成)
- [ ] `docs/EXPERIMENTS.md` - 实验规范 (本文档)
- [ ] `docs/API.md` - 接口文档
- [ ] `docs/RESULTS.md` - 实验结果报告

### 10.3 结果
- [ ] `results/scenario_a/` - MMMU 搜索结果
- [ ] `results/scenario_b/` - VQA-RAD 搜索结果
- [ ] `results/scenario_c/` - Edge 搜索结果
- [ ] `results/best_architectures/` - 最优架构代码
- [ ] `results/comparison/` - 与基线对比分析

### 10.4 演示
- [ ] Demo Notebook - 端到端演示
- [ ] 视频演示 - 搜索过程可视化
- [ ] 部署包 - 可直接运行的最优模型

---

*Document Version: 1.0*
*Last Updated: 2026-03-03*
*Status: Ready for Implementation*
