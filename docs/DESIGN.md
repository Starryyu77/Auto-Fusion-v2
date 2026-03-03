# AutoFusion 2.0: 系统架构设计文档

## 1. 概述 (Overview)

AutoFusion 2.0 是一种**完全无搜索空间（Search Space-Free）**的通用多模态神经架构搜索范式。通过双层闭环反馈机制，系统能够自动从原始数据文件夹生成最优的融合架构，无需人工预定义算子库或模板。

### 1.1 核心创新

| 创新点 | 描述 | 优势 |
|--------|------|------|
| **动态数据适配** | 自动嗅探数据维度，生成接口契约 | 跨任务一键泛化 |
| **内环自愈编译** | 基于代码报错的自动维度对齐 | 100% 编译成功率 |
| **外环拓扑演化** | 基于真实任务 Reward 的架构进化 | 突破人类设计局限 |
| **原生代码生成** | LLM 直接输出 PyTorch 类代码 | 无模板约束 |

### 1.2 系统边界

```
┌─────────────────────────────────────────────────────────────┐
│                        AutoFusion 2.0                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Input:      │  │  Core:       │  │  Output:     │        │
│  │  Raw Data    │→ │  Dual-Loop   │→ │  Best Arch   │        │
│  │  Folder      │  │  Controller  │  │  Code        │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│         ↑                                    ↓               │
│         └────────── Feedback Loop ←──────────┘               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 系统架构 (System Architecture)

### 2.1 三大流水线阶段

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AutoFusion 2.0 Pipeline                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Stage 1: Data Ingestion & Contract Generation                          │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ Raw Folder  │───→│ Data Adapter │───→│ API Contract │               │
│  │ (img+json)  │    │ (Extractor)  │    │ (Shape Spec) │               │
│  └─────────────┘    └──────────────┘    └──────────────┘               │
│                              ↓                                          │
│  Stage 2: Dual-Loop Search Engine                                       │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                    RL Controller                             │      │
│  │  ┌────────────────────────────────────────────────────────┐  │      │
│  │  │  【Inner Loop】Auto-Debugging                          │  │      │
│  │  │  LLM → Raw Code → Sandbox Dry-run → Error Feedback     │  │      │
│  │  │  (Compile Success Guarantee)                           │  │      │
│  │  └────────────────────────────────────────────────────────┘  │      │
│  │                           ↓ Success                          │      │
│  │  ┌────────────────────────────────────────────────────────┐  │      │
│  │  │  【Outer Loop】Performance Evolution                   │  │      │
│  │  │  Proxy Training → Metric Collection → Reward Update    │  │      │
│  │  │  (Architecture Topology Evolution)                     │  │      │
│  │  └────────────────────────────────────────────────────────┘  │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                              ↓                                          │
│  Stage 3: Full Training & Export                                        │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ Best Code   │───→│ Full Dataset │───→│ Final Model  │               │
│  │ (Top-1)     │    │ Training     │    │ Export       │               │
│  └─────────────┘    └──────────────┘    └──────────────┘               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心组件详细设计

### 3.1 动态数据适配器 (Dynamic Data Adapter)

#### 职责
- 解耦物理文件与底层张量
- 自动嗅探数据维度
- 生成标准化的 API 接口契约

#### 工作流程

```python
class DynamicDataAdapter:
    """
    动态数据适配器

    Input: /data/my_dataset/
           ├── images/
           ├── videos/ (optional)
           └── annotations.json

    Output: {
        "batch_dict": {"visual": tensor, "text": tensor, "label": tensor},
        "contract": {
            "visual": {"shape": [B, 576, 1024], "dtype": "float32"},
            "text": {"shape": [B, 77, 768], "dtype": "float32"},
            "label": {"shape": [B], "dtype": "int64"}
        }
    }
    """
```

#### 接口契约格式

```json
{
  "api_contract": {
    "version": "1.0",
    "input_specs": {
      "visual": {
        "description": "Visual features from frozen CLIP-ViT",
        "shape": ["B", 576, 1024],
        "dtype": "torch.float32",
        "source": "image"
      },
      "text": {
        "description": "Text features from frozen CLIP-Text",
        "shape": ["B", 77, 768],
        "dtype": "torch.float32",
        "source": "text"
      }
    },
    "output_spec": {
      "description": "Classification logits",
      "shape": ["B", "num_classes"],
      "dtype": "torch.float32"
    },
    "constraints": {
      "max_flops": 10000000,
      "max_params": 50000000
    }
  }
}
```

---

### 3.2 双层闭环控制器 (Dual-Loop Controller)

#### 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                   DualLoopController                        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Inner Loop (内环)                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │  LLM     │→ │  Code    │→ │  Sandbox │          │   │
│  │  │ Generate │  │  Exec    │  │  Dry-run │          │   │
│  │  └──────────┘  └──────────┘  └────┬─────┘          │   │
│  │                                    ↓                │   │
│  │                              ┌──────────┐          │   │
│  │                              │  Error?  │──Yes────→│   │
│  │                              └────┬─────┘          │   │
│  │                                   No               │   │
│  └───────────────────────────────────┼────────────────┘   │
│                                      ↓                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Outer Loop (外环)                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │  Proxy   │→ │  Metric  │→ │  Reward  │          │   │
│  │  │  Train   │  │  Collect │  │  Update  │          │   │
│  │  └──────────┘  └──────────┘  └────┬─────┘          │   │
│  │                                    ↓                │   │
│  │                              ┌──────────┐          │   │
│  │                              │  Better? │──Yes────→│   │
│  │                              └──────────┘          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 内环：沙盒自愈编译 (Inner Loop)

**目标**：确保 LLM 生成的代码能够无报错运行，并避免重复尝试相同的错误

**核心问题**：原始设计中，LLM 可能在 5 次重试中反复横跳（如 `permute()` → `transpose()` → `permute()`），因为没有历史记忆。

**解决方案**：引入**尝试历史记录 (Attempt History)**，每次失败都记录代码和错误，让 LLM 具备"历史记忆"。

**流程**：

```python
class InnerLoopSandbox:
    """
    内环沙盒自愈编译器（带历史记忆）

    1. LLM 生成原生 PyTorch 代码
    2. 在内存沙盒中 exec() 执行
    3. 灌入真实张量进行前向传播干跑
    4. 捕获 RuntimeError/SyntaxError
    5. 【关键】记录本次尝试 (代码 + 错误) 到历史
    6. 【关键】将所有历史反馈给 LLM，提示"DO NOT REPEAT"
    7. LLM 基于历史生成新代码，避免重复错误
    8. 重复直到成功
    """

    def self_healing_compile(self, initial_prompt: str) -> Tuple[str, int]:
        """
        自愈编译（带历史记忆）
        """
        self.attempt_history = []  # 重置历史

        for attempt in range(self.max_retries):
            # 生成代码（首次）或修复（重试）
            if attempt == 0:
                prompt = initial_prompt
            else:
                # 【关键】使用历史构造反馈 Prompt
                prompt = self._construct_error_prompt_with_history()

            code = self.llm.generate(prompt)

            # 尝试编译
            success, error = self._validate_code(code)

            if success:
                return code, attempt + 1

            # 【关键】记录失败到历史
            self.attempt_history.append(AttemptRecord(
                attempt_number=attempt + 1,
                code=code,
                error=error
            ))

        raise CompilationError(f"Failed after {self.max_retries} attempts")
```

**增强版错误反馈 Prompt 模板（带历史记忆）**：

```
【Attempt 3/5】
Your previous code generation attempts failed. Please analyze the history and try a different approach.

【Previous Attempts - DO NOT REPEAT THESE】

Attempt 1:
Code snippet: class AutoFusionLayer(nn.Module): def __init__...
Error: RuntimeError: permute(): invalid number of dimensions

Attempt 2:
Code snippet: class AutoFusionLayer(nn.Module): def __init__...
Error: RuntimeError: transpose(): invalid dimensions

【Most Recent Failure - Fix This】
Attempt 2 Code:
```python
{latest_code}
```

【Error Message】
```
{latest_error}
```

【Specific Guidance】
- Tensor dimension reordering issue. Try:
  * Use .reshape() or .view() instead of permute/transpose
  * Check the actual tensor shape with .shape before operations

【Fix Requirements】
1. Analyze the error history above - do NOT repeat previous failed approaches
2. If you tried permute() and it failed, try reshape() or adaptive pooling
3. Ensure all tensor dimensions match the contract
4. Keep the class name as 'AutoFusionLayer'
5. Only return the fixed code, no explanation

Provide the corrected code with a NEW approach:
```

**关键改进**：
1. **历史记录**：`attempt_history` 列表保存所有失败尝试
2. **去重提示**：Prompt 中明确标注 "DO NOT REPEAT THESE"
3. **针对性建议**：根据错误类型（如 `permute` vs `shape mismatch`）给出不同修复建议
4. **打破循环**：LLM 明确知道哪些路已经走过不通，被迫寻找新思路

#### 外环：性能演化 (Outer Loop)

**目标**：基于真实任务性能优化架构拓扑

**流程**：

```python
class OuterLoopEvolution:
    """
    外环性能演化器

    1. 接收内环编译成功的代码
    2. 在代理数据集上进行快速训练
    3. 收集 Accuracy, FLOPs, Params
    4. 计算 Reward
    5. 将性能反馈给 LLM 指导架构改进
    6. 迭代直到收敛
    """

    def evaluate_and_evolve(self, code: str, iteration: int) -> dict:
        """
        评估与演化

        Returns:
            {
                "reward": float,
                "accuracy": float,
                "flops": int,
                "params": int,
                "feedback": str  # 给 LLM 的自然语言反馈
            }
        """
        # 1. 快速训练
        model = self._instantiate_from_code(code)
        metrics = self.proxy_trainer.train(model, num_epochs=5)

        # 2. 计算效率指标
        flops, params = self._profile_model(model)

        # 3. 计算综合 Reward
        reward = self._calculate_reward(
            accuracy=metrics['accuracy'],
            flops=flops,
            params=params,
            constraints=self.constraints
        )

        # 4. 生成自然语言反馈
        feedback = self._generate_feedback(metrics, flops, params, iteration)

        return {
            "reward": reward,
            "accuracy": metrics['accuracy'],
            "flops": flops,
            "params": params,
            "feedback": feedback
        }
```

**Reward 函数设计**：

```python
def calculate_reward(self, accuracy, flops, params, constraints):
    """
    多目标奖励函数

    R = w1 * accuracy - w2 * efficiency_penalty - w3 * constraint_violation
    """
    # 准确率奖励 (0-1)
    acc_reward = accuracy

    # 效率奖励 (鼓励低 FLOPs)
    target_flops = constraints.get('max_flops', 10e6)
    flops_ratio = flops / target_flops
    efficiency_reward = max(0, 1 - flops_ratio)

    # 约束惩罚
    violation_penalty = 0
    if flops > target_flops:
        violation_penalty += (flops - target_flops) / target_flops
    if params > constraints.get('max_params', 50e6):
        violation_penalty += (params - constraints['max_params']) / constraints['max_params']

    # 综合奖励
    reward = (
        self.weights['accuracy'] * acc_reward +
        self.weights['efficiency'] * efficiency_reward -
        self.weights['constraint'] * violation_penalty
    )

    return reward
```

**性能反馈 Prompt 模板**：

```
当前架构性能评估结果：

【架构代码】
{current_code}

【性能指标】
- 准确率: {accuracy:.2%}
- 计算量: {flops:.2f}M FLOPs (限制: {max_flops:.2f}M)
- 参数量: {params:.2f}M (限制: {max_params:.2f}M)
- 综合奖励: {reward:.3f}

【历史最佳】
- 最佳奖励: {best_reward:.3f}
- 最佳架构: {best_arch_summary}

【优化建议】
{natural_language_feedback}

请生成改进后的架构代码。建议考虑：
1. 如果准确率低：增加模型容量或改进特征交互方式
2. 如果 FLOPs 超标：减少层数或通道数，使用更轻量的操作
3. 如果过拟合：添加 Dropout 或正则化
```

---

### 3.3 沙盒执行环境 (Sandbox Environment)

#### 安全隔离与显存保护

**核心风险**：在 200 轮搜索循环中，LLM 生成的代码可能引发：
- **显存泄漏**：在 `forward()` 中疯狂创建不释放的张量
- **OOM 崩溃**：显存溢出导致整个主进程崩溃
- **僵尸进程**：子进程未彻底清理，残留显存占用

**解决方案**：**三层显存防护机制**

```python
class SecureSandbox:
    """
    安全沙盒执行环境（带显存保护）

    - 限制可用模块
    - 禁止文件系统访问
    - 超时控制
    - 【关键】显存隔离与强制回收
    """

    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 1024,
        max_cpu_time: int = 60,
        max_vram_mb: int = 2048  # 单进程显存限制
    ):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.max_cpu_time = max_cpu_time
        self.max_vram_mb = max_vram_mb

    def execute(self, code: str, inputs: dict) -> Tuple[bool, Any]:
        """
        在隔离进程中执行代码，保证显存完全释放
        """
        # ========== 第一层：执行前清理 ==========
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # 使用 spawn 模式创建全新进程（避免 fork 的内存继承）
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()

        process = ctx.Process(
            target=self._execute_in_process,
            args=(code, inputs, queue),
            daemon=True  # 父进程死亡时自动 kill 子进程
        )

        try:
            process.start()
            process.join(timeout=self.timeout)

            if process.is_alive():
                # 超时：强制 kill
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()  # 强制终止
                    process.join()
                return False, "Execution timeout"

            # 获取结果
            success, result = queue.get_nowait()
            return success, result

        finally:
            # ========== 第二层：执行后强制清理 ==========
            # 【关键】确保子进程已死
            if process.is_alive():
                process.kill()
                process.join()

            # 【关键】强制回收显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    def _execute_in_process(self, code, inputs, queue):
        """
        在子进程中执行（完全隔离的显存上下文）
        """
        try:
            # 设置显存限制（CUDA）
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(
                    self.max_vram_mb / total_vram
                )
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 执行模型...
            model = AutoFusionLayer(...)
            output = model(**inputs)

            queue.put((True, output))

        except Exception as e:
            queue.put((False, str(e)))

        finally:
            # ========== 第三层：子进程退出前清理 ==========
            # 【关键】子进程必须在退出前释放所有显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
```

**三层防护机制详解**：

| 层级 | 触发时机 | 操作 | 目的 |
|-----|---------|------|------|
| **第一层** | 执行前 | `torch.cuda.empty_cache()` + `gc.collect()` | 清理上一轮残留，为新执行腾出空间 |
| **第二层** | 执行后 (finally) | `process.kill()` + `empty_cache()` | 确保子进程死亡，回收其占用的显存 |
| **第三层** | 子进程退出前 | `torch.cuda.empty_cache()` + `synchronize()` | 子进程自我清理，避免僵尸占用 |

**为什么使用 `spawn` 而非 `fork`**：
- `fork`：子进程继承父进程的显存上下文，可能导致 CUDA 上下文混乱
- `spawn`：全新进程，干净的 CUDA 上下文，真正的显存隔离

**OOM 防护**：
```python
# 在子进程中设置显存限制
torch.cuda.set_per_process_memory_fraction(0.3)  # 最多使用 30% 显存

# 捕获 OOM 错误
try:
    output = model(**inputs)
except RuntimeError as e:
    if "out of memory" in str(e):
        # 强制清理并返回错误
        torch.cuda.empty_cache()
        return False, "GPU OOM"
```

---

## 4. 工作流程时序图

```
┌─────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  User   │  │   Data      │  │    LLM      │  │   Inner     │  │   Outer     │
│         │  │  Adapter    │  │  Controller │  │   Loop      │  │   Loop      │
└────┬────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
     │              │                │                │                │
     │ 1. Provide   │                │                │                │
     │    Folder    │                │                │                │
     │─────────────>│                │                │                │
     │              │                │                │                │
     │              │ 2. Extract     │                │                │
     │              │    Features    │                │                │
     │              │────┐           │                │                │
     │              │    │           │                │                │
     │              │<───┘           │                │                │
     │              │                │                │                │
     │              │ 3. Sniff Shape │                │                │
     │              │────┐           │                │                │
     │              │    │           │                │                │
     │              │<───┘           │                │                │
     │              │                │                │                │
     │              │ 4. Generate    │                │                │
     │              │    Contract    │                │                │
     │              │───────────────────────────────>│                │
     │              │                │                │                │
     │              │                │ 5. Generate    │                │
     │              │                │    Raw Code    │                │
     │              │                │───────────────────────────────>│
     │              │                │                │                │
     │              │                │                │ 6. Sandbox     │
     │              │                │                │    Dry-run     │
     │              │                │                │────┐           │
     │              │                │                │    │           │
     │              │                │                │<───┘           │
     │              │                │                │                │
     │              │                │ 7. Error?      │                │
     │              │                │<───────────────────────────────│
     │              │                │                │                │
     │              │                │ 8. Fix Code    │                │
     │              │                │───────────────────────────────>│
     │              │                │                │ (Repeat 6-8)   │
     │              │                │                │    until OK    │
     │              │                │                │                │
     │              │                │                │ 9. Success     │
     │              │                │<───────────────────────────────│
     │              │                │                │                │
     │              │                │                │                │ 10. Proxy
     │              │                │<────────────────────────────────│ Train
     │              │                │                │                │────┐
     │              │                │                │                │    │
     │              │                │                │                │<───┘
     │              │                │                │                │
     │              │                │                │                │ 11. Metrics
     │              │                │<────────────────────────────────│ & Reward
     │              │                │                │                │
     │              │                │ 12. Evolve?    │                │
     │              │                │───────────────────────────────>│
     │              │                │                │                │
     │              │                │ 13. Generate   │                │
     │              │                │    New Code    │                │
     │              │                │───────────────────────────────>│
     │              │                │                │ (Repeat 6-13)  │
     │              │                │                │   100-200 iters│
     │              │                │                │                │
     │              │                │ 14. Best Arch  │                │
     │              │                │                │                │
     │              │                │<────────────────────────────────│
     │              │                │                │                │
     │              │                │ 15. Full Train │                │
     │              │                │────┐           │                │
     │              │                │    │           │                │
     │              │                │<───┘           │                │
     │              │                │                │                │
     │              │ 16. Return     │                │                │
     │              │     Best Model │                │                │
     │<─────────────│                │                │                │
     │              │                │                │                │
```

---

## 5. 关键设计决策

### 5.1 为什么选择双层闭环？

| 单层方案 | 双层方案 (AutoFusion 2.0) |
|---------|-------------------------|
| 编译错误直接浪费一次迭代 | 内环过滤所有语法/维度错误 |
| 编译成功率 10-30% | 编译成功率 100% |
| 奖励信号稀疏 | 外环只接收有效架构的奖励 |
| 搜索效率低 | 搜索效率高 3-5x |

### 5.2 为什么原生代码生成？

| 模板方案 (1.0) | 原生代码 (2.0) |
|---------------|---------------|
| 受限于预定义算子 | 无搜索空间限制 |
| 无法突破人类设计 | 可能发现全新架构范式 |
| 固定代码结构 | 动态拓扑适应任务需求 |
| 编译成功率 100% | 内环保障 100% 编译成功 |

### 5.3 数据接口设计

```python
# 标准化接口 - 解耦数据与模型
class AutoFusionLayer(nn.Module):
    """
    所有生成的架构必须实现此接口
    """
    def __init__(self, input_specs: Dict[str, TensorSpec]):
        """
        Args:
            input_specs: 输入张量规格，如
                {
                    "visual": TensorSpec(shape=[B, 576, 1024]),
                    "text": TensorSpec(shape=[B, 77, 768])
                }
        """
        super().__init__()

    def forward(self, visual: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual: [B, V, D_v] 视觉特征
            text: [B, T, D_t] 文本特征

        Returns:
            [B, num_classes] 分类 logits
        """
        raise NotImplementedError
```

---

## 6. 与 AutoFusion 1.0 的对比

| 特性 | AutoFusion 1.0 | AutoFusion 2.0 |
|------|---------------|----------------|
| **搜索空间** | 预定义 5 种模板 | 完全开放，原生代码 |
| **数据适配** | 硬编码数据加载器 | 动态数据适配器 |
| **编译保障** | 模板保证 100% | 内环自愈保证 100% |
| **架构创新** | 有限（模板内选择） | 无限（LLM 自由设计） |
| **跨任务泛化** | 需重写数据接口 | 更换文件夹即可 |
| **搜索效率** | 中等 | 高（双层过滤） |
| **可解释性** | 高（已知模板） | 中等（需分析代码） |

---

## 7. 风险与缓解策略

| 风险 | 影响 | 缓解策略 |
|-----|------|---------|
| 内环陷入无限修复 | 高 | 设置最大重试次数，超时回退 |
| LLM 生成危险代码 | 中 | 沙盒隔离，白名单模块限制 |
| 外环收敛过慢 | 中 | 代理数据集快速评估，早停机制 |
| API 成本过高 | 中 | 缓存成功代码，批量评估 |
| 架构过于复杂 | 低 | 在 Reward 中加入复杂度惩罚 |

---

## 8. 实现优先级

### Phase 1: 基础设施 (Week 1)
- [ ] DynamicDataAdapter 实现
- [ ] SecureSandbox 框架
- [ ] 基础 LLM Backend 接口

### Phase 2: 内环闭环 (Week 2)
- [ ] InnerLoopSandbox 实现
- [ ] 错误提取与反馈机制
- [ ] 自愈编译验证

### Phase 3: 外环闭环 (Week 3)
- [ ] OuterLoopEvolution 实现
- [ ] Reward 函数调优
- [ ] 代理训练加速

### Phase 4: 系统集成 (Week 4)
- [ ] DualLoopController 整合
- [ ] 端到端流水线测试
- [ ] 三个场景验证

---

*Document Version: 1.0*
*Last Updated: 2026-03-03*
*Status: Design Complete*
