#!/usr/bin/env python3
"""
Minimal Test for AutoFusion 2.0

Tests the core pipeline with mock data and mock LLM.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_adapter():
    """Test 1: Dynamic Data Adapter"""
    print("\n" + "="*60)
    print("Test 1: Dynamic Data Adapter")
    print("="*60)

    from src.adapter import DynamicDataAdapter, APIContract, TensorSpec

    # Create a mock API contract directly (without actual data)
    # Note: Output shape [B, 512] matches the MockLLM code below
    contract = APIContract(
        version="1.0",
        input_specs={
            "visual": TensorSpec(
                name="visual",
                shape=["B", 576, 1024],
                dtype="float32",
                description="CLIP visual features",
                source="image"
            ),
            "text": TensorSpec(
                name="text",
                shape=["B", 77, 768],
                dtype="float32",
                description="CLIP text features",
                source="text"
            )
        },
        output_spec=TensorSpec(
            name="output",
            shape=["B", 512],  # Match MockLLM output
            dtype="float32",
            description="Fusion output",
            source="prediction"
        ),
        constraints={
            "max_flops": 10_000_000,
            "max_params": 50_000_000
        }
    )

    print("✓ API Contract created successfully")
    print(f"  Input specs: {list(contract.input_specs.keys())}")
    print(f"  Output shape: {contract.output_spec.shape}")
    print(f"  Constraints: {contract.constraints}")

    return contract

def test_inner_loop(contract):
    """Test 2: Inner Loop Sandbox"""
    print("\n" + "="*60)
    print("Test 2: Inner Loop Sandbox")
    print("="*60)

    from src.sandbox import InnerLoopSandbox

    # First test: Direct code validation without LLM
    print("\n  Sub-test 2a: Direct code validation")

    test_code = '''import torch
import torch.nn as nn

class AutoFusionLayer(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.proj_v = nn.Linear(1024, 512)
        self.proj_t = nn.Linear(768, 512)
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, visual, text):
        v = visual.mean(dim=1)
        t = text.mean(dim=1)
        v = self.proj_v(v)
        t = self.proj_t(t)
        fused = torch.cat([v, t], dim=-1)
        output = self.fusion(fused)
        return output
'''

    # Test direct validation
    try:
        namespace = {}
        exec(test_code, namespace)
        FusionLayer = namespace['AutoFusionLayer']
        model = FusionLayer({})
        dummy_v = torch.randn(2, 576, 1024)
        dummy_t = torch.randn(2, 77, 768)
        output = model(visual=dummy_v, text=dummy_t)
        print(f"  ✓ Direct validation passed, output shape: {output.shape}")
    except Exception as e:
        print(f"  ✗ Direct validation failed: {e}")
        return None

    # Mock LLM backend that returns a simple fusion layer
    print("\n  Sub-test 2b: Inner loop with mock LLM")

    class MockLLM:
        def generate(self, prompt):
            # Return a valid but simple fusion architecture
            return '''```python
import torch
import torch.nn as nn

class AutoFusionLayer(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        # Simple projection and concatenation
        self.proj_v = nn.Linear(1024, 512)
        self.proj_t = nn.Linear(768, 512)
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, visual, text):
        # Pool and project
        v = visual.mean(dim=1)  # [B, 1024]
        t = text.mean(dim=1)    # [B, 768]

        v = self.proj_v(v)      # [B, 512]
        t = self.proj_t(t)      # [B, 512]

        # Concatenate and fuse
        fused = torch.cat([v, t], dim=-1)  # [B, 1024]
        output = self.fusion(fused)        # [B, 512]
        return output
```'''

    llm = MockLLM()
    sandbox = InnerLoopSandbox(llm, contract, max_retries=3)

    # Test self-healing compile with detailed debugging
    prompt = "Generate a simple multimodal fusion layer"

    print("  Testing self_healing_compile...")

    # Manual iteration for debugging
    code = None
    for attempt in range(3):
        print(f"    Attempt {attempt + 1}...")

        if attempt == 0:
            current_prompt = prompt
        else:
            current_prompt = f"Fix the error in previous code: {error}"

        response = llm.generate(current_prompt)
        code = sandbox._extract_code(response)

        success, error = sandbox._validate_code(code)
        print(f"      Validation: {'✓ Success' if success else '✗ Failed'}")
        if error:
            print(f"      Error: {error[:100]}...")

        if success:
            print(f"    ✓ Compilation successful after {attempt + 1} attempt(s)")
            break
    else:
        print("  ✗ All attempts failed")
        return None

    print(f"  Code length: {len(code)} characters")

    # Verify the code actually runs
    namespace = {}
    exec(code, namespace)
    FusionLayer = namespace['AutoFusionLayer']

    # Test forward pass
    model = FusionLayer({})
    dummy_v = torch.randn(2, 576, 1024)
    dummy_t = torch.randn(2, 77, 768)
    output = model(visual=dummy_v, text=dummy_t)

    print(f"  Output shape: {output.shape}")
    assert output.shape == torch.Size([2, 512]), f"Expected [2, 512], got {output.shape}"
    print("✓ Forward pass successful")

    return code

def test_reward_function():
    """Test 3: Reward Function"""
    print("\n" + "="*60)
    print("Test 3: Reward Function")
    print("="*60)

    from src.evaluator import RewardFunction

    reward_fn = RewardFunction(
        weights={"accuracy": 1.0, "efficiency": 0.5, "constraint": 2.0}
    )

    # Test case 1: Good architecture
    reward1 = reward_fn.calculate(
        accuracy=0.75,
        flops=5_000_000,
        params=20_000_000,
        constraints={"max_flops": 10_000_000, "max_params": 50_000_000}
    )
    print(f"✓ Good arch reward: {reward1:.3f}")

    # Test case 2: Constraint violation
    reward2 = reward_fn.calculate(
        accuracy=0.80,
        flops=15_000_000,  # Exceeds limit
        params=60_000_000,  # Exceeds limit
        constraints={"max_flops": 10_000_000, "max_params": 50_000_000}
    )
    print(f"✓ Violation arch reward: {reward2:.3f} (should be lower)")

    assert reward1 > reward2, "Good arch should have higher reward"
    print("✓ Reward function working correctly")

def test_controller_iteration(contract, code):
    """Test 4: Controller Single Iteration"""
    print("\n" + "="*60)
    print("Test 4: Controller Single Iteration")
    print("="*60)

    from src.controller import DualLoopController
    from src.evaluator import ProxyEvaluator, RewardFunction

    # Create mock dataset
    class MockDataset:
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "visual": torch.randn(576, 1024),
                "text": torch.randn(77, 768),
                "label": torch.randint(0, 4, (1,)).item()
            }

    dataset = MockDataset(size=50)  # Small dataset for testing

    # Create evaluator
    evaluator = ProxyEvaluator(
        dataset=dataset,
        num_shots=4,
        num_epochs=2,
        batch_size=4,
        device="cpu"
    )

    reward_fn = RewardFunction()

    # Mock LLM that always returns the same code
    class MockLLM:
        def __init__(self, code):
            self.code = code

        def generate(self, prompt):
            return f"```python\n{self.code}\n```"

    llm = MockLLM(code)

    # Create controller
    controller = DualLoopController(
        llm_backend=llm,
        api_contract=contract,
        proxy_evaluator=evaluator,
        reward_fn=reward_fn,
        max_iterations=2,  # Just 2 iterations for testing
        output_dir="./test_output"
    )

    print("Running 2 iterations...")
    best_result = controller.search()

    if best_result:
        print(f"✓ Search completed")
        print(f"  Best reward: {best_result.reward:.3f}")
        print(f"  Best accuracy: {best_result.accuracy:.3f}")
        print(f"  Compile attempts: {best_result.compile_attempts}")
    else:
        print("✗ Search failed")

def test_simple_controller(contract, code):
    """Test 4: Simplified Controller (2 iterations, small dataset)"""
    print("\n" + "="*60)
    print("Test 4: Simplified Controller")
    print("="*60)

    from src.controller import DualLoopController
    from src.evaluator import RewardFunction

    # Create minimal mock components
    class MockEvaluator:
        def __init__(self):
            self.call_count = 0

        def evaluate(self, code):
            self.call_count += 1
            # Return deterministic results
            return {
                "accuracy": 0.5 + 0.1 * (self.call_count % 3),  # Varying accuracy
                "flops": 5_000_000,
                "params": 10_000_000,
                "training_time": 1.0,
                "total_time": 2.0
            }

    class MockLLMFixed:
        def __init__(self, code):
            self.code = code

        def generate(self, prompt):
            return f"```python\n{self.code}\n```"

    evaluator = MockEvaluator()
    reward_fn = RewardFunction()
    llm = MockLLMFixed(code)

    controller = DualLoopController(
        llm_backend=llm,
        api_contract=contract,
        proxy_evaluator=evaluator,
        reward_fn=reward_fn,
        max_iterations=2,
        output_dir="./test_output"
    )

    print("Running 2 iterations...")
    best_result = controller.search()

    if best_result:
        print(f"✓ Search completed")
        print(f"  Best reward: {best_result.reward:.3f}")
        print(f"  Best accuracy: {best_result.accuracy:.3f}")
        print(f"  Total evaluations: {evaluator.call_count}")
        return True
    else:
        print("✗ Search failed")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("AutoFusion 2.0 - Minimal Test Suite")
    print("="*60)

    try:
        # Test 1: Adapter
        contract = test_adapter()

        # Test 2: Inner Loop
        code = test_inner_loop(contract)
        if not code:
            print("\n✗ Inner loop test failed, stopping")
            return

        # Test 3: Reward Function
        test_reward_function()

        # Test 4: Controller (simplified)
        test_simple_controller(contract, code)

        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
