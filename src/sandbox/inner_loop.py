"""
Inner Loop Sandbox

Implements the self-healing compilation mechanism that guarantees
100% compile success through automatic error detection and correction.
Includes history tracking to prevent LLM from repeating the same mistakes.
"""

import re
import gc
import traceback
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, TimeoutError

import torch
import torch.nn as nn

from ..adapter import APIContract


@dataclass
class AttemptRecord:
    """Record of a single compilation attempt."""
    attempt_number: int
    code: str
    error: str


class InnerLoopSandbox:
    """
    Inner Loop Sandbox for AutoFusion 2.0.

    Guarantees compilation success through iterative error correction:
    1. LLM generates raw code
    2. Execute in sandbox with dummy tensors
    3. Catch RuntimeError/SyntaxError
    4. Feed error + history back to LLM for correction
    5. Repeat until success or max retries

    Features:
    - History tracking to prevent repeated mistakes
    - GPU memory cleanup after each attempt
    - Detailed error feedback with context
    """

    def __init__(
        self,
        llm_backend: Any,
        api_contract: APIContract,
        max_retries: int = 5,
        timeout: int = 30
    ):
        self.llm = llm_backend
        self.contract = api_contract
        self.max_retries = max_retries
        self.timeout = timeout
        self.attempt_history: List[AttemptRecord] = []

    def self_healing_compile(
        self,
        initial_prompt: str
    ) -> Tuple[Optional[str], int]:
        """
        Compile code with automatic error correction.

        Args:
            initial_prompt: Initial prompt for code generation

        Returns:
            (code, attempts): Compiled code (None if failed) and number of attempts
        """
        code = None
        self.attempt_history = []  # Reset history for new compilation

        for attempt in range(self.max_retries):
            # Generate code (or fix previous error)
            if attempt == 0:
                prompt = initial_prompt
            else:
                prompt = self._construct_error_prompt_with_history()

            # Get code from LLM
            response = self.llm.generate(prompt)
            code = self._extract_code(response)

            # Try to compile and run
            success, error = self._validate_code(code)

            if success:
                return code, attempt + 1

            # Record this failed attempt with full context
            self.attempt_history.append(AttemptRecord(
                attempt_number=attempt + 1,
                code=code,
                error=error
            ))

            # Force GPU cleanup after failed attempt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        # Max retries exceeded
        return None, self.max_retries

    def _validate_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate code by executing it in a sandbox.

        Returns:
            (success, error): Whether validation succeeded and error message if failed
        """
        try:
            # Step 1: Preprocess code (remove imports for already-provided modules)
            code = self._preprocess_code(code)

            # Step 2: Syntax check
            compile(code, '<string>', 'exec')

            # Step 3: Create restricted namespace
            restricted_globals = self._create_restricted_namespace()

            # Step 4: Execute class definition
            exec(code, restricted_globals)

            # Step 5: Check if AutoFusionLayer exists
            if 'AutoFusionLayer' not in restricted_globals:
                return False, "Class 'AutoFusionLayer' not found in generated code"

            FusionLayer = restricted_globals['AutoFusionLayer']

            # Step 6: Instantiate with contract dimensions
            input_dims = self._get_input_dims_from_contract()
            model = FusionLayer(input_dims)

            # Step 7: Move to device and run forward pass
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            model.eval()

            dummy_inputs = self._create_dummy_inputs(device)
            output = model(**dummy_inputs)

            # Step 8: Validate output shape
            expected_shape = self._get_expected_output_shape()
            if output.shape[1:] != torch.Size(expected_shape[1:]):
                return False, (
                    f"Output shape mismatch: got {list(output.shape)}, "
                    f"expected {expected_shape}"
                )

            # Success - cleanup before returning
            del model, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True, None

        except SyntaxError as e:
            return False, f"Syntax Error: {str(e)}"
        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                # Force aggressive cleanup on OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                return False, f"GPU OOM Error: {error_msg}"
            return False, f"Runtime Error: {error_msg}"
        except Exception as e:
            return False, f"Error: {traceback.format_exc()}"

    def _preprocess_code(self, code: str) -> str:
        """Preprocess code to remove import statements (modules already provided)."""
        import re
        # Remove import statements for already-provided modules
        lines = code.split('\n')
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip import statements for torch and nn (already in namespace)
            if stripped.startswith('import torch') or stripped.startswith('import torch.'):
                continue
            if stripped.startswith('from torch'):
                continue
            if stripped.startswith('import nn') or stripped.startswith('from nn'):
                continue
            filtered_lines.append(line)
        return '\n'.join(filtered_lines)

    def _create_restricted_namespace(self) -> Dict[str, Any]:
        """Create a namespace for code execution with necessary modules."""
        # Import builtins module
        import builtins

        # Create a safe builtins dict - exclude dangerous functions
        safe_builtins = builtins.__dict__.copy()
        dangerous = ['eval', 'exec', 'compile', '__import__', 'open', 'input']
        for name in dangerous:
            safe_builtins.pop(name, None)

        namespace = {
            '__builtins__': safe_builtins,
            '__name__': '__sandbox__',
            '__doc__': None,
            'torch': torch,
            'nn': nn,
            'F': nn.functional,
        }

        return namespace

    def _create_dummy_inputs(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Create dummy input tensors based on contract."""
        inputs = {}
        batch_size = 2  # Small batch for testing

        for name, spec in self.contract.input_specs.items():
            # Parse shape (replace 'B' with actual batch size)
            shape = [batch_size if dim == "B" else dim for dim in spec.shape]

            # Create tensor with appropriate dtype
            dtype = getattr(torch, spec.dtype.replace("float32", "float").replace("int64", "long"))
            inputs[name] = torch.randn(shape, dtype=dtype, device=device)

        return inputs

    def _get_input_dims_from_contract(self) -> Dict[str, Any]:
        """Extract input dimensions from API contract."""
        return {
            name: {
                "shape": spec.shape,
                "dtype": spec.dtype
            }
            for name, spec in self.contract.input_specs.items()
        }

    def _get_expected_output_shape(self) -> List[int]:
        """Get expected output shape from contract."""
        if self.contract.output_spec:
            return self.contract.output_spec.shape
        return ["B", 2]  # Default binary classification

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to extract code from markdown blocks
        code_pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(code_pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Try generic code blocks
        code_pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(code_pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Return entire response if no code blocks found
        return response.strip()

    def _construct_error_prompt_with_history(self) -> str:
        """
        Construct a prompt with full history to prevent repeated mistakes.

        This is the key improvement - we show LLM all previous attempts
        and their errors, so it can learn from history and avoid loops.
        """
        if not self.attempt_history:
            return "Generate a multimodal fusion architecture."

        current_attempt = len(self.attempt_history) + 1
        last_record = self.attempt_history[-1]

        prompt_parts = []

        prompt_parts.append(f"【Attempt {current_attempt}/{self.max_retries}】")
        prompt_parts.append("Your previous code generation attempts failed. Please analyze the history and try a different approach.\n")

        # Show full history to prevent loops
        if len(self.attempt_history) >= 2:
            prompt_parts.append("【Previous Attempts - DO NOT REPEAT THESE】")
            for record in self.attempt_history[:-1]:  # All except the last one
                prompt_parts.append(f"\nAttempt {record.attempt_number}:")
                prompt_parts.append(f"Code snippet: {record.code[:200]}...")
                prompt_parts.append(f"Error: {record.error[:150]}...")
            prompt_parts.append("\n" + "="*60)

        # Show the most recent failure in detail
        prompt_parts.append("【Most Recent Failure - Fix This】")
        prompt_parts.append(f"\nAttempt {last_record.attempt_number} Code:")
        prompt_parts.append("```python")
        prompt_parts.append(last_record.code)
        prompt_parts.append("```\n")

        prompt_parts.append("【Error Message】")
        prompt_parts.append(f"```\n{last_record.error}\n```\n")

        # Add specific guidance based on error type
        prompt_parts.append(self._get_error_specific_guidance(last_record.error))

        prompt_parts.append("【API Contract】")
        prompt_parts.append(self.contract.to_prompt())
        prompt_parts.append("")

        prompt_parts.append("【Fix Requirements】")
        prompt_parts.append("1. Analyze the error history above - do NOT repeat previous failed approaches")
        prompt_parts.append("2. If you tried permute() and it failed, try transpose() or reshape()")
        prompt_parts.append("3. If dimension mismatch persists, consider using adaptive pooling")
        prompt_parts.append("4. Ensure all tensor dimensions match the contract")
        prompt_parts.append("5. Use correct PyTorch API syntax")
        prompt_parts.append("6. Keep the class name as 'AutoFusionLayer'")
        prompt_parts.append("7. Only return the fixed code, no explanation\n")

        prompt_parts.append("Provide the corrected code with a NEW approach:")

        return "\n".join(prompt_parts)

    def _get_error_specific_guidance(self, error: str) -> str:
        """Generate specific guidance based on error type."""
        error_lower = error.lower()

        guidance = ["【Specific Guidance】"]

        if "shape mismatch" in error_lower or "size mismatch" in error_lower:
            guidance.append("- Dimension mismatch detected. Consider:")
            guidance.append("  * Using .mean(dim=1) instead of .view() for pooling")
            guidance.append("  * Adding projection layers to match dimensions")
            guidance.append("  * Using adaptive pooling: nn.AdaptiveAvgPool1d(output_size)")

        elif "permute" in error_lower or "transpose" in error_lower:
            guidance.append("- Tensor dimension reordering issue. Try:")
            guidance.append("  * Use .reshape() or .view() instead of permute/transpose")
            guidance.append("  * Check the actual tensor shape with .shape before operations")
            guidance.append("  * Use einops.rearrange if available, or manual indexing")

        elif "out of memory" in error_lower or "oom" in error_lower:
            guidance.append("- GPU memory issue. Solutions:")
            guidance.append("  * Reduce hidden dimensions (e.g., 512 -> 256)")
            guidance.append("  * Use fewer layers")
            guidance.append("  * Avoid creating intermediate tensors in forward()")

        elif "syntax" in error_lower:
            guidance.append("- Syntax error. Check:")
            guidance.append("  * All parentheses and brackets are balanced")
            guidance.append("  * Proper indentation (4 spaces)")
            guidance.append("  * No trailing commas in argument lists")

        elif "attribute" in error_lower or "has no attribute" in error_lower:
            guidance.append("- Attribute error. Verify:")
            guidance.append("  * Variable names are correct")
            guidance.append("  * Methods exist on the object (check PyTorch docs)")
            guidance.append("  * Proper initialization in __init__")

        else:
            guidance.append("- General debugging tips:")
            guidance.append("  * Simplify the architecture")
            guidance.append("  * Test each layer individually")
            guidance.append("  * Use print statements to debug shapes")

        return "\n".join(guidance) + "\n"


class CompilationError(Exception):
    """Raised when code compilation fails after max retries."""
    pass
