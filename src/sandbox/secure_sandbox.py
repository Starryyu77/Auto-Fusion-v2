"""
Secure Sandbox Environment

Provides isolated execution environment for generated code with
resource limits, security constraints, and GPU memory management.
"""

import sys
import gc
import signal
import resource
import multiprocessing
from typing import Dict, Any, Optional, Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn


class SecureSandbox:
    """
    Secure sandbox for executing untrusted code with GPU memory protection.

    Features:
    - Resource limits (CPU time, memory)
    - Restricted module imports
    - Process isolation with forced cleanup
    - Timeout handling
    - GPU memory leak prevention
    """

    ALLOWED_MODULES = {
        'torch', 'torch.nn', 'torch.nn.functional',
        'math', 'numpy', 'typing'
    }

    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 1024,
        max_cpu_time: int = 60,
        max_vram_mb: Optional[int] = None  # Per-process VRAM limit
    ):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.max_cpu_time = max_cpu_time
        self.max_vram_mb = max_vram_mb or 2048  # Default 2GB per sandbox

    def execute(
        self,
        code: str,
        inputs: Dict[str, torch.Tensor]
    ) -> Tuple[bool, Any]:
        """
        Execute code in isolated process with guaranteed cleanup.

        Args:
            code: Python code to execute
            inputs: Input tensors for the model

        Returns:
            (success, result): Whether execution succeeded and output or error
        """
        # Pre-execution cleanup in parent process
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Use multiprocessing with 'spawn' to ensure clean state
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()

        process = ctx.Process(
            target=self._execute_in_process,
            args=(code, inputs, queue),
            daemon=True  # Force kill if parent dies
        )

        try:
            process.start()
            process.join(timeout=self.timeout)

            if process.is_alive():
                # Timeout - force kill
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join()
                return False, "Execution timeout"

            if process.exitcode != 0:
                return False, f"Process crashed with exit code {process.exitcode}"

            try:
                success, result = queue.get_nowait()
                return success, result
            except:
                return False, "Failed to get result from queue"

        finally:
            # Post-execution cleanup in parent process
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Ensure process is dead
            if process.is_alive():
                process.kill()
                process.join()

    def _execute_in_process(
        self,
        code: str,
        inputs: Dict[str, torch.Tensor],
        queue: multiprocessing.Queue
    ):
        """
        Execute code in a separate process with resource restrictions.
        This runs in a child process with isolated GPU context.
        """
        try:
            # Set resource limits (Unix only)
            if sys.platform != 'win32':
                # Limit memory
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (self.max_memory_mb * 1024 * 1024, -1)
                )
                # Limit CPU time
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (self.max_cpu_time, -1)
                )

            # Limit GPU memory if CUDA is available
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(
                    self.max_vram_mb / (torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
                )
                # Clear any residual GPU memory
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Create restricted namespace
            namespace = self._create_restricted_namespace()

            # Execute code
            exec(code, namespace)

            # Get model class
            if 'AutoFusionLayer' not in namespace:
                queue.put((False, "AutoFusionLayer class not found"))
                return

            ModelClass = namespace['AutoFusionLayer']

            # Instantiate and run
            input_dims = {k: v.shape for k, v in inputs.items()}
            model = ModelClass(input_dims)
            model.eval()

            with torch.no_grad():
                output = model(**inputs)

            queue.put((True, output))

        except Exception as e:
            queue.put((False, str(e)))

        finally:
            # CRITICAL: Force GPU cleanup before process exits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

    def _create_restricted_namespace(self) -> Dict[str, Any]:
        """Create a restricted namespace with only allowed modules."""
        namespace = {
            '__builtins__': {
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'slice': slice,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'int': int,
                'float': float,
                'str': str,
                'print': print,
                'Exception': Exception,
                'RuntimeError': RuntimeError,
                'ValueError': ValueError,
                'TypeError': TypeError,
            },
            'torch': torch,
            'nn': nn,
            'F': nn.functional,
        }

        # Add allowed modules if needed
        try:
            import math
            namespace['math'] = math
        except:
            pass

        try:
            import numpy as np
            namespace['np'] = np
        except:
            pass

        return namespace


@contextmanager
def timeout_context(seconds: int):
    """Context manager for timeout handling."""
    if sys.platform == 'win32':
        # Windows doesn't support signal-based timeouts
        yield
        return

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {seconds} seconds")

    # Set timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
