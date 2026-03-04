"""
Dual-Loop Controller

The core controller implementing the inner loop (auto-debugging) and
outer loop (performance evolution) feedback mechanisms.
"""

import os
import json
import time
import logging
import traceback
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime

import torch
import torch.nn as nn

from ..adapter import APIContract
from ..sandbox import InnerLoopSandbox
from ..evaluator import ProxyEvaluator, RewardFunction

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of a single architecture search iteration."""
    iteration: int
    code: str
    compile_success: bool
    compile_attempts: int
    accuracy: float
    flops: int
    params: int
    reward: float
    error_message: Optional[str] = None
    time_taken: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class DualLoopController:
    """
    Dual-Loop Controller for AutoFusion 2.0.

    Implements the core search algorithm with:
    - Inner Loop: Self-healing compilation (syntax + shape validation)
    - Outer Loop: Performance-based architecture evolution
    """

    def __init__(
        self,
        llm_backend: Any,  # LLM backend for code generation
        api_contract: APIContract,
        proxy_evaluator: ProxyEvaluator,
        reward_fn: RewardFunction,
        max_inner_retries: int = 5,
        max_iterations: int = 200,
        output_dir: str = "./results",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.llm = llm_backend
        self.contract = api_contract
        self.proxy_evaluator = proxy_evaluator
        self.reward_fn = reward_fn
        self.max_inner_retries = max_inner_retries
        self.max_iterations = max_iterations
        self.output_dir = output_dir
        self.device = device

        # Initialize components
        self.inner_loop = InnerLoopSandbox(
            llm_backend=llm_backend,
            api_contract=api_contract,
            max_retries=max_inner_retries
        )

        # State tracking
        self.history: List[SearchResult] = []
        self.best_result: Optional[SearchResult] = None
        self.iteration = 0

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def search(self) -> SearchResult:
        """
        Execute the dual-loop search.

        Returns:
            The best architecture found during the search.
        """
        logger.info("=" * 80)
        logger.info("AutoFusion 2.0: Dual-Loop Architecture Search")
        logger.info("=" * 80)
        logger.info(f"Max Iterations: {self.max_iterations}")
        logger.info(f"Max Inner Retries: {self.max_inner_retries}")
        logger.info(f"API Contract:\n{self.contract.to_prompt()}")
        logger.info("=" * 80)

        start_time = time.time()

        for iteration in range(1, self.max_iterations + 1):
            self.iteration = iteration
            iter_start = time.time()

            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration}/{self.max_iterations}")
            logger.info(f"{'='*60}")

            # Execute dual-loop for this iteration
            result = self._run_iteration(iteration)

            result.time_taken = time.time() - iter_start
            self.history.append(result)

            # Update best result
            if result.compile_success and (
                self.best_result is None or result.reward > self.best_result.reward
            ):
                self.best_result = result
                logger.info(f"🏆 New Best! Reward: {result.reward:.3f}")

            # Print summary
            self._print_iteration_summary(result)

            # Periodic checkpoint save
            if iteration % 10 == 0:
                self._save_checkpoint(iteration)

        total_time = time.time() - start_time
        logger.info(f"\n{'='*80}")
        logger.info("Search Complete!")
        logger.info(f"Total Time: {total_time/60:.1f} minutes")
        logger.info(f"Iterations: {self.max_iterations}")
        logger.info(f"Compile Success Rate: {self._get_compile_success_rate():.1%}")

        if self.best_result:
            logger.info(f"Best Reward: {self.best_result.reward:.3f}")
            logger.info(f"Best Accuracy: {self.best_result.accuracy:.2%}")
            logger.info(f"Best FLOPs: {self.best_result.flops/1e6:.1f}M")

        logger.info(f"{'='*80}")

        return self.best_result

    def _run_iteration(self, iteration: int) -> SearchResult:
        """
        Run a single iteration of the dual-loop search.

        Returns:
            SearchResult with metrics from this iteration.
        """
        # Step 1: Build prompt with history and contract
        prompt = self._build_prompt(iteration)

        # Step 2: Inner Loop - Self-healing compilation
        code, compile_attempts = self.inner_loop.self_healing_compile(prompt)

        if code is None:
            # Compilation failed after all retries
            return SearchResult(
                iteration=iteration,
                code="",
                compile_success=False,
                compile_attempts=compile_attempts,
                accuracy=0.0,
                flops=0,
                params=0,
                reward=0.0,
                error_message="Compilation failed after max retries"
            )

        logger.info(f"✅ Compilation succeeded after {compile_attempts} attempt(s)")

        # Step 3: Outer Loop - Performance evaluation
        try:
            metrics = self.proxy_evaluator.evaluate(code)
        except Exception as e:
            logger.warning(f"⚠️ Proxy evaluation failed: {e}")
            # Return a failed result but don't crash
            return SearchResult(
                iteration=iteration,
                code=code,
                compile_success=True,  # Code compiles but eval failed
                compile_attempts=compile_attempts,
                accuracy=0.0,
                flops=0,
                params=0,
                reward=0.0,
                error_message=f"Evaluation failed: {str(e)[:100]}"
            )

        # Step 4: Calculate reward
        reward = self.reward_fn.calculate(
            accuracy=metrics['accuracy'],
            flops=metrics['flops'],
            params=metrics['params'],
            constraints=self.contract.constraints
        )

        # Step 5: Generate feedback for next iteration
        feedback = self._generate_feedback(metrics, reward, iteration)

        return SearchResult(
            iteration=iteration,
            code=code,
            compile_success=True,
            compile_attempts=compile_attempts,
            accuracy=metrics['accuracy'],
            flops=metrics['flops'],
            params=metrics['params'],
            reward=reward
        )

    def _build_prompt(self, iteration: int) -> str:
        """Build the prompt for LLM code generation."""
        prompt_parts = []

        # System context
        prompt_parts.append("You are an expert neural architecture designer.")
        prompt_parts.append("Generate PyTorch code for a multimodal fusion architecture.\n")

        # API Contract
        prompt_parts.append(self.contract.to_prompt())
        prompt_parts.append("")

        # History feedback (if not first iteration)
        if self.history:
            prompt_parts.append("【Search History】")
            prompt_parts.append(f"Total iterations so far: {len(self.history)}")

            if self.best_result:
                prompt_parts.append(f"\nCurrent Best Architecture:")
                prompt_parts.append(f"- Reward: {self.best_result.reward:.3f}")
                prompt_parts.append(f"- Accuracy: {self.best_result.accuracy:.2%}")
                prompt_parts.append(f"- FLOPs: {self.best_result.flops/1e6:.1f}M")

            # Show recent results
            prompt_parts.append("\nRecent Results:")
            for result in self.history[-5:]:
                status = "✅" if result.compile_success else "❌"
                prompt_parts.append(
                    f"Iter {result.iteration}: {status} "
                    f"Reward={result.reward:.3f}, "
                    f"Acc={result.accuracy:.2%}"
                )

            prompt_parts.append("")

            # Generate feedback
            feedback = self._generate_strategy_feedback(iteration)
            prompt_parts.append(f"【Strategy Guidance】\n{feedback}\n")

        # Code generation instructions
        prompt_parts.append("【Code Generation Instructions】")
        prompt_parts.append("1. Create a class named 'AutoFusionLayer' inheriting from nn.Module")
        prompt_parts.append("2. __init__ should accept input_dims as parameter")
        prompt_parts.append("3. forward should accept inputs matching the API contract")
        prompt_parts.append("4. Use only standard PyTorch operations (nn.Module, nn.functional)")
        prompt_parts.append("5. Ensure output shape matches the contract")
        prompt_parts.append("6. Do not include training code or main blocks")
        prompt_parts.append("7. Only return the class definition, no explanation\n")

        prompt_parts.append("【Output Format】")
        prompt_parts.append("```python")
        prompt_parts.append("import torch")
        prompt_parts.append("import torch.nn as nn")
        prompt_parts.append("import torch.nn.functional as F")
        prompt_parts.append("")
        prompt_parts.append("class AutoFusionLayer(nn.Module):")
        prompt_parts.append("    def __init__(self, input_dims): ...")
        prompt_parts.append("    def forward(self, ...): ...")
        prompt_parts.append("```")

        return "\n".join(prompt_parts)

    def _generate_strategy_feedback(self, iteration: int) -> str:
        """Generate strategy guidance based on search progress."""
        progress = iteration / self.max_iterations

        if progress < 0.3:
            return (
                "Phase: EXPLORATION\n"
                "Focus on trying diverse architecture types:\n"
                "- Attention-based fusion (cross-modal attention)\n"
                "- Gating mechanisms (feature-wise gates)\n"
                "- Concatenation + MLP baselines\n"
                "Don't worry about perfection, explore the space."
            )
        elif progress < 0.7:
            return (
                "Phase: EXPLOITATION\n"
                "Focus on refining the best architectures found:\n"
                "- Improve the top-performing design\n"
                "- Adjust hidden dimensions and layer counts\n"
                "- Try architectural variants of successful patterns\n"
                "Balance exploration and exploitation."
            )
        else:
            return (
                "Phase: REFINEMENT\n"
                "Focus on fine-tuning for maximum performance:\n"
                "- Optimize the best architecture\n"
                "- Tune hyperparameters (dropout, normalization)\n"
                "- Ensure efficiency constraints are met\n"
                "Polish the final design."
            )

    def _generate_feedback(
        self,
        metrics: Dict[str, Any],
        reward: float,
        iteration: int
    ) -> str:
        """Generate natural language feedback based on metrics."""
        feedback_parts = []

        # Performance summary
        feedback_parts.append(f"Iteration {iteration} Results:")
        feedback_parts.append(f"- Accuracy: {metrics['accuracy']:.2%}")
        feedback_parts.append(f"- FLOPs: {metrics['flops']/1e6:.1f}M")
        feedback_parts.append(f"- Parameters: {metrics['params']/1e6:.1f}M")
        feedback_parts.append(f"- Reward: {reward:.3f}")

        # Comparison to best
        if self.best_result:
            reward_diff = reward - self.best_result.reward
            if reward_diff > 0:
                feedback_parts.append(f"✅ New best! (+{reward_diff:.3f})")
            else:
                feedback_parts.append(f"vs Best: {reward_diff:.3f}")

        return "\n".join(feedback_parts)

    def _print_iteration_summary(self, result: SearchResult):
        """Print a summary of the iteration."""
        logger.info(f"Iteration {result.iteration} Summary:")
        logger.info(f"  Compile: {'✅' if result.compile_success else '❌'} "
                   f"({result.compile_attempts} attempts)")

        if result.compile_success:
            logger.info(f"  Accuracy: {result.accuracy:.2%}")
            logger.info(f"  FLOPs: {result.flops/1e6:.1f}M")
            logger.info(f"  Reward: {result.reward:.3f}")

        if self.best_result:
            logger.info(f"  🏆 Best so far: {self.best_result.reward:.3f}")

        logger.info(f"  Time: {result.time_taken:.1f}s")

    def _save_checkpoint(self, iteration: int):
        """Save a checkpoint of the current search state."""
        checkpoint = {
            "iteration": iteration,
            "history": [r.to_dict() for r in self.history],
            "best_result": self.best_result.to_dict() if self.best_result else None,
            "api_contract": self.contract.to_dict()
        }

        checkpoint_path = os.path.join(
            self.output_dir, f"checkpoint_iter_{iteration}.json"
        )

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

    def _get_compile_success_rate(self) -> float:
        """Calculate the compile success rate."""
        if not self.history:
            return 0.0
        successes = sum(1 for r in self.history if r.compile_success)
        return successes / len(self.history)
