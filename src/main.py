#!/usr/bin/env python3
"""
AutoFusion 2.0: Main Entry Point

Usage:
    python src/main.py --data_dir ./data/mmmu --scenario high_dim_reasoning
    python src/main.py --config configs/scenario_a.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapter import DynamicDataAdapter
from src.controller import DualLoopController
from src.evaluator import ProxyEvaluator, RewardFunction

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AutoFusion 2.0: Search Space-Free NAS'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to data folder'
    )

    parser.add_argument(
        '--scenario',
        type=str,
        choices=['high_dim_reasoning', 'medical_vqa', 'edge_robotics'],
        default='high_dim_reasoning',
        help='Scenario type'
    )

    parser.add_argument(
        '--max_iterations',
        type=int,
        default=200,
        help='Maximum search iterations'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Output directory'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file (YAML)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if os.system('nvidia-smi') == 0 else 'cpu',
        help='Device to use'
    )

    return parser.parse_args()


def get_scenario_config(scenario: str) -> dict:
    """Get default configuration for a scenario."""
    configs = {
        "high_dim_reasoning": {
            "constraints": {
                "max_flops": 10_000_000,
                "max_params": 50_000_000,
                "target_accuracy": 0.45
            },
            "evaluator": {
                "num_shots": 16,
                "num_epochs": 5,
                "batch_size": 8
            }
        },
        "medical_vqa": {
            "constraints": {
                "max_flops": 50_000_000,
                "max_params": 100_000_000,
                "target_accuracy": 0.80
            },
            "evaluator": {
                "num_shots": 32,
                "num_epochs": 10,
                "batch_size": 8
            }
        },
        "edge_robotics": {
            "constraints": {
                "max_flops": 2_000_000,
                "max_params": 1_000_000,
                "target_accuracy": 0.90
            },
            "evaluator": {
                "num_shots": 64,
                "num_epochs": 5,
                "batch_size": 16
            }
        }
    }

    return configs.get(scenario, configs["high_dim_reasoning"])


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("AutoFusion 2.0: Search Space-Free NAS")
    logger.info("=" * 80)
    logger.info(f"Data Directory: {args.data_dir}")
    logger.info(f"Scenario: {args.scenario}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)

    # Step 1: Data Ingestion and Contract Generation
    logger.info("\n[Stage 1] Data Ingestion and Contract Generation")
    logger.info("-" * 60)

    adapter = DynamicDataAdapter(device=args.device)
    dataset, contract = adapter.ingest_folder(args.data_dir)

    # Add scenario constraints to contract
    scenario_config = get_scenario_config(args.scenario)
    contract.constraints = scenario_config["constraints"]

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"API Contract:\n{contract.to_prompt()}")

    # Step 2: Initialize Evaluator and Reward Function
    logger.info("\n[Stage 2] Initializing Evaluator")
    logger.info("-" * 60)

    eval_config = scenario_config["evaluator"]
    evaluator = ProxyEvaluator(
        dataset=dataset,
        num_shots=eval_config["num_shots"],
        num_epochs=eval_config["num_epochs"],
        batch_size=eval_config["batch_size"],
        device=args.device
    )

    reward_fn = RewardFunction(
        weights={"accuracy": 1.0, "efficiency": 0.5, "constraint": 2.0}
    )

    logger.info("✓ Evaluator initialized")

    # Step 3: Initialize LLM Backend
    logger.info("\n[Stage 3] Initializing LLM Backend")
    logger.info("-" * 60)

    # TODO: Implement actual LLM backend integration
    # For now, use a mock
    class MockLLMBackend:
        def generate(self, prompt: str) -> str:
            # This is a placeholder - integrate with actual LLM API
            return '''```python
import torch
import torch.nn as nn

class AutoFusionLayer(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.proj_v = nn.Linear(1024, 512)
        self.proj_t = nn.Linear(768, 512)
        self.fusion = nn.Linear(1024, 512)
        self.output = nn.Linear(512, 512)

    def forward(self, visual, text):
        v = visual.mean(dim=1)
        t = text.mean(dim=1)
        v = self.proj_v(v)
        t = self.proj_t(t)
        fused = torch.cat([v, t], dim=-1)
        fused = self.fusion(fused)
        return self.output(fused)
```'''

    llm_backend = MockLLMBackend()
    logger.info("✓ LLM Backend initialized (Mock)")

    # Step 4: Dual-Loop Search
    logger.info("\n[Stage 4] Starting Dual-Loop Search")
    logger.info("-" * 60)

    controller = DualLoopController(
        llm_backend=llm_backend,
        api_contract=contract,
        proxy_evaluator=evaluator,
        reward_fn=reward_fn,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir
    )

    best_result = controller.search()

    # Step 5: Final Output
    logger.info("\n[Stage 5] Final Results")
    logger.info("-" * 60)

    if best_result:
        logger.info(f"Best Architecture Found:")
        logger.info(f"  Reward: {best_result.reward:.3f}")
        logger.info(f"  Accuracy: {best_result.accuracy:.2%}")
        logger.info(f"  FLOPs: {best_result.flops/1e6:.1f}M")
        logger.info(f"  Params: {best_result.params/1e6:.1f}M")

        # Save best architecture
        output_path = Path(args.output_dir) / "best_architecture.py"
        with open(output_path, 'w') as f:
            f.write(best_result.code)
        logger.info(f"\nBest architecture saved to: {output_path}")
    else:
        logger.warning("No valid architecture found during search.")

    logger.info("\n" + "=" * 80)
    logger.info("AutoFusion 2.0 Search Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
