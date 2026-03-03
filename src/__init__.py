"""
AutoFusion 2.0: Search Space-Free Multimodal Architecture Search

A dual-loop feedback system for automatic neural architecture discovery.
"""

__version__ = "2.0.0"
__author__ = "AutoFusion Team"

from .adapter import DynamicDataAdapter
from .controller import DualLoopController
from .sandbox import SecureSandbox, InnerLoopSandbox
from .evaluator import ProxyEvaluator, RewardFunction

__all__ = [
    "DynamicDataAdapter",
    "DualLoopController",
    "SecureSandbox",
    "InnerLoopSandbox",
    "ProxyEvaluator",
    "RewardFunction",
]
