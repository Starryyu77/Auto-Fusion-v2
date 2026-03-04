"""
Proxy Evaluator

Fast evaluation of architectures using proxy datasets and few-shot learning.
Part of the outer loop for performance-based evolution.
"""

import re
import time
from typing import Dict, Any, Optional
from contextlib import redirect_stdout, redirect_stderr
import io

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np


class ProxyEvaluator:
    """
    Proxy Evaluator for fast architecture assessment.

    Uses:
    - Subset of training data (few-shot)
    - Few training epochs (5-10)
    - Small batch size
    - Early stopping
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_shots: int = 16,
        num_epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_time: int = 300  # Max 5 minutes per evaluation
    ):
        self.full_dataset = dataset
        self.num_shots = num_shots
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.max_time = max_time

    def evaluate(self, code: str) -> Dict[str, Any]:
        """
        Evaluate a generated architecture.

        Args:
            code: Python code string containing AutoFusionLayer class

        Returns:
            Dictionary with metrics:
                - accuracy: Top-1 accuracy
                - flops: Number of FLOPs
                - params: Number of parameters
                - training_time: Time spent training
        """
        start_time = time.time()

        # Create model from code
        model = self._instantiate_model(code)
        model = model.to(self.device)

        # Profile model
        flops, params = self._profile_model(model)

        # Create few-shot dataset
        train_loader, val_loader = self._create_dataloaders()

        # Train
        training_time = self._train_model(model, train_loader)

        # Evaluate
        accuracy = self._evaluate_model(model, val_loader)

        total_time = time.time() - start_time

        return {
            "accuracy": accuracy,
            "flops": flops,
            "params": params,
            "training_time": training_time,
            "total_time": total_time
        }

    def _instantiate_model(self, code: str) -> nn.Module:
        """Instantiate model from code string."""
        namespace = {}

        # Suppress stdout/stderr during model creation
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            exec(code, namespace)

        if 'AutoFusionLayer' not in namespace:
            raise ValueError("AutoFusionLayer class not found in code")

        FusionLayer = namespace['AutoFusionLayer']

        # Create a wrapper that handles the full forward pass
        class ModelWrapper(nn.Module):
            def __init__(self, fusion_layer, num_classes):
                super().__init__()
                self.fusion = fusion_layer
                self.classifier = nn.Linear(
                    self._get_classifier_input_dim(fusion_layer),
                    num_classes
                )

            def _get_classifier_input_dim(self, fusion_layer):
                # Infer output dimension by running a dummy forward
                # Note: Using 256 to match CLIP-ViT-L/14 patch embeddings (224/14=16, 16*16=256 patches)
                dummy_visual = torch.randn(1, 256, 1024)
                dummy_text = torch.randn(1, 77, 768)
                with torch.no_grad():
                    output = fusion_layer(visual=dummy_visual, text=dummy_text)
                return output.shape[-1]

            def forward(self, visual, text, label=None):
                fused = self.fusion(visual=visual, text=text)
                logits = self.classifier(fused)
                return logits

        # Create fusion layer with dummy input_dims
        # Note: 256 patches for CLIP-ViT-L/14 (224x224 image / 14x14 patch = 16x16 = 256)
        input_dims = {"visual": [256, 1024], "text": [77, 768]}
        fusion_layer = FusionLayer(input_dims)

        # Get number of classes from dataset
        num_classes = self._get_num_classes()

        return ModelWrapper(fusion_layer, num_classes)

    def _get_num_classes(self) -> int:
        """Get number of classes from dataset."""
        # Try to infer from dataset
        sample = self.full_dataset[0]
        if "label" in sample:
            # Count unique labels
            labels = []
            for i in range(min(len(self.full_dataset), 100)):
                labels.append(self.full_dataset[i]["label"].item())
            return len(set(labels))
        return 2  # Default binary

    def _create_dataloaders(self) -> tuple:
        """Create few-shot train and validation dataloaders."""
        # Sample few-shot subset
        indices = list(range(len(self.full_dataset)))
        np.random.shuffle(indices)

        # Use stratified sampling if possible
        train_indices = indices[:self.num_shots * 4]  # 4 classes x 16 shots
        val_indices = indices[self.num_shots * 4:self.num_shots * 4 + 64]

        train_subset = Subset(self.full_dataset, train_indices)
        val_subset = Subset(self.full_dataset, val_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        return train_loader, val_loader

    def _train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader
    ) -> float:
        """Train the model for few epochs."""
        start_time = time.time()

        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.num_epochs):
            for batch_idx, batch in enumerate(train_loader):
                # Check timeout
                if time.time() - start_time > self.max_time:
                    return time.time() - start_time

                # Get inputs
                visual = batch.get("visual", batch.get("image")).to(self.device)
                text = batch.get("text", batch.get("question")).to(self.device)
                labels = batch["label"].to(self.device)

                # Forward
                optimizer.zero_grad()
                logits = model(visual=visual, text=text)
                loss = criterion(logits, labels)

                # Backward
                loss.backward()
                optimizer.step()

        return time.time() - start_time

    def _evaluate_model(
        self,
        model: nn.Module,
        val_loader: DataLoader
    ) -> float:
        """Evaluate model on validation set."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                visual = batch.get("visual", batch.get("image")).to(self.device)
                text = batch.get("text", batch.get("question")).to(self.device)
                labels = batch["label"].to(self.device)

                logits = model(visual=visual, text=text)
                predictions = logits.argmax(dim=-1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return correct / total if total > 0 else 0.0

    def _profile_model(self, model: nn.Module) -> tuple:
        """Profile FLOPs and parameters of the model."""
        # Count parameters
        params = sum(p.numel() for p in model.parameters())

        # Estimate FLOPs (simplified)
        # Run a forward pass and count operations
        dummy_visual = torch.randn(1, 576, 1024).to(self.device)
        dummy_text = torch.randn(1, 77, 768).to(self.device)

        # Simple FLOPs estimation based on Linear layers
        flops = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # FLOPs = 2 * input_features * output_features (multiply-add)
                flops += 2 * module.in_features * module.out_features
            elif isinstance(module, nn.MultiheadAttention):
                # Approximate attention FLOPs
                flops += 2 * 576 * 576  # Simplified

        return flops, params
