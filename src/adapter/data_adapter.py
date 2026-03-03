"""
Dynamic Data Adapter

Automatically ingests raw data folders, extracts features using frozen backbones,
and generates API contracts for the LLM controller.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor


@dataclass
class TensorSpec:
    """Tensor specification for API contract."""
    name: str
    shape: List[Any]  # Can include "B" for batch dimension
    dtype: str
    description: str
    source: str


@dataclass
class APIContract:
    """API contract generated from data sniffing."""
    version: str = "1.0"
    input_specs: Dict[str, TensorSpec] = None
    output_spec: TensorSpec = None
    constraints: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "input_specs": {
                k: asdict(v) for k, v in self.input_specs.items()
            } if self.input_specs else {},
            "output_spec": asdict(self.output_spec) if self.output_spec else None,
            "constraints": self.constraints or {}
        }

    def to_prompt(self) -> str:
        """Convert to LLM prompt format."""
        lines = ["【API Interface Contract】", ""]

        lines.append("Input Specifications:")
        for name, spec in self.input_specs.items():
            lines.append(f"  - {name}:")
            lines.append(f"    Shape: {spec.shape}")
            lines.append(f"    Dtype: {spec.dtype}")
            lines.append(f"    Description: {spec.description}")
            lines.append("")

        lines.append("Output Specification:")
        if self.output_spec:
            lines.append(f"  Shape: {self.output_spec.shape}")
            lines.append(f"  Dtype: {self.output_spec.dtype}")
            lines.append("")

        lines.append("Constraints:")
        for key, value in (self.constraints or {}).items():
            lines.append(f"  - {key}: {value}")

        return "\n".join(lines)


class DynamicDataAdapter:
    """
    Dynamic Data Adapter for AutoFusion 2.0.

    Automatically adapts to various data formats and generates API contracts.
    """

    def __init__(
        self,
        vision_backbone: str = "clip-vit-l-14",
        text_backbone: str = "clip-text-l",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.vision_backbone = vision_backbone
        self.text_backbone = text_backbone
        self.device = device

        # Initialize frozen backbones
        self._init_backbones()

    def _init_backbones(self):
        """Initialize frozen feature extraction backbones."""
        # For CLIP-based models
        if "clip" in self.vision_backbone.lower():
            model_name = "openai/clip-vit-large-patch14"
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError(f"Backbone {self.vision_backbone} not supported")

    def ingest_folder(
        self,
        data_dir: str,
        annotations_file: Optional[str] = None
    ) -> Tuple[Dataset, APIContract]:
        """
        Ingest a data folder and generate API contract.

        Args:
            data_dir: Path to data folder containing images/videos and annotations
            annotations_file: Path to annotations JSON file (optional)

        Returns:
            (dataset, api_contract): PyTorch dataset and generated API contract
        """
        data_dir = Path(data_dir)

        # Auto-detect annotations file
        if annotations_file is None:
            annotations_file = self._find_annotations(data_dir)

        # Load annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        # Create dataset
        dataset = self._create_dataset(data_dir, annotations)

        # Sniff shapes from first batch
        contract = self._sniff_shapes(dataset, annotations)

        return dataset, contract

    def _find_annotations(self, data_dir: Path) -> str:
        """Auto-find annotations file in data directory."""
        candidates = [
            "annotations.json",
            "metadata.json",
            "labels.json",
            "data.json",
            "info.json"
        ]

        for candidate in candidates:
            path = data_dir / candidate
            if path.exists():
                return str(path)

        raise FileNotFoundError(
            f"No annotations file found in {data_dir}. "
            f"Please provide annotations_file parameter."
        )

    def _create_dataset(
        self,
        data_dir: Path,
        annotations: List[Dict]
    ) -> Dataset:
        """Create a PyTorch dataset from annotations."""
        return AutoFusionDataset(
            data_dir=data_dir,
            annotations=annotations,
            processor=self.clip_processor,
            device=self.device
        )

    def _sniff_shapes(
        self,
        dataset: Dataset,
        annotations: List[Dict]
    ) -> APIContract:
        """
        Extract tensor shapes from first batch.

        This is the key "Shape Sniffer" component that dynamically detects
        input dimensions without human intervention.
        """
        # Get a sample from dataset
        sample = dataset[0]

        input_specs = {}

        # Visual features
        if "visual" in sample:
            visual_tensor = sample["visual"]
            input_specs["visual"] = TensorSpec(
                name="visual",
                shape=["B"] + list(visual_tensor.shape[1:]),
                dtype=str(visual_tensor.dtype).replace("torch.", ""),
                description=f"Visual features from frozen {self.vision_backbone}",
                source="image"
            )

        # Text features
        if "text" in sample:
            text_tensor = sample["text"]
            input_specs["text"] = TensorSpec(
                name="text",
                shape=["B"] + list(text_tensor.shape[1:]),
                dtype=str(text_tensor.dtype).replace("torch.", ""),
                description=f"Text features from frozen {self.text_backbone}",
                source="text"
            )

        # Other modalities (sensor, audio, etc.)
        for key in sample.keys():
            if key not in ["visual", "text", "label"]:
                tensor = sample[key]
                input_specs[key] = TensorSpec(
                    name=key,
                    shape=["B"] + list(tensor.shape[1:]),
                    dtype=str(tensor.dtype).replace("torch.", ""),
                    description=f"{key} modality",
                    source=key
                )

        # Determine output shape from labels
        if "label" in sample:
            label_tensor = sample["label"]

            # Check if classification or regression
            if label_tensor.dim() == 0 or (label_tensor.dim() == 1 and label_tensor.shape[0] == 1):
                # Single label - get number of classes
                num_classes = self._get_num_classes(annotations)
                output_shape = ["B", num_classes]
            else:
                output_shape = ["B"] + list(label_tensor.shape[1:])

            output_spec = TensorSpec(
                name="output",
                shape=output_shape,
                dtype="float32",
                description="Model output logits",
                source="prediction"
            )
        else:
            output_spec = None

        return APIContract(
            input_specs=input_specs,
            output_spec=output_spec,
            constraints={}  # Will be filled by scenario config
        )

    def _get_num_classes(self, annotations: List[Dict]) -> int:
        """Extract number of classes from annotations."""
        # Try common label keys
        label_keys = ["label", "answer", "class", "category", "target"]

        for key in label_keys:
            if key in annotations[0]:
                labels = [ann[key] for ann in annotations]
                unique_labels = set(labels)
                return len(unique_labels)

        # Default to binary classification
        return 2


class AutoFusionDataset(Dataset):
    """
    Generic dataset for AutoFusion 2.0.

    Automatically handles various data formats and extracts features
    using frozen backbones.
    """

    def __init__(
        self,
        data_dir: Path,
        annotations: List[Dict],
        processor: CLIPProcessor,
        device: str = "cuda"
    ):
        self.data_dir = data_dir
        self.annotations = annotations
        self.processor = processor
        self.device = device

        # Load CLIP model for feature extraction
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(device)
        self.clip_model.eval()

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with extracted features."""
        ann = self.annotations[idx]

        result = {}

        # Extract visual features
        if "image" in ann:
            image_path = self.data_dir / ann["image"]
            result["visual"] = self._extract_visual(str(image_path))

        # Extract text features
        if "text" in ann or "question" in ann:
            text = ann.get("text") or ann.get("question", "")
            result["text"] = self._extract_text(text)

        # Extract other modalities
        for key in ["sensor", "audio", "video"]:
            if key in ann:
                result[key] = self._extract_other(key, ann[key])

        # Label
        for key in ["label", "answer", "class", "category", "target"]:
            if key in ann:
                result["label"] = torch.tensor(ann[key], dtype=torch.long)
                break

        return result

    def _extract_visual(self, image_path: str) -> torch.Tensor:
        """Extract visual features using frozen CLIP."""
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            vision_outputs = self.clip_model.vision_model(
                pixel_values=inputs["pixel_values"].to(self.device)
            )
            # Return patch embeddings (excluding CLS token)
            features = vision_outputs.last_hidden_state[:, 1:, :]

        return features.squeeze(0).cpu()  # Remove batch dim added by processor

    def _extract_text(self, text: str) -> torch.Tensor:
        """Extract text features using frozen CLIP."""
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )

        with torch.no_grad():
            text_outputs = self.clip_model.text_model(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device)
            )
            features = text_outputs.last_hidden_state

        return features.squeeze(0).cpu()

    def _extract_other(self, key: str, data: Any) -> torch.Tensor:
        """Extract other modality features."""
        # Default: convert to tensor
        if isinstance(data, (list, tuple)):
            return torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, dict):
            # Flatten dict to vector
            values = []
            for v in data.values():
                if isinstance(v, (list, tuple)):
                    values.extend(v)
                else:
                    values.append(v)
            return torch.tensor(values, dtype=torch.float32)
        else:
            return torch.tensor([data], dtype=torch.float32)


if __name__ == "__main__":
    # Example usage
    print("Dynamic Data Adapter Example")

    # Mock API Contract
    contract = APIContract(
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
            shape=["B", 4],
            dtype="float32",
            description="Classification logits",
            source="prediction"
        ),
        constraints={
            "max_flops": 10_000_000,
            "max_params": 50_000_000
        }
    )

    print(contract.to_prompt())
