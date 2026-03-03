#!/usr/bin/env python3
"""
Simplified Dataset Preparation for AutoFusion 2.0

Uses synthetic/sample data for quick testing.
Real datasets can be substituted later.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image

DATA_ROOT = Path(__file__).parent.parent / "data"


def prepare_mmmu_synthetic():
    """Create synthetic MMMU-style multimodal QA dataset."""
    print("\n" + "="*60)
    print("Preparing MMMU-Style Dataset (Scenario A)")
    print("="*60)

    target_dir = DATA_ROOT / "mmmu"
    target_dir.mkdir(parents=True, exist_ok=True)

    subjects = ["Math", "Physics", "Chemistry", "Computer_Science", "Biology"]
    num_samples_per_subject = 100

    annotations = []
    img_idx = 0

    for subject in subjects:
        img_dir = target_dir / "images" / subject
        img_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples_per_subject):
            # Create synthetic question image (diagram/formula)
            img_array = np.random.randint(200, 255, (448, 448, 3), dtype=np.uint8)

            # Add some structure based on subject
            if subject == "Math":
                # Add geometric shapes
                img_array[100:120, 100:300] = 0  # Horizontal line
                img_array[100:300, 100:120] = 0  # Vertical line
            elif subject == "Physics":
                # Add wave pattern
                for x in range(448):
                    y = int(224 + 50 * np.sin(x / 20))
                    if 0 <= y < 448:
                        img_array[y-2:y+2, x] = [255, 0, 0]

            img = Image.fromarray(img_array)
            img_path = img_dir / f"{i:04d}.jpg"
            img.save(img_path)

            # Create QA pair
            choices = [f"Option {chr(65+j)}" for j in range(4)]
            answer = np.random.choice(['A', 'B', 'C', 'D'])

            annotation = {
                "id": f"mmmu_{img_idx:05d}",
                "subject": subject,
                "image": str(img_path.relative_to(target_dir)),
                "question": f"What is the solution to this {subject} problem?",
                "choices": choices,
                "answer": answer,
                "hint": f"Think about {subject.lower()} principles."
            }
            annotations.append(annotation)
            img_idx += 1

    # Save annotations
    with open(target_dir / "annotations.json", 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"✓ MMMU-style dataset created")
    print(f"  Total samples: {len(annotations)}")
    print(f"  Subjects: {subjects}")
    print(f"  Annotations: {target_dir / 'annotations.json'}")

    return True


def prepare_vqa_rad_synthetic():
    """Create synthetic VQA-RAD style medical dataset."""
    print("\n" + "="*60)
    print("Preparing VQA-RAD-Style Dataset (Scenario B)")
    print("="*60)

    target_dir = DATA_ROOT / "vqa_rad"
    target_dir.mkdir(parents=True, exist_ok=True)

    body_parts = ["brain", "chest", "abdomen", "spine"]
    num_samples = 500

    annotations = []

    for i in range(num_samples):
        body_part = np.random.choice(body_parts)
        img_dir = target_dir / "images" / body_part
        img_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic medical image (grayscale with texture)
        img_array = np.random.randint(50, 200, (512, 512), dtype=np.uint8)

        # Add some medical-like structures
        if body_part == "brain":
            # Add circular structure (skull)
            center = (256, 256)
            for y in range(512):
                for x in range(512):
                    dist = np.sqrt((x-center[0])**2 + (y-center[1])**2)
                    if 180 < dist < 200:
                        img_array[y, x] = 30
        elif body_part == "chest":
            # Add lung-like regions
            img_array[100:400, 50:200] = np.random.randint(150, 180, (300, 150))
            img_array[100:400, 312:462] = np.random.randint(150, 180, (300, 150))

        img = Image.fromarray(img_array)
        img_path = img_dir / f"synpic{i:05d}.jpg"
        img.save(img_path)

        # Create Q&A
        question_types = ["what", "where", "is", "how"]
        q_type = np.random.choice(question_types)

        if q_type == "what":
            question = f"What abnormality is seen in this {body_part} image?"
            answer = np.random.choice(["Normal", "Tumor", "Inflammation", "Fracture"])
        elif q_type == "where":
            question = f"Where is the lesion located?"
            answer = np.random.choice(["Upper left", "Lower right", "Center", "No lesion"])
        elif q_type == "is":
            question = f"Is there a mass present?"
            answer = np.random.choice(["Yes", "No"])
        else:
            question = f"How severe is the condition?"
            answer = np.random.choice(["Mild", "Moderate", "Severe"])

        annotation = {
            "id": f"synpic{i:05d}",
            "image": str(img_path.relative_to(target_dir)),
            "question": question,
            "answer": answer,
            "question_type": q_type,
            "answer_type": "CLOSED" if q_type in ["is", "where"] else "OPEN",
            "body_part": body_part
        }
        annotations.append(annotation)

    # Save annotations
    with open(target_dir / "annotations.json", 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"✓ VQA-RAD-style dataset created")
    print(f"  Total samples: {len(annotations)}")
    print(f"  Body parts: {body_parts}")
    print(f"  Annotations: {target_dir / 'annotations.json'}")

    return True


def main():
    """Main entry point."""
    print("="*60)
    print("AutoFusion 2.0 - Dataset Preparation")
    print("="*60)
    print("\nNote: Using synthetic data for rapid prototyping.")
    print("Real datasets can be substituted for production runs.\n")

    results = {
        "mmmu": prepare_mmmu_synthetic(),
        "vqa_rad": prepare_vqa_rad_synthetic(),
    }

    # RoboSense already created by previous script
    if not (DATA_ROOT / "robo_sense" / "annotations.json").exists():
        print("\nRun: python scripts/download_datasets.py for RoboSense")

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    print("\n✓ All datasets ready for testing!")
    print(f"  Location: {DATA_ROOT}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
