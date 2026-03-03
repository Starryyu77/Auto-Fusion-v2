#!/usr/bin/env python3
"""
Dataset Download Script for AutoFusion 2.0

Downloads and prepares:
- Scenario A: MMMU (Massive Multi-discipline Multimodal Understanding)
- Scenario B: VQA-RAD (Visual Question Answering for Radiology)
- Scenario C: RoboSense (Robotics sensor data - using Habitat/Simulated)
"""

import os
import sys
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = Path(__file__).parent.parent / "data"


def create_directories():
    """Create data directory structure."""
    dirs = [
        DATA_ROOT / "mmmu",
        DATA_ROOT / "vqa_rad",
        DATA_ROOT / "robo_sense",
        DATA_ROOT / "cache"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {d}")


def download_mmmu():
    """
    Download MMMU dataset.

    MMMU is available from HuggingFace datasets.
    We'll use the validation split for our experiments.
    """
    print("\n" + "="*60)
    print("Downloading MMMU Dataset (Scenario A)")
    print("="*60)

    target_dir = DATA_ROOT / "mmmu"

    try:
        from datasets import load_dataset
        from datasets import concatenate_datasets

        print("Loading MMMU from HuggingFace...")

        # MMMU requires specifying a config (subject). We'll load multiple subjects
        subjects = ['Computer_Science', 'Math', 'Physics', 'Chemistry', 'Biology']

        datasets = []
        for subject in subjects:
            print(f"  Loading {subject}...")
            try:
                ds = load_dataset("MMMU/MMMU", subject, split="validation", cache_dir=str(DATA_ROOT / "cache"))
                datasets.append(ds)
            except Exception as e:
                print(f"    Warning: Could not load {subject}: {e}")

        if not datasets:
            raise ValueError("No subjects could be loaded")

        # Combine all subjects
        ds = concatenate_datasets(datasets)
        print(f"  Combined {len(datasets)} subjects, total samples: {len(ds)}")

        # Organize by subject
        subjects = set(ds['subject'])
        print(f"Found subjects: {subjects}")

        # Create annotations
        annotations = []
        image_count = 0

        for idx, item in enumerate(ds):
            # Save image if present
            image_path = None
            if 'image' in item and item['image'] is not None:
                img_dir = target_dir / "images" / item['subject']
                img_dir.mkdir(parents=True, exist_ok=True)

                image_path = img_dir / f"{idx}.jpg"
                if hasattr(item['image'], 'save'):
                    item['image'].save(image_path)
                    image_count += 1

            annotation = {
                "id": f"mmmu_{idx}",
                "subject": item['subject'],
                "question": item['question'],
                "choices": item['choices'],
                "answer": item['answer'],
                "image": str(image_path.relative_to(target_dir)) if image_path else None,
                "hint": item.get('hint', '')
            }
            annotations.append(annotation)

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(ds)} samples...")

        # Save annotations
        with open(target_dir / "annotations.json", 'w') as f:
            json.dump(annotations, f, indent=2)

        print(f"✓ MMMU downloaded successfully")
        print(f"  Total samples: {len(annotations)}")
        print(f"  Images saved: {image_count}")
        print(f"  Annotations: {target_dir / 'annotations.json'}")

        return True

    except ImportError:
        print("⚠ datasets library not installed. Installing...")
        os.system("pip install datasets -q")
        return download_mmmu()
    except Exception as e:
        print(f"✗ Error downloading MMMU: {e}")
        return False


def download_vqa_rad():
    """
    Download VQA-RAD dataset.

    Available from: https://github.com/abachaa/VQA-RAD
    """
    print("\n" + "="*60)
    print("Downloading VQA-RAD Dataset (Scenario B)")
    print("="*60)

    target_dir = DATA_ROOT / "vqa_rad"

    # VQA-RAD GitHub repository structure
    base_url = "https://raw.githubusercontent.com/abachaa/VQA-RAD/master"

    # Note: Images need to be obtained separately due to medical data licensing
    print("Downloading annotations...")

    # URL encode the filename to handle spaces
    encoded_filename = "VQA_RAD%20Dataset%20Public.json"
    url = f"{base_url}/{encoded_filename}"
    local_path = target_dir / "annotations.json"

    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"✓ Downloaded: annotations.json")
    except Exception as e:
        print(f"✗ Failed to download annotations: {e}")
        print("\n  Manual download required:")
        print(f"  1. Visit: https://github.com/abachaa/VQA-RAD")
        print(f"  2. Download the dataset")
        print(f"  3. Extract to: {target_dir}")
        return False

    # Process annotations
    with open(target_dir / "annotations.json", 'r') as f:
        data = json.load(f)

    print(f"✓ VQA-RAD annotations loaded")
    print(f"  Total Q&A pairs: {len(data)}")

    # Create README for data preparation
    readme = """
# VQA-RAD Data Preparation

Due to medical data licensing, images must be obtained separately:

1. Visit: https://github.com/abachaa/VQA-RAD
2. Follow their instructions to request dataset access
3. Place images in: ./images/
4. Expected structure:
   images/
   ├── synpicXXXXX.jpg
   ├── synpicXXXXX.jpg
   └── ...

Annotations have been downloaded to: annotations.json
"""
    with open(target_dir / "README.txt", 'w') as f:
        f.write(readme)

    print(f"  See: {target_dir / 'README.txt'}")

    return True


def create_robo_sense():
    """
    Create RoboSense dataset using Habitat Simulator or synthetic data.

    For this experiment, we'll create a synthetic robotics dataset
    that mimics real robot sensor fusion scenarios.
    """
    print("\n" + "="*60)
    print("Creating RoboSense Dataset (Scenario C)")
    print("="*60)

    target_dir = DATA_ROOT / "robo_sense"

    print("Generating synthetic robotics data...")

    import numpy as np
    from PIL import Image

    np.random.seed(42)

    num_samples = 2000
    annotations = []

    # Scene categories for indoor navigation
    categories = ["free", "obstacle", "cliff", "stuck", "docking"]

    print(f"Generating {num_samples} samples...")

    for idx in range(num_samples):
        # Generate category
        category_idx = np.random.choice(len(categories), p=[0.4, 0.3, 0.1, 0.1, 0.1])
        category = categories[category_idx]

        # Generate low-res camera image (simulating edge camera)
        img_dir = target_dir / "images"
        img_dir.mkdir(exist_ok=True)

        # Create synthetic image based on category
        if category == "free":
            # Clear path - mostly uniform color
            img_array = np.random.randint(180, 220, (224, 224, 3), dtype=np.uint8)
        elif category == "obstacle":
            # Has obstacle - add dark blob
            img_array = np.random.randint(180, 220, (224, 224, 3), dtype=np.uint8)
            center = (np.random.randint(50, 174), np.random.randint(50, 174))
            img_array[center[0]-30:center[0]+30, center[1]-30:center[1]+30] = np.random.randint(30, 80, (60, 60, 3))
        elif category == "cliff":
            # Edge detected - gradient
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)
            for i in range(224):
                img_array[i, :] = int(255 * (i / 224))
        elif category == "stuck":
            # Texture indicating stuck (repetitive pattern)
            base = np.random.randint(100, 150, (32, 32, 3), dtype=np.uint8)
            img_array = np.tile(base, (7, 7, 1))[:224, :224, :]
        else:  # docking
            # Target pattern
            img_array = np.random.randint(180, 220, (224, 224, 3), dtype=np.uint8)
            # Add marker
            img_array[100:124, 100:124] = [255, 0, 0]

        img = Image.fromarray(img_array)
        img_path = img_dir / f"frame_{idx:06d}.jpg"
        img.save(img_path)

        # Generate sensor data
        sensor_data = {
            "imu_accel": [float(x) for x in np.random.randn(3)],
            "imu_gyro": [float(x) for x in np.random.randn(3) * 0.1],
            "lidar_distances": [float(d) for d in np.random.uniform(0.1, 5.0, 16)],
            "bump_sensors": [int(b) for b in np.random.choice([0, 1], 4, p=[0.9, 0.1])],
            "battery_level": float(np.random.uniform(0.2, 1.0))
        }

        # Adjust sensor data based on category
        if category == "obstacle":
            sensor_data["lidar_distances"][0:4] = [0.3, 0.3, 0.35, 0.4]  # Close obstacles ahead
        elif category == "cliff":
            sensor_data["lidar_distances"][8:12] = [5.0] * 4  # Large distance = drop
        elif category == "stuck":
            sensor_data["imu_accel"][2] = 0.0  # No Z acceleration = not moving
            sensor_data["bump_sensors"] = [1, 1, 1, 1]  # All bumpers triggered

        annotation = {
            "id": f"frame_{idx:06d}",
            "image": f"images/frame_{idx:06d}.jpg",
            "sensor": sensor_data,
            "label": category,
            "label_idx": category_idx
        }
        annotations.append(annotation)

        if (idx + 1) % 500 == 0:
            print(f"  Generated {idx + 1}/{num_samples} samples...")

    # Save annotations
    with open(target_dir / "annotations.json", 'w') as f:
        json.dump(annotations, f, indent=2)

    # Create split
    split_idx = int(0.8 * num_samples)
    train_data = annotations[:split_idx]
    val_data = annotations[split_idx:]

    with open(target_dir / "train.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(target_dir / "val.json", 'w') as f:
        json.dump(val_data, f, indent=2)

    print(f"✓ RoboSense dataset created")
    print(f"  Total samples: {num_samples}")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Categories: {categories}")

    return True


def verify_datasets():
    """Verify all datasets are properly downloaded."""
    print("\n" + "="*60)
    print("Dataset Verification")
    print("="*60)

    datasets = {
        "mmmu": ["annotations.json"],
        "vqa_rad": ["annotations.json"],
        "robo_sense": ["annotations.json", "train.json", "val.json"]
    }

    all_ok = True
    for ds_name, required_files in datasets.items():
        ds_dir = DATA_ROOT / ds_name
        print(f"\n{ds_name}:")

        for fname in required_files:
            fpath = ds_dir / fname
            if fpath.exists():
                size = fpath.stat().st_size
                print(f"  ✓ {fname} ({size:,} bytes)")
            else:
                print(f"  ✗ {fname} MISSING")
                all_ok = False

    return all_ok


def main():
    """Main entry point."""
    print("="*60)
    print("AutoFusion 2.0 - Dataset Download")
    print("="*60)

    # Create directories
    create_directories()

    # Download/Create datasets
    results = {
        "mmmu": download_mmmu(),
        "vqa_rad": download_vqa_rad(),
        "robo_sense": create_robo_sense()
    }

    # Verify
    all_ok = verify_datasets()

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for ds_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {ds_name}")

    if all_ok:
        print("\n✓ All datasets ready!")
        print(f"  Data location: {DATA_ROOT}")
        return 0
    else:
        print("\n⚠ Some datasets require manual intervention")
        print("  See error messages above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
