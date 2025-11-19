#!/usr/bin/env python3
"""
Trial Inference Script for OCTO Model

This script validates that OCTO can be loaded and run successfully
on a small sample from the Open X-Embodiment dataset (Bridge V2).

It serves as a quick sanity check before running full benchmark evaluations.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add OCTO to path
OCTO_PATH = Path(__file__).parent.parent.parent / "models" / "octo"
sys.path.insert(0, str(OCTO_PATH))

try:
    import numpy as np
    import jax
    import matplotlib.pyplot as plt
    from PIL import Image
    import requests
    from octo.model.octo_model import OctoModel
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease install dependencies:")
    print("  cd models/octo && pip install -r requirements.txt")
    sys.exit(1)


def download_sample_image(save_dir="./temp"):
    """Download a sample image from Bridge V2 dataset"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Sample image URL from BridgeV2
    IMAGE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
    
    print(f"Downloading sample image from Bridge V2...")
    try:
        response = requests.get(IMAGE_URL, stream=True, timeout=30)
        response.raise_for_status()
        
        img = Image.open(response.raw).resize((256, 256))
        img_array = np.array(img)
        
        # Save for inspection
        img_path = os.path.join(save_dir, "sample_image.jpg")
        img.save(img_path)
        print(f"✓ Image downloaded and saved to {img_path}")
        
        return img_array
    except Exception as e:
        print(f"✗ Failed to download image: {e}")
        print("Creating a synthetic image for testing...")
        # Create a synthetic RGB image
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        return img_array


def load_octo_model(model_name="hf://rail-berkeley/octo-small-1.5"):
    """Load OCTO model from HuggingFace"""
    print(f"\nLoading OCTO model: {model_name}")
    print("This may take a few minutes on first run...")
    
    start_time = time.time()
    try:
        model = OctoModel.load_pretrained(model_name)
        load_time = time.time() - start_time
        
        print(f"✓ Model loaded successfully in {load_time:.2f}s")
        print(f"\nModel Specification:")
        print(model.get_pretty_spec())
        
        return model, load_time
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise


def run_inference(model, image, language_instruction="pick up the fork"):
    """Run inference on a single image"""
    print(f"\n Running inference...")
    print(f"  Language instruction: '{language_instruction}'")
    
    try:
        # Prepare observation (add batch + time dimensions)
        img_input = image[np.newaxis, np.newaxis, ...]  # [1, 1, 256, 256, 3]
        observation = {
            "image_primary": img_input,
            "timestep_pad_mask": np.array([[True]])
        }
        
        # Create task from language instruction
        task = model.create_tasks(texts=[language_instruction])
        
        # Sample actions
        start_time = time.time()
        action = model.sample_actions(
            observation,
            task,
            unnormalization_statistics=model.dataset_statistics.get("bridge_dataset", {}).get("action"),
            rng=jax.random.PRNGKey(0)
        )
        inference_time = time.time() - start_time
        
        print(f"✓ Inference completed in {inference_time*1000:.2f}ms")
        print(f"\nAction output shape: {action.shape}")
        print(f"Action values:\n{action}")
        
        return {
            "action": action.tolist() if hasattr(action, 'tolist') else action,
            "action_shape": list(action.shape),
            "inference_time_ms": inference_time * 1000,
            "language_instruction": language_instruction
        }
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        raise


def save_results(results, output_dir="../../results/octo/trial"):
    """Save trial results to JSON"""
    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"trial_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    return results_file


def main():
    print("="*70)
    print("OCTO Model Trial Inference Script")
    print("="*70)
    
    # Configuration
    model_name = "hf://rail-berkeley/octo-small-1.5"  # 27M parameters
    language_instructions = [
        "pick up the fork",
        "open the drawer",
        "close the drawer"
    ]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "test_type": "trial_inference",
        "success": False
    }
    
    try:
        # Step 1: Download sample image
        image = download_sample_image()
        results["image_shape"] = list(image.shape)
        
        # Step 2: Load OCTO model
        model, load_time = load_octo_model(model_name)
        results["model_load_time_s"] = load_time
        
        # Step 3: Run inference on multiple instructions
        inference_results = []
        for instruction in language_instructions:
            result = run_inference(model, image, instruction)
            inference_results.append(result)
        
        results["inference_results"] = inference_results
        results["success"] = True
        
        # Calculate summary statistics
        avg_inference_time = np.mean([r["inference_time_ms"] for r in inference_results])
        results["average_inference_time_ms"] = avg_inference_time
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"✓ Model: {model_name}")
        print(f"✓ Model load time: {load_time:.2f}s")
        print(f"✓ Average inference time: {avg_inference_time:.2f}ms")
        print(f"✓ Tested {len(language_instructions)} language instructions")
        print(f"✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Trial failed with error: {e}")
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
    
    # Save results
    results_file = save_results(results)
    
    if results["success"]:
        print("\n🎉 OCTO model is working correctly!")
        print("   You can now proceed with full benchmark evaluation.")
        return 0
    else:
        print("\n❌ Trial inference failed. Please check the error above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
