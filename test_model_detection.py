#!/usr/bin/env python3
"""
Test script for model detection functionality
"""
import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the ModelDetector class
from main import ModelDetector

def test_model_detection():
    """Test various model detection scenarios"""
    
    test_cases = [
        # Standard Diffusers models
        ("stabilityai/stable-diffusion-2-1", "diffusers_pipeline"),
        ("runwayml/stable-diffusion-v1-5", "diffusers_pipeline"),
        ("stabilityai/stable-diffusion-xl-base-1.0", "diffusers_pipeline"),
        
        # LoRA models (examples - these may not exist)
        ("some-user/my-lora-model", "diffusers_pipeline"),  # Will fallback to diffusers
        
        # URL examples
        ("https://civitai.com/api/download/models/123456", "civitai_url"),
        ("https://huggingface.co/some-model/resolve/main/model.ckpt", "checkpoint_url"),
        ("https://huggingface.co/some-model/resolve/main/model.safetensors", "safetensors_url"),
        
        # File path examples (these won't exist, but we can test the logic)
        ("/path/to/model.ckpt", "checkpoint_file"),
        ("/path/to/model.safetensors", "safetensors_file"),
    ]
    
    print("🧪 Testing Model Detection System")
    print("=" * 50)
    
    for model_id, expected_type in test_cases:
        try:
            detected_type, metadata = ModelDetector.detect_model_type(model_id)
            status = "✅" if detected_type == expected_type else "⚠️"
            print(f"{status} {model_id}")
            print(f"   Expected: {expected_type}")
            print(f"   Detected: {detected_type}")
            print(f"   Metadata: {metadata}")
            print()
        except Exception as e:
            print(f"❌ {model_id}")
            print(f"   Error: {e}")
            print()
    
    print("🧪 Testing LoRA Base Model Detection")
    print("=" * 50)
    
    lora_test_cases = [
        "some-user/sdxl-lora",
        "some-user/sd15-lora", 
        "some-user/sd21-lora",
        "some-user/unknown-lora"
    ]
    
    for lora_id in lora_test_cases:
        try:
            base_model = ModelDetector.detect_lora_base_model(lora_id, {})
            print(f"✅ {lora_id} -> {base_model}")
        except Exception as e:
            print(f"❌ {lora_id} -> Error: {e}")

if __name__ == "__main__":
    test_model_detection()