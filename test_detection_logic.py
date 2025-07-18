#!/usr/bin/env python3
"""
Test just the detection logic without dependencies
"""
import os

# Mock the dependencies
class MockHfApi:
    def model_info(self, model_id, token=None):
        # Mock different model types for testing
        class MockRepoInfo:
            def __init__(self, files):
                self.siblings = [MockFile(f) for f in files]
        
        class MockFile:
            def __init__(self, name):
                self.rfilename = name
        
        # Mock responses for different model types
        if "stable-diffusion" in model_id and "lora" not in model_id:
            return MockRepoInfo(['model_index.json', 'text_encoder/config.json', 'unet/config.json', 'vae/config.json'])
        elif "lora" in model_id:
            return MockRepoInfo(['pytorch_lora_weights.safetensors', 'README.md'])
        else:
            return MockRepoInfo(['model.safetensors', 'config.json'])

def mock_detect_model_type(model_id: str):
    """Simplified model detection logic"""
    print(f"🔍 Detecting model type for: {model_id}")
    
    # Check if it's a direct URL or file path
    if model_id.startswith(('http://', 'https://')):
        if 'civitai.com' in model_id:
            return 'civitai_url', {'url': model_id}
        elif model_id.endswith('.ckpt'):
            return 'checkpoint_url', {'url': model_id}
        elif model_id.endswith('.safetensors'):
            return 'safetensors_url', {'url': model_id}
        else:
            return 'unknown_url', {'url': model_id}
    
    # Check if it's a local file path
    if os.path.exists(model_id):
        if model_id.endswith('.ckpt'):
            return 'checkpoint_file', {'path': model_id}
        elif model_id.endswith('.safetensors'):
            return 'safetensors_file', {'path': model_id}
        else:
            return 'unknown_file', {'path': model_id}
    
    # Must be a HuggingFace model ID - check the repository
    try:
        api = MockHfApi()
        repo_info = api.model_info(model_id, token=None)
        files = [f.rfilename for f in repo_info.siblings]
        
        # Check for diffusers pipeline structure
        if 'model_index.json' in files:
            return 'diffusers_pipeline', {'repo_id': model_id, 'files': files}
        
        # Check for LoRA indicators
        if any(f.endswith('.safetensors') for f in files) and len(files) < 10:
            # Likely a LoRA - small number of files with safetensors
            if 'pytorch_lora_weights.safetensors' in files:
                return 'lora_diffusers', {'repo_id': model_id, 'files': files}
            else:
                return 'lora_generic', {'repo_id': model_id, 'files': files}
        
        # Check for single checkpoint file
        ckpt_files = [f for f in files if f.endswith('.ckpt')]
        safetensors_files = [f for f in files if f.endswith('.safetensors')]
        
        if ckpt_files:
            return 'checkpoint_hf', {'repo_id': model_id, 'files': files, 'checkpoint_file': ckpt_files[0]}
        elif safetensors_files and 'model_index.json' not in files:
            return 'safetensors_hf', {'repo_id': model_id, 'files': files, 'safetensors_file': safetensors_files[0]}
        
        # Default to trying as diffusers pipeline
        return 'diffusers_pipeline', {'repo_id': model_id, 'files': files}
        
    except Exception as e:
        print(f"⚠️ Could not access HuggingFace model info: {e}")
        # Fallback to diffusers pipeline attempt
        return 'diffusers_pipeline', {'repo_id': model_id}

def test_detection_logic():
    """Test the detection logic"""
    
    test_cases = [
        # Standard Diffusers models
        ("stabilityai/stable-diffusion-2-1", "diffusers_pipeline"),
        ("runwayml/stable-diffusion-v1-5", "diffusers_pipeline"),
        
        # LoRA models  
        ("some-user/my-lora-model", "lora_diffusers"),
        
        # URL examples
        ("https://civitai.com/api/download/models/123456", "civitai_url"),
        ("https://huggingface.co/some-model/resolve/main/model.ckpt", "checkpoint_url"),
        ("https://huggingface.co/some-model/resolve/main/model.safetensors", "safetensors_url"),
        
        # File path examples
        ("/path/to/model.ckpt", "checkpoint_file"),
        ("/path/to/model.safetensors", "safetensors_file"),
    ]
    
    print("🧪 Testing Model Detection Logic")
    print("=" * 50)
    
    for model_id, expected_type in test_cases:
        try:
            detected_type, metadata = mock_detect_model_type(model_id)
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

if __name__ == "__main__":
    test_detection_logic()