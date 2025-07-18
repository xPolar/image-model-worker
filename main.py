"""
RunPod serverless worker for Universal Image Generation Models
Supports: HuggingFace Diffusers, LoRA, Checkpoints, CivitAI models
"""
import os, tempfile, json, inspect, base64, io, re, time, hashlib
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from PIL import Image
import runpod
from runpod.serverless.utils.rp_validator import validate
from huggingface_hub import HfApi, model_info, hf_hub_download
from pathlib import Path
import requests
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------
# 0.  Configuration and Cache Setup
# ---------------------------------------------------------------------
MODEL_ID  = os.getenv("MODEL_ID") or "stabilityai/stable-diffusion-2-1"
HF_TOKEN  = os.getenv("HF_TOKEN")        # optional private models
DTYPE     = torch.float16 if torch.cuda.is_available() else torch.float32
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/model_cache")

# Ensure cache directory exists
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

class ModelCache:
    """Simple file-based model metadata caching"""
    
    @staticmethod
    def get_cache_path(model_id: str) -> str:
        """Generate cache file path for model metadata"""
        # Create a safe filename from model ID
        safe_id = hashlib.md5(model_id.encode()).hexdigest()
        return os.path.join(CACHE_DIR, f"model_{safe_id}.json")
    
    @staticmethod
    def save_model_info(model_id: str, model_type: str, metadata: Dict, base_model: Optional[str] = None):
        """Save model information to cache"""
        cache_path = ModelCache.get_cache_path(model_id)
        cache_data = {
            "model_id": model_id,
            "model_type": model_type,
            "metadata": metadata,
            "base_model": base_model,
            "timestamp": time.time(),
            "cache_version": "1.0"
        }
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            print(f"📁 Cached model info: {cache_path}")
        except Exception as e:
            print(f"⚠️ Failed to cache model info: {e}")
    
    @staticmethod
    def load_model_info(model_id: str) -> Optional[Tuple[str, Dict, Optional[str]]]:
        """Load model information from cache"""
        cache_path = ModelCache.get_cache_path(model_id)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is not too old (24 hours)
            if time.time() - cache_data.get("timestamp", 0) > 86400:
                print(f"⏰ Cache expired for {model_id}")
                return None
            
            print(f"📁 Loaded cached info for {model_id}")
            return cache_data["model_type"], cache_data["metadata"], cache_data.get("base_model")
            
        except Exception as e:
            print(f"⚠️ Failed to load cached model info: {e}")
            return None
    
    @staticmethod
    def get_cache_stats() -> Dict:
        """Get cache statistics"""
        if not os.path.exists(CACHE_DIR):
            return {"cached_models": 0, "cache_size_mb": 0}
        
        try:
            files = [f for f in os.listdir(CACHE_DIR) if f.startswith("model_") and f.endswith(".json")]
            total_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in files)
            
            return {
                "cached_models": len(files),
                "cache_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_dir": CACHE_DIR
            }
        except Exception as e:
            print(f"⚠️ Failed to get cache stats: {e}")
            return {"cached_models": 0, "cache_size_mb": 0, "error": str(e)}

class ModelDetector:
    """Detects model type and determines appropriate loading strategy"""
    
    @staticmethod
    def detect_model_type(model_id: str) -> Tuple[str, Dict]:
        """Detect model type from MODEL_ID
        Returns: (model_type, metadata)
        """
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
            api = HfApi()
            repo_info = api.model_info(model_id, token=HF_TOKEN)
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

    @staticmethod
    def detect_lora_base_model(model_id: str, metadata: Dict) -> Optional[str]:
        """Detect base model for LoRA adapters"""
        try:
            # Try to get model card or readme for base model info
            api = HfApi()
            try:
                readme = api.model_info(model_id, token=HF_TOKEN).cardData
                if readme and 'base_model' in readme:
                    return readme['base_model']
            except:
                pass
            
            # Common base models for different LoRA types
            common_bases = {
                'sd15': 'runwayml/stable-diffusion-v1-5',
                'sd1.5': 'runwayml/stable-diffusion-v1-5', 
                'sdxl': 'stabilityai/stable-diffusion-xl-base-1.0',
                'sd21': 'stabilityai/stable-diffusion-2-1'
            }
            
            # Check model name for hints
            model_name = model_id.lower()
            for key, base in common_bases.items():
                if key in model_name:
                    return base
            
            # Default to SD 1.5 for unknown LoRAs
            return 'runwayml/stable-diffusion-v1-5'
            
        except Exception as e:
            print(f"⚠️ Could not detect base model: {e}")
            return 'runwayml/stable-diffusion-v1-5'

# Try to load from cache first
cached_info = ModelCache.load_model_info(MODEL_ID)
if cached_info:
    model_type, model_metadata, cached_base_model = cached_info
    print(f"📁 Using cached model info: {model_type}")
else:
    # Detect model type and cache it
    model_type, model_metadata = ModelDetector.detect_model_type(MODEL_ID)
    print(f"🔍 Detected model type: {model_type}")
    print(f"📋 Model metadata: {model_metadata}")
    
    cached_base_model = None
    if model_type in ['lora_diffusers', 'lora_generic']:
        cached_base_model = ModelDetector.detect_lora_base_model(MODEL_ID, model_metadata)
    
    # Cache the detected information
    ModelCache.save_model_info(MODEL_ID, model_type, model_metadata, cached_base_model)

# Load model based on detected type
PIPE = None
base_model_id = None

try:
    if model_type == 'diffusers_pipeline':
        print(f"📦 Loading Diffusers pipeline: {MODEL_ID}")
        PIPE = DiffusionPipeline.from_pretrained(
                   MODEL_ID,
                   torch_dtype=DTYPE,
                   use_safetensors=True,
                   token=HF_TOKEN
               ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    elif model_type in ['lora_diffusers', 'lora_generic']:
        print(f"🎯 Loading LoRA model: {MODEL_ID}")
        # Use cached base model or detect it
        base_model_id = cached_base_model or ModelDetector.detect_lora_base_model(MODEL_ID, model_metadata)
        print(f"📦 Loading base model: {base_model_id}")
        
        # Load base model first
        PIPE = DiffusionPipeline.from_pretrained(
                   base_model_id,
                   torch_dtype=DTYPE,
                   use_safetensors=True,
                   token=HF_TOKEN
               ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load LoRA weights
        print(f"🔗 Loading LoRA weights from: {MODEL_ID}")
        PIPE.load_lora_weights(MODEL_ID, token=HF_TOKEN)
    
    elif model_type in ['checkpoint_hf', 'safetensors_hf']:
        print(f"💾 Loading checkpoint/safetensors model: {MODEL_ID}")
        # Use single file loading
        file_name = model_metadata.get('checkpoint_file') or model_metadata.get('safetensors_file')
        PIPE = StableDiffusionPipeline.from_single_file(
                   hf_hub_download(MODEL_ID, file_name, token=HF_TOKEN),
                   torch_dtype=DTYPE,
                   use_safetensors=file_name.endswith('.safetensors')
               ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    elif model_type in ['checkpoint_url', 'safetensors_url']:
        print(f"🌐 Loading model from URL: {MODEL_ID}")
        # Download file first
        response = requests.get(MODEL_ID)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(suffix='.safetensors' if 'safetensors' in model_type else '.ckpt', delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            
            PIPE = StableDiffusionPipeline.from_single_file(
                       tmp_path,
                       torch_dtype=DTYPE,
                       use_safetensors=model_type == 'safetensors_url'
                   ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Cleanup
            os.unlink(tmp_path)
        else:
            raise ValueError(f"Could not download model from {MODEL_ID}")
    
    elif model_type in ['checkpoint_file', 'safetensors_file']:
        print(f"📁 Loading local file: {MODEL_ID}")
        PIPE = StableDiffusionPipeline.from_single_file(
                   MODEL_ID,
                   torch_dtype=DTYPE,
                   use_safetensors=model_type == 'safetensors_file'
               ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    else:
        print(f"⚠️ Unknown model type {model_type}, trying as Diffusers pipeline")
        PIPE = DiffusionPipeline.from_pretrained(
                   MODEL_ID,
                   torch_dtype=DTYPE,
                   use_safetensors=True,
                   token=HF_TOKEN
               ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configure pipeline
    PIPE.set_progress_bar_config(disable=True)
    
    # Validate this is an image generation pipeline
    if not hasattr(PIPE, '__call__'):
        raise ValueError(f"Pipeline {MODEL_ID} is not callable")
    
    # Test run to ensure it produces images (with error handling)
    try:
        test_output = PIPE("test", num_inference_steps=1, guidance_scale=1.0, height=64, width=64)
        if not isinstance(test_output.images[0], Image.Image):
            raise ValueError(f"Pipeline {MODEL_ID} does not generate PIL Images - this worker only supports image generation models")
    except Exception as test_error:
        print(f"⚠️ Test run failed, but continuing: {test_error}")
        # Continue anyway as some models may need specific parameters
        
except Exception as e:
    print(f"❌ Failed to load image generation model {MODEL_ID}: {e}")
    raise

# Print loading success and cache stats
print(f"✅ Loaded image generation model: {MODEL_ID} (type: {model_type})")
if base_model_id:
    print(f"✅ Base model: {base_model_id}")
print(f"✅ Device: {PIPE.device}")
print(f"✅ Dtype: {DTYPE}")

# Print cache statistics
cache_stats = ModelCache.get_cache_stats()
print(f"📁 Cache stats: {cache_stats['cached_models']} models, {cache_stats['cache_size_mb']} MB")

# ---------------------------------------------------------------------
# 1.  Enhanced dynamic schema generation
# ---------------------------------------------------------------------
class SchemaGenerator:
    """Enhanced schema generation for different model types"""
    
    @staticmethod
    def generate_schema(pipeline, model_type: str) -> Dict:
        """Generate comprehensive schema based on pipeline and model type"""
        print(f"📝 Generating schema for {model_type} pipeline")
        
        sig = inspect.signature(pipeline.__call__)
        schema = {}
        
        for name, param in sig.parameters.items():
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                schema_entry = SchemaGenerator._analyze_parameter(name, param, model_type)
                if schema_entry:
                    schema[name] = schema_entry
        
        # Add model-specific enhancements
        schema = SchemaGenerator._enhance_schema(schema, model_type, pipeline)
        
        return schema
    
    @staticmethod
    def _analyze_parameter(name: str, param, model_type: str) -> Optional[Dict]:
        """Analyze individual parameter and determine its properties"""
        # Skip certain parameters
        if name in ['self', 'args', 'kwargs']:
            return None
            
        # Determine parameter type
        param_type = SchemaGenerator._determine_type(name, param)
        
        # Create base schema entry
        schema_entry = {
            "type": param_type,
            "required": param.default is param.empty,
            "default": None if param.default is param.empty else param.default
        }
        
        # Add parameter-specific metadata
        schema_entry.update(SchemaGenerator._get_parameter_metadata(name, param_type))
        
        return schema_entry
    
    @staticmethod
    def _determine_type(name: str, param):
        """Determine parameter type with enhanced logic"""
        # Check annotation first
        if hasattr(param, 'annotation') and param.annotation != param.empty:
            if param.annotation is int:
                return int
            elif param.annotation is float:
                return float
            elif param.annotation is str:
                return str
            elif param.annotation is bool:
                return bool
            elif hasattr(param.annotation, '__origin__'):
                # Handle List, Optional, Union types
                if param.annotation.__origin__ is list:
                    return list
        
        # Check default value type
        if hasattr(param, 'default') and param.default is not param.empty:
            return type(param.default)
        
        # Parameter name-based type inference
        int_params = {
            'height', 'width', 'num_inference_steps', 'num_images_per_prompt',
            'seed', 'steps', 'batch_size', 'num_images', 'clip_skip'
        }
        float_params = {
            'guidance_scale', 'strength', 'noise_level', 'controlnet_conditioning_scale',
            'aesthetic_score', 'negative_aesthetic_score', 'cfg_scale', 'denoising_strength'
        }
        bool_params = {
            'return_dict', 'output_type', 'do_classifier_free_guidance'
        }
        
        if name in int_params:
            return int
        elif name in float_params:
            return float
        elif name in bool_params:
            return bool
        elif name in ['prompt', 'negative_prompt']:
            return str
        elif 'image' in name:
            return 'image'  # Special type for images
        
        # Default to string
        return str
    
    @staticmethod
    def _get_parameter_metadata(name: str, param_type) -> Dict:
        """Get additional metadata for parameters"""
        metadata = {}
        
        # Add descriptions and constraints
        param_info = {
            'prompt': {
                'description': 'Text prompt for image generation',
                'example': 'A beautiful sunset over mountains'
            },
            'negative_prompt': {
                'description': 'Negative text prompt to avoid certain features', 
                'example': 'blurry, low quality, distorted'
            },
            'height': {
                'description': 'Height of generated image in pixels',
                'min': 64, 'max': 2048, 'default': 512
            },
            'width': {
                'description': 'Width of generated image in pixels',
                'min': 64, 'max': 2048, 'default': 512
            },
            'num_inference_steps': {
                'description': 'Number of denoising steps',
                'min': 1, 'max': 150, 'default': 20
            },
            'guidance_scale': {
                'description': 'How closely to follow the prompt',
                'min': 0.0, 'max': 30.0, 'default': 7.5
            },
            'num_images_per_prompt': {
                'description': 'Number of images to generate per prompt',
                'min': 1, 'max': 10, 'default': 1
            },
            'strength': {
                'description': 'How much to transform the input image (img2img)',
                'min': 0.0, 'max': 1.0, 'default': 0.8
            }
        }
        
        if name in param_info:
            metadata.update(param_info[name])
        
        return metadata
    
    @staticmethod
    def _enhance_schema(schema: Dict, model_type: str, pipeline) -> Dict:
        """Add model-type specific enhancements to schema"""
        # Ensure prompt is always required
        if 'prompt' in schema:
            schema['prompt']['required'] = True
        
        # Add seed parameter if missing (most models support it)
        if 'seed' not in schema:
            schema['seed'] = {
                'type': int,
                'required': False,
                'default': None,
                'description': 'Random seed for reproducible generation',
                'min': 0,
                'max': 2**32 - 1
            }
        
        # Model-specific enhancements
        if model_type in ['lora_diffusers', 'lora_generic']:
            # LoRA models might have additional scaling parameters
            if hasattr(pipeline, 'set_adapters') or hasattr(pipeline, 'set_lora_scale'):
                schema['lora_scale'] = {
                    'type': float,
                    'required': False,
                    'default': 1.0,
                    'description': 'LoRA adapter strength',
                    'min': 0.0,
                    'max': 2.0
                }
        
        return schema

# Generate schema for the loaded pipeline
SIG = inspect.signature(PIPE.__call__)
INPUT_SCHEMA = SchemaGenerator.generate_schema(PIPE, model_type)

print(f"📝 Generated schema with {len(INPUT_SCHEMA)} parameters:")
for param_name, param_info in INPUT_SCHEMA.items():
    req_str = "required" if param_info.get('required') else "optional"
    print(f"  - {param_name} ({param_info['type'].__name__ if hasattr(param_info['type'], '__name__') else param_info['type']}) - {req_str}")

# ---------------------------------------------------------------------
class InferenceHandler:
    """Enhanced inference handler with better error handling"""
    
    @staticmethod
    def validate_input(payload: Dict) -> Tuple[bool, Dict, list]:
        """Enhanced input validation"""
        validated = validate(payload, INPUT_SCHEMA)
        
        if "errors" in validated:
            return False, {}, validated["errors"]
        
        args = validated["validated_input"]
        errors = []
        
        # Additional validation
        if 'height' in args and 'width' in args:
            # Ensure dimensions are multiples of 8 (common requirement)
            if args['height'] % 8 != 0:
                args['height'] = (args['height'] // 8) * 8
                print(f"⚠️ Adjusted height to {args['height']} (must be multiple of 8)")
            
            if args['width'] % 8 != 0:
                args['width'] = (args['width'] // 8) * 8
                print(f"⚠️ Adjusted width to {args['width']} (must be multiple of 8)")
        
        return True, args, errors
    
    @staticmethod
    def handle_inference(args: Dict, seed: int) -> Tuple[bool, any, str]:
        """Handle the actual inference with error handling"""
        try:
            print(f"🎨 Starting inference with seed {seed}")
            print(f"📋 Parameters: {list(args.keys())}")
            
            result = PIPE(**args)
            return True, result, None
            
        except torch.cuda.OutOfMemoryError as e:
            error_msg = f"GPU out of memory. Try reducing image size or batch size. Error: {str(e)}"
            print(f"❌ {error_msg}")
            return False, None, error_msg
            
        except ValueError as e:
            error_msg = f"Invalid parameter values: {str(e)}"
            print(f"❌ {error_msg}")
            return False, None, error_msg
            
        except Exception as e:
            error_msg = f"Inference failed: {str(e)}"
            print(f"❌ {error_msg}")
            return False, None, error_msg
    
    @staticmethod
    def process_output(result, seed: int) -> Tuple[bool, list, str]:
        """Process and validate output"""
        try:
            if not hasattr(result, 'images') or not result.images:
                return False, [], "Pipeline did not return images"
            
            outputs = []
            for idx, image in enumerate(result.images):
                if not isinstance(image, Image.Image):
                    return False, [], f"Output {idx} is not a PIL Image"
                
                # Convert PIL Image to base64
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                outputs.append({
                    "image_base64": img_base64,
                    "seed": seed + idx,
                    "width": image.width,
                    "height": image.height,
                    "format": "PNG"
                })
            
            return True, outputs, None
            
        except Exception as e:
            return False, [], f"Failed to process output: {str(e)}"

def handler(job):
    """Main handler function"""
    try:
        payload = job["input"]
        print(f"📨 Received job with payload keys: {list(payload.keys())}")
        
        # Validate input
        is_valid, args, errors = InferenceHandler.validate_input(payload)
        if not is_valid:
            return {"errors": errors}
        
        # Enhanced seed handling
        seed = payload.get("seed")
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big") % (2**32)  # Use 32-bit seed
        
        if "generator" in SIG.parameters:
            args["generator"] = torch.Generator(device=PIPE.device).manual_seed(seed)
        
        # LoRA scale handling if applicable
        if model_type in ['lora_diffusers', 'lora_generic'] and 'lora_scale' in payload:
            lora_scale = payload['lora_scale']
            if hasattr(PIPE, 'set_lora_scale'):
                PIPE.set_lora_scale(lora_scale)
            # Remove from args as it's handled separately
            args.pop('lora_scale', None)
        
        # Run inference
        success, result, error_msg = InferenceHandler.handle_inference(args, seed)
        if not success:
            return {"errors": [error_msg]}
        
        # Process output
        success, outputs, error_msg = InferenceHandler.process_output(result, seed)
        if not success:
            return {"errors": [error_msg]}
        
        # Return successful result with cache stats
        return {
            "model_id": MODEL_ID,
            "model_type": model_type,
            "base_model": base_model_id,
            "seed": seed,
            "outputs": outputs,
            "num_outputs": len(outputs),
            "cache_stats": ModelCache.get_cache_stats()
        }
        
    except Exception as e:
        error_msg = f"Handler failed: {str(e)}"
        print(f"❌ {error_msg}")
        return {"errors": [error_msg]}


# fire it up
runpod.serverless.start({"handler": handler})
