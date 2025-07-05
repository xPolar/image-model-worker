"""
RunPod serverless worker for 🤗 Diffusers IMAGE GENERATION models only
"""
import os, tempfile, json, inspect, base64, io
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import runpod
from runpod.serverless.utils.rp_validator import validate

# ---------------------------------------------------------------------
# 0.  load once per container – ΔT here amortises across jobs
# ---------------------------------------------------------------------
MODEL_ID  = os.getenv("MODEL_ID") or "stabilityai/stable-diffusion-2-1"
HF_TOKEN  = os.getenv("HF_TOKEN")        # optional private models
DTYPE     = torch.float16 if torch.cuda.is_available() else torch.float32

try:
    PIPE = DiffusionPipeline.from_pretrained(
               MODEL_ID,
               torch_dtype=DTYPE,
               use_safetensors=True,
               token=HF_TOKEN
           ).to("cuda" if torch.cuda.is_available() else "cpu")
    PIPE.set_progress_bar_config(disable=True)
    
    # Validate this is an image generation pipeline
    if not hasattr(PIPE, '__call__'):
        raise ValueError(f"Pipeline {MODEL_ID} is not callable")
        
    # Test run to ensure it produces images
    test_output = PIPE("test", num_inference_steps=1, guidance_scale=1.0, height=64, width=64)
    if not isinstance(test_output.images[0], Image.Image):
        raise ValueError(f"Pipeline {MODEL_ID} does not generate PIL Images - this worker only supports image generation models")
        
except Exception as e:
    print(f"❌ Failed to load image generation model {MODEL_ID}: {e}")
    raise

print(f"✅ Loaded image generation model: {MODEL_ID}")

# ---------------------------------------------------------------------
# 1.  dynamic schema → we discover the pipeline's __call__ signature
# ---------------------------------------------------------------------
SIG = inspect.signature(PIPE.__call__)
INPUT_SCHEMA = {}
for name, p in SIG.parameters.items():
    if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
        # Fix type detection - many parameters don't have proper annotations
        param_type = str  # Default to string
        if p.annotation is int or (hasattr(p, 'default') and isinstance(p.default, int)):
            param_type = int
        elif p.annotation is float or (hasattr(p, 'default') and isinstance(p.default, float)):
            param_type = float
        elif name in ['height', 'width', 'num_inference_steps', 'num_images_per_prompt']:
            param_type = int  # Common integer parameters
            
        INPUT_SCHEMA[name] = {"type": param_type,
                              "required": p.default is p.empty,
                              "default": None if p.default is p.empty else p.default}

# ---------------------------------------------------------------------
def handler(job):
    """Run one inference job and return result + metadata"""
    payload = job["input"]
    validated = validate(payload, INPUT_SCHEMA)
    if "errors" in validated:
        return {"errors": validated["errors"]}

    args = validated["validated_input"]

    # seed handling (works for every pipeline that accepts 'generator')
    seed = payload.get("seed", int.from_bytes(os.urandom(2), "big"))
    if "generator" in SIG.parameters:
        args["generator"] = torch.Generator(device=PIPE.device).manual_seed(seed)

    # ------- infer -------
    try:
        out = PIPE(**args)
    except Exception as e:
        return {"errors": [f"Inference failed: {str(e)}"]}

    # -------- return images as base64 -------
    if not hasattr(out, 'images') or not out.images:
        return {"errors": ["Pipeline did not return images"]}
        
    results = []
    for idx, image in enumerate(out.images):
        if not isinstance(image, Image.Image):
            return {"errors": [f"Output {idx} is not a PIL Image"]}
            
        # Convert PIL Image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        results.append({
            "image_base64": img_base64,
            "seed": seed + idx
        })

    return {"model_id": MODEL_ID, "outputs": results}

# fire it up
runpod.serverless.start({"handler": handler})
