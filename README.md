# Universal Image Model Worker

A plug-and-play RunPod template that supports any image generation model from HuggingFace and CivitAI through a simple `MODEL_ID` environment variable.

## Features

✅ **Universal Model Support**
- HuggingFace Diffusers pipelines
- LoRA adapters (with automatic base model detection)
- Checkpoint files (.ckpt) 
- Safetensors files
- CivitAI model URLs
- Local model files

✅ **Smart Model Detection**
- Automatic model type detection
- Intelligent LoRA base model inference
- Fallback strategies for unknown models

✅ **Enhanced Performance**
- File-based metadata caching
- Memory-optimized loading
- Comprehensive error handling

✅ **Developer-Friendly API**
- Dynamic schema generation
- Detailed error messages
- Enhanced response metadata

## Quick Start

### 1. Set Model ID
```bash
export MODEL_ID="stabilityai/stable-diffusion-2-1"
```

### 2. Run the Worker
```bash
python main.py
```

## Supported Model Types

### Standard Diffusers Models
```bash
MODEL_ID="stabilityai/stable-diffusion-2-1"
MODEL_ID="runwayml/stable-diffusion-v1-5" 
MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0"
```

### LoRA Models
```bash
MODEL_ID="some-user/my-lora-model"  # Base model auto-detected
```

### Checkpoint Files
```bash
MODEL_ID="https://civitai.com/api/download/models/123456"
MODEL_ID="/path/to/local/model.ckpt"
MODEL_ID="username/repo-name"  # HF repo with .ckpt file
```

### Safetensors Files
```bash
MODEL_ID="https://example.com/model.safetensors"
MODEL_ID="/path/to/local/model.safetensors"
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_ID` | Model identifier (HF repo, URL, or file path) | `stabilityai/stable-diffusion-2-1` |
| `HF_TOKEN` | HuggingFace access token (for private models) | None |
| `CACHE_DIR` | Directory for model metadata cache | `/tmp/model_cache` |

## API Request Format

```json
{
  "prompt": "A beautiful sunset over mountains",
  "negative_prompt": "blurry, low quality",
  "height": 512,
  "width": 512,
  "num_inference_steps": 20,
  "guidance_scale": 7.5,
  "num_images_per_prompt": 1,
  "seed": 42,
  "lora_scale": 1.0
}
```

## API Response Format

```json
{
  "model_id": "stabilityai/stable-diffusion-2-1",
  "model_type": "diffusers_pipeline",
  "base_model": null,
  "seed": 42,
  "num_outputs": 1,
  "outputs": [
    {
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
      "seed": 42,
      "width": 512,
      "height": 512,
      "format": "PNG"
    }
  ],
  "cache_stats": {
    "cached_models": 3,
    "cache_size_mb": 0.1,
    "cache_dir": "/tmp/model_cache"
  }
}
```

## Model-Specific Features

### LoRA Models
- Automatic base model detection and loading
- `lora_scale` parameter for controlling adapter strength (0.0-2.0)
- Support for popular LoRA repositories

### Checkpoint Models  
- Automatic conversion to Diffusers format
- Security validation for .ckpt files
- Metadata extraction

### Error Handling
- GPU out of memory detection with helpful messages
- Invalid parameter validation
- Model loading failure recovery
- Network error handling

## Docker Usage

```dockerfile
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .
ENV MODEL_ID="stabilityai/stable-diffusion-2-1"

CMD ["python", "main.py"]
```

## Caching

The worker caches model metadata to improve startup times:
- Cache location: `/tmp/model_cache` (configurable)
- Cache duration: 24 hours
- Automatic cache cleanup
- Cache statistics in API responses

## Troubleshooting

### Common Issues

**Model not loading:**
- Check if MODEL_ID is correct
- Verify HF_TOKEN if using private models
- Check internet connection for URL-based models

**GPU out of memory:**
- Reduce image dimensions (height/width)
- Decrease num_images_per_prompt
- Use CPU inference (automatic fallback)

**LoRA not working:**
- Verify LoRA model format
- Check if base model is compatible
- Adjust lora_scale parameter

### Debug Information
The worker provides detailed logging:
- Model detection process
- Loading steps and timings
- Cache hit/miss information
- Parameter validation results

## Performance Tips

1. **Use caching**: Keep CACHE_DIR persistent for faster restarts
2. **Optimize dimensions**: Use multiples of 8 for height/width
3. **Batch requests**: Use num_images_per_prompt for multiple images
4. **Monitor memory**: Check logs for memory usage patterns

## Contributing

This is a plug-and-play template designed for RunPod deployment. The code is self-contained with no external database dependencies.

## License

MIT License - Use freely for commercial and non-commercial projects.