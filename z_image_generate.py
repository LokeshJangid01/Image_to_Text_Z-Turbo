import argparse
import os
import shutil
import traceback
from pathlib import Path

import torch
from diffusers import DiffusionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an image with FLUX.1-dev.")
    parser.add_argument(
        "--prompt",
        default=(
            "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, "
            "red floral forehead pattern. Elaborate high bun, golden phoenix headdress, "
            "red flowers, beads. Holds round folding fan with lady, trees, bird. "
            "Neon lightning-bolt lamp, bright yellow glow, above extended left palm. "
            "Soft-lit outdoor night background, silhouetted tiered pagoda, "
            "blurred colorful distant lights."
        ),
        help="Text prompt for image generation.",
    )
    parser.add_argument("--height", type=int, default=512, help="Image height.")
    parser.add_argument("--width", type=int, default=512, help="Image width.")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps.")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Execution device.",
    )
    parser.add_argument("--output", default="example.png", help="Output image path.")
    parser.add_argument(
        "--model-id",
        default="black-forest-labs/FLUX.1-dev",
        help="Hugging Face model ID (e.g., black-forest-labs/FLUX.1-dev).",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("MODEL_CACHE_DIR", ".hf_cache"),
        help="Directory for Hugging Face model cache/downloads.",
    )
    parser.add_argument("--cpu-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--gpu-offload",
        action="store_true",
        help="Enable model CPU offload on CUDA to reduce VRAM usage.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Delete local cache dir and force re-download model files.",
    )
    return parser


def resolve_dtype(device: str, cpu_dtype: str) -> torch.dtype:
    if device == "cuda":
        return torch.float16
    return torch.float32 if cpu_dtype == "float32" else torch.bfloat16


def main() -> None:
    args = build_parser().parse_args()
    model_id = args.model_id

    cache_dir = Path(args.cache_dir).resolve()
    if args.force_redownload and cache_dir.exists():
        print(f"Force redownload enabled. Deleting cache dir: {cache_dir}", flush=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
    hub_cache_dir = cache_dir / "hub"
    transformers_cache_dir = cache_dir / "transformers"
    cache_dir.mkdir(parents=True, exist_ok=True)
    hub_cache_dir.mkdir(parents=True, exist_ok=True)
    transformers_cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache_dir))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")

    dtype = resolve_dtype(args.device, args.cpu_dtype)
    print(f"Loading model: {model_id}", flush=True)
    print(f"Device: {args.device} | dtype: {dtype}", flush=True)
    print(f"Image: {args.width}x{args.height} | Steps: {args.steps} | Guidance: {args.guidance}", flush=True)

    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=str(hub_cache_dir),
            torch_dtype=dtype,
            force_download=args.force_redownload,
            use_safetensors=True,
        )
    except Exception:
        if args.device == "cpu" and dtype == torch.bfloat16:
            print("bfloat16 load failed on CPU, retrying with float32...", flush=True)
            dtype = torch.float32
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                cache_dir=str(hub_cache_dir),
                torch_dtype=dtype,
                force_download=args.force_redownload,
                use_safetensors=True,
            )
        else:
            raise

    if args.device == "cuda":
        if args.gpu_offload:
            # More aggressive than model_cpu_offload for tight VRAM (e.g. T4).
            if hasattr(pipe, "enable_sequential_cpu_offload"):
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.enable_model_cpu_offload()
        else:
            pipe.to(args.device)
    else:
        pipe.to(args.device)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "vae") and pipe.vae is not None:
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    try:
        with torch.inference_mode():
            print("Starting generation...", flush=True)
            image = pipe(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
            ).images[0]
    except torch.OutOfMemoryError:
        if args.device != "cuda":
            raise
        print("CUDA OOM detected. Retrying with lower-memory settings...", flush=True)
        torch.cuda.empty_cache()
        retry_width = max(512, (args.width // 2 // 8) * 8)
        retry_height = max(512, (args.height // 2 // 8) * 8)
        retry_steps = min(args.steps, 20)
        with torch.inference_mode():
            image = pipe(
                prompt=args.prompt,
                height=retry_height,
                width=retry_width,
                num_inference_steps=retry_steps,
                guidance_scale=args.guidance,
                generator=generator,
            ).images[0]

    output_path = Path(args.output)
    image.save(output_path)
    print(f"Saved image to: {output_path.resolve()}")
    print(f"Cache dir used: {cache_dir}")
    print(f"Device used: {args.device}")
    print(f"Dtype used: {dtype}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
        raise SystemExit(130)
    except Exception:
        print("\nGeneration failed with an exception:")
        traceback.print_exc()
        raise SystemExit(1)
    except BaseException:
        print("\nGeneration ended due to a non-standard fatal error:")
        traceback.print_exc()
        raise SystemExit(1)
