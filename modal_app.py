from pathlib import Path

import modal

app = modal.App("text-to-image-fastapi-flux")

volume = modal.Volume.from_name("text-to-image-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands("python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision")
    .pip_install_from_requirements("modal_requirements.txt")
    .add_local_dir("static", remote_path="/root/static")
    .add_local_file("app.py", remote_path="/root/app.py")
    .add_local_file("z_image_generate.py", remote_path="/root/z_image_generate.py")
)


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 30,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.asgi_app()
def fastapi_app():
    import os

    os.environ["GENERATED_IMAGE_DIR"] = "/data/flux_generated_images"
    os.environ["GENERATION_DEVICE"] = "cuda"
    os.environ["FORCE_MODEL_REDOWNLOAD_ONCE"] = "1"
    os.environ["MODEL_REDOWNLOAD_MARKER"] = "/data/.model_redownload_done_flux_v1"
    os.environ["MODEL_CACHE_DIR"] = "/data/model_cache_flux"
    from app import app as fastapi_instance

    Path("/data/flux_generated_images").mkdir(parents=True, exist_ok=True)
    return fastapi_instance


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 10,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def check_model_cache(model_id: str = "black-forest-labs/FLUX.1-dev"):
    from pathlib import Path

    from huggingface_hub import snapshot_download

    model_path = Path(
        snapshot_download(
            repo_id=model_id,
            local_files_only=False,
        )
    )
    files = [p for p in model_path.rglob("*") if p.is_file()]
    safetensors = [p for p in files if p.suffix == ".safetensors"]
    bin_files = [p for p in files if p.suffix == ".bin"]
    total_bytes = sum(p.stat().st_size for p in files)
    return {
        "model_id": model_id,
        "model_path": str(model_path),
        "file_count": len(files),
        "safetensors_count": len(safetensors),
        "bin_count": len(bin_files),
        "total_size_gb": round(total_bytes / (1024**3), 3),
        "has_model_index_json": (model_path / "model_index.json").exists(),
    }
