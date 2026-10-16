import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
IMAGE_DIR = Path(os.getenv("GENERATED_IMAGE_DIR", str(BASE_DIR / "z_generated_images")))
PYTHON_EXE = BASE_DIR / "env" / "python.exe"
FORCE_MODEL_REDOWNLOAD_ONCE = os.getenv("FORCE_MODEL_REDOWNLOAD_ONCE", "0") == "1"
MODEL_REDOWNLOAD_MARKER = Path(os.getenv("MODEL_REDOWNLOAD_MARKER", str(BASE_DIR / ".model_redownload_done")))

STATIC_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Z Image FastAPI")
app.mount("/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt for image generation")
    height: int = 512
    width: int = 512
    steps: int = 28
    guidance: float = 3.5
    seed: int = 42
    device: str = os.getenv("GENERATION_DEVICE", "cpu")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/generate")
def create_image(payload: GenerateRequest) -> dict:
    python_exe = PYTHON_EXE if PYTHON_EXE.exists() else Path(sys.executable)

    filename = f"zimg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.png"
    output_path = IMAGE_DIR / filename

    command = [
        str(python_exe),
        str(BASE_DIR / "z_image_generate.py"),
        "--prompt",
        payload.prompt.strip(),
        "--height",
        str(payload.height),
        "--width",
        str(payload.width),
        "--steps",
        str(payload.steps),
        "--guidance",
        str(payload.guidance),
        "--seed",
        str(payload.seed),
        "--device",
        payload.device,
        "--output",
        str(output_path),
    ]
    if payload.device == "cuda":
        command.append("--gpu-offload")
    force_redownload_this_run = FORCE_MODEL_REDOWNLOAD_ONCE and not MODEL_REDOWNLOAD_MARKER.exists()
    if force_redownload_this_run:
        command.append("--force-redownload")

    try:
        proc = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start generation: {exc}") from exc

    if proc.returncode != 0:
        error_text = (proc.stderr or proc.stdout or "Unknown generation error").strip()
        raise HTTPException(status_code=500, detail=error_text)

    if force_redownload_this_run:
        MODEL_REDOWNLOAD_MARKER.parent.mkdir(parents=True, exist_ok=True)
        MODEL_REDOWNLOAD_MARKER.write_text("done", encoding="utf-8")

    return {
        "message": "Image generated successfully",
        "filename": filename,
        "image_url": f"/images/{filename}",
    }
