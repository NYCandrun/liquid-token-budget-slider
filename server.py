"""
server.py — local inference backend for the LFM2.5-VL-450M browser demo.

What this does
--------------
* Serves the sibling ./web directory as static files at "/", so hitting
  http://localhost:8000/ returns web/index.html.
* GET  /health -> reports whether the model is loaded and which device it is on.
                  The browser polls this to choose "real model" vs "simulated" mode.
* POST /infer  -> accepts a base64 scene image + prompt + image-token budget,
                  runs the vision-language model, and returns the description,
                  the measured generate() latency, and the *actual* number of
                  image (vision) tokens the processor produced.

The model is loaded lazily and exactly once, on the first /infer call, so the
server starts instantly and /health can answer immediately (model_loaded=false)
before the weights are resident.

Model API notes (LiquidAI/LFM2.5-VL-450M)
-----------------------------------------
* Processor:  AutoProcessor
* Model:      AutoModelForImageTextToText  (concrete class: Lfm2VlForConditionalGeneration)
* The image-token budget is an *image-processor* kwarg named `max_image_tokens`
  (part of Lfm2VlProcessorKwargs; companion floor `min_image_tokens`). It can be
  passed to `apply_chat_template(...)` per call, which is what we do so the budget
  can change on every request. Documented envelope: min 64, max 256; raising
  max_image_tokens above 256 is supported (a single large image can reach ~1020).
* Each image placeholder in `input_ids` is `config.image_token_id` (default 396),
  so the true vision-token count == number of those ids in the encoded input.

Transformers version: pinned floor is >=4.57.0 (empirically the earliest release
that exposes the lfm2_vl module). Liquid's current model card recommends v5.1+;
if the model fails to load, `pip install -U transformers`.

Run:  python server.py     (serves on 0.0.0.0:8000)
"""

from __future__ import annotations

import base64
import io
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

MODEL_ID = "LiquidAI/LFM2.5-VL-450M"

# Directory that holds the static browser demo (index.html, etc.). Resolved
# relative to this file so it works regardless of the current working directory.
WEB_DIR = Path(__file__).resolve().parent / "web"

# Scene footage lives in data/scenes/. Serving it lets the browser load the same
# `data/scenes/<id>.jpg` path in real-model mode that it uses on file:// — and lets
# the page fetch the frame bytes to POST to /infer.
SCENES_DIR = Path(__file__).resolve().parent / "data" / "scenes"

# Default image-token budget if the client does not send one.
DEFAULT_MAX_IMAGE_TOKENS = 256

# How many new text tokens the model may emit per description.
MAX_NEW_TOKENS = 96


# ----------------------------------------------------------------------------
# Lazy, one-time model loader
# ----------------------------------------------------------------------------
#
# We keep the heavy imports (torch/transformers) *inside* the loader so that the
# process — and therefore /health — comes up even if those packages are slow to
# import or, in "simulated" mode, not installed at all.

class ModelBundle:
    """Holds the processor + model + device, loaded lazily and thread-safely."""

    def __init__(self) -> None:
        self.processor = None
        self.model = None
        self.device: str = "cpu"
        self.image_token_id: int | None = None
        self._loaded = False
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self) -> None:
        """Load the model exactly once. Safe to call from multiple requests."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:  # re-check under the lock (double-checked locking)
                return

            # Heavy imports happen here, on first use only.
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            # ---- Pick a device: prefer Apple MPS (fp16), fall back to CPU -----
            if torch.backends.mps.is_available():
                device = "mps"
                dtype = torch.float16
            elif torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
            else:
                device = "cpu"
                dtype = torch.float32  # fp16 on CPU is slow/unsupported for many ops

            processor = AutoProcessor.from_pretrained(MODEL_ID)

            # Single-image, variable-resolution mode: turn OFF tiling so the per-image
            # token budget (max_image_tokens) directly controls the vision-token count.
            # With splitting ON, a large frame is cut into 512px tiles and the token
            # count is dominated by the tile count, which swamps the budget knob
            # (verified on transformers 5.12: caps 64/128/256 all yielded ~1600 tokens).
            # OFF gives the clean budget -> tokens -> latency tradeoff this demo is about,
            # matching Liquid's published token<->resolution figures (e.g. 256x384 -> 96).
            processor.image_processor.do_image_splitting = False

            # `torch_dtype` is the compatible spelling across transformers 4.57 -> 5.x
            # (v5 renamed it to `dtype` but keeps `torch_dtype` as a working alias).
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
            )
            model.to(device)
            model.eval()

            # The image placeholder token id; used to count vision tokens.
            # Read from config so we don't hard-code the default (396).
            image_token_id = getattr(model.config, "image_token_id", None)

            self.processor = processor
            self.model = model
            self.device = device
            self.image_token_id = image_token_id
            self._loaded = True


BUNDLE = ModelBundle()

# Serializes the (set budget -> encode -> generate) critical section: the processor
# is shared mutable state, and the single model instance can't run parallel generate
# on MPS. Fine for a local, single-user demo.
INFER_LOCK = threading.Lock()


# ----------------------------------------------------------------------------
# Request / response schemas
# ----------------------------------------------------------------------------

class InferRequest(BaseModel):
    # Base64-encoded image bytes. A "data:image/...;base64,XXXX" data URL is also
    # accepted — we strip the prefix before decoding.
    image_b64: str
    prompt: str = Field(default="Describe the scene in one sentence.")
    # Floor is 64: the processor's min_image_tokens defaults to 64 and upscales to
    # meet it, so a smaller cap can't actually lower the vision-token count. We only
    # send max_image_tokens (below), so pinning the schema floor keeps the contract honest.
    max_image_tokens: int = Field(default=DEFAULT_MAX_IMAGE_TOKENS, ge=64, le=4096)


class InferResponse(BaseModel):
    description: str
    latency_ms: float
    vision_tokens: int
    device: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _decode_image(image_b64: str) -> Image.Image:
    """Decode a base64 string (optionally a data URL) into an RGB PIL image."""
    if not image_b64:
        raise HTTPException(status_code=400, detail="image_b64 is empty")

    # Accept full data URLs like "data:image/png;base64,iVBORw0K..."
    if image_b64.startswith("data:"):
        _, _, image_b64 = image_b64.partition(",")

    try:
        raw = base64.b64decode(image_b64, validate=False)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - surface a clean 400 to the client
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc


def _count_vision_tokens(inputs, image_token_id: int | None) -> int:
    """Count the actual number of image/vision tokens in the encoded input.

    Preferred method: count occurrences of `image_token_id` in `input_ids`.
    Fallbacks handle processor variants that don't expose the placeholder id.
    """
    input_ids = inputs.get("input_ids")

    # Primary path: count the image placeholder token in the prompt.
    if image_token_id is not None and input_ids is not None:
        try:
            return int((input_ids == image_token_id).sum().item())
        except Exception:  # noqa: BLE001 - fall through to the heuristics below
            pass

    # Fallback: only trust an EXPLICIT token count. (image_sizes / spatial_shapes are
    # pixel dimensions, not token counts — their product is a pixel area, so we must
    # not derive a token count from them.)
    val = inputs.get("num_image_tokens")
    if val is not None:
        try:
            import torch  # local import; only reached when model is loaded

            t = val if isinstance(val, torch.Tensor) else torch.as_tensor(val)
            return int(t.sum().item())
        except Exception:  # noqa: BLE001
            pass

    return 0


# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------

app = FastAPI(title="LFM2.5-VL-450M local demo")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Cheap, always-fast readiness probe. The browser polls this on load."""
    return HealthResponse(
        status="ok",
        model_loaded=BUNDLE.loaded,
        device=BUNDLE.device,
    )


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest) -> InferResponse:
    """Run one vision-language description over the posted scene image."""
    # Load the model on first use (subsequent calls are a no-op).
    try:
        BUNDLE.ensure_loaded()
    except Exception as exc:  # noqa: BLE001 - e.g. weights missing / OOM
        raise HTTPException(status_code=503, detail=f"Model failed to load: {exc}") from exc

    import torch  # safe: model is loaded, so torch is importable

    image = _decode_image(req.image_b64)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": req.prompt},
            ],
        }
    ]

    # Apply the per-image vision-token budget by setting it ON THE IMAGE PROCESSOR.
    # In transformers 5.x, passing max_image_tokens through apply_chat_template does
    # NOT reach the image processor (it warns and is silently ignored), so we set the
    # attribute directly. do_image_splitting was turned OFF at load, so this is a
    # single, variable-resolution image whose token count the budget controls. The
    # /infer schema pins max_image_tokens >= 64 (the processor's min floor), so the
    # requested budget is always honored.
    #
    # Everything from setting the budget through generate() runs under INFER_LOCK.
    with INFER_LOCK:
        BUNDLE.processor.image_processor.max_image_tokens = int(req.max_image_tokens)

        inputs = BUNDLE.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Count the true vision-token usage *before* moving tensors to the device.
        vision_tokens = _count_vision_tokens(inputs, BUNDLE.image_token_id)

        inputs = inputs.to(BUNDLE.model.device)
        prompt_len = inputs["input_ids"].shape[-1]

        # ---- Measure latency strictly around generate() --------------------
        # Flush queued async GPU work first so the timer captures only generate().
        if BUNDLE.device == "cuda":
            torch.cuda.synchronize()
        elif BUNDLE.device == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            generated = BUNDLE.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        if BUNDLE.device == "cuda":
            torch.cuda.synchronize()
        elif BUNDLE.device == "mps":
            torch.mps.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Decode only the newly generated tokens (strip the prompt prefix).
        new_tokens = generated[:, prompt_len:]

    description = BUNDLE.processor.batch_decode(
        new_tokens, skip_special_tokens=True
    )[0].strip()

    return InferResponse(
        description=description,
        latency_ms=round(latency_ms, 2),
        vision_tokens=vision_tokens,
        device=BUNDLE.device,
    )


# ----------------------------------------------------------------------------
# Static files — mounted LAST so they don't shadow /health and /infer.
# ----------------------------------------------------------------------------
# Narrower mount FIRST: scene footage at /data/scenes/<id>.jpg, so the UI can load
# the frame (and read its bytes to POST) at the same path it uses on file://.
if SCENES_DIR.is_dir():
    app.mount("/data/scenes", StaticFiles(directory=str(SCENES_DIR)), name="scenes")

# Catch-all LAST. `html=True` makes StaticFiles serve index.html for "/".
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print("Starting LFM2.5-VL-450M demo server on http://localhost:8000 ...")
    print("The model loads lazily on the first slider move; the first response")
    print("will be slower while weights download/load. Leave this window open.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
