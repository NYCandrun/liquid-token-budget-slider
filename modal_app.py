"""
modal_app.py — OPTIONAL self-host of LFM2.5-VL-450M as an HTTPS web endpoint on Modal.

NOT USED BY THE DEMO AND NOT DEPLOYED BY DEFAULT. The demo is **local-only**: the live model
runs against server.py on your own machine, and the public GitHub Pages site always runs in
simulated mode. This file is kept only as an optional recipe if you deliberately want to
self-host on a cloud GPU.

SECURITY WARNING: as written this deploys a **public, UNAUTHENTICATED** GPU endpoint
(ALLOW_ORIGINS = "*"). Anyone who discovers the *.modal.run URL can invoke it and burn
GPU-seconds billed to your Modal account (a "denial of wallet"). A public, no-signup demo
cannot hide an auth token in its client JS, so if you do deploy this: set a Modal spend limit,
cap autoscaling (max_containers), and prefer taking it down (`modal app stop <name>`) when not
actively demoing. That is exactly why the demo does not use it.

HONESTY NOTE: this runs on a **cloud GPU**, which is the opposite of the demo's on-device
thesis. It produces real captions, real vision-token counts, and the real
budget -> tokens -> detection behaviour — but the latency it reports is cloud GPU + network,
NOT the edge/on-device latency the product story is about.

Endpoints (same contract as server.py):
  GET  /health -> {status, model_loaded, device}
  POST /infer  -> {image_b64, prompt, max_image_tokens} -> {description, latency_ms, vision_tokens, device}

Deploy (only if you accept the warning above):  modal deploy modal_app.py
"""
import base64
import io
import threading
import time

import modal

MODEL_ID = "LiquidAI/LFM2.5-VL-450M"
ALLOW_ORIGINS = ["*"]  # demo endpoint; set to ["https://nycandrun.github.io"] to lock it down


def _download():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "torchvision",              # required by the LFM2-VL image processor
        "transformers>=4.57.0",
        "accelerate",
        "pillow",
        "huggingface_hub",
        "hf_transfer",
        "fastapi[standard]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(_download)        # bake the weights into the image for fast cold starts
)

app = modal.App("lfm2-vl-450m-demo", image=image)


@app.cls(gpu="T4", scaledown_window=300, timeout=600)
class Model:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        # Single-image, variable-resolution: OFF so the token budget directly controls
        # the vision-token count (splitting would swamp the knob). Same as server.py.
        self.processor.image_processor.do_image_splitting = False
        self.model = (
            AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )
        self.image_token_id = getattr(self.model.config, "image_token_id", None)
        self.lock = threading.Lock()  # serialize the shared-processor critical section

    @modal.asgi_app()
    def web(self):
        import torch
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from PIL import Image
        from pydantic import BaseModel, Field

        webapp = FastAPI(title="LFM2.5-VL-450M on Modal")
        webapp.add_middleware(
            CORSMiddleware,
            allow_origins=ALLOW_ORIGINS,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        class InferReq(BaseModel):
            image_b64: str
            prompt: str = Field(default="Describe the scene in one sentence.")
            max_image_tokens: int = Field(default=256, ge=64, le=4096)

        def _decode(b64: str) -> Image.Image:
            if b64.startswith("data:"):
                b64 = b64.split(",", 1)[1]
            try:
                return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail=f"bad image: {exc}") from exc

        @webapp.get("/health")
        def health():
            return {"status": "ok", "model_loaded": True, "device": self.device}

        @webapp.post("/infer")
        def infer(req: InferReq):
            image = _decode(req.image_b64)
            conversation = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": req.prompt},
                ]}
            ]
            with self.lock:
                # Budget is set on the image processor (apply_chat_template ignores it in
                # transformers 5.x). do_image_splitting is OFF (set at load).
                self.processor.image_processor.max_image_tokens = int(req.max_image_tokens)
                inputs = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                )
                vision_tokens = 0
                if self.image_token_id is not None:
                    vision_tokens = int((inputs["input_ids"] == self.image_token_id).sum())
                inputs = inputs.to(self.device)
                prompt_len = inputs["input_ids"].shape[-1]

                if self.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    generated = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - start) * 1000.0
                description = self.processor.batch_decode(
                    generated[:, prompt_len:], skip_special_tokens=True
                )[0].strip()

            return {
                "description": description,
                "latency_ms": round(latency_ms, 2),
                "vision_tokens": vision_tokens,
                "device": self.device,
            }

        return webapp
