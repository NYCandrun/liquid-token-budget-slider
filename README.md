# Token-Budget Slider Demo

An interactive demo for **LFM2.5-VL-450M**, a small on-device vision-language model from Liquid AI. It turns a front-door camera frame into a **structured event** a control loop can act on — `{"event":"package_placed","where":"front_door","conf":0.9}` — and puts the model's **vision-token budget** on a slider. The point: **fewer vision tokens = less compute, memory and energy per frame**, so you dial the fewest tokens that still gets the event. (Companion to the "cameras as a sensor" control-loop one-pager.)

You'll see, as you drag the budget:
- the **structured event** the camera emits — front and center,
- the **cost per frame** — vision tokens, compute/energy, latency — all dropping as you spend fewer tokens,
- an **event-confidence vs cost** curve (the sweet spot is the fewest tokens past the confidence elbow),
- and the caption — "what the model saw".

Honesty up front: in **simulated** mode the event + confidence are an *illustrative fine-tuned target*. Run the model locally (see [Try it](#try-it)) and the raw LFM2.5-VL-450M returns valid JSON but a **noisy** event/confidence — locking those in is a fine-tune. Vision-token counts are exact and compute/memory/energy scale with them.

---

## Try it

**Live demo (no install):** **[nycandrun.github.io/liquid-token-budget-slider](https://nycandrun.github.io/liquid-token-budget-slider/)** — opens in any browser and runs in **simulated** mode (clearly badged), so you can drag the slider and feel the tradeoff immediately. Nothing is installed; nothing leaves your machine.

**Run the real model (the app auto-connects to it):**

1. **Download this repo** — `git clone https://github.com/NYCandrun/liquid-token-budget-slider` (or download the ZIP and unzip).
2. **Set up and start the server** (full steps in [Quick start B](#quick-start-b--real-model-runs-on-your-mac-on-localhost)):
   ```bash
   cd liquid-token-budget-slider
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python server.py
   ```
3. **Open [http://localhost:8000](http://localhost:8000).** The page detects the running server and switches to the **LIVE MODEL** automatically — in Chrome on a Mac, and in any other browser, with no configuration. Drop your own frames into `data/scenes/` to run the model on real footage.

**The live model is local-only.** There is no hosted/cloud endpoint, so the public demo link always runs in **simulated** mode — to see the real model, run `server.py` on your own machine (above). *(The repo keeps an optional [`modal_app.py`](modal_app.py) if you ever want to self-host on a cloud GPU, but deploying it publishes a **public, unauthenticated** GPU endpoint billed to your account, so it is left undeployed — see the warning at the top of that file.)*

> The app is a single page: **`web/index.html`** is the entry point (the `index.html` at the repo root just redirects to it), and `server.py` serves it at `http://localhost:8000`. The public link above is the shareable **simulated** demo; the live model runs locally via `server.py`.

---

## Two ways to run

**Mode A — Simulated (no installation).** Just open the demo in a browser. It uses realistic, clearly-labeled illustrative numbers and scene descriptions instead of running the real model. Works offline, opens in seconds, nothing to install. Use this to see and present the demo quickly. A "SIMULATED" badge is always visible in this mode so no one mistakes the illustrative numbers for a measured benchmark.

**Mode B — Real model (runs on your Mac).** Runs the actual LFM2.5-VL-450M model locally so the latency and descriptions are real. Requires a short setup (below) and an Apple Silicon Mac. Use this to get genuine numbers on real footage.

The demo page is the same in both modes: it automatically uses the real model **if** the local model server is running, and falls back to simulated mode (with the visible "SIMULATED" badge) if it isn't.

---

## Quick start A — Simulated mode (no installation)

1. Get the project folder (unzip the folder you received, or `git clone <repo-url>` if you were given a link).
2. Open `web/index.html` in any web browser — double-click it, or drag it into a browser window.
3. Drag the slider. That's it.

No Python, no server, no internet needed. A "SIMULATED" badge is shown so it's clear the numbers are illustrative. The three scenes use real front-door frames that ship with the repo (public-domain — see [Bring your own footage](#bring-your-own-footage)); the slider, readouts, and curve all work with no setup.

---

## Quick start B — Real model (runs on your Mac, on localhost)

> This runs the real model on your machine. The model runs in a small local program (the "server"); the demo page talks to it at `http://localhost:8000`. Nothing you do ever leaves your computer.

**Prerequisites**
- macOS on **Apple Silicon** (M1 or newer). This build uses Apple's GPU (MPS) through PyTorch; on an Intel Mac it will run on CPU and be slow. (See [What this build is](#what-this-build-is--and-what-it-is-not) for why.)
- **Python 3.10–3.12** installed. Check with `python3 --version`. If you don't have it, install from [python.org](https://www.python.org/downloads/).
- Internet access **once**, for the first-run model download (~900 MB). After that it runs offline. (A smaller quantized variant exists, but it belongs to the vendor-runtime path in the scope section below and is not wired into this build.)

**1. Open a terminal in the project folder**
```
cd liquid-token-budget-slider
```
(Unzip the folder first if you were given a zip, or `git clone <repo-url> && cd liquid-token-budget-slider` if you were given a link.)

**2. Create the bundled Python environment**
This makes a self-contained Python setup inside the project folder (`./.venv`) so it never touches the rest of your system.
```
python3 -m venv .venv
source .venv/bin/activate
```
You'll know it worked when your terminal prompt shows `(.venv)`.

**3. Install the dependencies**
```
pip install -r requirements.txt
```
This installs the pinned backend packages (`transformers>=4.57.0`, `torch`, `accelerate`, `pillow`, `fastapi`, `uvicorn`, `huggingface_hub`). The `transformers>=4.57.0` version is required — the model won't load on older versions.

**4. Start the model server**
```
python server.py
```
The first time, this downloads the model (`LiquidAI/LFM2.5-VL-450M`) and loads it into memory. No account or login is required. Wait until it prints that it's running at **`http://localhost:8000`**. Leave this terminal window open — the server needs to keep running.

**5. Open the demo**
Open **`http://localhost:8000`** in your browser. The same page serves as the UI and is already connected to the model server, so there's a single address and no CORS/file issues. When it's using the real model, the "SIMULATED" badge is gone.

Drag the slider — each move sends the current setting to the server, which runs the model and returns a real description, a measured latency, and the actual token count.

---

## Bring your own footage

The demo ships with **three real front-door frames** (public-domain, from one fixed camera — see [`data/scenes/SOURCES.md`](data/scenes/SOURCES.md)):

```
data/scenes/package.jpg        # a box on the step
data/scenes/small-parcel.jpg   # a small parcel at the door
data/scenes/two-parcels.jpg    # a box + a small parcel
```

They drive the three demo scenes. The token budget doesn't change *whether* the model sees the package — a 450M model gets these clear frames right at any budget — it changes the **per-frame cost** (fewer tokens → less compute, memory, energy, latency) and how confident/complete the read is. That framing was verified by running the real model locally — see [`server.py`](server.py); detection/counting/OCR were too noisy on a 450M model to fake a clean "it breaks at low budget" story. Replace any of these (keep the filenames) with **your own** doorbell frames and reload. The event, confidence and cost curve in simulated mode are **illustrative**; in real-model mode the event comes from the actual model (and is where the fine-tune matters).

---

## How to use the demo

- **The slider** sets the vision-token budget — how much the model "looks." Left = fewer tokens (faster, coarser); right = more tokens (slower, sharper).
- **Scene picker** — switch between doorbell scenes (a package on the step, a person at the door, two people, an empty porch with branch motion). Different scenes break down at different points.
- **Latency** — the big number; watch it fall as you move the slider left.
- **Token count** — confirms the slider actually changed how many tokens the image used (in real-model mode this is the *actual* count the processor produced, which can differ slightly from the slider stop).
- **Description** — the model's read of the scene; notice it gets vaguer at low budgets.
- **Accuracy curve** — shows, across all budget settings, where the model stays correct and where it starts to fail. The current setting is marked on it. In simulated mode this curve is illustrative and labeled as such; the real curve comes from the labeled sweep in `data/` (see [About the accuracy numbers](#about-the-accuracy-numbers)).

The interesting move is to push the slider down until the model *just* starts getting the scene wrong — that's the speed/quality tradeoff point for that scene, and you found it yourself.

---

## What this build is — and what it is not

**This version is built for the transformers / PyTorch runtime.** The model runs through Hugging Face `transformers` on PyTorch, using Apple's MPS (GPU) backend on an Apple Silicon Mac, or CPU as a fallback. The vision-token budget is set with the processor's image-token setting. This is the standard, reference-correct way to run the model and is what every step above assumes.

**This build will NOT exercise a custom NPU or silicon accelerator.** If your goal is to benchmark a new chip whose value is its dedicated AI accelerator (an NPU, DSP, or vendor accelerator), this build will run on that board's CPU or general GPU — **not** the accelerator — so the speed numbers would not reflect the hardware you're trying to test.

**Testing a custom-NPU chip requires evolving the backend to a vendor runtime.** That is a different backend than the one in this project, and it is deliberately out of scope here. It would mean:
- Converting the model to the chip's format (e.g. GGUF, ONNX, or the vendor's own format) instead of the standard PyTorch checkpoint.
- Running it through the vendor's runtime/SDK (e.g. TensorRT on Jetson, ONNX Runtime with a vendor execution provider, Qualcomm QNN, or the silicon partner's toolchain) instead of PyTorch.
- **Rebuilding the token-budget knob** on that path. The image-token setting lives in the transformers software; a converted model does not carry it, so the equivalent control (adjusting image resolution before the model sees it) has to be re-implemented.

In short: **transformers/PyTorch today; a vendor-runtime version is a future step and is required before any custom-NPU chip measurement means anything.** Treat any latency this build reports on non-NPU hardware as the *shape* of the tradeoff, not a chip's real number.

---

## Front end

The front end is a single web page in `web/`. In simulated mode it opens by double-click with nothing to install. In real-model mode it's served by the model server at `http://localhost:8000`, so the UI and the model share one address (no CORS or `file://` issues). The page holds no model — it only draws the controls and readouts and talks to the server.

## Architecture

The demo has two parts:
- a **UI** (the browser page) — this holds no model, and
- a **model server** (`server.py`) — this loads LFM2.5-VL-450M once and keeps it ready.

They talk to each other over `http://localhost:8000`. Each time you move the slider, the UI sends the current scene image, a prompt, and the budget value; the server runs the model and sends back the description, the measured latency, and the actual token count. Because they communicate over a network address, the server does not have to be on the same machine as the UI (see below).

## Running the model on other hardware

Since the UI reaches the server by network address, you can run `server.py` **on a separate test board** and point the UI (still on your laptop) at that board's address instead of `localhost`. The UI needs no changes. The server, though, only ports as far as PyTorch does:
- A board where PyTorch runs (e.g. a Jetson dev kit, or a generic ARM/x86 Linux board) will run this build — but on CPU/GPU, **not** a custom NPU (see the scope section above).
- A chip whose accelerator you actually want to measure needs the vendor-runtime version, not this project's backend.

## Building it out: comparing multiple models

This demo is intentionally **single-model** — its whole point is the interactive token-budget slider. That knob is a **first-class, documented control** in LFM2.5-VL-450M (`min_image_tokens` / `max_image_tokens`, set at inference time). Some other models of this size expose image-*resolution* controls that indirectly change token counts, but few surface a clean, documented per-image token budget you set at run time — so a single slider doesn't translate across them cleanly, and cramming several models onto the slider screen tends to muddy both. If you want to compare several models against each other, build it as a **separate mode**.

The honest version of a comparison is a **held-constant harness**: it runs the same prompts and the same labeled image set across the models that can share one runtime, and reports each model's latency and rubric-graded accuracy side by side. "Held-constant" is the credibility requirement — same runtime, same precision, same hardware, same prompts, same eval set — because a technical reviewer will look for exactly the place a comparison was rigged.

The practical wrinkle is that not every model drops into the same lane cleanly, so split the field into two groups:

- **In-lane (put them on the chart):** models that load through the same standard runtime and precision this project uses (transformers, fp16). These get scored directly against each other.
- **Next-step (name them, don't force them on):** models that can't share the lane — for example ones that ship only in a quantized format, run only on a different vendor runtime, or need their own custom loader. Listing these as "next-step comparisons" is more honest than scoring them on a setup their makers didn't intend, which reads as rigged and costs credibility. Some models also simply lack a capability (e.g. they can't return bounding boxes or structured output) — that's a "not supported" mark, not a low score.

How this fits the code here: the UI-plus-model-server design already generalizes — the server would load several in-lane models and the UI would show their outputs, latencies, and accuracy side by side on the same scene, as a distinct "comparison" view alongside the single-model slider. The next-step models are documented as future integrations rather than wired in.

## About the accuracy numbers

Latency and token count are measured live by the server. **Accuracy is not measured live** — there's no way to know the "right answer" for an arbitrary live frame. Instead it's measured once, ahead of time, on a small set of doorbell images that have known correct answers (~20–40 images, scored with a fixed rubric). The demo shows that pre-computed curve and labels it clearly, so live speed and pre-measured accuracy are never confused.

In simulated mode there is no eval run behind the curve, so the curve shown is **illustrative** (clearly labeled). The real curve is produced by `data/eval_sweep.py`, which runs the model over the labeled frames in `data/scenes/` at each budget and scores them against `data/rubric.md`. Run that sweep on your own footage to replace the illustrative curve with a measured one.

## Project layout (reference)

Some of these files may be added over time; this is the intended structure.
```
liquid-token-budget-slider/
├── README.md
├── token-budget-slider-PRD.md   # product rationale behind the demo
├── requirements.txt             # backend dependencies (transformers>=4.57, torch, ...)
├── server.py                    # local model server (transformers/PyTorch backend)
├── web/                         # browser front end (the demo page)
│   └── index.html
├── data/                        # doorbell scenes + labels for the accuracy curve
│   ├── scenes.json              # scene definitions + illustrative per-budget data
│   ├── scenes/                  # sample front-door frames (package/small-parcel/two-parcels) + SOURCES.md
│   ├── rubric.md                # how accuracy is scored
│   └── eval_sweep.py            # runs the real model over the labeled set → measured curve
└── .venv/                       # the bundled Python environment (created in setup)
```

## Troubleshooting

- **The page shows a "SIMULATED" badge when I expected the real model** — the model server isn't running or isn't reachable. Make sure `python server.py` is running (step 4) and that you opened `http://localhost:8000` (not the file directly).
- **"transformers can't find the model" / model won't load** — your `transformers` is older than 4.57.0. Re-run `pip install -r requirements.txt` with the `.venv` activated.
- **Very slow** — confirm you're on an Apple Silicon Mac so the GPU (MPS) is used; on Intel/CPU it will be slow. Remember laptop latency shows the shape of the tradeoff, not a camera chip's real number.
- **`command not found: python3` or `pip`** — Python isn't installed or the `.venv` isn't activated. Install Python 3.10–3.12 and re-run step 2 (the prompt should show `(.venv)`).
- **Scenes show placeholders** — you haven't added footage yet. Drop your frames into `data/scenes/` (see [Bring your own footage](#bring-your-own-footage)) and reload.

## Notes

The vision-token budget (`min_image_tokens` / `max_image_tokens`) is documented on Liquid AI's LFM2-VL model pages and blog. This demo targets the `LiquidAI/LFM2.5-VL-450M` model. See the accompanying PRD (`token-budget-slider-PRD.md`) for the product rationale behind the demo.
