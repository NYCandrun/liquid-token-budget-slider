# PRD — Token-Budget Slider Demo

**Status:** Draft v1 · 2026-07-01
**Delivery target:** runs in a web browser; backend built for the transformers/PyTorch runtime.
**Setup & run instructions:** see the repo `README.md`.

---

## 0. Background (read first)

This section defines everything the rest of the document assumes, so it can be read with no prior context.

**The model.** *LFM2.5-VL-450M* is a small vision-language model made by Liquid AI. A vision-language model takes an image (and a text prompt) and returns text — for example, looking at a doorbell camera frame and answering "a person left a package on the porch." "450M" refers to its size (450 million parameters), which is small enough to run directly on a device such as a camera or a laptop, rather than in the cloud.

**The knob this demo is about.** Before the model reasons about an image, it converts the image into a number of *vision tokens* — the model's internal units for "looking." This model lets you choose how many vision tokens to spend per image at run time, without retraining it, via a first-class, documented control (`min_image_tokens` / `max_image_tokens`). Fewer tokens is faster and lighter but coarser; more tokens is slower but more detailed. This adjustable budget is the feature the demo showcases. Some competing models of this size let you change image *resolution* (which indirectly changes token counts), but few expose a clean, documented per-image token budget you set at inference time — so it is a differentiator, just not a unique-in-the-world one, and the demo should not over-claim it as such.

**Why it matters.** Cameras and other edge devices have a fixed, tight budget for speed, memory, and power. A model you can dial to fit that budget — trading a little accuracy for a lot of speed when needed — is directly useful to a team deciding which model to put on a device. The token budget turns an abstract "speed vs. quality" tension into a control someone can actually operate.

**Intended audience for the demo.** A technically sophisticated evaluator — for example, an engineering lead choosing a model for a home-camera product. This person distrusts marketing benchmarks and trusts things they can operate and reproduce themselves. The demo is designed for that person to pick up and drive.

## 1. Overview

An interactive demo that turns the model's tunable vision-token budget into a single on-screen slider. As the user drags it, the demo shows — live, on a doorbell scene — the model's description of the scene, the measured latency, the number of vision tokens used, and where that setting sits on an accuracy curve. The goal is to let the evaluator *feel* the speed-versus-quality tradeoff and find its limits themselves, which is more convincing than a static benchmark chart.

## 2. Problem & insight

When choosing a small on-device model, the hard question isn't "device or cloud" — it's *which* small model. The deciding advantages are hard to convey on a slide. This model's tunable token budget is one such advantage: a direct speed/quality dial, set at run time, with no retraining, exposed as a first-class documented control that most competitors at this size don't surface as cleanly. Because it maps precisely to the fixed speed/memory/power envelope of a real device, the most persuasive way to communicate it is to hand the evaluator the dial and let them turn it. A slider makes the tradeoff tangible and puts the evaluator in control.

## 3. Goals

- Let a user set the vision-token budget and see latency, the description, and the token count change live.
- Show where each setting lands on an accuracy-versus-budget curve, so the tradeoff is visible, not merely asserted.
- Run offline, which also demonstrates that no image data leaves the device.
- Be reproducible by the evaluator on their own hardware and footage, so the demo doubles as an evaluation tool.

## 4. Non-goals

- Not a comparison against other models — this demo is one model, one knob.
- Not a video/event demo — it works on single still frames only.
- Not a production alerting system or an integration with any camera platform.

## 5. Target users

- **Presenter** (e.g. a product or sales engineer): drives the slider to show the tradeoff in one gesture.
- **Skeptical evaluator** (e.g. an ML/engineering lead): takes the controls, pushes the budget down until quality breaks, and finds the tradeoff point for a scene themselves.
- **Evaluator's engineer, afterward:** swaps in their own footage and hardware to produce their own speed/quality result.

## 6. Core experience

A doorbell scene on screen, a slider labeled "detail vs. speed," and live readouts:

- **Description** — the model's output for the current scene ("a package is on the porch step").
- **Latency** — a large number that visibly drops as the budget goes down.
- **Vision-token count** — the tokens the image actually used, confirming the knob did something.
- **Accuracy curve** — a small chart, with the current setting marked, showing where accuracy holds and where it drops off.

Nice-to-have: the scene image visibly coarsens as the budget drops, making "fewer tokens" tangible (a visual *metaphor* for reduced internal resolution, not a literal render of what the model sees); and a hardware selector (a device-class preset) that rescales the latency to match a chosen target device.

## 7. Requirements

**Must have**
- Slider bound to the vision-token budget, roughly 8 discrete stops.
- Live latency and token-count readouts for each setting.
- Scene picker (package on step, person at door, two people, empty porch with branch motion).
- Accuracy-versus-budget curve with the current point marked.
- Fully offline operation.

**Should have**
- Hardware presets that rescale the latency figure to a chosen device class (clearly labeled illustrative).
- Visible image coarsening tied to the token budget.

**Could have**
- The "real model" mode described in §9 (runs actual inference locally).
- A recorded-run fallback for unreliable demo hardware.

## 8. How it runs: front end and setup

One front end (a browser page) over one backend, in a single codebase.

**Browser.** A web page that, in simulated mode (§9), opens by double-click with nothing to install, and in real-model mode is served by the local model server at `http://localhost:8000`. Best for handing the evaluator the controls. In real-model mode the UI and model share one address, so there are no CORS or `file://` issues.

**Bundled Python environment.** The project ships its own self-contained Python environment (a `./.venv` folder created during setup) rather than relying on the user's system Python, so setup is reproducible. Dependencies are pinned in `requirements.txt` (`transformers>=4.57.0` — a hard requirement, since the model does not exist in earlier versions — plus `torch`, `accelerate`, `pillow`, `fastapi`, `uvicorn`, `huggingface_hub`). The first run downloads the model (~900 MB); after that it runs offline. (A smaller quantized variant exists but belongs to the Tier 2 vendor-runtime path and is not wired into this build.)

## 8a. Backend runtime tiers

This build implements one runtime. The others are named so the scope is explicit.

- **Tier 1 — transformers / PyTorch (this build).** The model runs through Hugging Face `transformers` on PyTorch, using Apple's GPU (MPS) backend on Apple Silicon, or CPU otherwise. The token budget is the processor's image-token setting. This is the standard, reference-correct path and the only one this project implements. It will **not** use a device's custom NPU — it runs on CPU/GPU regardless.
- **Tier 2 — Vendor runtime (future, required for custom-NPU chips).** To benchmark a chip whose value is its dedicated AI accelerator, the backend must be evolved: convert the model to the chip's format (GGUF/ONNX/vendor format), run it through the vendor's SDK (TensorRT, ONNX Runtime with a vendor execution provider, Qualcomm QNN, or a silicon-partner toolchain), and **rebuild the token-budget control** there (via image resolution/preprocessing, since the image-token setting lives in the transformers software and is not carried by a converted model). Deliberately out of scope for this project.
- **Tier 3 — MLX (optional, Mac-only).** Faster on Apple Silicon but Apple-only, and exposes the token knob less cleanly; not relevant to testing on other hardware.

## 8b. Testing on other hardware (including a new chip)

Because the UI reaches the backend by network address, the model server can run **on a separate test board** while the UI stays on the laptop — just point the front end at the board's address instead of `localhost`, with no front-end changes. How far the backend ports depends on its tier: Tier 1 runs anywhere PyTorch runs (e.g. a Jetson dev kit or a generic ARM/x86 Linux board) but on CPU/GPU, not a custom NPU; measuring a chip's dedicated accelerator requires the Tier 2 vendor-runtime version. Any latency this build reports on non-NPU hardware is the *shape* of the tradeoff, not that chip's real number.

## 9. Data & accuracy honesty

Latency and token count are measured live. Accuracy requires knowing the correct answer, which can't be done for an arbitrary live frame, so it is measured once, ahead of time, on a small labeled set of doorbell frames (~20–40, each with a known answer and a fixed scoring rubric). The demo displays that pre-computed curve and labels it as pre-measured, so live speed and pre-measured accuracy are never blurred together — a distinction a technical evaluator will notice and trust.

For a first working version, the demo runs in a clearly labeled **simulated mode** (believable pre-set latency and scene descriptions that degrade as the budget drops) so it is demoable with zero setup. In simulated mode there is no eval run behind the accuracy curve, so the curve is **illustrative** and labeled as such — not a measured result. The **real local-model mode** produces genuine latency/token numbers, and the labeled sweep (`data/eval_sweep.py` + `data/rubric.md`) produces the genuine accuracy curve on the user's own footage.

## 10. Honest caveat to state up front

Latency depends on the hardware. On a laptop, the demo shows the *shape* of the tradeoff; a real device number comes from running on that actual device's chip — which is exactly the next step the demo sets up. Stating this plainly, rather than implying laptop numbers are device numbers, is what earns a technical evaluator's trust.

## 11. Success criteria

- Dragging the slider visibly and meaningfully changes latency.
- The token-count readout tracks the setting, proving the knob is live.
- The accuracy curve is defensible and clearly labeled (illustrative in simulated mode, pre-measured in real mode).
- The demo runs offline, with no data leaving the machine.
- An evaluator could reproduce it on their own footage and hardware.

## 12. Rough phasing

- **P1:** Browser app in simulated mode — slider, readouts, scenes, curve. Demoable end-to-end.
- **P2:** Real local-model mode on the Tier 1 transformers/PyTorch backend (Mac) + the labeled accuracy sweep + the bundled Python environment.
- **P3 (future, out of current scope):** Tier 2 vendor-runtime backend — required before any custom-NPU chip measurement is meaningful.

## 13. Open questions

- Demo hardware: a laptop, or a device-class board for real numbers?
- Footage: the evaluator supplies their own doorbell frames (the demo is built for drop-in footage in `data/scenes/`); staged/public images are not shipped.
- Expected deliverable depth: a working browser demo now, with the transformers/PyTorch backend scaffolded alongside it.

---

*The tunable image-token budget (`min_image_tokens` / `max_image_tokens`) is documented on Liquid AI's LFM2-VL model pages and blog.*
