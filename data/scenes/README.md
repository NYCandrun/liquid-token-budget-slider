# Scene frames

The demo ships with three real front-door / doorbell-camera frames (all from one fixed
camera). The token budget changes how much **detail** the model's description has, not
whether it's correct (verified on the real model — see `../../modal_app.py`):

| file | scene | low budget → high budget |
|---|---|---|
| `package.jpg`      | Box on step  | "a box on the step" → materials, colour, the potted plant |
| `small-parcel.jpg` | Small parcel | "a white item" → "a small white padded parcel on the black doormat" |
| `two-parcels.jpg`  | Two parcels  | mentions the box → adds the second small parcel and surroundings |

These are public-domain (CC0) — see [`SOURCES.md`](SOURCES.md).

**Bring your own footage:** replace any of these `.jpg`s (keep the filenames) with your own
doorbell frames and reload — they appear automatically. To add scenes, edit the `DATA.scenes`
array in `web/index.html` and `data/scenes.json`. For a real measured accuracy curve, add a
labeled set and run `../eval_sweep.py` with the server running — see `../rubric.md`.
