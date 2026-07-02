# Description-detail rubric

The demo's curve is **description detail** vs. token budget — how complete and specific
the model's caption is — **not** a pass/fail accuracy. We checked on the real model
(run locally — see `../server.py`): this 450M model gets the scene *right* at every budget on these
clear frames; what the token budget changes is the level of **detail**. Detection,
counting, and OCR were all too noisy to score as a clean accuracy curve (the model
hallucinates label text and detects the tiny parcel inconsistently), so detail is the
honest axis.

> In simulated mode the curve is **illustrative** (labeled as such in the UI). Run
> `eval_sweep.py` on your own footage to replace it with a measured detail curve.

## Scenes and ground truth

| scene id       | ground truth                                            | what the budget changes                |
|----------------|---------------------------------------------------------|----------------------------------------|
| `package`      | A cardboard box on the step in front of the blue door.  | coarse "a box on the step" → materials, colour, the plant beside it |
| `small-parcel` | A small white parcel at the base of the blue door.      | "a white item" → "a small white padded parcel on the black doormat" |
| `two-parcels`  | A cardboard box and a small parcel on the step.         | mentions the box; the small parcel + surroundings appear as detail rises |

## Scoring detail (per frame, per budget)

Score each description for **completeness**, not correctness (all are correct). Take a
fixed checklist of salient, verifiable details per frame — e.g. for `two-parcels`:
{box present, a second small parcel, on the brick step, blue door, potted plant} — and
count how many the caption includes. Detail score = fraction of the checklist present.
Any *contradicted* specific (a wrong colour or count) scores that item zero. Keep the
checklist, prompt, judge, and frame set fixed across budgets and models so the
comparison is held-constant.

## Labeled set

Put frames under `data/scenes/` (see the repo README, "Bring your own footage"), each
with a detail checklist. A defensible set is ~20–40 frames across the three scene types
and a range of lighting/weather.
