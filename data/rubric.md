# Accuracy rubric

Accuracy in this demo is **not** a live measurement — you cannot know the correct
answer for an arbitrary live frame. It is measured once, ahead of time, over a small
labeled set of doorbell frames with known answers, scored with the fixed rubric below.
`eval_sweep.py` runs that sweep across every vision-token budget and writes the curve
that the UI displays.

> In the shipped **simulated** mode there is no sweep behind the curve — the curve is
> **illustrative** (and labeled as such in the UI). Run `eval_sweep.py` on your own
> footage to replace it with a measured curve.

## Scene types and their ground truth

| scene id       | ground truth                                              | the hard part                          |
|----------------|-----------------------------------------------------------|----------------------------------------|
| `package`      | A parcel is on the porch step; no person present.         | easy — should hold even at low budgets |
| `person`       | One person standing at the front door.                    | presence is easy; detail needs budget  |
| `two-people`   | Two people at the door.                                   | **counting** — undercounts at low budget |
| `empty-branch` | Empty porch; a tree branch is moving. No person/package.  | **false positives** — hallucinates a person at low budget |

## Scoring (per frame, per budget)

Each generated description is graded against the frame's label on three checks. A
frame **passes** (counts as correct) only if all three hold; accuracy at a budget is
the fraction of frames that pass at that budget.

1. **Primary subject present / absent correct** — the description asserts the right
   thing exists (a package, a person, two people) or correctly asserts the porch is
   empty. A false positive (claiming a person when there is none) fails here.
2. **Count correct** — where the label has a count (0, 1, 2 people), the description's
   count matches. "A person" for a two-person frame fails.
3. **No contradicted detail** — any specific detail the description volunteers
   (clothing, object type, action) must not contradict the label. Vagueness does not
   fail this check; a wrong specific does.

A grader (human, or an LLM judge given the label + rubric) applies checks 1–3 and
records pass/fail. Keep the grader, prompt, and label set fixed across all budgets and
all models so the comparison is held-constant.

## Labeled set

Put labeled frames under `data/scenes/` (see the repo README, "Bring your own
footage"). A defensible set is ~20–40 frames spread across the four scene types and a
range of lighting/weather. Each frame needs a one-line label following the ground-truth
column above.
