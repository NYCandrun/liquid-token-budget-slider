# Drop your footage here

The demo loads one image per scene from this folder. Add your doorbell frames with
these exact names (`.jpg` or `.png`) and reload the page — they appear automatically:

```
package.jpg        # a package left on the step (no person)
person.jpg         # one person at the door
two-people.jpg     # two people at the door
empty-branch.jpg   # empty porch, a branch moving (the tricky false-positive scene)
```

Until a file is present, that scene shows a clearly-marked placeholder frame; the
slider, readouts, and accuracy curve still work (in simulated mode). For a real
measured accuracy curve, add a labeled set (~20–40 frames) and run `../eval_sweep.py`
with the server running — see `../rubric.md`.

No footage is shipped in this repo, so nothing here is copyrighted to anyone but you.
