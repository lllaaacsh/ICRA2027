# Website video manifest

All clips: H.264 MP4, `yuv420p`, fast-start, no audio, square 480x480 (teaser wall 1600x800).
Replace a clip in place with the same filename; no HTML change is needed.

## Playback speed (shown on the page — keep these in sync if a clip is re-cut)

| Setting | Speed | How it is derived |
|---|---|---|
| Physical UR5 | 1.75x | Source clips were exported at 1.75x by the authors |
| Franka Kitchen | 1x | One frame per control step, dt = 0.08 s, encoded at 12.5 steps/s |
| Meta-World MT10 | 0.5x | One frame per env step, dt = 0.0125 s (80 Hz), encoded at 40 steps/s |

## Teaser

- `teaser-wall.mp4`, `teaser-wall-poster.jpg` — 14 s zoom-out wall of 32 rollouts (built by `make_wall.py`)

## Physical UR5 (two trials per task)

`trail-real-<task>-trial{1,2}.mp4` for: stacking-blocks, packing-blocks, packing-objects,
pushing-piles, pressing-buttons, packing-unseen-blocks, packing-unseen-objects.
Source numbering: 2/3 stacking, 1/4 packing blocks, 5/6 packing objects, 11/12 pushing piles,
9/10 pressing buttons, 13/14 packing unseen blocks, 7/8 packing unseen objects.

## Franka Kitchen (left and right camera views)

`trail-franka-<task>-{left,right}.mp4` for: turn-knob, open-door, flip-switch, open-microwave, slide-door.

## Meta-World MT10

`trail-metaworld-<task>.mp4` for: reach, push, pick-place, door-open, drawer-open, drawer-close,
button-press, peg-insert, window-open, window-close.
