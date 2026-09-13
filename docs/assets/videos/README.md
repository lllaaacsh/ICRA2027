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

## Head-to-head comparison (Meta-World, same initial states)

`cmp-<task>-ep<N>.mp4`, 1440x480, TRAIL | DINOv3 | R3M side by side, 0.5x (40 steps/s).

- Built on BSC by `LARM/metaworld_bc/record_comparison_videos.py` (job script
  `bsc/sbatch_record_mt10_comparison.sh`), then stitched locally by
  `LARM/metaworld_bc/compose_comparison_videos.py <ffmpeg> <raw_dir> <out_dir>`.
- Each method replays its evaluation episodes in order at seed 999999. The job fails if any
  replayed outcome differs from that method's evaluation json, or if initial observations
  differ across methods.
- Checkpoints: TRAIL `mt10_deltaz_ablation_pooled_full`, DINOv3 `mt10_dino_official_chunk1`,
  R3M `mt10_r3m_resnet50_chunk1`.
- Selection rule: the first two episodes where TRAIL succeeds and both baselines fail, plus the
  first where TRAIL fails and a baseline succeeds — Push episodes 1, 2, 6; Pick-Place 5, 9, 12.
- Shorter clips hold their last frame so the three columns loop together.
