# DATA_NOTES: IMPACT v1.1, Model A Reassembly, front view

Status: **M1 review gate.** Items marked ✅ were verified on real files. Items marked ⏳ still need the Hugging Face
release bundles. Run `notebooks/01_download_and_discover.ipynb` and paste back its 📋 outputs to close them.

**Sources inspected (2026-11-03).** A clone of the IMPACT GitHub repo (`Kratos-Wen/IMPACT`, commit `4fed5fa`). Its
`dataset/` folder ships the v1.1 task annotations: PSR, ASR, TAS, PPR, ATR, AF-S, CV and the official splits. Hugging Face
and arXiv were not reachable from the build sandbox, so the release bundles (metadata, videos, features) are not
inspected yet. One real front-view frame was available: the IMPACT project page's `annotation_demo.mp4`, a
half-resolution front-view panel. The pictorial "Manual Book" figure was also available.

The adapters (`src/sopverify/data/impact/`) find files by name pattern, so they work on both the GitHub `dataset/` layout
and the release `annotations/` layout. `tests/test_data_impact.py` runs them on the real files.

---

## 5.2 #1: Annotation file formats ✅

All frame indices are **0-based frames of the front video at 30 fps**. `view_start` is 0 for all 45 pool executions.
Conversion: `t = frame / fps`, implemented in `adapters.frames_to_seconds` and unit-tested for 25, 29.97 and 30 fps.

| Type | File | Format |
|---|---|---|
| TAS-S | `TAS/groundTruth_TAS-S/<vid>.txt` (release: `annotations/TAS-S/…` ⏳) | text, **one label per frame** (5214 lines for a 5214-frame video); labels in `mapping_TAS-S.txt` (26 classes, `0 NULL` … `25 FINISH_ANGLE_GRINDER_ASSEMBLY`) |
| PSR | `PSR/labels/{train,val,test}/<vid>/PSR_labels.csv` | rows `<frame>.jpg,<category_id>,<description>`, one per completion event |
| PSR (errors) | `…/PSR_labels_with_errors.csv` | same format, plus `Incorrectly installed X` events |
| PSR (raw) | `…/PSR_labels_raw.csv` | `<frame>.jpg,<s0>,…,<s16>`: the ASR state vector at each keyframe |
| ASR | `ASR/annotations/<vid>_asr.json` | JSON: `fps`, `frame_count`, `view_start/end`, `meta_data.resolution`, `components[17]`, `state_sequence[{frame, state[17]}]`, states ∈ {−1, 0, 1} |
| PPR-L/R | `PPR/groundTruth_PPR_{L,R}/<vid>.txt` | text, one label per frame: `NORMAL`, `ANOMALY` or `RECOVERY` (`mapping_PPR.txt`) |
| ATR-L/R | `ATR/groundTruth_ATR_{L,R}/<vid>.npy` | uint8 `[T, 6]`, one column per anomaly type (`mapping_ATR.txt`: temporal, spatial, handling, wrong part, wrong tool, procedural); masks in `mask_ATR_*` |

Annotations are **per view**: every view has its own file (`<execution>_<view>`). V1 reads only `_front`.

Excerpts, from `AL07EJ17_Reassembly_A_002_front`:

```text
PSR_labels.csv               PSR_labels_with_errors.csv              ASR state_sequence (frames)
982.jpg,3,Install gearbox_housing      653.jpg,4,Incorrectly installed gearbox_housing   0:   all 0
982.jpg,6,Install drive_shaft          982.jpg,3,Install gearbox_housing                 653: housing/shaft/gear/M6 nut = -1
982.jpg,9,Install bevel_gear           1461.jpg,13,Incorrectly installed adapter_plate   982: housing/shaft/gear/M6 nut = 1
982.jpg,48,Install M6_nut              …                                                 …
1910.jpg,24,Install screw_adaptor_lowright                                                5176: all 1
…
5176.jpg,0,Install anti_vibration_handle
```

**ASR state −1 is mostly transitional.** In this execution *every* component passes through −1 ("Incorrectly installed")
on its way from 0 to 1. So −1 means "placed but not yet fastened" as well as "misinstalled". A final −1 is a real
misinstallation. This affects ground truth (D-005).

## 5.2 #2: Prerequisite graph ⚠️ released, but not usable for this SOP

- `dataset/PSR/labels/procedure_info_IMPACT.json` lists the **51 PSR categories = 17 components × {Install,
  Incorrectly installed, Remove}**. Its `expected_*` fields are all false, and it contains no edges.
- The only released graph is `tasks/PSR/gemini_3_1_pro/configs/procedure_graph.json`. It is **mined from data** (92 videos,
  disassembly and reassembly, `min_confidence 0.9`), with 53 nodes of types `install_ok / recover_ok / remove_ok`, 102
  edges and **zero install→install edges**. The other PSR baselines learn a graph from each training split.
- Decision D-004: use a **manual** graph (drafted, pending review). `inferred_all` and `inferred_demos` are E6 ablations.

## 5.2 #3: Instruction manual / SOP text ⚠️ pictorial only

No text manual is released. The project page has a pictorial "Manual Book" (`website/assets/figures/Manual_Book.svg`)
showing the standardized order. Reassembly: rotor (drive shaft with bevel gear) into the housing and M6 nut tightened
with a wrench → adapter plate screwed on → bearing plate pushed in and screwed → locking lever with spring → anti-vibration
handle last. `procedures/impact_reassembly_A/sop.md` is drafted from this and from the step labels, and is marked DRAFT.

## 5.2 #4: Metadata ⏳ (expertise) / ✅ (rest)

- Model A vs B, Reassembly vs Disassembly and trial number come from the execution id
  (`<participant>_<task>_<A|B>_<NNN>`). ASR also states `workflow: assemble` and `model_type: CG15-125BL`.
- Per-execution fps, frame count and resolution are in every ASR JSON (`meta_data`).
- **Expert vs novice is not in the GitHub annotations.** It should be in the release's `metadata/` ⏳. The catalog has a
  best-effort parser, plus `data.expertise_overrides` in the config to fill it in by hand. Until then every execution is
  `unknown`, and the demo rule falls back to "complete + low anomaly" (D-003).

## 5.2 #5: Official splits ✅ (recorded, not used)

- `S1`–`S4` per task: `TAS/splits_TAS-S/{train,val,test}.split{1..4}.bundle` (one `<execution>_<view>` per line), same for
  PPR, ATR and CV. S1 IID, S2 cross-subject, S3 cross-configuration, S4 exo→ego (`docs/BENCHMARK.md`).
- PSR/ASR use only front-view `split1`, in folder form (`PSR/labels/{train,val,test}/`). Model A Reassembly in split1:
  **31 train, 5 val, 9 test**.
- V1 uses its own participant-disjoint dev/test split (spec 9.2).

## 5.2 #6: VideoMAEv2 feature layout ⏳ (documented, to verify)

From `docs/FEATURES.md` and `tools/features/extract_videomaev2.py`: VideoMAEv2-G (`OpenGVLab/VideoMAEv2-giant`), **1408-D,
stored `1408 × T`, frame-aligned**. There is one feature per video frame, from a 16-frame clip centred on it (border frames
repeated), so feature index k corresponds to frame k at time k/30 s. The bundle is 21.6 GiB for all five views, and there is
no per-view bundle. `shipped.py` transposes when needed and resamples onto the 2 fps grid by nearest frame. Notebook 01 §8
checks `T == n_frames` for every pool file. Notebook 02 §4 checks alignment by cross-correlating feature-change signals.

## 5.2 #7: Official evaluation code ✅

`tasks/PSR/gemini_3_1_pro/evaluate_psr.py` (Step F1, delay, POS) and `tasks/ASR/gemini_3_1_pro/evaluate_asr.py`. The PSR
function `determine_step_metrics(gt, pred, proc_info)` accepts `{frame, id, conf}` entries. `sopverify.eval.official`
converts `psr` step-set verdicts into that format (F1 and delay). POS is computed inside `main()` and is not reusable as is.

## 5.2 #8: Counts for Model A Reassembly, front view ✅

| Quantity | Value |
|---|---|
| Executions with front annotations (TAS-S + PSR + ASR) | **45** |
| Participants | **13** (3–4 executions each) |
| Experts | ⏳ metadata (paper: 4 experts, 9 novices across the dataset) |
| Resolution / fps (ASR metadata) | **1280×720** for all 45 (spec said 1280×800, D-002), 30 fps |
| Duration | min 132 s, **median 224 s**, max 1025 s (spec assumed about 20 min) |
| Executions with zero ANOMALY phases on both hands | **0**. Median anomaly segment 1–4 s; anomaly time 1–48 % per hand |
| Demo candidates (all parts installed, no Remove, ≤ 5 % anomaly per hand) | **12** from 8 participants (D-003) |
| Executions not ending fully installed | **6**: LE06AS03 × 4 (5–6 components at −1, adapter-plate group never installed), KI05KO01_003 and KJ03JM25_003 (one component at 0) |
| Executions with Remove events (rework) | **4** |
| PSR install events per execution | 17 in 38 executions; 11–32 otherwise (rework, missing groups) |
| Distinct completion instants per execution | **7** in 40 of 45 executions (several parts are logged at the same frame) |

**Ground truth under the draft manual graphs** (`sopverify procedure stats`, D-014):

| Step set | Compliant | Non-compliant | Not-done step instances | Order-violation instances | Executions with a violation |
|---|---:|---:|---:|---:|---:|
| grouped (5) | 36 | 9 | 8 | 10 | 8 |
| psr (17) | 36 | 9 | 24 | 25 | 8 |
| tas_s (13, 5 verifiable) | 39 | 6 | 20 | 7 | 5 |

Real negatives are scarce, which matches the spec's risk table, so truncation negatives (spec 9.4) carry most of the
not-done evaluation. Order support is below 10 on the test split, so it is reported with a warning.

## Other findings that shaped the design

- **Co-occurring events, which define the grouped steps (D-007).** Install events cluster into 7 instants:
  {gearbox_housing, drive_shaft, bevel_gear, M6_nut} (43/45), {screw_adaptor_lowright, M4_nut_plate_lowright},
  {adapter_plate, screw_adaptor_topleft, M4_nut_plate_topleft}, {bearing_screw_topleft},
  {bearing_plate, bearing_screw_lowright}, {screw_lever, spring, lever, washer} (44/45), {anti_vibration_handle}.
  A plate counts as "installed" when its last screw is in.
- **Typical order** (37 of 45 executions): rotor → adapter plate → bearing plate → locking lever → handle.
- **The last step barely appears on camera (D-006).** The anti-vibration handle completes a median of **2.4 s** (min 0.7 s)
  before the recording ends, and the locking lever 13.6 s before. The spec defaults (3 s positive margin, 5 s minimum ON
  run) would make the handle unlearnable and undetectable.
- **TAS-S in reassembly** uses 13 labels, plus 5 disassembly labels during rework. Only 5 of the 13 describe a lasting
  end-state (D-009).
- **Front view**: the camera faces the participant across the table. The device is assembled over a grey area at the
  table centre (about x 575–690, y 450–570 on a 1280×720 frame), with part bins along the bottom edge. Draft ROI
  `[440, 300, 480, 360]`, pending confirmation on a full-resolution frame (M2 gate).
- **License**: the dataset is CC BY-NC-SA 4.0 (`LICENSE-DATA`). No frames, features or annotations are committed, apart
  from label names in `vocab.yaml`.
