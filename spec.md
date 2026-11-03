# spec.md — SOP Compliance Checker, Version 1

**Project codename:** `sopverify`
**Target builder:** Claude Code
**Dataset for V1:** IMPACT v1.1 — Model A (Fein CG15-125BL) **Reassembly**, **front exocentric view only**
**Compute:** Google Colab Pro (A100 / G4 GPU) · **Storage:** GitHub repo (code) + Google Drive (data, caches, runs)
**Rigor level:** Solid prototype — typed modules, unit tests, reproducible configs, evaluation harness, logging

---

## 0. Instructions for the builder (read first)

1. Read this whole spec before writing code. Work **milestone by milestone** (Section 13). At the end of each milestone, run its acceptance checks, then update `docs/PROGRESS.md` with what was done, what was verified, and what remains.
2. **Never invent dataset formats.** Section 5 lists what is known about IMPACT and what is not. Inspect the real files, record findings in `docs/DATA_NOTES.md`, and write adapters against what you actually find. If the real data contradicts this spec, the data wins: record the conflict and your resolution in `docs/DECISIONS.md`.
3. **Human review gates** are marked **[GATE]**. At a gate, stop, summarize what you need reviewed, and wait for the user before continuing.
4. **Environment split.** You (Claude Code) write code, tests and notebooks in the repo. Anything that needs the GPU or the full dataset runs in **Colab notebooks** that you write and the user executes. Those notebooks `pip install` this repo from GitHub. Notebook outputs the user pastes back (tables, logs) go into `docs/`. Everything that does not need the GPU or the full data must be runnable and tested locally.
5. **Test-label firewall.** The inference pipeline must never read annotations of the video it is judging. Only the `eval` package reads ground truth for test videos (enforcement in Section 9.1).
6. Keep everything **config-driven and resumable**. Colab sessions reset: every stage must skip work whose outputs already exist in the cache, keyed by a config hash.

---

## 1. Goal

Build an end-to-end system with this contract. It is given:

- an **SOP** (a structured list of steps with prerequisites),
- **K = 2–3 expert demonstration videos** of that SOP, with step timing for the demos only,
- a **new video** of someone performing the SOP, from the same fixed camera,

and it outputs, for every step, whether it was **done**, **not done**, **done out of order**, or **can't tell**, with a timestamp and an evidence keyframe. It also gives an overall **compliant / non-compliant / needs review** verdict.

V1 is built and evaluated on IMPACT, but the design must be **SOP-agnostic**. Everything specific to one procedure lives in a *procedure package* (Section 7.1). A future procedure, such as the user's own SOP, must only need a new package plus demo videos, with no code changes.

### Core design principle

**Verify end-states, not actions.** For each step, learn from the demo frames what the workspace looks like *after* that step is complete. Then detect when that end-state appears in the test video, and check the completion order against the SOP's prerequisite graph.

Fine-grained hand-action recognition is out of scope for V1. The IMPACT paper shows it is the hardest layer, and 2–3 demos cannot support it.

---

## 2. Scope

### In scope (V1)
- Offline analysis of recorded front-view videos.
- **Configurable step granularity:** `grouped` (default), `tas_s` (IMPACT's coarse steps) and `psr` (IMPACT's step-completion events).
- Few-shot state scorers built on frozen features. No backbone fine-tuning.
- Two feature families, compared:
  - IMPACT's shipped VideoMAEv2 features (full frame), and
  - self-extracted DINOv2 features (full frame and workspace crop).
- An open-weight VLM (Qwen3-VL family), used both as an alternative state scorer and as an optional verifier for uncertain steps.
- A rule-based compliance checker, JSON + HTML reports, and a **Gradio** web app (runs inside Colab with a share link).
- An evaluation harness with baselines, ablations and a written results summary.

### Non-goals (V1)
- Real-time or streaming inference.
- Egocentric, depth, audio or gaze inputs. Multi-view fusion.
- Hand-level action recognition, tool-identity checks, anomaly-type classification.
- Fine-tuning video backbones or VLMs.
- Automatic onboarding of a new SOP from text alone.
- Production packaging (Docker, CI/CD, auth).

---

## 3. Definitions

| Term | Meaning |
|---|---|
| **Procedure package** | Folder with the SOP text, step sets, prerequisite graph, ROI config and GT-mapping rules for one procedure. |
| **Step set** | One granularity of steps (`grouped`, `tas_s`, `psr`). Each step has an id, a name, a description, prerequisites (`requires`), a `required` flag, a `visually_verifiable` flag and a GT mapping. |
| **Prerequisite graph** | DAG over the steps of a step set. Edge `p → s` means `p` must be complete before `s` completes. Steps without a path between them may occur in either order. |
| **Completion time** | The time at which a step's end-state first appears and persists. |
| **Demo** | One of the K expert executions. Its annotations may be used to train scorers. |
| **Test / dev execution** | An execution being judged. Its annotations are used **only** by `eval`. |
| **Score trace** | Array `[T, S]` holding the per-step probability that step *s* is complete at time *t*, on a uniform time grid. |
| **Step verdict** | One of `DONE`, `NOT_DONE`, `OUT_OF_ORDER`, `UNCERTAIN`, `NOT_VERIFIABLE`. |
| **Video verdict** | One of `COMPLIANT`, `NON_COMPLIANT`, `NEEDS_REVIEW`. |

---

## 4. Environment and storage

### 4.1 Layout on Google Drive
```
/content/drive/MyDrive/sopverify/
├── data/IMPACT-v1.1/        # extracted dataset (only the bundles we need)
├── cache/                   # frames, features, score traces, keyed by config hash
├── runs/<run_id>/           # one folder per experiment run
└── exports/                 # HTML reports, results tables
```
Colab notebooks copy the videos they need to local scratch (`/content/scratch`) before decoding, because Drive I/O is slow, and write outputs back to Drive.

### 4.2 Paths config (overridable by environment variables)
```yaml
paths:
  data_root:   ${SOPVERIFY_DATA_ROOT:-/content/drive/MyDrive/sopverify/data/IMPACT-v1.1}
  cache_root:  ${SOPVERIFY_CACHE:-/content/drive/MyDrive/sopverify/cache}
  runs_root:   ${SOPVERIFY_RUNS:-/content/drive/MyDrive/sopverify/runs}
  scratch:     /content/scratch
```

### 4.3 Dependencies
- **Python version:** whatever Colab currently ships (≥ 3.10). Use the preinstalled `torch`.
- **Core:** `numpy`, `scipy`, `pandas`, `scikit-learn`, `pydantic>=2`, `pyyaml`, `networkx`, `av` (PyAV) for decoding, `opencv-python-headless`, `transformers`, `pillow`, `tqdm`, `jinja2`, `plotly`, `typer` (CLI).
- **App:** `gradio`.
- **Optional extra `[vlm]`:** `vllm` (preferred on A100) or `transformers` + `qwen-vl-utils` as a fallback.
- **Dev:** `pytest`, `ruff`, `mypy` (lenient).
- Pin versions in `pyproject.toml` once confirmed working on Colab. Record the working set in `docs/DECISIONS.md`.

### 4.4 Models (all configurable)
- **Image encoder:** default `facebook/dinov2-base`; option `facebook/dinov2-large`. Verify the Hugging Face IDs at build time.
- **VLM:** default a Qwen3-VL ~8B Instruct checkpoint. The IMPACT paper evaluated Qwen3-VL-8B. Verify the exact Hugging Face ID at build time and record it. Run in bf16 on the A100.

---

## 5. Data: IMPACT v1.1

### 5.1 Known facts (from the paper and the Hugging Face dataset card; verify)
- **Download tooling:**
  ```bash
  git clone https://github.com/Kratos-Wen/IMPACT.git
  pip install -r requirements-tools.txt
  python tools/download_impact.py --list
  python tools/download_impact.py --include <bundle> --output downloads
  python tools/verify_release.py <zip>
  ```
  A Google Drive mirror also exists (linked from the dataset card).
- **Full release size:** about 201 GB. **Download only:** the annotations bundle, front-view RGB videos (selectively, if the bundle keys allow it), the VideoMAEv2 features (front view) and the 2.8 GB quick-start sample.
- **Release layout:**
  ```
  IMPACT-v1.1/
  ├── annotations/{TAS-S,TAS-B,AF-S,ASR,PSR,PPR,ATR}/
  ├── videos/{ego,front,left,right,top}/
  ├── depth/{front,left,right,top}/
  ├── features/{I3D,MViTv2,VideoMAEv2}/   # VideoMAEv2-G is 1408-D
  └── metadata/
  ```
- **Identifiers:**
  - The canonical id is `<execution_id>_<view>`.
  - Execution ids look like `AL07EJ17_Reassembly_A_002`, i.e. `<participant>_<Disassembly|Reassembly>_<A|B>_<NNN>`.
  - Mapping files exist. For example, `annotations/PPR/mapping_PPR.txt` contains lines like `0 NORMAL`, `1 ANOMALY`, `2 RECOVERY`.
- **Content and labels:**
  - The front view is 1280×800 at 30 fps.
  - TAS-S has 26 coarse step labels.
  - PSR has 51 completion-event categories, derived from a partial-order prerequisite graph.
  - ASR has 17 component instances with states in {−1 misinstalled, 0 uninstalled, 1 installed}.
  - PPR holds per-hand phases.
  - ATR holds six non-exclusive anomaly types.
  - Four participants are experts and nine are novices.
- **Version:** v1.1 (July 2026) revalidated annotations and updated some RGB videos. Use v1.1 consistently. The paper's numbers are from v1.0.
- **License:** the dataset card says **CC BY-NC-SA 4.0**, while the paper says CC BY 4.0. Treat it as non-commercial and share-alike. Do not redistribute derived data or features publicly. Record this in the README.

### 5.2 Unknowns to resolve in M1 (document all of them in `docs/DATA_NOTES.md`)
1. **Annotation file formats** for TAS-S, PSR, ASR, PPR and ATR: extension, columns, frame index vs seconds, which fps/frame basis, one file per execution or per view.
2. Whether a **prerequisite graph file** is released for PSR, and in what format.
3. Whether the **instruction manual / SOP text** is released.
4. **Metadata:** how to tell expert from novice, Model A from B, and Reassembly from Disassembly; whether there is a per-execution duration and fps.
5. **Official split files** (S1–S4): names and location. Not used for our splits, but record them.
6. **VideoMAEv2 feature layout:** file format, shape, temporal stride and window, and how feature index maps to video frame/time. Check the repository's `docs/FEATURES.md`.
7. **Official evaluation code** for PSR and ASR: location, inputs, and whether it can be called on our predictions.
8. **Counts for Model A Reassembly, front view:**
   - number of executions, participants and experts;
   - executions with zero ANOMALY phases (candidate "clean" demos);
   - number of GT order violations and never-completed steps (Section 9.3);
   - any misinstalled (−1) final states.

### 5.3 Data subset for V1
**Pool** = all `*_Reassembly_A_*` executions that have a front-view video plus TAS-S, PSR and ASR annotations. Expect roughly 40–46 executions; confirm in M1.

---

## 6. Architecture and repository layout

### 6.1 Pipeline

```
procedure package ─────────────────────────────────────────────┐
demo videos + demo annotations ─► demo label builder ─┐        │
                                                       ▼        ▼
test video ─► decode @ sample_fps ─► ROI crop ─► features ─► state scorer ─► score trace [T,S]
                                                                                │
                                                                    temporal smoothing (per step, independent)
                                                                                │
                                                                     step completion estimates
                                                                                │
                                                          (optional) VLM verifier for UNCERTAIN steps
                                                                                │
                                                                    compliance checker (rules + graph)
                                                                                │
                                                          verdict JSON ─► HTML report ─► Gradio app
```

### 6.2 Repository layout
```
sopverify/
├── README.md                     # quickstart (Colab), license notes
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   └── experiments/*.yaml        # one file per experiment in Section 9.6
├── procedures/
│   └── impact_reassembly_A/
│       ├── procedure.yaml
│       ├── sop.md                # human-readable SOP text
│       ├── step_sets/{grouped,tas_s,psr}.yaml
│       └── roi.yaml
├── src/sopverify/
│   ├── config.py                 # pydantic config models, config hashing
│   ├── procedure/                # schema.py, loader.py, graph.py (DAG checks, transitive reduction, inference)
│   ├── data/impact/              # download.py, discover.py, adapters.py, catalog.py, splits.py
│   ├── video/                    # decode.py, timegrid.py, roi.py
│   ├── features/                 # base.py, dinov2.py, shipped.py, cache.py
│   ├── state/                    # labels.py (demo labels), scorers/{prototype,linear,knn,time_prior,oracle,vlm}.py
│   ├── temporal/                 # smoothing.py
│   ├── vlm/                      # backend.py (vllm|transformers), prompts.py, scorer_impl.py, verifier.py
│   ├── compliance/               # verdicts.py, checker.py
│   ├── report/                   # json_report.py, html_report.py, templates/
│   ├── eval/                     # ground_truth.py, truncation.py, metrics.py, runner.py, tables.py
│   ├── pipeline.py               # run(video, procedure, demos, config) -> VerdictReport
│   ├── cli.py                    # typer CLI
│   └── app.py                    # Gradio app
├── notebooks/
│   ├── 00_setup_colab.ipynb
│   ├── 01_download_and_discover.ipynb
│   ├── 02_extract_features.ipynb
│   ├── 03_run_experiments.ipynb
│   ├── 04_vlm.ipynb
│   └── 05_app.ipynb
├── tests/
└── docs/
    ├── DATA_NOTES.md  DECISIONS.md  PROGRESS.md  RESULTS.md
```

---

## 7. Data schemas

All schemas are pydantic models. JSON and YAML files are validated on load.

### 7.1 Procedure package — `procedure.yaml`
```yaml
procedure_id: impact_reassembly_A
name: "Angle grinder reassembly (Fein CG15-125BL)"
sop_text: sop.md
default_step_set: grouped
step_sets:
  grouped: step_sets/grouped.yaml
  tas_s:   step_sets/tas_s.yaml
  psr:     step_sets/psr.yaml
graph_source: released        # released | manual | inferred_all | inferred_demos  (see 8.2)
roi: roi.yaml
camera: front
```

### 7.2 Step set — `step_sets/<name>.yaml`
```yaml
step_set: grouped
steps:
  - id: G03
    name: "Mount adaptor plate"
    description: "Place adaptor plate on gearbox housing and fasten both screws."
    requires: [G01, G02]
    required: true
    visually_verifiable: true
    roi: null                    # optional per-step crop override [x, y, w, h]
    gt_mapping:                  # how eval derives ground truth from IMPACT annotations
      psr_events: [ "<psr_event_id>", "<psr_event_id>" ]   # complete when ALL listed events complete
```

**GT mapping rules per step set:**
- **`psr`:** one step per PSR event; `gt_mapping.psr_events: [id]`.
- **`tas_s`:** one step per TAS-S label; `gt_mapping.tas_s_labels: [label]`. Completion = end frame of the **last** segment with that label. Document this rule in `DECISIONS.md`.
- **`grouped`:** each step lists the PSR events it covers. Completion = the time the **last** member event completes. Target 8–15 steps, each with a clearly visible end-state from the front view.

**[GATE]** The `grouped` step set is drafted by the builder from label names and annotation statistics, then **reviewed and edited by the user** before it is used in any experiment.

### 7.3 Canonical execution record (internal, produced by IMPACT adapters)
```json
{
  "execution_id": "AL07EJ17_Reassembly_A_002",
  "participant": "AL07EJ17",
  "task": "Reassembly", "model": "A", "trial": 2,
  "expertise": "expert",
  "view": "front", "video_path": "...", "fps": 30.0, "n_frames": 37800, "duration_s": 1260.0,
  "psr_events":     [{"event_id": "...", "time_s": 41.2}],
  "tas_s_segments": [{"label": "...", "start_s": 0.0, "end_s": 12.4}],
  "asr_states":     {"times_s": [...], "components": [...], "states": [[...]]},
  "phases":         [{"hand": "L", "phase": "ANOMALY", "start_s": 300.1, "end_s": 312.0}],
  "anomaly_types":  [{"start_s": 300.1, "end_s": 312.0, "types": ["temporal"]}],
  "has_anomaly": true
}
```
All internal times are **seconds**. Adapters convert from whatever frame basis the annotations use (see 5.2). Unit-test the conversion.

### 7.4 Score trace (cache file `.npz`)
`times_s [T]`, `scores [T, S]` (float32, 0–1), `step_ids [S]`, plus metadata `scorer`, `config_hash`, `execution_id`.

### 7.5 Verdict report (JSON)
```json
{
  "execution_id": "...", "procedure_id": "impact_reassembly_A", "step_set": "grouped",
  "video_verdict": "NON_COMPLIANT",
  "summary": {"done": 10, "not_done": 1, "out_of_order": 1, "uncertain": 0, "not_verifiable": 1},
  "steps": [
    {
      "step_id": "G03", "name": "Mount adaptor plate",
      "verdict": "OUT_OF_ORDER",
      "completion_time_s": 412.5,
      "confidence": 0.87,
      "violated_prerequisites": ["G02"],
      "flags": ["possible_rework"],
      "evidence_frame": "evidence/G03.jpg",
      "verifier": {"used": false}
    }
  ],
  "scorer": "linear_probe:dinov2-base:roi",
  "config_hash": "…", "git_commit": "…", "runtime_s": 6.1
}
```

---

## 8. Component specifications

### 8.1 Data adapters (`data/impact/`)
- **`discover.py`:** walks the extracted release and writes a machine-readable inventory (`cache/inventory.parquet`) plus a human summary for `DATA_NOTES.md`.
- **`adapters.py`:** converts raw annotations into canonical records (7.3). One function per annotation type.
- **`catalog.py`:** queries such as "all Model A Reassembly front-view executions" or "clean expert executions" (zero ANOMALY phases on either hand).
- **CLI:** `sopverify data summarize` prints a table with one row per execution (participant, expertise, duration, #anomaly segments, #PSR events, has_front_video, has_features).

### 8.2 Procedure package and graph (`procedure/`)
- **Validation:**
  - unique step ids;
  - every id in `requires` exists;
  - the graph is a **DAG** (report any cycle);
  - every step is reachable;
  - `gt_mapping` ids exist in the annotation vocabulary.
- **`graph_source` options:**
  - **`released`:** load IMPACT's PSR graph if it exists (M1). For `grouped`, derive edges: `A requires B` if any member event of A has a path from any member event of B. Then apply transitive reduction.
  - **`manual`:** hand-written `requires` in the step-set YAML (reviewed by the user).
  - **`inferred_all`:** infer `p → s` if `p` completed before `s` in 100% of pool executions where both completed, with at least `min_support` executions (default 5). Then apply transitive reduction. This uses procedure knowledge from all executions; flag it in `RESULTS.md` as "oracle procedure knowledge."
  - **`inferred_demos`:** the same rule using demos only. Expect it to over-constrain the graph, producing false OUT_OF_ORDER flags. Keep it as an ablation.
- **Rendering:** render the graph to SVG for the app and the report.
- **[GATE]** Show the user the chosen graph for the `grouped` set (rendered) before M5.

### 8.3 Video and time grid (`video/`)
- Decode the front-view video with PyAV and sample at `sample_fps` (default **2.0**) onto a uniform grid `times_s`.
- **ROI:**
  - one static workspace box per procedure (`roi.yaml`, front-view pixel coordinates), with optional per-step overrides;
  - provide a notebook cell and an app helper that show one frame with the box overlaid, so the user can set or confirm it. **[GATE]** The user confirms the ROI.
- **Frame cache:** store frames as a per-video uint8 array, resized to the encoder's input. Do not store thousands of JPEGs on Drive.

### 8.4 Features (`features/`)
- **`FeatureExtractor` interface:** `extract(execution) -> (times_s [T], feats [T, D])`, cached by `(execution_id, extractor_name, crop, sample_fps, model_id)`.
- **`dinov2`:**
  - CLS token, optionally concatenated with mean-pooled patch tokens (config);
  - `crop: full | roi`;
  - L2-normalize;
  - batch size tuned for the A100.
- **`shipped_videomaev2`:**
  - load IMPACT's released features for the front view;
  - **resample onto the same `times_s` grid** with nearest-neighbor resampling, after determining the feature-to-time mapping in M1;
  - full frame only (no crops possible).
- **Performance target:** DINOv2-base features for one ~20-minute video at 2 fps in under 2 minutes on an A100, excluding decode.

### 8.5 Demo label builder (`state/labels.py`)
For each demo, each step `s` with GT completion time `c_s`:
- **Positive frames:** `t ≥ c_s + pos_margin_s` (default 3 s).
- **Negative frames:** `t ≤ c_s − neg_margin_s` (default 10 s).
- **Excluded:** the band in between, which is the transition and is often occluded by hands.
- **Never completed in a demo:** all frames are negatives for that step. Log it; this should be rare in clean demos.

Output a design matrix per step: features, labels and demo id (for leave-one-demo-out calibration).

### 8.6 State scorers (`state/scorers/`)
**Common interface:**
```python
class StateScorer(Protocol):
    name: str
    def fit(self, demos: list[DemoData], steps: list[Step]) -> None: ...
    def score(self, times_s: np.ndarray, feats: np.ndarray | None, execution: ExecutionRef) -> np.ndarray:  # [T, S] in [0,1]
        ...
```

**Implementations:**
1. **`prototype`** (no training): per step, compute the mean of positive features (after-prototype) and of negative features (before-prototype). Score = `sigmoid(α · (cos(x, after) − cos(x, before)))`. α is calibrated on demos.
2. **`linear_probe`:** per step, class-balanced L2 logistic regression on standardized features (sklearn). Choose C by leave-one-demo-out when K ≥ 2.
3. **`knn`:** per step, the fraction of positive labels among the k nearest demo frames (cosine). k is configurable (default 15).
4. **`time_prior`** (baseline): `p_s(t) = 1` if `t / duration ≥ mean normalized completion time of s across demos`, else 0. This is a sanity baseline: a visual scorer must beat "progress clock."
5. **`oracle`** (testing only): builds traces from GT. Used to verify that the checker and metrics are correct. Must be refused by `pipeline.run` unless `allow_oracle=True` (eval-only).
6. **`vlm`:** see 8.9.

### 8.7 Temporal smoothing (`temporal/smoothing.py`)
Applied **independently per step**. The prerequisite graph is **not** used here, so that order violations stay detectable.

1. Median filter, window `median_window_s` (default 2.5 s).
2. Hysteresis with `tau_on` (0.6), `tau_off` (0.4) and minimum duration `min_on_s` (5 s); this yields ON runs.
3. **Per step output:**
   - `final_on`: whether the last sustained ON run reaches the end of the video;
   - `completion_time_s`: start of the final ON run, if `final_on`;
   - `confidence`: mean smoothed score over the final ON run, or `1 − max smoothed score` when there is no ON run;
   - `max_score`, the full list of ON runs, and the flag `possible_rework` when there is more than one ON run or an ON run that ends before the video ends.
4. **Evidence frame:** the frame at `completion_time_s + evidence_offset_s` (default 3 s), cropped to the ROI and saved as JPEG.

All thresholds live in config and are tuned on **dev** only.

### 8.8 Compliance checker (`compliance/checker.py`)
Deterministic rules, applied in order.

**Per step `s`:**
1. If `visually_verifiable == false` → `NOT_VERIFIABLE`.
2. If `final_on` → provisionally `DONE`.
3. Else if `max_score < tau_low` (default 0.3) → `NOT_DONE`.
4. Else → `UNCERTAIN`.

**Order check, for each provisionally DONE step `s` and each `p ∈ requires(s)`:**
- If `p` is DONE and `completion(s) < completion(p) − order_tolerance_s` (default 5 s) → `s` becomes `OUT_OF_ORDER`, with `violated_prerequisites += [p]`.
- If `p` is NOT_DONE → `s` becomes `OUT_OF_ORDER` (done without its prerequisite). `p` stays NOT_DONE.
- If `p` is UNCERTAIN → `s` keeps DONE, flagged `prereq_uncertain`.

**Video verdict:**
- `NON_COMPLIANT` if any `required` step is `NOT_DONE` or `OUT_OF_ORDER`;
- else `NEEDS_REVIEW` if any `required` step is `UNCERTAIN`;
- else `COMPLIANT`.

`NOT_VERIFIABLE` steps never change the verdict. The report lists them under "manual check required."

**Testing:** the checker must be exhaustively unit-tested with table-driven cases, including diamonds in the graph, missing prerequisites, ties within tolerance, and all-UNCERTAIN.

### 8.9 VLM components (`vlm/`)
**Backend.** `VLMBackend` with two implementations: `vllm` (preferred) and `transformers`.
- Use deterministic decoding (temperature 0).
- Request JSON output. Use constrained or guided decoding if the backend supports it; otherwise parse robustly and retry up to 2 times.
- Log every prompt and response to the run folder.

**VLM scorer (alternative to feature scorers):**
- Sample keyframes every `vlm_keyframe_s` (default 10 s); use ROI crops at most 768 px on the long side.
- Per keyframe, prompt with the SOP step list (id, name, end-state description). Optionally add 1 reference "after" crop per step from a demo (`in_context_refs: true`).
- Ask for a JSON object `{step_id: {"complete": true|false|null, "confidence": 0-1}}`.
- Convert answers to probabilities: `null` → 0.5, true → confidence, false → 1 − confidence.
- Interpolate onto `times_s` (step-hold) and pass through the **same** smoothing and checker.
- Batch keyframes. If the prompt exceeds the context limit, split the step list into chunks.

**VLM verifier (optional post-step).** For each `UNCERTAIN` step (and optionally low-confidence `NOT_DONE` steps):
- **Input:** the step text, 2 demo "after" reference crops, 2 demo "before" reference crops, and 3 test crops (last 10% of the video, plus the time of `max_score`).
- **Question:** is the step's end-state present at the end of the test video? Answer `yes | no | cannot_tell` with a short rationale.
- **Mapping:** yes → DONE (completion_time_s = time of max_score); no → NOT_DONE; cannot_tell → stays UNCERTAIN.
- Record the result under `verifier`. Then re-run the order check and the video verdict.

**Resource budget:** a full VLM-scorer run over all test executions should finish in under 2 hours on an A100. Measure it and report it.

### 8.10 Pipeline entry point (`pipeline.py`)
```python
def run(video: ExecutionRef | Path, procedure: Procedure, demos: list[DemoRef], cfg: Config) -> VerdictReport
```
- Accepts an IMPACT execution reference or an uploaded video path. An uploaded video must come from the same camera setup; warn if its resolution differs from 1280×800.
- Never touches GT for `video`.
- Writes the verdict JSON, evidence frames and the score trace to `runs/<run_id>/<execution_id>/`.

### 8.11 Reports (`report/`)
**JSON:** schema 7.5.

**HTML (single self-contained file, Jinja2 + Plotly inline):**
- a header with the video verdict badge;
- a step table (verdict colors, times, confidence, flags);
- a timeline chart showing per-step score traces with thresholds and predicted completion markers;
- **GT overlay** (GT completion markers, anomaly spans) when the video is an IMPACT execution and `show_gt: true`. Label it clearly as evaluation-only;
- a gallery of evidence keyframes;
- the rendered prerequisite graph with nodes colored by verdict;
- the config and git hash.

### 8.12 CLI (`cli.py`)
```
sopverify data summarize
sopverify procedure validate procedures/impact_reassembly_A --step-set grouped
sopverify features extract --config configs/default.yaml --executions pool
sopverify run --execution AL07EJ17_Reassembly_A_002 --config configs/default.yaml
sopverify eval --config configs/experiments/E3_scorers.yaml
sopverify report --run runs/<run_id>
sopverify app --share
```

---

## 9. Evaluation protocol (`eval/`)

### 9.1 Test-label firewall
- Only `eval/ground_truth.py` may read annotations of non-demo executions.
- `pipeline.py`, `state/` and `vlm/` receive demo annotations through `DemoData` objects only.
- Add a test that statically checks that `pipeline`, `state`, `temporal`, `compliance` and `vlm` do not import `sopverify.eval`.

### 9.2 Splits
- **Demos (user choice: any clean expert trials):** draw K executions from the clean expert executions (zero ANOMALY phases). K ∈ {1, 2, 3}, **default K = 2**. Repeat over `n_demo_seeds` (default 5) different draws and report mean ± std.
- **Dev / test:** split the remaining pool executions **participant-disjoint** into dev (~30%) and test (~70%), with a fixed seed. Tune all thresholds and hyperparameters on dev only. Report test once per configuration.
- **Participant overlap:** demos may share participants with dev/test. Report test metrics separately for **seen-participant** and **unseen-participant** test executions. Also provide an optional `demo_mode: held_out_participant` that removes demo participants from dev/test.

### 9.3 Ground truth per execution and step set
- **`gt_completion_s[s]`:** the final completion time from `gt_mapping`. Null if never completed. If a step's events complete, are undone and complete again, use the final completion and mark `gt_rework`.
- **`gt_done[s]`:** completion exists and the step's end-state holds at the end of the video. Use ASR final states where the mapping allows; otherwise completion exists.
- **`gt_order_violation[s]`:** true if `gt_completion(s) < gt_completion(p)` for some `p ∈ requires(s)` under the active graph.
- **`gt_video_compliant`:** all required `gt_done`, and no `gt_order_violation`. Optionally (config `final_state_check`), also no component in ASR state −1 at the end.
- **Diagnostics:** in M1/M4, report how many executions, steps and order violations each GT category actually contains. This tells us whether real negatives are scarce.

### 9.4 Synthetic negatives by truncation
IMPACT participants usually finish the procedure, so real "not done" cases may be rare. Create **truncated variants** of each dev/test execution:
- Cut at a time chosen between GT completions so that `m ∈ {1, 2, 3}` required steps remain incomplete. Use a deterministic seed and `n_truncations_per_video` (default 3).
- Implement truncation by **slicing cached features and traces**. No re-encoding.
- Recompute GT on the truncated range. Recompute `time_prior` with the truncated duration. The baseline sees the shorter video, which is honest.
- Report metrics on originals and truncations separately, and combined.

### 9.5 Metrics (`eval/metrics.py`)
**Step level, over (execution, step) pairs:**
- Treat **not-done detection** as the positive class: precision, recall and F1, where the prediction is `NOT_DONE` (with `OUT_OF_ORDER` counting as done).
- DONE precision and recall.
- **Coverage:** the fraction of verifiable steps not UNCERTAIN. Report metrics on decided steps, and a variant where UNCERTAIN counts as wrong.
- **Timing:** MAE of `completion_time_s` vs GT for true-positive DONE steps; % within ±5 s, ±15 s and ±30 s.

**Order:** precision and recall of `OUT_OF_ORDER` vs `gt_order_violation`. Report support counts; if support is below 10, say so explicitly.

**False alarms:** the mean number of non-DONE verdicts per GT-compliant execution.

**Video level:** the 3×2 confusion matrix (predicted COMPLIANT / NON_COMPLIANT / NEEDS_REVIEW vs GT compliant / non-compliant); accuracy on decided executions; review rate.

**Per-step breakdown:** a table with one row per step (recall, false-alarm rate, timing error). This drives error analysis.

**Official comparability:** for `step_set: psr`, also run IMPACT's official PSR evaluator (Step F1, delay τ, POS) if it is usable, and note that our setting is few-shot and front-view only.

### 9.6 Experiments (each one is a config in `configs/experiments/`)
| ID | Question | Variants |
|---|---|---|
| E0 | Are the checker and metrics correct? | `oracle` scorer → must give perfect step and video metrics on originals and truncations |
| E1 | Sanity baseline | `time_prior` |
| E2 | Features | VideoMAEv2 (shipped, full) vs DINOv2 full vs DINOv2 ROI (fixed scorer: `linear_probe`) |
| E3 | Scorers | `prototype` vs `linear_probe` vs `knn` (best features from E2) |
| E4 | Number of demos | K = 1, 2, 3 (best of E2 × E3) |
| E5 | Granularity | `grouped` vs `tas_s` vs `psr` |
| E6 | Graph source | `released`/`manual` vs `inferred_all` vs `inferred_demos` (effect on OUT_OF_ORDER false alarms) |
| E7 | VLM | VLM scorer (with and without in-context refs) vs best feature scorer; best feature scorer + VLM verifier |

**Run outputs.** Every run writes `runs/<run_id>/{config.yaml, metrics.json, per_step.csv, verdicts/*.json, git_commit.txt}`. `eval/tables.py` aggregates runs into Markdown tables for `docs/RESULTS.md`.

---

## 10. Gradio app (`app.py`)

Launch from Colab: `sopverify app --share`. Use the run's cached features when available.

**Tab "Run":**
- choose the procedure and step set;
- choose the input, either an IMPACT execution from a dropdown (dev/test pool; demos marked) or an uploaded MP4;
- choose a scorer preset (best feature scorer; feature + VLM verifier; VLM scorer);
- click Run, with a progress bar.
- **Uploaded videos:** decode, crop with the procedure ROI, extract DINOv2 features on the GPU (if available) and run. Show a warning that uploads must match the camera setup.

**Tab "Report":**
- the video verdict badge;
- the step table;
- the timeline chart, with a GT overlay toggle for IMPACT videos;
- an evidence gallery;
- the graph view;
- download buttons for the JSON and HTML report.

**Tab "Procedure":** read-only view of the SOP text, steps and rendered graph; ROI preview on a sample frame.

**Tab "Results":** render `docs/RESULTS.md` tables.

**Performance:** with cached features and no VLM, an IMPACT execution must return a report in under 15 s.

---

## 11. Configuration (`configs/default.yaml`, abridged)
```yaml
procedure: procedures/impact_reassembly_A
step_set: grouped
demos: {k: 2, mode: any_clean_expert, n_seeds: 5, seed: 0}
splits: {dev_fraction: 0.3, participant_disjoint: true, seed: 0}
video: {sample_fps: 2.0}
features: {extractor: dinov2, model_id: facebook/dinov2-base, crop: roi, pooling: cls}
labels: {pos_margin_s: 3.0, neg_margin_s: 10.0}
scorer: {name: linear_probe, knn_k: 15}
smoothing: {median_window_s: 2.5, tau_on: 0.6, tau_off: 0.4, min_on_s: 5.0, evidence_offset_s: 3.0}
checker: {tau_low: 0.3, order_tolerance_s: 5.0, final_state_check: false}
vlm: {enabled: false, backend: vllm, model_id: "<Qwen3-VL ~8B Instruct; verify id>", keyframe_s: 10, in_context_refs: true, verifier: false}
eval: {truncation: {enabled: true, n_per_video: 3, m_choices: [1, 2, 3], seed: 0}}
report: {html: true, show_gt: true}
```
**Config hashing:** each stage has a config hash computed only from the fields that stage depends on. Caches are keyed by it.

---

## 12. Testing
- **Fast suite** (`pytest -m "not data and not gpu"`), under 60 s, runs locally with **synthetic fixtures** only:
  - procedure schema and DAG validation (cycles, unknown ids, transitive reduction, grouped-graph derivation);
  - graph inference (`inferred_*`) on toy orderings;
  - time conversions in the adapters (frame ↔ seconds, different fps);
  - demo label builder (margins, excluded band, never-completed steps);
  - smoothing (hysteresis, min duration, rework detection, final_on);
  - checker (table-driven, ≥ 20 cases);
  - truncation GT;
  - metrics on hand-computed examples;
  - JSON schema round-trip;
  - the firewall import test.
- **Integration test:** synthetic features with a planted per-step signal → the full pipeline → expected verdicts, including one planted order violation.
- **`@pytest.mark.data`:** adapter tests on the IMPACT 2.8 GB sample. Skipped when the data is absent.
- **`@pytest.mark.gpu`:** feature-extractor smoke test. Skipped without CUDA.

---

## 13. Milestones and acceptance criteria

**M0 — Scaffold**
- Repo layout, `pyproject.toml`, config models, CLI skeleton, logging, `docs/*` stubs, `00_setup_colab.ipynb` (mount Drive, clone/pull, `pip install -e .[vlm]`, set environment variables).
- ✅ The fast test suite runs (even if nearly empty). `sopverify --help` works. The setup notebook runs on Colab.

**M1 — Data discovery and adapters**
- `01_download_and_discover.ipynb`: download the annotations bundle, the sample, the front-view videos and the front-view VideoMAEv2 features (selective bundles if available), and verify checksums.
- Write `discover.py`, `adapters.py`, `catalog.py`.
- Fill `docs/DATA_NOTES.md` with every item in 5.2, including real file excerpts and the counts in 5.2 #8.
- ✅ `sopverify data summarize` prints the pool table. Adapter tests pass on the sample. Every 5.2 item is answered or explicitly marked "not released."
- **[GATE]** The user reviews `DATA_NOTES.md`.

**M2 — Procedure package**
- Build the `psr` and `tas_s` step sets from the vocabularies.
- Draft the `grouped` step set (8–15 steps) with GT mappings and a proposed `visually_verifiable` flag per step.
- Build the graph per `graph_source` (released if available).
- Write `sop.md` from the released manual if available; otherwise draft it from step names and mark it as drafted.
- ROI proposal with an overlay image.
- ✅ `sopverify procedure validate` passes for all three step sets. The graph renders.
- **[GATE]** The user approves the grouped steps, the graph and the ROI.

**M3 — Video, time grid and features**
- `02_extract_features.ipynb`: decode at 2 fps, crop the ROI, extract DINOv2 (full and ROI) for the pool, and align the shipped VideoMAEv2 features to the grid. All cached and resumable.
- ✅ The cache covers 100% of the pool. An alignment check shows that shipped features and video frames share the time grid (document the method). The timing target in 8.4 is met or the gap is explained.

**M4 — Ground truth, eval harness and sanity baselines**
- `ground_truth.py`, `truncation.py`, `metrics.py`, `runner.py`, `tables.py`; experiments E0 and E1.
- ✅ E0 (oracle) gives perfect metrics. E1 results appear in `RESULTS.md`. GT diagnostics (9.3) are reported.

**M5 — Few-shot state pipeline**
- Label builder, the scorers (`prototype`, `linear_probe`, `knn`), smoothing, checker, `pipeline.run`, JSON reports; experiments E2–E6 with threshold tuning on dev.
- ✅ All experiments run end to end. `RESULTS.md` has tables with mean ± std over demo seeds, plus seen/unseen-participant breakdowns. The best configuration is identified and saved as `configs/best.yaml`.

**M6 — VLM**
- `vlm/` backend, scorer and verifier; `04_vlm.ipynb`; experiment E7.
- ✅ E7 is in `RESULTS.md`, with runtime per video and the number of JSON parse failures.

**M7 — Reports and app**
- HTML report; Gradio app with all four tabs; `05_app.ipynb`.
- ✅ The app runs in Colab with a share link. Both an IMPACT execution and an uploaded clip produce reports. The performance target in Section 10 is met with cached features.

**M8 — Error analysis and write-up**
- `RESULTS.md`:
  - headline table (best feature pipeline, best + verifier, VLM scorer, time prior);
  - per-step breakdown;
  - 5 annotated failure cases with evidence frames;
  - a recommendation of which steps to mark `NOT_VERIFIABLE` from the front view;
  - suggested V2 changes.
- ✅ The user can reproduce the headline table from a clean Colab session using the README alone.

---

## 14. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Annotation formats, graph or manual differ from assumptions | M1 discovery; adapters; DECISIONS log; `manual` and `inferred_*` graph fallbacks |
| Few real "not done" or order-violation cases in IMPACT | Truncation negatives (9.4); report GT support counts; never over-claim order-detection results with tiny support |
| Per-step classifiers learn "how far along the video is" instead of the specific end-state | ROI crops; `time_prior` baseline must be beaten; inspect per-step results; stretch goal: per-component (ASR) scorers |
| Hands and tools occlude the workspace | Excluded transition band in the labels; median filter + hysteresis + minimum duration; evidence offset after completion |
| Small parts (screws, washers) not visible from the front view | Grouped steps defined by visible end-states; per-step ROI overrides; mark `NOT_VERIFIABLE` after M8 analysis |
| Inferred graph is over-constrained | E6 ablation; prefer the released or user-reviewed graph |
| Demo participants overlap with test | Seen/unseen breakdown; optional held-out-participant mode |
| VLM latency, context limits, malformed JSON | Keyframe-based scoring, chunked step lists, constrained decoding, retries, logging |
| Colab resets and Drive I/O | Config-hash caches, resumable stages, local scratch for decoding |
| License (CC BY-NC-SA 4.0 on the dataset card) | Research use only; no public redistribution of data or derived features; README notice |

---

## 15. Open questions (resolve at the marked gates)
1. Final `grouped` step list and which steps are `visually_verifiable` (M2 gate).
2. Graph source if IMPACT does not release the PSR graph: manual vs inferred (M2 gate).
3. ROI box coordinates (M2 gate).
4. **Target numbers** for calling V1 successful. Proposed for dev: not-done recall ≥ 0.90 at ≤ 1.0 false alarm per compliant video, with review rate ≤ 20%. The user confirms or edits these before M5 tuning.
5. **Stretch goals,** if time allows after M8:
   - per-component ASR scorers;
   - adding the top view;
   - a hand-occlusion frame filter;
   - HACT-style reference-execution conditioning.

---

## 16. References
- Wen et al., *IMPACT: A Dataset for Multi-Granularity Human Procedural Action Understanding in Industrial Assembly*, arXiv:2604.10409 (2026). Repo: github.com/Kratos-Wen/IMPACT. Data: Hugging Face `KratosWen/IMPACT` (v1.1).
- Schoonbeek et al., *IndustReal: procedure step recognition*, WACV 2024. This is the PSR task definition.
- Wen et al., *HACT: Hand-Aware Transition Modeling for Bimanual Procedural Anomaly Detection*, arXiv:2609.21207 (2026). Uses 2 reference executions + a step list on IMPACT Reassembly.
- Wen et al., *INSPECT: Learning Robot View Selection from Assistant Use*, arXiv:2609.20615 (2026). Supported / contradicted / insufficient claim verification.
- Ozsoy et al., *The Unreasonable Effectiveness of VLMs for Zero-shot Procedural Mistake Detection*, arXiv:2606.21579 (2026). VLM baseline recipe.
