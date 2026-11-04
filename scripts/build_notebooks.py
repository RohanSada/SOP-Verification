"""Generate notebooks/*.ipynb from the cell lists below (keeps the shared setup cells identical).

Usage: python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"
BRANCH = "claude/nifty-mccarthy-5suut3"
REPO_URL = "https://github.com/RohanSada/SOP-Verification.git"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def setup_cells(extras: str = "app,vlm,dev") -> list[dict]:
    return [
        md("## Setup\nMounts Drive, clones or updates the repo, installs it and sets the `SOPVERIFY_*` paths. "
           "Safe to re-run after a runtime reset."),
        code('from google.colab import drive\ndrive.mount("/content/drive")'),
        code(f"""%%bash
set -e
REPO=/content/SOP-Verification
BRANCH={BRANCH}   # switch to main once the branch is merged
if [ -d $REPO/.git ]; then
  git -C $REPO fetch -q origin $BRANCH && git -C $REPO checkout -q $BRANCH && git -C $REPO pull -q --ff-only origin $BRANCH
else
  git clone -q -b $BRANCH {REPO_URL} $REPO
fi
pip install -q -e "$REPO[{extras}]"
git -C $REPO log --oneline -1"""),
        code('''import os, sys
sys.path.insert(0, "/content/SOP-Verification/src")   # editable install without a runtime restart
os.chdir("/content/SOP-Verification")
from sopverify.colab import setup_env, show
ROOT = "/content/drive/MyDrive/sopverify"
ENV = setup_env(ROOT)
try:  # optional Hugging Face token (Colab secret named HF_TOKEN), in case the dataset is gated
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    pass
ENV'''),
    ]


NOTEBOOKS: dict[str, list[dict]] = {}

NOTEBOOKS["00_setup_colab"] = [
    md("""# 00 · Colab setup
Run this once per new Colab environment (Runtime → Change runtime type → **A100 GPU**).
It mounts Drive, installs the repo and checks that everything imports.
**Paste back:** the version block printed at the end, so it can be recorded in `docs/DECISIONS.md`."""),
    *setup_cells(),
    code("!sopverify --help\n!sopverify info"),
    code("!python -m pytest -q -o addopts=\"\" -m \"not data and not gpu\" 2>&1 | tail -3"),
    md("GPU check: downloads `facebook/dinov2-base` (~350 MB) and encodes a few frames on the A100."),
    code("!python -m pytest -q -o addopts=\"\" -m gpu -rs 2>&1 | tail -3"),
    code('''import importlib, platform, subprocess
pkgs = ["torch", "transformers", "numpy", "scipy", "pandas", "sklearn", "pydantic", "av", "cv2", "plotly", "gradio",
        "networkx", "huggingface_hub"]
print("python", platform.python_version())
for p in pkgs:
    try:
        m = importlib.import_module(p); print(p, getattr(m, "__version__", "?"))
    except Exception as e:
        print(p, "NOT INSTALLED", e)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)'''),
]

NOTEBOOKS["01_download_and_discover"] = [
    md("""# 01 · Download and discover IMPACT (M1)
Downloads only what V1 needs and inventories it:
1. the annotations bundle (8 MiB, SHA256-checked) and the 2.8 GB quick-start sample (to local scratch, for adapter tests);
2. the 45 Model A Reassembly **front-view videos** and **VideoMAEv2 features**, pulled out of the big remote zips with
   HTTP range requests (no full 13–29 GiB / 21.6 GiB downloads; each file is CRC-checked).

**Paste back** the outputs marked 📋 so they can go into `docs/DATA_NOTES.md`."""),
    *setup_cells("app,dev"),
    md("### 1. Annotations bundle"),
    code("!sopverify data download annotations\n!ls $SOPVERIFY_DATA_ROOT; ls $SOPVERIFY_DATA_ROOT/annotations | head"),
    md("### 2. Inventory 📋"),
    code('!sopverify data discover --out "$SOPVERIFY_EXPORTS/inventory_summary.md"'),
    md("### 3. Metadata (expert vs novice) 📋\nThe GitHub copy of the annotations has no expertise labels; they should be in `metadata/`."),
    code('''from sopverify.data.impact.catalog import Catalog
cat = Catalog.from_paths(os.environ["SOPVERIFY_DATA_ROOT"], None)
print("metadata files:", len(cat.index.metadata_files))
for p in cat.index.metadata_files[:20]:
    print("\\n==", p); print(open(p, errors="ignore").read()[:1500])
print("\\nexpertise parsed:", cat.expertise)'''),
    md("### 4. Pool summary 📋\nExpect **45 executions, 13 participants**. If the count is far lower, the next cell "
       "copies the same v1.1 annotations from the IMPACT GitHub repo (front view only, ~30 MB) and re-counts."),
    code('!sopverify data summarize --out "$SOPVERIFY_EXPORTS/pool_summary.csv"'),
    code('''from sopverify.colab import copy_github_annotations
n = len(Catalog.from_paths(os.environ["SOPVERIFY_DATA_ROOT"], None).execution_ids())
print("pool executions recognised:", n)
if n < 40:
    !rm -rf /content/scratch/IMPACT && git clone -q --depth 1 https://github.com/Kratos-Wen/IMPACT.git /content/scratch/IMPACT
    copied = copy_github_annotations("/content/scratch/IMPACT/dataset", os.environ["SOPVERIFY_DATA_ROOT"] + "/annotations_github")
    print("copied", copied, "files from the IMPACT GitHub repo")
    cat = Catalog.from_paths(os.environ["SOPVERIFY_DATA_ROOT"], None)
    print("pool executions now:", len(cat.execution_ids()))'''),
    md("### 5. Quick-start sample → adapter tests on real files"),
    code('''!sopverify data download sample
!ls /content/scratch/impact_sample
# The sample root holds annotations, videos and features: the data tests use all three.
os.environ["SOPVERIFY_TEST_ANNOTATIONS"] = "/content/scratch/impact_sample"
!python -m pytest -q -o addopts="" -m data -rs 2>&1 | tail -8'''),
    md("### 6. What is inside the big bundles? (zip directory only) 📋"),
    code('''from sopverify.data.impact.download import remote_listing
pool = cat.execution_ids()
for key in ["videos:front", "features:VideoMAEv2"]:
    lst = remote_listing(key)
    want = [(n, s) for n, s in lst if any(f"{e}_front" in n for e in pool)]
    print(key, "members:", len(lst), "| pool members:", len(want), "| pool size: %.1f GB" % (sum(s for _, s in want) / 1e9))
    print("  e.g.", want[:2])'''),
    md("### 7. Fetch pool videos and VideoMAEv2 features (resumable)"),
    code("!sopverify data download videos\n!sopverify data download features --model VideoMAEv2"),
    md("### 8. Verify resolution, fps and feature layout 📋"),
    code('''import numpy as np, pandas as pd
from sopverify.video.decode import probe
from sopverify.features.shipped import load_feature_array
cat = Catalog.from_paths(os.environ["SOPVERIFY_DATA_ROOT"], None)
rows = []
for e in cat.execution_ids():
    ref = cat.ref(e); row = {"execution": e, "asr_frames": ref.n_frames}
    if ref.video_path:
        i = probe(ref.video_path); row.update(w=i.width, h=i.height, fps=round(i.fps, 3), video_frames=i.n_frames)
    if "VideoMAEv2" in ref.shipped_features:
        raw = np.load(ref.shipped_features["VideoMAEv2"], mmap_mode="r"); row.update(feat_shape=str(raw.shape))
        row["feat_T"] = load_feature_array(ref.shipped_features["VideoMAEv2"], 1408).shape[0]
    rows.append(row)
df = pd.DataFrame(rows); print(df.to_string())
print(df[["w", "h", "fps"]].value_counts() if "w" in df else "no videos")
if "feat_T" in df: print("feat_T - asr_frames:", (df.feat_T - df.asr_frames).describe())'''),
    code('!sopverify data summarize | tail -3\n!du -sh "$SOPVERIFY_DATA_ROOT"'),
    md("### 9. Done 📋\nSend back: the outputs of steps 2, 3, 4, 6 and 8, plus the two lines above. "
       "The files `inventory_summary.md` and `pool_summary.csv` are also saved in `MyDrive/sopverify/exports/`."),
]

NOTEBOOKS["02_extract_features"] = [
    md("""# 02 · ROI, frames and features (M2 gate + M3)
1. **Confirm the workspace crop (ROI)** on real frames. Edit the box until it covers the work area, then put
   it in `procedures/impact_reassembly_A/roi.yaml` (set `confirmed: true`) and commit, or paste it back.
2. Extract DINOv2 features (ROI and full frame) and resample the shipped VideoMAEv2 features onto the 2 fps grid.
   Everything is cached on Drive by config hash and skips finished executions.
3. Check the temporal alignment of shipped features vs our frames."""),
    *setup_cells("app,dev"),
    md("### 0. Expertise labels 📋\nThe only metadata file in the release is `NASA_TLX_anonymized.xlsx`. This prints "
       "its sheets so we can see whether it says who is an expert. Takes a few seconds."),
    code('''import glob, pandas as pd
from sopverify.data.impact.catalog import Catalog
for p in sorted(glob.glob(os.environ["SOPVERIFY_DATA_ROOT"] + "/metadata/*.xls*")):
    for name, df in pd.read_excel(p, sheet_name=None).items():
        print(f"== {p.split('/')[-1]} | sheet {name!r} | {df.shape[0]} rows x {df.shape[1]} columns")
        print("columns:", list(df.columns))
        print(df.head(20).to_string(), "\n")
print("expertise parsed:", Catalog.from_paths(os.environ["SOPVERIFY_DATA_ROOT"], None).expertise)'''),
    md("### 1. ROI preview"),
    code('''from sopverify.data.impact.catalog import Catalog
from sopverify.procedure.loader import load_procedure
from sopverify.video.decode import read_frames_at
from sopverify.video.roi import overlay_box
cat = Catalog.from_paths(os.environ["SOPVERIFY_DATA_ROOT"], None)
proc = load_procedure("procedures/impact_reassembly_A")
BOX = proc.roi.box          # try other boxes here, e.g. [420, 280, 520, 400]
from PIL import Image
ids = cat.execution_ids()
for i, (e, t) in enumerate([(ids[0], 20), (ids[len(ids) // 2], 120), (ids[-1], 200)]):
    ref = cat.ref(e)
    fr = read_frames_at(ref.video_path, [min(t, ref.duration_s - 1)])[0]
    img = overlay_box(fr, BOX)
    show(img, f"{e} t={t}s  box={BOX}")
    Image.fromarray(img).save(f"{os.environ['SOPVERIFY_EXPORTS']}/roi_preview_{i}.jpg", quality=85)
print("saved roi_preview_0..2.jpg in MyDrive/sopverify/exports/")'''),
    md("### 2. Features (resumable). Target: DINOv2-base < 2 min per 20-min video on A100, excluding decode."),
    code("!sopverify features extract --crop roi"),
    code("!sopverify features extract --crop full"),
    code("!sopverify features extract --extractor shipped_videomaev2"),
    md("### 3. Coverage and timing 📋"),
    code('''import time, numpy as np
from sopverify.config import load_config
from sopverify.features.provider import FeatureProvider
base = load_config()
variants = {"dinov2_roi": {"features.crop": "roi"}, "dinov2_full": {"features.crop": "full"},
            "videomae": {"features.extractor": "shipped_videomaev2"}}
for name, ov in variants.items():
    fp = FeatureProvider(base.with_overrides(ov), proc.roi.box, proc.roi.frame_size)
    have = sum(fp.has(cat.ref(e)) for e in ids)
    print(f"{name}: {have}/{len(ids)} cached  key={fp.key_hash()}")
# Encoder speed on frames already in memory (excludes decode):
fp = FeatureProvider(base, proc.roi.box, proc.roi.frame_size)
ref = cat.ref(ids[0]); frames = fp.frames(ref, fp.grid(ref))
t0 = time.time(); fp.extractor().encode(np.asarray(frames)); dt = time.time() - t0
print(f"DINOv2 encode: {len(frames)} frames in {dt:.1f}s -> {dt / len(frames) * 2400:.0f}s per 20-min video at 2 fps")'''),
    md("### 4. Alignment check: shipped VideoMAEv2 vs our frame grid 📋\nBoth streams should change at the same "
       "moments; the best lag of their change signals should be ~0 s."),
    code('''from sopverify.features.alignment import alignment_report
fa = FeatureProvider(base.with_overrides({"features.crop": "full"}), proc.roi.box)
fb = FeatureProvider(base.with_overrides({"features.extractor": "shipped_videomaev2"}), proc.roi.box)
rep = []
for e in ids[:10]:
    r = alignment_report(fa.get(cat.ref(e))[1], fb.get(cat.ref(e))[1], base.video.sample_fps)
    rep.append({"execution": e, **r}); print(e, r)
import pandas as pd; print(pd.DataFrame(rep)[["best_lag_s", "peak_corr"]].describe())'''),
]

NOTEBOOKS["03_run_experiments"] = [
    md("""# 03 · Experiments E0–E6 (M4, M5)
Each experiment writes `runs/<experiment>__<variant>/` on Drive (config, metrics, per-step CSV, verdicts, traces).
Thresholds are tuned on **dev only**; test is reported once per configuration.
After E2 and E3, pass the winners to later experiments with `--set` (shown below).
**Paste back** the tables printed at the end (or the `exports/RESULTS_tables.md` file)."""),
    *setup_cells("app,dev"),
    md("### E0 oracle (must be perfect) and E1 time prior (no GPU needed)"),
    code("!sopverify eval -c configs/experiments/E0_oracle.yaml\n!sopverify eval -c configs/experiments/E1_time_prior.yaml"),
    md("### E2 features (linear probe)"),
    code("!sopverify eval -c configs/experiments/E2_features.yaml"),
    md("### E3 scorers on the E2 winner\nThe winner is picked on the **dev** split only (targets first, then "
       "not-done recall, then F1)."),
    code('''from sopverify.eval.tables import pick_best
RUNS = os.environ["SOPVERIFY_RUNS"]
b2 = pick_best(RUNS, "E2_features"); print("E2 winner:", b2["variant"], b2["dev"])
f = b2["config"]["features"]
BEST_FEATURES = f"--set features.extractor={f['extractor']} --set features.crop={f['crop']}"
print(BEST_FEATURES)
!sopverify eval -c configs/experiments/E3_scorers.yaml $BEST_FEATURES'''),
    md("### E4–E6 on the E2 × E3 winner"),
    code('''b3 = pick_best(RUNS, "E3_scorers"); print("E3 winner:", b3["variant"], b3["dev"])
BEST = BEST_FEATURES + f" --set scorer.name={b3['config']['scorer']['name']}"
print(BEST)
!sopverify eval -c configs/experiments/E4_num_demos.yaml $BEST
!sopverify eval -c configs/experiments/E5_granularity.yaml $BEST
!sopverify eval -c configs/experiments/E6_graph.yaml $BEST'''),
    md("### Tables 📋"),
    code('!sopverify tables --out "$SOPVERIFY_EXPORTS/RESULTS_tables.md"\n'
         'from IPython.display import Markdown, display\n'
         'display(Markdown(open(os.environ["SOPVERIFY_EXPORTS"] + "/RESULTS_tables.md").read()))'),
    md("### Save the best configuration 📋\nThe E3 winner's full config (tuned thresholds included) becomes the "
       "candidate `configs/best.yaml`. Send me `exports/best.yaml` together with the tables."),
    code('''import shutil
shutil.copy(b3["config_path"], f"{os.environ['SOPVERIFY_EXPORTS']}/best.yaml")
print(open(b3["config_path"]).read())'''),
]

NOTEBOOKS["04_vlm"] = [
    md("""# 04 · VLM scorer and verifier (M6, E7)
Use a **fresh A100 runtime**: vLLM installs its own PyTorch build. Budget: a full VLM-scorer run over all test
executions should take < 2 h; runtime per video and JSON parse failures are written to each run's `metrics.json`."""),
    *setup_cells("app,vlm,dev"),
    md("### Install vLLM\nTakes a few minutes and replaces PyTorch. When it finishes: **Runtime → Restart session**, "
       "then run the Setup cells again (skip this install cell)."),
    code("!pip install -q vllm\n!pip show vllm torch | grep -E '^(Name|Version)'"),
    md("### Verify the model id"),
    code('''from huggingface_hub import model_info
from sopverify.config import load_config
cfg = load_config()
info = model_info(cfg.vlm.model_id); print(info.id, info.sha, info.pipeline_tag)'''),
    md("### Smoke test: one keyframe"),
    code('''import json
from sopverify.data.impact.catalog import Catalog
from sopverify.procedure.loader import load_procedure
from sopverify.vlm.backend import make_backend
from sopverify.vlm.frames import VideoFrameSource
from sopverify.vlm.prompts import scorer_request
cat = Catalog.from_paths(os.environ["SOPVERIFY_DATA_ROOT"], None)
proc = load_procedure("procedures/impact_reassembly_A")
backend = make_backend(cfg.vlm)
frames = VideoFrameSource(proc.roi.box, proc.roi.frame_size, cfg.vlm.max_long_side)
ref = cat.ref(cat.execution_ids()[0])
img = frames.crops(ref.video_path, [ref.duration_s - 1], ref.fps)[0]
print(backend.generate([scorer_request(proc.steps("grouped"), img, ref.duration_s - 1)])[0])'''),
    md("### E7"),
    code('''import time
from sopverify.eval.runner import run_experiment
t0 = time.time()
res = run_experiment("configs/experiments/E7_vlm.yaml", vlm_backend=backend)
print(f"total {time.time() - t0:.0f}s")
for v, m in res.items():
    print(v, m["runtime_s"], m.get("vlm"))'''),
    code('!sopverify tables --experiments E7_vlm --out "$SOPVERIFY_EXPORTS/RESULTS_E7.md"\n'
         'print(open(os.environ["SOPVERIFY_EXPORTS"] + "/RESULTS_E7.md").read())'),
]

NOTEBOOKS["05_app"] = [
    md("""# 05 · Gradio app (M7)
Launches the app with a public share link. Uses cached features when available; with cached features and no VLM,
an IMPACT execution should return its report in under 15 s. The VLM presets need the vLLM setup from notebook 04.
Use **configs/best.yaml** (falls back to default.yaml)."""),
    *setup_cells("app,vlm,dev"),
    code('''from sopverify.app import launch
launch(share=True)   # stop the cell to shut the app down'''),
]


def build() -> None:
    OUT.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                                           "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 5}
        for i, c in enumerate(nb["cells"]):
            c["id"] = f"{name[:2]}-{i:02d}"
        (OUT / f"{name}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
        print("wrote", OUT / f"{name}.ipynb", len(cells), "cells")


if __name__ == "__main__":
    build()
