# RUNBOOK: what to run, in order, and what to send back

You run the notebooks in Google Colab and send the 📋 outputs back in chat. Paste the text, or attach the files from
`MyDrive/sopverify/exports/`. Everything is cached on Drive, so after a disconnect you re-run the **Setup** cells and
then the cell that was interrupted: finished work is skipped.

## Before you start (one time, ~10 min)

1. **Drive space:** at least **20 GB free** in Google Drive. The pool's videos and features take about 10–14 GB.
2. **Hugging Face token:**
   - Sign in at huggingface.co and open https://huggingface.co/datasets/KratosWen/IMPACT. If the page shows an
     "agree / access repository" box, accept it.
   - Settings → Access Tokens → **Create new token**, type **Read**. Copy it.
3. **Put the token in Colab:** open any Colab notebook → key icon in the left sidebar (**Secrets**) → **Add new secret**,
   name `HF_TOKEN`, value = the token → switch **Notebook access** on. If a notebook later asks to grant access to
   `HF_TOKEN`, click **Grant**.

**Opening a notebook:** colab.research.google.com → File → Open notebook → **GitHub** tab → enter
`RohanSada/SOP-Verification` → branch `claude/nifty-mccarthy-5suut3` → pick the notebook. A direct link also works:
`https://colab.research.google.com/github/RohanSada/SOP-Verification/blob/claude/nifty-mccarthy-5suut3/notebooks/<name>.ipynb`

**Runtime type:** Runtime → Change runtime type. Use **A100 GPU** for notebooks 00, 02, 04 and 05. Use a standard **CPU**
runtime for 01 and 03, which saves compute units: the A100 costs roughly 12 units per hour, so check the Resources panel.

**Running:** Runtime → Run all, or Shift+Enter cell by cell. The first cell asks to connect Google Drive: allow it.

---

## Done so far
- ✅ Step 1 `00_setup_colab` and Step 2 `01_download_and_discover` (2026-11-03): all 45 front videos (14.7 GB) and
  VideoMAEv2 files (2.2 GB) are on Drive, every check passed.
- ✅ Step 3 reviews (2026-11-04): DATA_NOTES, the 5 grouped steps, the graph (G5 needs only G1) and the SOP text approved.

## Step 4: notebook `02_extract_features` (A100, ~45–75 min)
Open it from GitHub (so you get the latest cells), choose **A100 GPU**, then run the cells **one section at a time**,
not Run all. Section 1 may need a decision.

| Section | What happens | What you should see |
|---|---|---|
| Setup (4 cells) | Drive, code, paths | Last line shows the latest commit; a dict of paths |
| 0 📋 | Prints the only metadata file (`NASA_TLX_anonymized.xlsx`) | Sheet names, columns and the first rows, then `expertise parsed: {...}` |
| 1 📋 | Three frames from different executions with the green crop box | The box should cover the grey work area, the hands and the device in all three |
| 2 | Three extraction cells: DINOv2 on the crop, DINOv2 on the full frame, shipped VideoMAEv2 | A progress bar per cell, ending `cached 45/45; failed 0`. The first DINOv2 cell copies the videos to local disk and decodes them (the slow part, ~15–25 min); the second reuses the copies; the third takes ~1 min |
| 3 📋 | Coverage and speed | `45/45 cached` for all three, and a DINOv2 speed line |
| 4 📋 | Alignment check | A table where `best_lag_s` is 0 or very close (±0.5 s) |

**Decision at section 1:**
- If the box covers the work area in all three frames, tell me **"ROI approved"** and go straight on to section 2.
- If not, change `BOX = [x, y, w, h]` at the top of that cell and re-run it until it looks right. x and y are the box's
  top-left corner, in pixels of a 1280×720 frame; the yellow ruler marks every 160 px across and 120 px down. Then **stop**
  and send me the final numbers plus `exports/roi_preview_0.jpg` to `_2.jpg`. I'll update the repo. After that, re-run the
  Setup cells (they pull my change) and continue from section 2. The feature cache is tied to the box, so extracting
  with the old box would be wasted.

If the runtime disconnects during section 2, re-run Setup and the same cell: finished videos are skipped.

📋 **Send me:** sections 0, 1 (your decision or new box), 3 and 4, and any error text.

## Step 5: notebook `03_run_experiments` (CPU runtime, ~1–2 h)
Start after Step 4 shows `45/45 cached` for all three feature types (and after my reply if you changed the box).
Features are read from Drive, so no GPU is needed. Choose the standard CPU runtime, then **Runtime → Run all**.

| Cell | What happens | Rough time |
|---|---|---|
| E0, E1 | Oracle check (must be all 1.00) and time-prior baseline | ~3 min |
| E2 | Three feature types with the linear probe | ~10–15 min |
| E3 | Prints `E2 winner: …`, then prototype / linear probe / kNN on those features | ~10–15 min |
| E4–E6 | Prints `E3 winner: …`, then 1/2/3 demos, the three step sets and the three graph sources | ~30–60 min |
| Tables | Shows all result tables and saves `exports/RESULTS_tables.md` | <1 min |
| Best config | Copies the E3 winner's config to `exports/best.yaml` | <1 min |

Every experiment line ends with something like `E2_features__dinov2_roi: {'nd_recall': …}`. If a cell fails, send me the
error and don't continue. Later cells use earlier results.

📋 **Send me:**
- The two lines starting `E2 winner:` and `E3 winner:`.
- `exports/RESULTS_tables.md` and `exports/best.yaml` (attach them).
- `per_step.csv` and `per_step_dev.csv` from `MyDrive/sopverify/runs/E3_scorers__<winner>/`, where `<winner>` is the
  name printed after `E3 winner:`.

I'll then commit `configs/best.yaml`, fill in `docs/RESULTS.md`, and tell you if anything should be re-run.

## Step 6: notebook `04_vlm` (fresh A100 runtime, ~2–3 h)
Start after Step 5. This tests the Qwen3-VL model as a scorer and as a checker for uncertain steps (experiment E7).
1. Open from GitHub, choose **A100 GPU**, run the 4 Setup cells.
2. Run **Install vLLM** (~5 min). It replaces PyTorch, so then do **Runtime → Restart session**, run the 4 Setup cells
   again, and continue from the cell **after** the install cell. Don't re-run the install.
3. 📋 **Model check:** prints `Qwen/Qwen3-VL-8B-Instruct`, a revision hash and `image-text-to-text`.
4. 📋 **Smoke test:** loads the model (~3–5 min the first time), then prints one JSON answer with G1–G5.
5. **E7:** the long cell (~1.5–2.5 h). Keep the tab open.
6. 📋 **Tables:** prints the E7 table and saves `exports/RESULTS_E7.md`.

If vLLM fails to install or load (an error in step 2 or 4), send me the error; I'll switch you to the slower fallback.

📋 **Send me:** the outputs of 3, 4 and 5 (the per-variant timing lines) and `exports/RESULTS_E7.md`.

## Step 7: notebook `05_app` (A100, ~15 min, after I push `configs/best.yaml`)
1. Open from GitHub, A100, **Run all**. The last cell prints a `https://….gradio.live` link. Open it, and keep it private
   (IMPACT frames are non-commercial).
2. **Run** tab: pick an execution marked `[test]`, keep the "Best feature scorer" preset, click **Run**. The **Report** tab
   opens with the verdict, step table, timeline, evidence frames and graph.
3. Tick **Show ground truth** to compare against the annotations.
4. Upload test: download one front video from Drive (`MyDrive/sopverify/data/IMPACT-v1.1/videos/front/`, ~300 MB),
   choose **Upload video**, select it and click **Run**. Expect 1–3 min, since its features are computed on the fly.
5. Stop the last cell to shut the app down.

📋 **Send me:** the "Done in X s" time for both runs, screenshots of the Report tab, and any errors.

## Step 8: wrap-up
I write the final analysis in `docs/RESULTS.md`: headline table, per-step breakdown, failure cases, steps to mark not
verifiable, and V2 ideas. For the failure cases I may ask you to run a couple of named executions in the app and send
their `report.html` files.
