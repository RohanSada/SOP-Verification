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

## Step 1: notebook `00_setup_colab` (A100, ~10 min)
It installs the code and runs the 136 fast tests.

Check: the pytest line ends in `passed` with no `failed`, and `sopverify info` shows `CUDA True (NVIDIA A100…)`.

📋 **Send me:** the pytest summary line and the whole output of the last cell (Python, package versions, GPU).

## Step 2: notebook `01_download_and_discover` (CPU, ~1–2 h, mostly downloading)

| Section | What happens | What you should see |
|---|---|---|
| 1 | Downloads the 8 MiB annotations bundle (checksum-verified) into Drive | `extracted … into …/IMPACT-v1.1`, then a folder listing |
| 2 📋 | Inventories every file found | A table of kinds (asr, psr, tas_s, …) with counts |
| 3 📋 | Prints the `metadata/` files, which should say who is expert or novice | File contents plus `expertise parsed: {...}` |
| 4 📋 | Pool table | **45 executions, 13 participants**. If far fewer, the next cell copies the annotations from the IMPACT GitHub repo and re-counts; send me that output too |
| 5 📋 | Downloads the 2.8 GB sample to local disk and runs the real-data tests | `N passed` (some skips are fine; `-rs` lists why) |
| 6 📋 | Reads the table of contents of the big video and feature archives (no full download) | Member counts, pool size in GB, two example names per archive |
| 7 | Downloads only the 45 front-view videos and VideoMAEv2 feature files (resumable) | Lines like `extracted videos/front/…mp4`, then `45 of 45 files present` twice |
| 8 📋 | Checks resolution, fps and feature shapes for every video | A table, plus `w h fps` counts (expected 1280 720 30) and `feat_T - asr_frames` (expected ~0) |
| 9 📋 | Final counts and Drive usage | Last lines of the summary and `du -sh` |

If section 7 is interrupted, re-run the Setup cells and section 7; finished files are skipped.

📋 **Send me:** outputs of sections 2, 3, 4 (plus the fallback cell if it ran), 5, 6, 8 and 9. Also attach
`exports/inventory_summary.md` and `exports/pool_summary.csv`. On errors, paste the full error text.

## Step 3: reviews (you can do these while step 2 downloads)
Read on GitHub (branch `claude/nifty-mccarthy-5suut3`) and reply in chat:
1. `docs/DATA_NOTES.md`: anything wrong or surprising?
2. `procedures/impact_reassembly_A/step_sets/grouped.yaml`: approve the 5 steps, or tell me the edits.
3. `procedures/impact_reassembly_A/graph_grouped.svg`: should **G5 (handle)** only need G1 (current draft), or come last
   (after G4, as the pictorial manual shows)?
4. `procedures/impact_reassembly_A/sop.md`: OK as a draft?

Wait for my confirmation before step 5. Experiment results depend on these choices.

## Step 4: notebook `02_extract_features` (A100, ~1–2 h)
1. **Section 1 (ROI):** three frames with a green box. The box should cover the grey work area, the hands and the device
   in every frame, without much empty background.
   - Looks right → tell me **"ROI approved"** and continue with section 2.
   - Needs changes → edit `BOX = [x, y, w, h]` in that cell and re-run until it looks right. Send me the final numbers and
     the images `exports/roi_preview_0.jpg` … `_2.jpg`. **Wait until I push the new box** before section 2, because the
     feature cache is keyed by the box.
2. **Section 2:** three extraction cells: DINOv2 crop, DINOv2 full frame, shipped VideoMAEv2. Resumable.
3. **Section 3 📋:** expect `45/45 cached` for all three, plus a speed line.
4. **Section 4 📋:** the alignment table. `best_lag_s` should be about 0.

📋 **Send me:** sections 3 and 4.

## Step 5: notebook `03_run_experiments` (CPU, ~1–3 h, after my go-ahead from step 3)
It runs E0, E1 and E2, picks the E2 winner on the dev split, runs E3, picks the E3 winner, then runs E4, E5 and E6. It
writes the tables and saves `exports/best.yaml`.

📋 **Send me:**
- The "E2 winner" and "E3 winner" lines.
- `exports/RESULTS_tables.md` (attach it).
- `exports/best.yaml`.
- From `MyDrive/sopverify/runs/E3_scorers__<winner>/`: `per_step.csv` and `per_step_dev.csv`.
- Any error text.

I then commit `configs/best.yaml` and update `docs/RESULTS.md`.

## Step 6: notebook `04_vlm` (fresh A100 runtime, ~2–3 h)
1. Run Setup, then the **Install vLLM** cell. Then **Runtime → Restart session**, run Setup again, and continue below the
   install cell.
2. **Model check 📋:** prints the model id and revision.
3. **Smoke test 📋:** prints one JSON answer from the model.
4. **E7:** the long cell. Prints runtime and VLM statistics per variant.
5. **Tables 📋:** prints `RESULTS_E7.md`.

If vLLM fails to install or load, send me the error. There is a transformers fallback.

📋 **Send me:** outputs of 2–5 (attach `exports/RESULTS_E7.md`).

## Step 7: notebook `05_app` (A100, ~15 min, after I push `configs/best.yaml`)
1. Run all. Open the `https://….gradio.live` link it prints. Share it only privately: IMPACT frames are non-commercial.
2. **Run** tab → pick an execution marked `[test]` → **Run** → the **Report** tab opens.
3. Try an upload: download one front video from Drive (`MyDrive/sopverify/data/IMPACT-v1.1/videos/front/…`), choose
   **Upload video** and run it.

📋 **Send me:** the "Done in X s" time for both runs, screenshots of the Report tab, and any errors.

## Step 8: wrap-up
I write the M8 analysis (headline table, per-step breakdown, failure cases, which steps to mark not verifiable, V2
ideas). For the failure cases I may ask you to run `sopverify run --execution <id>` for a few named executions and send
the `report.html` files.
