# Adding a new procedure

A new SOP needs a procedure package and a few demo videos, with no code changes.

1. **Create `procedures/<your_id>/`** by copying `impact_reassembly_A/`, then edit:
   - `procedure.yaml`: id, name, `default_step_set`, `graph_source: manual`, `camera`.
   - `step_sets/<name>.yaml`: one entry per step. Give each step an `id`, a `name`, an `end_state` (what the workspace
     looks like once the step is complete; the VLM reads this), `requires`, `required` and `visually_verifiable`.
     Pick steps whose **end-state is visible** from your camera.
   - `gt_mapping`: how your annotations say a step is complete. For your own data, the simplest form is one event per
     step (`psr_events: ["install:<step>"]`) with completion times recorded per demo (see below).
   - `roi.yaml`: the workspace box `[x, y, w, h]` in your camera's pixels, plus `frame_size`.
   - `vocab.yaml`: the event ids your mappings use.
   - `sop.md`: the human-readable SOP.
2. **Validate:** `sopverify procedure validate procedures/<your_id>` and `sopverify procedure render procedures/<your_id>`.
3. **Demos.** Record 2–3 clean executions with the same fixed camera, and note when each step's end-state first appears.
   The demo code consumes `DemoData(execution_id, times_s, feats, completion_s={step_id: seconds}, duration_s, video_path)`,
   so a small adapter that reads your completion times and builds `DemoData` is all that's dataset-specific. Pass those
   to `pipeline.run(video_path, procedure, demos, cfg)`.
4. **Check the crop** with `sopverify procedure roi-preview procedures/<your_id> --video demo1.mp4`.
5. **Thresholds.** The defaults come from IMPACT. With only a few videos, start from `configs/best.yaml` and check the
   timeline in the HTML report (`report.html`) to adjust `smoothing.tau_on` and `checker.tau_low`.

Keep the camera fixed between demos and test videos. The ROI is rescaled when the resolution differs, but a moved
camera breaks the learned end-states.
