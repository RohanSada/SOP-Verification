from pathlib import Path

import pytest
from pydantic import ValidationError

from sopverify.config import Config, deep_merge, expand_dotted, expand_env, load_config, load_experiment


def test_env_expansion(monkeypatch):
    monkeypatch.setenv("SV_X", "/tmp/x")
    assert expand_env("${SV_X:-/default}/a") == "/tmp/x/a"
    monkeypatch.delenv("SV_X")
    assert expand_env("${SV_X:-/default}/a") == "/default/a"
    assert expand_env("${SV_X}") == ""


def test_paths_resolved(monkeypatch):
    monkeypatch.setenv("SOPVERIFY_DATA_ROOT", "/d")
    monkeypatch.delenv("SOPVERIFY_ANNOTATIONS", raising=False)
    p = Config().paths_resolved()
    assert p["data_root"] == Path("/d") and p["annotations_root"] == Path("/d/annotations")


def test_dotted_and_merge():
    assert expand_dotted({"a.b": 1, "a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}
    assert expand_dotted({"eval": {"tune": {"grid": {"smoothing.tau_on": [1]}}}})["eval"]["tune"]["grid"] == {
        "smoothing.tau_on": [1]}
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def test_default_config_loads(repo):
    cfg = load_config(repo / "configs" / "default.yaml")
    assert cfg.step_set == "grouped" and cfg.video.sample_fps == 2.0
    assert cfg.video.expected_resolution == (1280, 720)


def test_overrides_and_validation():
    cfg = Config().with_overrides({"smoothing.tau_on": 0.7, "scorer": {"name": "knn"}})
    assert cfg.smoothing.tau_on == 0.7 and cfg.scorer.name == "knn"
    with pytest.raises(ValidationError):
        Config().with_overrides({"smoothing.not_a_field": 1})


def test_stage_hash_depends_only_on_stage_fields():
    a = Config()
    b = a.with_overrides({"smoothing.tau_on": 0.9, "checker.tau_low": 0.1})
    c = a.with_overrides({"features.model_id": "facebook/dinov2-large"})
    assert a.stage_hash("features") == b.stage_hash("features")
    assert a.stage_hash("features") != c.stage_hash("features")
    assert a.stage_hash("features", roi=[1, 2, 3, 4]) != a.stage_hash("features", roi=[1, 2, 3, 5])
    shipped = a.with_overrides({"features.extractor": "shipped_videomaev2"})
    assert shipped.stage_hash("features") == shipped.with_overrides({"features.crop": "full"}).stage_hash("features")


@pytest.mark.parametrize("name", sorted(p.name for p in (Path(__file__).parents[1] / "configs" / "experiments").glob("E*.yaml")))
def test_experiment_configs_load(repo, name):
    exp = load_experiment(repo / "configs" / "experiments" / name)
    assert exp.variants
    for cfg in exp.variants.values():
        assert cfg.run_name and cfg.run_name.startswith(exp.name)


def test_experiment_base_overrides(repo):
    from sopverify.config import parse_set_args

    ov = parse_set_args(["features.crop=full", "demos.k=3"])
    assert ov == {"features.crop": "full", "demos.k": 3}
    exp = load_experiment(repo / "configs" / "experiments" / "E3_scorers.yaml", overrides=ov)
    assert all(c.features.crop == "full" and c.demos.k == 3 for c in exp.variants.values())


def test_copy_github_annotations(tmp_path):
    from sopverify.colab import copy_github_annotations

    src = tmp_path / "dataset"
    for rel in ("ASR/annotations/X_front_asr.json", "ASR/annotations/X_left_asr.json", "PSR/labels/train/X_front/PSR_labels.csv",
                "TAS/groundTruth_TAS-S/X_front.txt", "TAS/groundTruth_TAS-S/X_ego.txt", "TAS/mapping_TAS-S.txt"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text("x")
    n = copy_github_annotations(str(src), str(tmp_path / "out"))
    assert n == 4 and not (tmp_path / "out/ASR/annotations/X_left_asr.json").exists()


def test_pick_best(tmp_path):
    from sopverify.eval.tables import pick_best
    from sopverify.utils import write_json, write_yaml

    def run(name, rec, fa):
        d = tmp_path / name
        res = {"dev": {"all": {"all": {"nd_recall": {"mean": rec}, "nd_f1": {"mean": rec}, "fa_per_compliant": {"mean": fa},
                                       "review_rate": {"mean": 0.1}}}}}
        write_json(d / "metrics.json", {"experiment": "E2", "variant": name, "run_name": name, "results": res})
        write_yaml(d / "config.yaml", {"features": {"crop": name}})

    run("a", 0.95, 3.0)   # best recall but too many false alarms
    run("b", 0.80, 0.5)
    run("c", 0.70, 0.2)
    best = pick_best(tmp_path, "E2")
    assert best["variant"] == "b" and best["config"]["features"]["crop"] == "b"
