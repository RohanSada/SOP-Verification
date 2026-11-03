"""Adapter tests on real IMPACT annotations (skipped unless SOPVERIFY_TEST_ANNOTATIONS is set)."""

import pytest

from sopverify.config import load_config
from sopverify.data.impact.catalog import Catalog
from sopverify.data.impact.splits import make_split_plan
from sopverify.eval.ground_truth import compute_gt
from sopverify.procedure.loader import load_procedure

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def catalog(impact_annotations):
    return Catalog.from_paths(None, impact_annotations)


def test_sample_execution(catalog):
    r = catalog.record("AL07EJ17_Reassembly_A_002")
    assert r.fps == 30.0 and r.n_frames == 5214 and r.duration_s == pytest.approx(173.8)
    assert len(r.psr_events) == 17 and r.psr_events[0].time_s == pytest.approx(982 / 30)
    assert r.final_component_states()["anti_vibration_handle"] == 1
    assert r.tas_s_segments and all(s.end_s <= r.duration_s + 1e-6 for s in r.tas_s_segments)
    assert r.has_anomaly and set(r.anomaly_frac) == {"L", "R"}


def test_pool_and_vocab(catalog, repo):
    ids = catalog.execution_ids()
    assert len(ids) == 45
    proc = load_procedure(repo / "procedures" / "impact_reassembly_A")
    vocab_ids = {v["id"] for v in catalog.psr_vocab}
    assert vocab_ids == proc.vocab.psr_ids()
    assert set(catalog.tas_s_mapping.values()) == set(proc.vocab.tas_s_labels)


def test_ref_has_no_annotations(catalog):
    ref = catalog.ref("AL07EJ17_Reassembly_A_002")
    assert ref.duration_s == pytest.approx(173.8) and not hasattr(ref, "psr_events")


def test_split_plan_and_gt(catalog, repo):
    cfg = load_config(repo / "configs" / "default.yaml")
    recs = catalog.records(catalog.execution_ids())
    plan = make_split_plan(recs, cfg.demos, cfg.splits)
    assert len(plan.candidates) >= cfg.demos.k + 1
    proc = load_procedure(repo / "procedures" / "impact_reassembly_A")
    req = proc.requires("grouped")
    gts = [compute_gt(r, proc.steps("grouped"), req) for r in recs]
    assert sum(not g.compliant for g in gts) >= 4  # LE06AS03 never finishes the adapter plate
