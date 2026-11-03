import networkx as nx
import pytest

from helpers import chain_steps, step, write_toy_procedure
from sopverify.procedure import graph as G
from sopverify.procedure.loader import GraphSourceUnavailable, load_procedure
from sopverify.procedure.schema import GTMapping, Step


def test_validation_detects_problems():
    steps = [step("A"), step("A"), step("B", ["Z"]), step("C", ["D"]), step("D", ["C"])]
    errs = " | ".join(G.validate_steps(steps))
    assert "duplicate" in errs and "unknown step 'Z'" in errs and "cycle" in errs


def test_validation_vocab(tmp_path):
    p = load_procedure(write_toy_procedure(tmp_path))
    assert p.validate() == []
    bad = [Step(id="A", name="a", gt_mapping=GTMapping(psr_events=["install:nope"]))]
    assert any("not in vocabulary" in e for e in G.validate_steps(bad, p.vocab))
    assert any("gt_mapping is empty" in e for e in G.validate_steps([Step(id="A", name="a")]))


def test_transitive_reduction_and_requires(tmp_path):
    steps = [step("A"), step("B", ["A"]), step("C", ["A", "B"])]
    p = load_procedure(write_toy_procedure(tmp_path, steps))
    assert p.requires() == {"A": [], "B": ["A"], "C": ["B"]}


def test_derive_group_graph():
    ev = nx.DiGraph([("e1", "e2"), ("e2", "e3"), ("e3", "e4"), ("e1", "e5")])
    groups = [Step(id="G1", name="g1", gt_mapping=GTMapping(psr_events=["e1", "e2"])),
              Step(id="G2", name="g2", gt_mapping=GTMapping(psr_events=["e3"])),
              Step(id="G3", name="g3", gt_mapping=GTMapping(psr_events=["e4"])),
              Step(id="G4", name="g4", gt_mapping=GTMapping(psr_events=["e5"]))]
    assert G.derive_group_graph(groups, ev) == {"G1": [], "G2": ["G1"], "G3": ["G2"], "G4": ["G1"]}
    cyc = [Step(id="X", name="x", gt_mapping=GTMapping(psr_events=["e1", "e4"])),
           Step(id="Y", name="y", gt_mapping=GTMapping(psr_events=["e2"]))]
    with pytest.raises(ValueError):
        G.derive_group_graph(cyc, ev)


def test_infer_graph_toy_orderings():
    comps = [{"A": 1, "B": 2, "C": 3}] * 5 + [{"A": 1, "B": 3, "C": 2}]
    req, sup = G.infer_graph(comps, ["A", "B", "C"], min_support=5)
    assert req == {"A": [], "B": ["A"], "C": ["A"]}  # B/C order varies -> no edge
    assert sup[("A", "B")] == 6
    req2, _ = G.infer_graph(comps[:4], ["A", "B", "C"], min_support=5)  # support too low
    assert req2 == {"A": [], "B": [], "C": []}
    ties = [{"A": 1, "B": 1}] * 6
    assert G.infer_graph(ties, ["A", "B"], 5)[0] == {"A": [], "B": []}
    missing = [{"A": 1, "B": 2}] * 5 + [{"A": None, "B": 1}]
    assert G.infer_graph(missing, ["A", "B"], 5)[0] == {"A": [], "B": ["A"]}


def test_graph_sources(tmp_path):
    p = load_procedure(write_toy_procedure(tmp_path, chain_steps()))
    with pytest.raises(GraphSourceUnavailable):
        p.requires(source="released")
    comps = [{"A": 1, "B": 2, "C": 3, "D": 4}] * 5
    assert p.requires(source="inferred_all", completions=comps) == {"A": [], "B": ["A"], "C": ["B"], "D": ["C"]}


def test_render_svg(tmp_path):
    p = load_procedure(write_toy_procedure(tmp_path))
    svg = p.render_graph(colors={"A": "#123456"})
    assert svg.startswith("<svg") and "#123456" in svg and svg.count("<rect") == 4
    with pytest.raises(ValueError):
        G.render_svg({"A": ["B"], "B": ["A"]})


def test_impact_package_validates(repo):
    p = load_procedure(repo / "procedures" / "impact_reassembly_A")
    assert set(p.step_sets) == {"grouped", "tas_s", "psr"}
    for name in p.step_sets:
        assert p.validate(name) == [], name
    assert len(p.steps("grouped")) == 5 and len(p.steps("psr")) == 17 and len(p.steps("tas_s")) == 13
    assert sum(s.visually_verifiable for s in p.steps("tas_s")) == 5
    assert p.requires("grouped") == {"G1": [], "G2": ["G1"], "G3": ["G2"], "G4": ["G3"], "G5": ["G1"]}
    assert len(p.roi.box) == 4
