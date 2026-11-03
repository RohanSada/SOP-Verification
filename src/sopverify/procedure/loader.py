"""Load and validate a procedure package; resolve its prerequisite graph per graph_source."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from sopverify.procedure import graph as G
from sopverify.procedure.schema import GraphSource, ProcedureSpec, ROIConfig, Step, StepSet, Vocabulary
from sopverify.utils import read_json, read_yaml


class GraphSourceUnavailable(RuntimeError):
    pass


@dataclass
class Procedure:
    root: Path
    spec: ProcedureSpec
    sop_text: str
    step_sets: dict[str, StepSet]
    roi: ROIConfig
    vocab: Vocabulary | None = None
    released_event_graph: nx.DiGraph | None = None
    _validated: dict[str, list[str]] = field(default_factory=dict)

    @property
    def procedure_id(self) -> str:
        return self.spec.procedure_id

    def step_set(self, name: str | None = None) -> StepSet:
        name = name or self.spec.default_step_set
        if name not in self.step_sets:
            raise KeyError(f"procedure {self.procedure_id} has no step set {name!r}; has {sorted(self.step_sets)}")
        return self.step_sets[name]

    def steps(self, name: str | None = None) -> list[Step]:
        return list(self.step_set(name).steps)

    def validate(self, name: str | None = None) -> list[str]:
        ss = self.step_set(name)
        return G.validate_steps(ss.steps, self.vocab, ss.step_set)

    def requires(
        self,
        name: str | None = None,
        source: GraphSource | None = None,
        completions: Sequence[Mapping[str, float | None]] | None = None,
        min_support: int = 5,
    ) -> dict[str, list[str]]:
        """Prerequisites per step for the requested graph source.

        * manual: the `requires` lists in the step-set YAML.
        * released: derived from the dataset's event-level graph (spec 8.2).
        * inferred_all / inferred_demos: inferred from `completions` (the caller decides
          whose completions they are: all pool executions or demos only).
        """
        steps = self.steps(name)
        source = source or self.spec.graph_source
        ids = [s.id for s in steps]
        if source == "manual":
            g = G.requires_graph(steps)
            return G.to_requires(G.transitive_reduction(g), ids)
        if source == "released":
            if self.released_event_graph is None:
                raise GraphSourceUnavailable(
                    f"procedure {self.procedure_id} has no usable released graph (see DECISIONS D-004)"
                )
            if all(s.gt_mapping.psr_events and len(s.gt_mapping.psr_events) == 1 for s in steps):
                sub = self.released_event_graph.subgraph(
                    [e for s in steps for e in (s.gt_mapping.psr_events or [])]
                ).copy()
                ev2step = {s.gt_mapping.psr_events[0]: s.id for s in steps}  # type: ignore[index]
                sub = nx.relabel_nodes(sub, ev2step)
                sub.add_nodes_from(ids)
                return G.to_requires(G.transitive_reduction(sub), ids)
            return G.derive_group_graph(steps, self.released_event_graph)
        if source in ("inferred_all", "inferred_demos"):
            if completions is None:
                raise ValueError(f"graph source {source} needs completions")
            req, _ = G.infer_graph(completions, ids, min_support=min_support)
            return req
        raise ValueError(f"unknown graph source {source!r}")

    def roi_for(self, step_id: str | None = None) -> list[int]:
        if step_id:
            st = self.step_set().by_id().get(step_id)
            if st is not None and st.roi:
                return list(st.roi)
            if step_id in self.roi.step_overrides:
                return list(self.roi.step_overrides[step_id])
        return list(self.roi.box)

    def render_graph(self, name: str | None = None, source: GraphSource | None = None,
                     colors: Mapping[str, str] | None = None,
                     completions: Sequence[Mapping[str, float | None]] | None = None) -> str:
        steps = self.steps(name)
        req = self.requires(name, source, completions)
        return G.render_svg(req, labels={s.id: s.name for s in steps}, colors=colors,
                            order=[s.id for s in steps], title=f"{self.spec.name} [{name or self.spec.default_step_set}]")


def load_procedure(path: str | Path) -> Procedure:
    root = Path(path)
    if root.is_file():
        root = root.parent
    spec = ProcedureSpec.model_validate(read_yaml(root / "procedure.yaml"))
    sop_path = root / spec.sop_text
    sop_text = sop_path.read_text() if sop_path.exists() else ""
    step_sets = {}
    for name, rel in spec.step_sets.items():
        ss = StepSet.model_validate(read_yaml(root / rel))
        if ss.step_set != name:
            raise ValueError(f"{rel}: step_set is {ss.step_set!r} but procedure.yaml lists it as {name!r}")
        step_sets[name] = ss
    if spec.default_step_set not in step_sets:
        raise ValueError(f"default_step_set {spec.default_step_set!r} not among {sorted(step_sets)}")
    roi = ROIConfig.model_validate(read_yaml(root / spec.roi))
    vocab = None
    if spec.vocabulary and (root / spec.vocabulary).exists():
        vocab = Vocabulary.model_validate(read_yaml(root / spec.vocabulary))
    released = None
    if spec.released_graph:
        p = root / spec.released_graph
        if p.exists():
            d = read_json(p) if p.suffix == ".json" else read_yaml(p)
            released = nx.DiGraph()
            released.add_nodes_from(d.get("nodes", []))
            released.add_edges_from(tuple(e) for e in d.get("edges", []))
    return Procedure(root=root, spec=spec, sop_text=sop_text, step_sets=step_sets, roi=roi,
                     vocab=vocab, released_event_graph=released)
