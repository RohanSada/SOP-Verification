"""State scorers and the factory used by the pipeline and eval."""

from __future__ import annotations

from sopverify.config import Config
from sopverify.state.scorers.base import BaseScorer, StateScorer


def make_scorer(cfg: Config, **kwargs) -> BaseScorer:
    """Build the scorer named in `cfg.scorer.name`. Oracle needs `gt_fn`; VLM needs `backend` etc."""
    name = cfg.scorer.name
    if name == "prototype":
        from sopverify.state.scorers.prototype import PrototypeScorer

        return PrototypeScorer(cfg.scorer, cfg.labels)
    if name == "linear_probe":
        from sopverify.state.scorers.linear import LinearProbeScorer

        return LinearProbeScorer(cfg.scorer, cfg.labels)
    if name == "knn":
        from sopverify.state.scorers.knn import KNNScorer

        return KNNScorer(cfg.scorer, cfg.labels)
    if name == "time_prior":
        from sopverify.state.scorers.time_prior import TimePriorScorer

        return TimePriorScorer()
    if name == "oracle":
        from sopverify.state.scorers.oracle import OracleScorer

        if "gt_fn" not in kwargs:
            raise ValueError("the oracle scorer is evaluation-only and needs gt_fn from sopverify.eval")
        return OracleScorer(kwargs["gt_fn"])
    if name == "vlm":
        from sopverify.state.scorers.vlm import make_vlm_scorer

        return make_vlm_scorer(cfg, **kwargs)
    raise ValueError(f"unknown scorer {name!r}")


__all__ = ["BaseScorer", "StateScorer", "make_scorer"]
