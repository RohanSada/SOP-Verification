"""Demo selection and participant-disjoint dev/test splits (spec 9.2, decision D-003).

Order of operations:
1. Split the pool's participants into dev/test once (fixed seed, participant-disjoint).
2. For each demo seed, draw K demos from the demo candidates (whole pool).
3. The eval set for that seed is dev/test minus the drawn demos (and minus all of the
   demo participants' executions when `held_out_participant` is on).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np

from sopverify.config import DemosConfig, SplitsConfig
from sopverify.data.records import ExecutionRecord
from sopverify.logging_utils import get_logger

log = get_logger(__name__)


def is_complete(rec: ExecutionRecord, n_components: int | None = None) -> bool:
    """All components installed (state 1) at the end, and no Remove events (no rework)."""
    final = rec.final_component_states()
    if not final:
        return False
    if n_components is not None and len(final) != n_components:
        return False
    return all(v == 1 for v in final.values()) and rec.n_remove_events() == 0


def demo_candidates(records: list[ExecutionRecord], cfg: DemosConfig) -> list[str]:
    """Executions eligible as demonstrations under `cfg.mode` and `cfg.expertise`."""
    out: list[ExecutionRecord] = []
    for r in records:
        if not is_complete(r):
            continue
        if cfg.mode == "complete_low_anomaly":
            if not r.anomaly_frac or max(r.anomaly_frac.values()) > cfg.max_anomaly_frac + 1e-9:
                continue
        if cfg.mode == "experts_only" and r.expertise != "expert":
            continue
        out.append(r)
    if cfg.expertise == "expert_only":
        out = [r for r in out if r.expertise == "expert"]
    elif cfg.expertise == "prefer_expert":
        experts = [r for r in out if r.expertise == "expert"]
        # Only narrow to experts if that still leaves room for different draws per seed.
        if len(experts) >= cfg.k + 1:
            out = experts
        elif any(r.expertise != "unknown" for r in records):
            log.info("only %d expert demo candidates (< k+1); using all %d candidates", len(experts), len(out))
    ids = sorted(r.execution_id for r in out)
    if len(ids) < cfg.k:
        log.warning("only %d demo candidates for k=%d", len(ids), cfg.k)
    return ids


def participant_split(records: list[ExecutionRecord], cfg: SplitsConfig) -> tuple[list[str], list[str]]:
    """Participant-disjoint dev/test split with ~dev_fraction of executions in dev."""
    by_part: dict[str, list[str]] = {}
    for r in records:
        by_part.setdefault(r.participant, []).append(r.execution_id)
    rng = np.random.default_rng(cfg.seed)
    parts = sorted(by_part)
    if not cfg.participant_disjoint:
        ids = sorted(r.execution_id for r in records)
        perm = rng.permutation(len(ids))
        n_dev = int(round(cfg.dev_fraction * len(ids)))
        dev = sorted(ids[i] for i in perm[:n_dev])
        return dev, sorted(set(ids) - set(dev))
    order = [parts[i] for i in rng.permutation(len(parts))]
    total = len(records)
    dev: list[str] = []
    for p in order:
        if len(dev) >= cfg.dev_fraction * total:
            break
        dev.extend(by_part[p])
    test = sorted(r.execution_id for r in records if r.execution_id not in set(dev))
    return sorted(dev), test


def draw_demos(
    candidates: list[str], records: dict[str, ExecutionRecord], cfg: DemosConfig
) -> list[list[str]]:
    """`n_seeds` draws of K demos (distinct participants when possible)."""
    if cfg.fixed_ids:
        return [list(cfg.fixed_ids)]
    draws: list[list[str]] = []
    for s in range(cfg.n_seeds):
        rng = np.random.default_rng([cfg.seed, s, zlib.crc32(",".join(candidates).encode())])
        perm = [candidates[i] for i in rng.permutation(len(candidates))]
        chosen: list[str] = []
        used: set[str] = set()
        for eid in perm:
            p = records[eid].participant
            if cfg.distinct_participants and p in used:
                continue
            chosen.append(eid)
            used.add(p)
            if len(chosen) == cfg.k:
                break
        if len(chosen) < cfg.k:  # not enough distinct participants: fill up
            chosen += [e for e in perm if e not in chosen][: cfg.k - len(chosen)]
        draws.append(sorted(chosen))
    return draws


@dataclass
class SplitPlan:
    pool: list[str]
    dev: list[str]
    test: list[str]
    candidates: list[str]
    demo_draws: list[list[str]]
    participants: dict[str, str] = field(default_factory=dict)  # execution -> participant

    def eval_ids(self, split: str, seed_idx: int, held_out_participant: bool = False) -> list[str]:
        base = self.dev if split == "dev" else self.test
        demos = set(self.demo_draws[seed_idx])
        demo_parts = {self.participants[d] for d in demos}
        out = [e for e in base if e not in demos]
        if held_out_participant:
            out = [e for e in out if self.participants[e] not in demo_parts]
        return out

    def is_seen(self, execution_id: str, seed_idx: int) -> bool:
        demo_parts = {self.participants[d] for d in self.demo_draws[seed_idx]}
        return self.participants[execution_id] in demo_parts

    def to_dict(self) -> dict:
        return {
            "pool": self.pool,
            "dev": self.dev,
            "test": self.test,
            "candidates": self.candidates,
            "demo_draws": self.demo_draws,
        }


def make_split_plan(records: list[ExecutionRecord], demos: DemosConfig, splits: SplitsConfig) -> SplitPlan:
    by_id = {r.execution_id: r for r in records}
    cands = demo_candidates(records, demos)
    dev, test = participant_split(records, splits)
    draws = draw_demos(cands, by_id, demos)
    return SplitPlan(
        pool=sorted(by_id),
        dev=dev,
        test=test,
        candidates=cands,
        demo_draws=draws,
        participants={e: r.participant for e, r in by_id.items()},
    )
