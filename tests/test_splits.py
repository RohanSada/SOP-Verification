from sopverify.config import DemosConfig, SplitsConfig
from sopverify.data.impact.splits import demo_candidates, is_complete, make_split_plan

from helpers import make_record


def _pool():
    recs = []
    for p in range(6):
        for t in range(3):
            eid = f"PA{p:02d}XX{t:02d}_Reassembly_A_00{t + 1}"
            frac = {"L": 0.01 * (p + 1), "R": 0.01}
            recs.append(make_record(eid, final_states={c: 1 for c in "abcd"}, anomaly_frac=frac))
    recs.append(make_record("PZ99ZZ99_Reassembly_A_001", installs={"a": 1.0}, anomaly_frac={"L": 0, "R": 0}))
    return recs


def test_complete_and_candidates():
    recs = _pool()
    assert is_complete(recs[0]) and not is_complete(recs[-1])
    c = demo_candidates(recs, DemosConfig(mode="complete_low_anomaly", max_anomaly_frac=0.03))
    assert all(x.startswith(("PA00", "PA01", "PA02")) for x in c) and len(c) == 9
    assert len(demo_candidates(recs, DemosConfig(mode="any_complete"))) == 18


def test_split_plan_participant_disjoint_and_seen():
    recs = _pool()
    plan = make_split_plan(recs, DemosConfig(k=2, n_seeds=4, mode="any_complete"), SplitsConfig(dev_fraction=0.3, seed=1))
    dev_p = {plan.participants[e] for e in plan.dev}
    test_p = {plan.participants[e] for e in plan.test}
    assert not dev_p & test_p and len(plan.dev) + len(plan.test) == len(recs)
    assert 0.2 <= len(plan.dev) / len(recs) <= 0.45
    assert len(plan.demo_draws) == 4 and all(len(d) == 2 for d in plan.demo_draws)
    for s, draw in enumerate(plan.demo_draws):
        assert len({plan.participants[d] for d in draw}) == 2  # distinct participants
        ev = plan.eval_ids("test", s) + plan.eval_ids("dev", s)
        assert not set(draw) & set(ev)
        held = plan.eval_ids("test", s, held_out_participant=True)
        assert not any(plan.is_seen(e, s) for e in held)
    again = make_split_plan(recs, DemosConfig(k=2, n_seeds=4, mode="any_complete"), SplitsConfig(dev_fraction=0.3, seed=1))
    assert again.demo_draws == plan.demo_draws and again.dev == plan.dev
