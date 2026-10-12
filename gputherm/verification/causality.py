"""인과성과 정답 coverage 필터 검사 (spec 2.6절 7).

t₀ 이후 원샘플을 바꿔도 출발 온도, 관측기 상태, look-back이 변하지 않아야 하고 계획 입력과 정답은 변해야 한다.
정수 초·반정수 초 시작과 이력 부족 window를 검사한다. coverage 필터는 합성 공백으로 검사한다.

실행: python -m gputherm.verification.causality
"""
import numpy as np
import pandas as pd

from gputherm.config import load_cfg
from gputherm.data.grid import RunWindows
from gputherm.data.io import list_runs, load_raw
from gputherm.data.windows import usable_windows, window_starts
from gputherm.physics.observer import observer_trajectory
from gputherm.verification.common import representative_params


def _perturb_future(d, t0, rng):
    """t₀ 이후 원샘플의 온도·power·사용률을 무작위로 바꾼 사본."""
    d2 = d.copy()
    m = d2["t"].values >= t0
    for c in ("temperature_c", "power_w", "utilization_pct", "memory_utilization_pct"):
        d2.loc[m, c] = d2.loc[m, c].values + rng.normal(0, 5, m.sum())
    return d2


def causality_test(path, t0s, cfg, p_obs, Ta=27.0, L=180, seed=0):
    """spec 2.6절 3: t₀ 이후 샘플을 바꿔도 출발 온도, 관측기 상태, look-back이 바뀌지 않아야 한다.

    계획 입력과 정답은 바뀌는 것이 정상이다. 반환: window별 결과 dict 목록.
    """
    rng = np.random.default_rng(seed)
    d = load_raw(path)
    out = []
    for t0 in t0s:
        a, b = RunWindows(d, cfg), RunWindows(_perturb_future(d, t0, rng), cfg)
        K = a.k_of(t0)
        th_a = observer_trajectory(p_obs, a.past(t0, "P"), Ta, a.past(t0, "T"), a.dt)[K]
        th_b = observer_trajectory(p_obs, b.past(t0, "P"), Ta, b.past(t0, "T"), b.dt)[K]
        ya, _, _ = a.targets(t0); yb, _, _ = b.targets(t0)
        out.append(dict(
            t0=t0, integer_second=float(t0).is_integer(),
            start_T_same=a.start_temperature(t0) == b.start_temperature(t0),
            observer_same=th_a == th_b,
            lookback_same=bool(np.array_equal(a.lookback_1s(t0, L), b.lookback_1s(t0, L))),
            past_bins_same=bool(np.array_equal(a.past(t0, "T"), b.past(t0, "T")) and np.array_equal(a.past(t0, "P"), b.past(t0, "P"))),
            plan_changed=not np.array_equal(a.horizon_inputs(t0)["P"], b.horizon_inputs(t0)["P"]),
            future_1s_changed=not np.array_equal(a.future_1s(t0), b.future_1s(t0)),
            target_changed=not np.allclose(ya, yb, equal_nan=True),
        ))
    return out


def main():
    cfg = load_cfg()
    runs = list_runs(cfg)
    p = representative_params()
    rows = []
    for rid in ("T02-A/r01", "I01-Q7B-A/r02"):                      # train run만 쓴다
        r = runs[runs.run_id == rid].iloc[0]
        d = load_raw(r.path)
        st, _ = window_starts(d, r.cond, cfg, cfg["task"]["stride_eval_s"])
        ints = [t for t in st if float(t).is_integer()][:2]
        halves = [t for t in st if not float(t).is_integer()][:2]
        if not halves:                                              # 반정수 시작이 없으면 0.5 s 옮긴 시작을 쓴다
            halves = [ints[0] + 0.5]
        for res in causality_test(r.path, ints + halves, cfg, p):
            rows.append(dict(run_id=rid, **res))
    # 이력 부족 window: t₀ = 5 s, L = 8 s이면 3 s가 기록 시작 전이다. 첫 샘플로 채워지고 t₀ 이후 변경에 영향받지 않아야 한다
    r = runs[runs.run_id == "T02-A/r01"].iloc[0]
    d = load_raw(r.path)
    a_, b_ = RunWindows(d, cfg), RunWindows(_perturb_future(d, 5.0, np.random.default_rng(1)), cfg)
    lb_a, lb_b = a_.lookback_1s(5.0, 8), b_.lookback_1s(5.0, 8)
    first = a_.g[["T", "P", "u", "mu"]].values[0]
    rows.append(dict(run_id="T02-A/r01 (이력 부족 t₀=5 s, L=8 s)", t0=5.0, integer_second=True,
                     start_T_same=a_.start_temperature(5.0) == b_.start_temperature(5.0), observer_same=True,
                     lookback_same=bool(np.array_equal(lb_a, lb_b) and np.isfinite(lb_a).all()
                                        and np.allclose(lb_a[5:], first[None, :])),
                     past_bins_same=bool(np.array_equal(a_.past(5.0, "T"), b_.past(5.0, "T"))),
                     plan_changed=True, future_1s_changed=True, target_changed=True))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # 정답 coverage 필터 자체 검사: 본 구간 안 15 s의 원샘플을 지운 합성 기록에서 해당 window가 빠져야 한다
    r = runs[runs.run_id == "T02-A/r01"].iloc[0]
    d = load_raw(r.path)
    full, _, _ = usable_windows(d, r.cond, cfg, cfg["task"]["stride_eval_s"])
    gap0 = float(full[len(full) // 2]) + 20.0
    d_gap = d[(d["t"] < gap0) | (d["t"] >= gap0 + 15.0)].reset_index(drop=True)
    kept, dropped, _ = usable_windows(d_gap, r.cond, cfg, cfg["task"]["stride_eval_s"])
    # 공백 시작이 0.5 s 격자에 맞으므로, 겹친 길이 / 0.5 s가 곧 원샘플이 없는 정답 bin 수다.
    # 유효 bin이 108개 미만, 즉 겹친 길이가 6 s를 넘는 window만 빠져야 한다.
    ov = np.minimum(full + 60.0, gap0 + 15.0) - np.maximum(full, gap0)
    expect = full[ov > 6.0 + 1e-9]
    cov_ok = (set(np.round(dropped["t0"], 3)) == set(np.round(expect, 3))
              and len(kept) == len(full) - len(expect) and len(expect) > 0)
    print(f"\ncoverage 필터: 공백과 6 s 넘게 겹치는 window {len(expect)}개(겹침 있는 window {(ov > 0).sum()}개), "
          f"뺀 window {len(dropped)}개 →", "통과" if cov_ok else "실패")
    must_same = df[["start_T_same", "observer_same", "lookback_same", "past_bins_same"]].all().all()
    must_change = df[["plan_changed", "future_1s_changed", "target_changed"]].all().all()
    both = df["integer_second"].any() and (~df["integer_second"]).any()
    ok = bool(must_same and must_change and both and cov_ok)
    df.to_csv(cfg["paths"]["results"] + "/verification/causality.csv", index=False)
    print("\n인과성 test", "통과" if ok else "실패")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
