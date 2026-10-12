"""run 단위 형식·오염 검사와 제외 규칙 (spec 2.5절, 2.7절).

test run에는 형식 검사와 window 수만 계산한다(spec 0절 규칙 1).
"""
import numpy as np
import pandas as pd

from gputherm.data.io import load_raw, md5
from gputherm.data.windows import usable_windows


PARSE_CHECK = {
    "T02-A": (578, 48.3), "T02-B": (579, 49.2), "T02-32B-A": (595, 49.3), "T02-32B-A2": (596, 49.2),
    "T04-A": (578, 48.4), "I01-Q7B-A": (424, 40.2), "T04-D": (598, 49.7), "I01-8B-A": (432, 40.7),
    "I02-A": (588, 48.6),
}


PARSE_TOL = (10.0, 0.5)


def phase_runs(phase):
    """연속된 같은 phase를 하나로 묶은 순서."""
    p = phase.values
    keep = np.r_[True, p[1:] != p[:-1]]
    return list(p[keep])


def phase_order_ok(seq, cfg):
    """warmup → measure → 본 구간 phase들 → cooldown 순서인지 확인."""
    main = set(cfg["phases"]["main"])
    if len(seq) < 4 or seq[0] != "warmup" or seq[1] != "measure" or seq[-1] != "cooldown":
        return False
    return all(s in main for s in seq[2:-1])


def inspect_run(row, cfg):
    """run 하나의 형식·오염 검사와 window 수. train·validation이면 본 구간 요약 통계도 계산한다."""
    d = load_raw(row.path)
    t, ts = d["t"].values, d["timestamp_ms"].values
    gaps = np.diff(t)
    seq = phase_runs(d["phase"])
    warm = d[d["phase"] == cfg["phases"]["warmup"]]
    cool = d[d["phase"] == cfg["phases"]["cooldown"]]
    t_main0 = t[d["phase"].isin(cfg["phases"]["main"]).values].min()
    out = dict(
        run_id=row.run_id, cond=row.cond, run=row.run, split=row.split, md5=md5(row.path),
        start_utc=pd.to_datetime(ts[0], unit="ms").isoformat(), end_utc=pd.to_datetime(ts[-1], unit="ms").isoformat(),
        n_rows=len(d), duration_s=round(t[-1], 1), n_nan=int(d.drop(columns=["phase"]).isna().sum().sum()),
        n_dup_ts=int((gaps == 0).sum()), monotonic=bool((gaps >= 0).all()),
        median_dt_ms=round(float(np.median(gaps)) * 1e3, 2), max_gap_s=round(float(gaps.max()), 2),
        n_gap_over_0p5s=int((gaps > 0.5).sum()),
        phase_sequence=" > ".join(seq), phase_order_ok=phase_order_ok(seq, cfg),
        warmup_s=round(warm["t"].max() - warm["t"].min(), 1),
        history_before_main_s=round(t_main0, 1),
        # 반복 control (spec 2.7절): warmup 마지막 30 s 평균 온도와 power, cooldown 끝 10 s 평균 온도
        warmup_plateau_c=warm[warm["t"] >= warm["t"].max() - 30]["temperature_c"].mean(),
        warmup_plateau_p=warm[warm["t"] >= warm["t"].max() - 30]["power_w"].mean(),
        cooldown_end_c=cool[cool["t"] >= cool["t"].max() - 10]["temperature_c"].mean(),
        cooldown_mean_p=round(cool["power_w"].mean(), 1), cooldown_mean_util=round(cool["utilization_pct"].mean(), 2),
    )
    ev, ev_drop, ev_short = usable_windows(d, row.cond, cfg, cfg["task"]["stride_eval_s"])
    tr, tr_drop, _ = usable_windows(d, row.cond, cfg, cfg["task"]["stride_train_s"])
    out.update(n_eval_windows=len(ev), n_train_windows=len(tr),
               n_eval_low_coverage=len(ev_drop), n_train_low_coverage=len(tr_drop),
               n_eval_short_history=ev_short)
    if row.split in ("train", "val"):
        m = d["phase"].isin(cfg["phases"]["main"]).values
        out.update(main_mean_p=d["power_w"][m].mean(), main_mean_t=d["temperature_c"][m].mean(),
                   main_p_min=round(d["power_w"][m].min(), 1), main_p_max=round(d["power_w"][m].max(), 1),
                   main_t_min=d["temperature_c"][m].min(), main_t_max=d["temperature_c"][m].max(),
                   all_p_min=round(d["power_w"].min(), 1), all_p_max=round(d["power_w"].max(), 1),
                   all_t_min=d["temperature_c"].min(), all_t_max=d["temperature_c"].max())
    return out


def exclusion_reasons(r, plateau_median):
    """spec 2.7절 제외 규칙 (평가 전 고정)."""
    reasons = []
    if not r["phase_order_ok"]:
        reasons.append("phase 순서 깨짐")
    if r["n_nan"] / max(r["n_rows"], 1) > 0.01:
        reasons.append("결측 1 % 초과")
    if r["cooldown_mean_p"] > 50 or r["cooldown_mean_util"] > 5:
        reasons.append("다른 작업과 동시 실행 흔적(cooldown)")
    candidate = abs(r["warmup_plateau_c"] - plateau_median) >= 2.0
    return reasons, candidate
