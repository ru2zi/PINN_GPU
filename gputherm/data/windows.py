"""window 시작 시각과 정답 coverage 필터 (spec 2.6절 3–4, 7절).

inventory, 학습, 평가는 모두 `usable_windows`로 window를 만든다.
"""
import numpy as np
import pandas as pd


def eval_bins(d, cfg):
    """0.5 s 평가 격자. bin k = [0.5k, 0.5(k+1)). 원샘플 수와 '모든 원샘플이 본 구간'인지 표시한다."""
    dt = cfg["grid"]["eval_s"]
    k = np.floor(d["t"].values / dt).astype(int)
    main = d["phase"].isin(cfg["phases"]["main"]).values
    n = np.bincount(k, minlength=k.max() + 1)
    n_main = np.bincount(k, weights=main, minlength=k.max() + 1)
    return n, (n > 0) & (n_main == n)


def window_starts(d, cond, cfg, stride_s):
    """평가 또는 학습 window의 시작 시각 t₀ 목록 (spec 2.6절 3, 7절).

    horizon 60 s의 모든 0.5 s bin이 본 구간 mask 안에 있어야 한다. 원샘플이 없는 bin은 본 구간 사이의
    공백이면 mask를 끊지 않는다(정답 coverage 규칙으로 따로 처리). I01-32B는 첫 inference* 시점부터 60 s를 뺀다.
    """
    dt = cfg["grid"]["eval_s"]
    h = int(round(cfg["task"]["horizon_s"] / dt))
    n, main_bin = eval_bins(d, cfg)
    t = d["t"].values
    is_main = d["phase"].isin(cfg["phases"]["main"]).values
    first_main, last_main = t[is_main].min(), t[is_main].max()
    k0 = int(np.ceil(first_main / dt))
    k1 = int(np.floor(last_main / dt))          # 마지막 본 구간 bin (이 bin 끝까지 포함하지 않음)
    drop = cfg["masks"]["drop_after_first_inference_s"].get(cond)
    if drop is not None:
        t_inf = t[d["phase"].str.startswith("inference").values].min()
        k0 = max(k0, int(np.ceil((t_inf + drop) / dt)))
    in_main = np.zeros_like(main_bin)
    in_main[k0:k1] = True
    in_main &= (n == 0) | main_bin              # 원샘플이 있는 bin은 모두 본 구간이어야 함
    step = int(round(stride_s / dt))
    starts = [k for k in range(k0, k1 - h + 1, step) if in_main[k:k + h].all()]
    return np.array(starts) * dt, n


def target_coverage(n, start_s, cfg):
    """window별 유효 정답 bin 수 (spec 2.6절 4: 원샘플 25개 이상인 bin만 유효)."""
    dt = cfg["grid"]["eval_s"]
    h = int(round(cfg["task"]["horizon_s"] / dt))
    k = np.round(np.asarray(start_s) / dt).astype(int)
    valid = n >= cfg["targets"]["min_samples_per_bin"]
    return np.array([valid[i:i + h].sum() for i in k])


def usable_windows(d, cond, cfg, stride_s):
    """학습·평가에 쓰는 window 시작 목록 (spec 2.6절 4, 7절). inventory, 학습, 평가가 모두 이 함수를 쓴다.

    1. 본 구간 mask 조건(window_starts)
    2. 정답 coverage: 유효 정답 bin(원샘플 25개 이상)이 120개 중 108개 미만이면 뺀다
    이력 부족(t₀ < 최대 look-back)은 빼지 않는다. 그런 window는 첫 샘플 값으로 채우고 개수를 기록한다(7절).
    반환: (쓸 시작 시각 배열, 뺀 window DataFrame[t0, reason], 이력 부족 개수)
    """
    starts, n = window_starts(d, cond, cfg, stride_s)
    h = int(round(cfg["task"]["horizon_s"] / cfg["grid"]["eval_s"]))
    cov = target_coverage(n, starts, cfg) if len(starts) else np.array([], int)
    low = cov < cfg["targets"]["min_valid_frac"] * h
    dropped = pd.DataFrame({"t0": starts[low], "reason": [f"정답 coverage {c}/{h}" for c in cov[low]]})
    kept = starts[~low]
    short = int((kept < max(cfg["task"]["lookback_candidates_s"])).sum())
    return kept, dropped, short
