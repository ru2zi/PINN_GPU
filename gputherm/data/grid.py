"""0.1 s 입력 격자와 window 입력·정답 (spec 2.6절).

bin k는 [0.1k, 0.1(k+1)) s의 샘플 평균이다. 빈 bin은 앞 값으로 채우며(인과성 유지), 채운 값은 모델 입력에만 쓴다.
평가 정답은 원샘플에서 직접 만든다.
"""
import numpy as np


COLS = {"P": "power_w", "u": "utilization_pct", "mu": "memory_utilization_pct", "T": "temperature_c"}


def grid(d, dt=0.1):
    """원시 기록 → dt 격자 bin 평균(앞 값 채움). 반환 DataFrame(index = bin 번호), 열 P, u, mu, T, filled, phase."""
    k = np.floor(d["t"].values / dt + 1e-9).astype(int)
    g = d.assign(k=k).groupby("k")
    out = g[list(COLS.values())].mean().rename(columns={v: c for c, v in COLS.items()})
    out["phase"] = g["phase"].last()
    out = out.reindex(np.arange(k.max() + 1))
    out["filled"] = out["P"].isna()
    out[["P", "u", "mu", "T", "phase"]] = out[["P", "u", "mu", "T", "phase"]].ffill()
    return out


class RunWindows:
    """run 하나의 격자와 window 입력. 관측기 궤적은 run마다 한 번 계산해 두고 window별로 잘라 쓴다."""

    def __init__(self, d, cfg, dt=0.1):
        self.cfg, self.dt = cfg, dt
        self.d = d
        self.g = grid(d, dt)
        self.t_raw = d["t"].values
        self.T_raw = d["temperature_c"].values

    def k_of(self, t0):
        """시각 t₀에 해당하는 0.1 s bin 경계 번호."""
        return int(round(t0 / self.dt))

    def start_temperature(self, t0, rule="last_0p1"):
        """출발 상태 T_g(t₀). 원샘플 시각 기준으로 정의해 입력 격자(0.01 s, 0.1 s)와 무관하게 같다.

        last_0p1: [t₀ − 0.1, t₀) 원샘플 평균(기본, spec 2.6절 3), last_0p5: [t₀ − 0.5, t₀) 평균, last_sample: t₀ 직전 원샘플.
        구간에 원샘플이 없으면 t₀ 직전 원샘플을 쓴다(앞 값 채움과 같다).
        """
        before = self.t_raw < t0
        if rule == "last_sample":
            return float(self.T_raw[before][-1])
        w = {"last_0p1": 0.1, "last_0p5": 0.5}[rule]
        sel = before & (self.t_raw >= t0 - w - 1e-9)
        return float(self.T_raw[sel].mean()) if sel.any() else float(self.T_raw[before][-1])

    def past(self, t0, col):
        """t₀까지 끝난 0.1 s bin들의 값 (관측기 입력)."""
        return self.g[col].values[:self.k_of(t0)]

    def horizon_inputs(self, t0, H=60.0):
        """[t₀, t₀ + H)의 0.1 s bin 입력 (계획값으로 쓰는 측정값)."""
        K, n = self.k_of(t0), int(round(H / self.dt))
        return {c: self.g[c].values[K:K + n] for c in ("P", "u", "mu")}

    def horizon_inputs_raw(self, t0, dt, H=60.0, col="power_w"):
        """[t₀, t₀ + H)를 dt 간격 구간으로 나눠 원샘플을 직접 평균한 입력. 빈 구간은 앞 값으로 채운다.

        입력 격자 비교(6.2절)용. dt마다 run 격자를 거치지 않고 t₀ 기준으로 집계하므로, 0.25 s처럼
        0.1 s의 정수배가 아닌 격자도 정확히 만든다.
        """
        n = int(round(H / dt))
        if abs(n * dt - H) > 1e-9:
            raise ValueError(f"H={H}가 dt={dt}의 정수배가 아니다")
        v = self.d[col].values
        idx = np.floor((self.t_raw - t0) / dt + 1e-9).astype(int)
        sel = (idx >= 0) & (idx < n)
        cnt = np.bincount(idx[sel], minlength=n)
        s = np.bincount(idx[sel], weights=v[sel], minlength=n)
        out = np.full(n, np.nan)
        out[cnt > 0] = s[cnt > 0] / cnt[cnt > 0]
        prev = v[self.t_raw < t0][-1]
        for i in range(n):
            if np.isnan(out[i]):
                out[i] = prev
            prev = out[i]
        return out

    def lookback_1s(self, t0, L):
        """DLinear-X 과거 입력. j번째 1 s bin = [t₀ − j − 1, t₀ − j), j = 0..L−1. 반환 (L, 4), 행 0이 가장 최근."""
        K, m = self.k_of(t0), int(round(1.0 / self.dt))
        v = self.g[["T", "P", "u", "mu"]].values[:K]
        need = m * L - K
        if need > 0:                                   # 기록 시작 전은 첫 샘플 값으로 채운다
            v = np.vstack([np.repeat(v[:1], need, axis=0), v])
        K = len(v)
        return np.stack([v[K - m * (j + 1):K - m * j].mean(axis=0) for j in range(L)])

    def future_1s(self, t0, H=60):
        """DLinear-X 미래 입력. j번째 1 s bin = [t₀ + j, t₀ + j + 1). 반환 (H, 3)."""
        K, m = self.k_of(t0), int(round(1.0 / self.dt))
        v = self.g[["P", "u", "mu"]].values
        return np.stack([v[K + m * j:K + m * (j + 1)].mean(axis=0) for j in range(H)])

    def targets(self, t0, H=60.0, dt_eval=0.5):
        """원샘플에서 직접 만든 정답: bin k = [t₀ + 0.5k, t₀ + 0.5(k+1))의 평균 T_g와 원샘플 수."""
        n = int(round(H / dt_eval))
        idx = np.floor((self.t_raw - t0) / dt_eval + 1e-9).astype(int)
        sel = (idx >= 0) & (idx < n)
        cnt = np.bincount(idx[sel], minlength=n)
        s = np.bincount(idx[sel], weights=self.T_raw[sel], minlength=n)
        with np.errstate(invalid="ignore"):
            y = s / cnt
        valid = cnt >= self.cfg["targets"]["min_samples_per_bin"]
        return y, valid, cnt
