"""cooldown 곡선의 시간 척도 (spec 5.5절 척도의 근거). train·validation run만 쓴다.

cooldown 온도(0.5 s 구간 평균)를 지수 1개와 지수 2개 모델로 fit해 비교한다. 측정이 구간 평균이므로
모델도 구간 [t, t+Δ)의 시간 평균 A(τ/Δ)(e^(−t/τ) − e^(−(t+Δ)/τ))로 계산한다(구간 중앙값으로 근사하지 않는다).
  1개: T_a + A e^(−t/τ)            (t ≥ 5 s 구간으로 느린 성분만)
  2개: T_a + A e^(−t/τ_s) + B e^(−t/τ_f)   (전체 구간)
진폭 A, B는 원래 시간 응답의 진폭이다.
첫 0.5 s 구간이 느린 성분 외삽보다 얼마나 높은지(빠른 성분의 크기)도 기록한다.
빠른 시간상수는 0.5 s 구간 평균으로 추정하므로 power 해상도 근처에서는 신뢰도가 낮다.

실행: python -m gputherm.verification.cooldown_modes
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from gputherm.config import load_cfg
from gputherm.data.io import list_runs, load_raw
from gputherm.verification.common import save


DT = 0.5                                     # 측정 온도를 평균하는 구간 길이


def bin_mean_exp(t, A, tau, dt=DT):
    """구간 [t, t+Δ)에서 A e^(−s/τ)의 시간 평균 = A (τ/Δ)(e^(−t/τ) − e^(−(t+Δ)/τ))."""
    return A * tau / dt * (np.exp(-t / tau) - np.exp(-(t + dt) / tau))


def f1(t, Ta, A, tau):
    """지수 1개 모델의 구간 평균. t는 구간 시작 시각."""
    return Ta + bin_mean_exp(t, A, tau)


def f2(t, Ta, A, tau_s, B, tau_f):
    """지수 2개 모델의 구간 평균. 진폭 A, B는 원래 시간 응답의 진폭이다."""
    return Ta + bin_mean_exp(t, A, tau_s) + bin_mean_exp(t, B, tau_f)


def fit_run(d):
    c = d[d["phase"] == "cooldown"]
    t = c["t"].values - c["t"].values[0]
    k = (t // DT).astype(int)
    g = pd.Series(c["temperature_c"].values).groupby(k).agg(["mean", "size"])
    g = g[g["size"] >= 25]                        # 원샘플이 부족한 bin은 뺀다(정답 규칙과 같은 기준)
    Tb = g["mean"].to_numpy()
    tb = g.index.to_numpy() * DT                  # 실제 bin 번호에서 구간 시작 시각을 만든다(공백이 있어도 시간축 유지)
    P_before = float(d.loc[d["phase"] != "cooldown", "power_w"].values[-200:].mean())
    late = tb >= 5
    p1, _ = curve_fit(f1, tb[late], Tb[late], p0=[27, 15, 27], maxfev=20000)
    p2, _ = curve_fit(f2, tb, Tb, p0=[p1[0], p1[1], p1[2], 2, 1],
                      bounds=([0, 0, 5, -50, 0.05], [60, 80, 300, 80, 5]), maxfev=50000)
    p1all, _ = curve_fit(f1, tb, Tb, p0=[27, 15, 27], maxfev=20000)
    rms = lambda f, p: float(np.sqrt(np.mean((f(tb, *p) - Tb) ** 2)))
    return dict(P_before_W=P_before, tau_1exp_all_s=float(p1all[2]), rms_1exp_all_K=rms(f1, p1all),
                rms_2exp_K=rms(f2, p2), Ta_2exp_C=float(p2[0]), slow_amp_K=float(p2[1]), tau_slow_s=float(p2[2]),
                fast_amp_K=float(p2[3]), tau_fast_s=float(p2[4]),
                excess_first_0p5s_K=float(Tb[0] - f1(tb[0], *p1)))


def main():
    cfg = load_cfg()
    runs = list_runs(cfg)
    rows = []
    for r in runs[runs["split"].isin(["train", "val"])].itertuples():
        rows.append(dict(run_id=r.run_id, **fit_run(load_raw(r.path))))
    df = pd.DataFrame(rows)
    df.to_csv(cfg["paths"]["results"] + "/verification/cooldown_modes.csv", index=False)
    hi = df[df["P_before_W"] > 300]                    # 높은 power에서 내려온 run (빠른 성분이 잘 보임)
    q = lambda s: dict(median=float(s.median()), min=float(s.min()), max=float(s.max()))
    eta = hi["fast_amp_K"] / (hi["fast_amp_K"] + hi["slow_amp_K"])
    save("cooldown_modes", dict(
        n_runs=len(df), n_high_power=len(hi),
        rms_1exp_all_K=q(df["rms_1exp_all_K"]), rms_2exp_K=q(df["rms_2exp_K"]),
        tau_1exp_all_s=q(df["tau_1exp_all_s"]),
        high_power=dict(tau_slow_s=q(hi["tau_slow_s"]), tau_fast_s=q(hi["tau_fast_s"]), slow_amp_K=q(hi["slow_amp_K"]),
                        fast_amp_K=q(hi["fast_amp_K"]), fast_share=q(eta), Ta_2exp_C=q(hi["Ta_2exp_C"]),
                        excess_first_0p5s_K=q(hi["excess_first_0p5s_K"]), P_before_W=q(hi["P_before_W"])),
        note="train·validation만. 빠른 시간상수는 0.5 s 구간 평균으로 추정해 power 해상도 근처에서 신뢰도가 낮다.",
    ))


if __name__ == "__main__":
    main()
