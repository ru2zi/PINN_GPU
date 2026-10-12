"""6.2 수렴과 입력 격자 검증 (M16). 실행: python -m gputherm.verification.refine

(1) RK4 시간 간격 수렴 (후보 R_ha(T_g)용): train run 측정 power, 대표 감소형 R_ha(T),
    dt = 0.1/k (k = 1, 2, 4, 8), 기준해 k = 16. 입력 격자(0.1 s)의 약수만 써서 입력 경계가 step 경계에 온다.
    통과 기준: 관측 order 3.5–4.5, dt = 0.1 s 오차 0.01 K 이하. 모두 float64.
(2) 입력 격자 (RC와 후보 α용): validation run 하나의 평가 window마다 관측기 상태에서 60 s를 예측하되,
    입력은 원샘플을 t₀ 기준 Δ = 0.5, 0.25, 0.1 s 구간으로 직접 평균하고 실제 Δ로 정확히 적분한다. 출력은 0.5 s 구간 평균(적분).
    통과 기준: Δ = 0.1 s 대비 Δ = 0.25 s의 horizon RMS 차이(window 평균) 0.05 K 이하.
"""
import numpy as np

from gputherm.config import load_cfg
from gputherm.data.grid import RunWindows
from gputherm.data.io import list_runs, load_raw
from gputherm.data.windows import window_starts
from gputherm.physics.discretize import bin_average
from gputherm.physics.ladder import steady_state
from gputherm.physics.observer import observer_trajectory
from gputherm.physics.simulate import rk4, simulate_exact
from gputherm.verification.common import TA, representative_params, save_and_check, train_trace


def rha_decreasing(p):
    R0 = p["Rha"]
    return lambda T: R0 / (1 + 0.3 / (1 + np.exp(-(T - 40.0) / 3.0)))


def rk4_convergence(p):
    P, _, _ = train_trace()
    x0 = steady_state(p, P[0], TA)
    f = rha_decreasing(p)
    ref = rk4(p, P, TA, x0, 0.1, substeps=16, Rha_fn=f)
    ks = [1, 2, 4, 8]
    err = [float(np.abs(rk4(p, P, TA, x0, 0.1, substeps=k, Rha_fn=f) - ref).max()) for k in ks]
    order = [float(np.log2(err[i] / err[i + 1])) for i in range(len(ks) - 1)]
    return dict(dt_s=[0.1 / k for k in ks], max_abs_error_K=err, observed_order=order,
                criterion="order 3.5–4.5 and error at dt 0.1 s ≤ 0.01 K",
                passed=bool(all(3.5 <= o <= 4.5 for o in order) and err[0] <= 0.01))


def predict_window(p, rw, t0, Th0, delta, H=60.0, out_dt=0.5):
    """관측기 상태에서 출발해 입력 격자 Δ로 60 s 예측. 출력은 0.5 s 구간 평균(적분으로 계산).

    입력은 원샘플을 t₀ 기준 Δ 구간으로 직접 평균해 만들고, 실제 Δ로 정확히 적분한다.
    0.5 s 출력 구간에는 Δ = 0.25 s 입력 두 구간, Δ = 0.1 s 입력 다섯 구간이 들어간다.
    """
    P = rw.horizon_inputs_raw(t0, delta, H)
    x0 = np.array([rw.start_temperature(t0), Th0])
    _, I = simulate_exact(p, P, TA, x0, delta, return_integral=True)
    return bin_average(I[:, 0], delta, out_dt)


def input_grid(p, run_id="I02-A/r01"):
    cfg = load_cfg()
    runs = list_runs(cfg)
    r = runs[runs.run_id == run_id].iloc[0]
    assert r.split == "val"
    d = load_raw(r.path)
    rw = RunWindows(d, cfg)
    starts, _ = window_starts(d, r.cond, cfg, cfg["task"]["stride_eval_s"])
    Th = observer_trajectory(p, rw.g["P"].values, TA, rw.g["T"].values, 0.1)
    pred = {dl: [] for dl in (0.5, 0.25, 0.1)}
    for t0 in starts:
        Th0 = Th[rw.k_of(t0)]
        for dl in pred:
            pred[dl].append(predict_window(p, rw, t0, Th0, dl))
    pred = {k: np.array(v) for k, v in pred.items()}
    rms = lambda a, b: float(np.sqrt(((a - b) ** 2).mean(axis=1)).mean())
    d025, d05 = rms(pred[0.25], pred[0.1]), rms(pred[0.5], pred[0.1])
    return dict(run_id=run_id, n_windows=len(starts), rms_diff_0p25_vs_0p1_K=d025, rms_diff_0p5_vs_0p1_K=d05,
                criterion_K=0.05, passed=d025 <= 0.05)


def main():
    p = representative_params()
    res = dict(params=p, rk4=rk4_convergence(p), input_grid=input_grid(p))
    res["passed"] = res["rk4"]["passed"] and res["input_grid"]["passed"]
    save_and_check("refine", res)


if __name__ == "__main__":
    main()
