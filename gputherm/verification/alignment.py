"""6.6 시간 정렬 검증 (전처리 파이프라인의 정확성). 실행: python -m gputherm.verification.alignment

알려진 사다리형 RC(대표 parameter)에 train run의 측정 power를 넣어 참 온도 궤적을 만들고(0.01 s 정확 이산화),
같은 참값 궤적을 유지한 채 관측 기록만 단계적으로 바꿔 실제 데이터와 같은 파이프라인에 넣는다.

참 입력: 측정 power trace를 0.01 s 격자에서 앞 값으로 유지한 계단 함수(NVML의 약 0.5 s 갱신은 이미 들어 있다).
단계 1: 참 온도를 원샘플 시각에서 그대로 기록(반올림 없음). 모델 입력도 참 입력(0.01 s). → 시간 정렬의 영향
단계 2: 단계 1과 같은 기록, 모델 입력은 파이프라인의 0.1 s bin 평균. → 입력 표현의 영향
단계 3: 단계 2 + 온도를 정수 °C로 반올림. → 양자화의 영향
단계 4: 단계 3 + 샘플 공백 3곳(1.5, 3, 4.5 s). → 실제 기록과 같은 조건
각 단계에서 참 parameter로 예측한 값과, 그 기록으로 fit한 parameter로 예측한 값을 참값의 0.5 s 구간 평균과 비교한다.
단계별 차이는 서로 영향을 주므로 독립 성분처럼 더하지 않는다.

통과 기준(전처리 정확성): 단계 1, 참 parameter의 60 s 예측 RMS 0.02 K 이하.
선택 규칙: 단계 4, 참 parameter에서 RMS가 가장 작은 출발 상태 정의. 차이가 0.01 K 미만이면 last_0p1.
"""
import numpy as np

from gputherm.config import load_cfg
from gputherm.data.grid import RunWindows
from gputherm.data.io import list_runs, load_raw
from gputherm.data.windows import usable_windows
from gputherm.estimation.rc_fit import WindowSet, multistart
from gputherm.physics.discretize import bin_average
from gputherm.physics.ladder import ladder_to_alpha, steady_state, time_constants
from gputherm.physics.observer import observer_trajectory
from gputherm.physics.simulate import simulate_exact
from gputherm.verification.common import TA, representative_params, save_and_check


RUN = "T02-A/r01"
RULES = ("last_0p1", "last_0p5", "last_sample")


def truth(d, p, dt=0.01):
    """참 입력(0.01 s 앞 값 유지)과 참 궤적. 반환: 격자 시각, 참 입력, 참 T_g(격자 시각), step별 T_g 적분."""
    t = d["t"].values
    n = int(np.floor(t[-1] / dt))
    grid_t = np.arange(n + 1) * dt
    idx = np.searchsorted(t, grid_t[:-1], side="right") - 1          # 각 step 시작 시각 이전의 마지막 원샘플
    P = d["power_w"].values[np.clip(idx, 0, None)]
    x0 = steady_state(p, P[0], TA)
    x, I = simulate_exact(p, P, TA, x0, dt, return_integral=True)
    return grid_t, P, x[:, 0], I[:, 0]


def make_record(d, grid_t, Tg, stage, rng):
    """참 궤적을 원샘플 시각에서 기록한 사본. 단계 3부터 정수 반올림, 단계 4는 샘플 공백 3곳."""
    r = d.copy()
    r["temperature_c"] = np.interp(r["t"].values, grid_t, Tg)
    if stage >= 3:
        r["temperature_c"] = np.round(r["temperature_c"])
    if stage >= 4:
        main = r["phase"].isin(["train", "checkpoint"]).values
        tm = r["t"].values[main]
        drop = np.zeros(len(r), bool)
        for c, w in zip(np.quantile(tm, [0.2, 0.5, 0.8]), (1.5, 3.0, 4.5)):
            drop |= (r["t"].values >= c) & (r["t"].values < c + w)
        r = r[~drop].reset_index(drop=True)
    return r


def truth_bins(grid_t, I, t0, dt=0.01, H=60.0, out_dt=0.5):
    k0 = int(round(t0 / dt))
    return bin_average(I[k0:k0 + int(round(H / dt))], dt, out_dt)


def predict_numpy(p, rw, t0, rule, P_obs, T_obs, dt_in, P_hor):
    """관측기(입력 격자 dt_in) → 출발 상태 → 60 s 예측(구간 평균)."""
    Th = observer_trajectory(p, P_obs, TA, T_obs, dt_in)
    K = int(round(t0 / dt_in))
    x0 = np.array([rw.start_temperature(t0, rule), Th[K]])
    _, I = simulate_exact(p, P_hor, TA, x0, dt_in, return_integral=True)
    return bin_average(I[:, 0], dt_in)


def rms_vs_truth(pred, tru):
    return float(np.mean(np.sqrt(np.mean((np.asarray(pred) - np.asarray(tru)) ** 2, axis=1))))


def main():
    cfg = load_cfg()
    runs = list_runs(cfg)
    r = runs[runs.run_id == RUN].iloc[0]
    assert r.split == "train"
    d = load_raw(r.path)
    p = representative_params()
    grid_t, P01, Tg01, I01 = truth(d, p)
    rng = np.random.default_rng(0)
    out = dict(run=RUN, params=p, Ta=TA, tau=time_constants(p).tolist(), stages={})
    preds_true = {}

    for stage in (1, 2, 3, 4):
        rec = make_record(d, grid_t, Tg01, stage, rng)
        rw = RunWindows(rec, cfg, dt=0.1)
        ev, dropped, _ = usable_windows(rec, r.cond, cfg, cfg["task"]["stride_eval_s"])
        tru = [truth_bins(grid_t, I01, t0) for t0 in ev]
        res = dict(n_windows=len(ev), n_dropped=len(dropped))
        if stage == 1:
            # 참 입력(0.01 s)으로 관측기와 예측: 시간 정렬만 남는다
            rw01 = RunWindows(rec, cfg, dt=0.01)
            n_h = 6000
            for rule in RULES:
                pr = []
                for t0 in ev:
                    K = int(round(t0 / 0.01))
                    pr.append(predict_numpy(p, rw, t0, rule, P01[:K], rw01.g["T"].values[:K], 0.01, P01[K:K + n_h]))
                res[f"true_params_rms_K_{rule}"] = rms_vs_truth(pr, tru)
        else:
            Pg, Tgr = rw.g["P"].values, rw.g["T"].values
            for rule in RULES:
                pr = []
                for t0 in ev:
                    K = rw.k_of(t0)
                    pr.append(predict_numpy(p, rw, t0, rule, Pg[:K], Tgr[:K], 0.1, Pg[K:K + 600]))
                res[f"true_params_rms_K_{rule}"] = rms_vs_truth(pr, tru)
                if rule == "last_0p1":
                    preds_true[stage] = np.array(pr)
            # 같은 기록으로 fit (학습 window, 기본 출발 상태)
            tr, _, _ = usable_windows(rec, r.cond, cfg, cfg["task"]["stride_train_s"])
            ws = WindowSet([(RUN, rw, tr)])
            good, log = multistart(ws, n_starts=8, seed=stage, max_iter=300)
            best = good[0]
            th = best["theta"]
            pf = dict(Cg=float(np.exp(th[0])), Ch=float(np.exp(th[1])), Rgh=float(np.exp(th[2])), Rha=float(np.exp(th[3])), alpha=1.0)
            pr = []
            for t0 in ev:
                K = rw.k_of(t0)
                Th = observer_trajectory(pf, Pg[:K], float(th[4]), Tgr[:K], 0.1)
                x0 = np.array([rw.start_temperature(t0), Th[K]])
                _, I = simulate_exact(pf, Pg[K:K + 600], float(th[4]), x0, 0.1, return_integral=True)
                pr.append(bin_average(I[:, 0], 0.1))
            tau_f = time_constants(pf)
            res.update(fitted_rms_K=rms_vs_truth(pr, tru), fitted_tau=tau_f.tolist(), fitted_Ta=float(th[4]),
                       fit_converged=bool(best["converged"]), fit_reason=best["reason"], fit_loss=best["loss"],
                       fit_starts_ok=int(sum(l["ok"] for l in log)), fit_starts_converged=int(sum(l.get("converged", False) for l in log)),
                       tau_fast_rel_err=float(abs(tau_f[0] / time_constants(p)[0] - 1)),
                       tau_slow_rel_err=float(abs(tau_f[1] / time_constants(p)[1] - 1)))
        out["stages"][stage] = res
        print(stage, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "fitted_tau"}, flush=True)

    # 인접 단계 사이의 예측 차이(참 parameter, last_0p1)
    out["adjacent_prediction_rms_K"] = {f"{a}->{b}": rms_vs_truth(preds_true[a], preds_true[b])
                                        for a, b in ((2, 3), (3, 4)) if preds_true[a].shape == preds_true[b].shape}

    # α_ref 동등성(단계 4 기록, 관측기 포함)
    rec4 = make_record(d, grid_t, Tg01, 4, rng)
    rw4 = RunWindows(rec4, cfg, dt=0.1)
    ev4, _, _ = usable_windows(rec4, r.cond, cfg, cfg["task"]["stride_eval_s"])
    q = ladder_to_alpha(p, 0.75)
    Pg, Tgr = rw4.g["P"].values, rw4.g["T"].values
    d_alpha = []
    for t0 in ev4:
        K = rw4.k_of(t0)
        a1 = predict_numpy(p, rw4, t0, "last_0p1", Pg[:K], Tgr[:K], 0.1, Pg[K:K + 600])
        Th = observer_trajectory(q, Pg[:K], TA, Tgr[:K], 0.1)
        x0 = np.array([rw4.start_temperature(t0), Th[K]])
        _, I = simulate_exact(q, Pg[K:K + 600], TA, x0, 0.1, return_integral=True)
        d_alpha.append(np.sqrt(np.mean((a1 - bin_average(I[:, 0], 0.1)) ** 2)))
    out["alpha_ref_equivalence"] = dict(alpha_ref=0.75, params=q, mean_window_rms_diff_K=float(np.mean(d_alpha)),
                                        max_window_rms_diff_K=float(np.max(d_alpha)))

    s1 = out["stages"][1]["true_params_rms_K_last_0p1"]
    s4 = {rule: out["stages"][4][f"true_params_rms_K_{rule}"] for rule in RULES}
    best_rule = min(s4, key=s4.get)
    chosen = "last_0p1" if s4["last_0p1"] - s4[best_rule] < 0.01 else best_rule
    out.update(alignment_error_stage1_K=s1, criterion_K=0.02, chosen_start_rule=chosen, passed=bool(s1 <= 0.02))
    save_and_check("alignment", out)


if __name__ == "__main__":
    main()
