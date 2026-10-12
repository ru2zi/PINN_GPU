"""RC 추정 공통 구현의 자체 검사 (spec 6.7절).

batch와 window별 개별 실행의 일치, 무효 정답의 배제, float32 거부, 실패 초기값의 격리, 종료 사유 기록을 확인한다.

실행: python -m gputherm.verification.fit_selftest
"""
import numpy as np
import torch

from gputherm.config import load_cfg
from gputherm.data.grid import RunWindows
from gputherm.data.io import list_runs, load_raw
from gputherm.data.windows import usable_windows
from gputherm.estimation.rc_fit import WindowSet, loss, predict, theta_from_params
from gputherm.physics.discretize import F64
from gputherm.physics.simulate import simulate_exact_torch
from gputherm.verification.common import TA, representative_params, save_and_check


def main():
    """(1) batch 예측·손실이 window별 개별 실행을 합친 결과와 같은지, (2) 무효 정답(NaN)이 손실에 섞이지 않는지,
    (3) float32 입력을 거부하는지."""
    cfg = load_cfg()
    runs = list_runs(cfg)
    items = []
    for rid in ("T02-A/r01", "I01-Q7B-A/r02"):
        r = runs[runs.run_id == rid].iloc[0]
        d = load_raw(r.path)
        st, _, _ = usable_windows(d, r.cond, cfg, cfg["task"]["stride_eval_s"])
        items.append((rid, RunWindows(d, cfg), st[:6]))
    ws = WindowSet(items)
    theta = theta_from_params(representative_params(), TA)

    # (1) batch 대 개별
    pb = predict(theta, ws)
    pi = torch.cat([predict(theta, ws.subset(torch.tensor([i]))) for i in range(len(ws))])
    batch_diff = float((pb - pi).abs().max())
    L_b = float(loss(theta, ws))
    sq = [((predict(theta, ws.subset(torch.tensor([i])))[ws.M[i:i + 1]] - ws.Y[i][ws.M[i]]) ** 2) for i in range(len(ws))]
    L_i = float(torch.cat(sq).mean())

    # (2) 무효 정답: 일부 bin을 무효로 만들고 그 자리 정답을 NaN으로 바꿔도 손실이 유한하고, 같은 bin을 뺀 계산과 같아야 한다
    ws2 = ws.subset(torch.arange(len(ws)))
    ws2.M = ws.M.clone(); ws2.Y = ws.Y.clone()
    ws2.M[0, 10:30] = False; ws2.Y[0, 10:30] = float("nan")
    L_nan = float(loss(theta, ws2))
    ws3 = ws.subset(torch.arange(len(ws)))
    ws3.M = ws2.M.clone(); ws3.Y = ws.Y.clone()
    L_ref = float(loss(theta, ws3))
    nan_ok = np.isfinite(L_nan) and abs(L_nan - L_ref) < 1e-12

    # (3) float32 거부
    try:
        simulate_exact_torch(representative_params(), ws.P_hor[:1].float(), TA, torch.zeros(1, 2, dtype=F64), 0.1)
        f32_rejected = False
    except TypeError:
        f32_rejected = True

    # (4) multistart: 실패하는 초기값(NaN)이 섞여도 나머지가 실행되고, 결과에 NaN이 남지 않아야 한다
    import gputherm.estimation.rc_fit as me
    orig = me.initial_thetas
    bad = theta.clone(); bad[0] = float("nan")
    me.initial_thetas = lambda n, seed=0: [theta.clone(), bad, theta.clone()]
    try:
        good, log = me.multistart(ws, n_starts=3, max_iter=3)
    finally:
        me.initial_thetas = orig
    ms_ok = (len(log) == 3 and len(good) == 2 and not log[1]["ok"] and all(np.isfinite(g["loss"]) for g in good))
    # 종료 사유: 반복·평가 제한으로 끝나면 수렴으로 표시하지 않고, 사유가 반복 수·평가 수와 일치해야 한다
    def consistent(r, max_iter):
        max_eval = int(max_iter * 1.25)
        want = ("반복 제한" if r["n_iter"] >= max_iter else "평가 제한" if r["func_evals"] >= max_eval
                else "gradient 기준" if r["grad_norm"] <= 1e-10 else "변화량 기준")
        return r["reason"] == want and r["converged"] == (want in ("gradient 기준", "변화량 기준"))
    r3 = me.fit(ws, theta.clone(), max_iter=3)                         # 제한에 걸려 끝나는 경우
    r30 = me.fit(ws, theta.clone(), max_iter=30)
    reason_ok = (consistent(r3, 3) and not r3["converged"] and consistent(r30, 30)
                 and r30["func_evals"] >= r30["n_iter"])

    res = dict(n_windows=len(ws), batch_vs_individual_max_abs_K=batch_diff, multistart_failure_isolated=bool(ms_ok),
               termination_reason_ok=bool(reason_ok),
               fit_max_iter3=dict(n_iter=r3["n_iter"], func_evals=r3["func_evals"], reason=r3["reason"], converged=r3["converged"]),
               fit_max_iter30=dict(n_iter=r30["n_iter"], func_evals=r30["func_evals"], reason=r30["reason"], converged=r30["converged"]), loss_batch=L_b, loss_individual=L_i,
               loss_rel_diff=abs(L_b - L_i) / L_i, nan_target_loss=L_nan, nan_reference_loss=L_ref, nan_handled=bool(nan_ok),
               float32_rejected=f32_rejected)
    res["passed"] = bool(batch_diff < 1e-10 and res["loss_rel_diff"] < 1e-12 and nan_ok and f32_rejected and ms_ok and reason_ok)
    save_and_check("fit_selftest", res)


if __name__ == "__main__":
    main()
