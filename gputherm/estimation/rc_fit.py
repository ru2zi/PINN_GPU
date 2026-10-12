"""60 s window loss에 의한 사다리형 RC 추정 (spec 5.6절).

학습 window마다 관측기 상태와 출발 온도에서 60 s를 예측하고, 유효 정답 bin만 골라 평균 제곱 오차를 구한다.
parameter는 θ = [log C_g, log C_h, log R_gh, log R_ha, T_a]이며 α = 1에서 추정한다.
6.6절 시간 정렬, 6.4절 gate, 12절 5단계가 같은 구현을 쓴다.
"""
import numpy as np
import torch

from gputherm.physics.discretize import F64
from gputherm.physics.ladder import ladder_from_scales
from gputherm.physics.observer import observer_trajectory_torch
from gputherm.physics.simulate import simulate_lti_torch


NAMES = ["Cg", "Ch", "Rgh", "Rha"]


class WindowSet:
    """여러 run의 window를 모은 학습·평가 묶음.

    runs: [(name, RunWindows, starts)] 목록. 관측기 입력(run 전체 0.1 s 격자), window별 출발 온도,
    horizon 입력(0.1 s 600개), 정답(0.5 s 120개)과 유효 표시를 float64 tensor로 만든다.
    """

    def __init__(self, runs, start_rule="last_0p1", H=60.0):
        self.names, self.P_obs, self.T_obs = [], [], []
        run_idx, K0, Tg0, P_hor, Y, M = [], [], [], [], [], []
        for r, (name, rw, starts) in enumerate(runs):
            self.names.append(name)
            self.P_obs.append(torch.as_tensor(rw.g["P"].values, dtype=F64))
            self.T_obs.append(torch.as_tensor(rw.g["T"].values, dtype=F64))
            for t0 in starts:
                y, valid, _ = rw.targets(t0, H)
                run_idx.append(r); K0.append(rw.k_of(t0))
                Tg0.append(rw.start_temperature(t0, start_rule))
                P_hor.append(rw.horizon_inputs(t0, H)["P"])
                Y.append(np.where(valid, y, 0.0)); M.append(valid)       # 무효 위치 값은 쓰지 않는다(선택으로 제외)
        self.run_idx = torch.as_tensor(run_idx)
        self.K0 = torch.as_tensor(K0)
        self.Tg0 = torch.as_tensor(np.array(Tg0), dtype=F64)
        self.P_hor = torch.as_tensor(np.array(P_hor), dtype=F64)
        self.Y = torch.as_tensor(np.array(Y), dtype=F64)
        self.M = torch.as_tensor(np.array(M), dtype=torch.bool)

    def __len__(self):
        return len(self.Tg0)

    def subset(self, idx):
        """window 일부만 담은 사본(관측기 입력은 공유)."""
        w = object.__new__(WindowSet)
        w.names, w.P_obs, w.T_obs = self.names, self.P_obs, self.T_obs
        for k in ("run_idx", "K0", "Tg0", "P_hor", "Y", "M"):
            setattr(w, k, getattr(self, k)[idx])
        return w


def params_from_theta(theta):
    """θ = [log C_g, log C_h, log R_gh, log R_ha, T_a] → parameter dict (α = 1)."""
    return dict(Cg=torch.exp(theta[0]), Ch=torch.exp(theta[1]), Rgh=torch.exp(theta[2]), Rha=torch.exp(theta[3]), alpha=1.0)


def predict(theta, ws, alpha=1.0):
    """window별 60 s 예측(0.5 s 구간 평균, (N, 120)). 관측기는 run마다 한 번 계산해 window별로 자른다."""
    p = params_from_theta(theta)
    p["alpha"] = alpha
    Ta = theta[4]
    Th0 = torch.empty(len(ws), dtype=F64)
    for r in range(len(ws.P_obs)):
        sel = ws.run_idx == r
        if sel.any():
            traj = observer_trajectory_torch(p, ws.P_obs[r], Ta, ws.T_obs[r], 0.1, alpha=alpha)
            Th0 = Th0.index_put((sel.nonzero()[:, 0],), traj[ws.K0[sel]])
    x0 = torch.stack([ws.Tg0, Th0], dim=1)
    return simulate_lti_torch(p, ws.P_hor, Ta, x0, 0.1, alpha_seq=alpha)


def loss(theta, ws, alpha=1.0):
    """유효 정답 bin만 골라 계산한 평균 제곱 오차 [K²].

    가중 방식(spec 5.6절): 모든 유효 bin에 같은 가중치를 준다. 따라서 유효 bin이 120개인 window가 108개인 window보다
    비중이 조금 크다. RC와 RC-NN의 학습 손실은 같은 정의를 쓴다. 평가 집계(window별 RMSE의 run 평균, 9.1절)는
    별도로 정의하며 이 손실과 다르다(예: window RMSE가 0 K와 2 K이면 window 평균 1 K, 전체 bin RMSE √2 K).
    """
    pred = predict(theta, ws, alpha)
    d = pred[ws.M] - ws.Y[ws.M]
    return (d * d).mean()


def theta_from_params(p, Ta):
    """parameter dict와 T_a → θ (float64 tensor)."""
    return torch.tensor([np.log(p["Cg"]), np.log(p["Ch"]), np.log(p["Rgh"]), np.log(p["Rha"]), Ta], dtype=F64)


def initial_thetas(n, seed=0, Ta_range=(25.0, 29.0)):
    """척도 범위에서 Latin hypercube로 초기값을 만든다. 만들 수 없는 조합(복소근 등)은 다시 뽑는다.

    범위: τ_fast 0.2–10 s, τ_slow 15–100 s(로그), R_ss 0.03–0.05 K/W, R_gh 비율 0.2–0.9, T_a 25–29 °C.
    """
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        m = n - len(out)
        u = (rng.permuted(np.tile(np.arange(m), (5, 1)), axis=1).T + rng.random((m, 5))) / m
        for tf, ts, rs, fr, ta in zip(10 ** (np.log10(0.2) + u[:, 0] * np.log10(50)), 10 ** (np.log10(15) + u[:, 1] * np.log10(100 / 15)),
                                      0.03 + 0.02 * u[:, 2], 0.2 + 0.7 * u[:, 3], Ta_range[0] + (Ta_range[1] - Ta_range[0]) * u[:, 4]):
            try:
                out.append(theta_from_params(ladder_from_scales(tf, ts, rs, frac_gh=fr), ta))
            except ValueError:
                continue
    return out[:n]


def fit(ws, theta0, max_iter=200, alpha=1.0):
    """L-BFGS(strong Wolfe)로 window loss를 최소화.

    반환 dict: theta, loss, grad_norm(최종 gradient의 최대 절댓값), n_iter(L-BFGS 반복 수), func_evals(손실 평가 수),
    reason(종료 사유: 반복 제한, 평가 제한, gradient 기준, 변화량 기준), finite(결과가 유한한지),
    converged(유한하고 반복·평가 제한이 아니라 수렴 기준으로 끝났는지).
    """
    th = theta0.clone().requires_grad_(True)
    tol_grad, max_eval = 1e-10, int(max_iter * 1.25)                  # max_eval은 torch 기본값과 같다
    opt = torch.optim.LBFGS([th], lr=1.0, max_iter=max_iter, max_eval=max_eval, line_search_fn="strong_wolfe",
                            tolerance_grad=tol_grad, tolerance_change=1e-14, history_size=20)
    n_eval = [0]

    def closure():
        opt.zero_grad()
        L = loss(th, ws, alpha)
        if not torch.isfinite(L):
            raise FloatingPointError("손실이 유한하지 않다")
        L.backward()
        n_eval[0] += 1
        return L

    opt.step(closure)
    th_f = th.detach().clone().requires_grad_(True)
    L = loss(th_f, ws, alpha)
    L.backward()
    g = float(th_f.grad.abs().max())
    st = opt.state[opt._params[0]]
    n_iter, func_evals = int(st.get("n_iter", 0)), int(st.get("func_evals", n_eval[0]))
    if n_iter >= max_iter:
        reason = "반복 제한"
    elif func_evals >= max_eval:
        reason = "평가 제한"
    elif g <= tol_grad:
        reason = "gradient 기준"
    else:
        reason = "변화량 기준"
    finite = bool(torch.isfinite(th_f).all() and np.isfinite(float(L)) and np.isfinite(g))
    converged = finite and reason in ("gradient 기준", "변화량 기준")
    return dict(theta=th_f.detach(), loss=float(L), grad_norm=g, n_iter=n_iter, func_evals=func_evals,
                reason=reason, finite=finite, converged=converged)


def multistart(ws, n_starts=20, seed=0, max_iter=200, alpha=1.0):
    """초기값 n_starts개로 fit한다. 초기값마다 성공·실패를 기록하고, parameter·loss·gradient가 유한한 결과만 비교한다.

    반환: (유한한 결과를 loss 순으로 정렬한 목록, 초기값별 기록 목록). 한 초기값의 실패가 나머지를 멈추지 않는다.
    """
    good, log = [], []
    for i, th0 in enumerate(initial_thetas(n_starts, seed)):
        try:
            r = fit(ws, th0, max_iter, alpha)
        except (FloatingPointError, RuntimeError, ValueError) as e:
            log.append(dict(start=i, ok=False, error=f"{type(e).__name__}: {e}", theta0=th0.tolist()))
            continue
        log.append(dict(start=i, ok=r["finite"], converged=r["converged"], loss=r["loss"], grad_norm=r["grad_norm"],
                        n_iter=r["n_iter"], func_evals=r["func_evals"], reason=r["reason"],
                        theta0=th0.tolist(), theta=r["theta"].tolist()))
        if r["finite"]:
            good.append(dict(start=i, **r))
    return sorted(good, key=lambda r: r["loss"]), log
