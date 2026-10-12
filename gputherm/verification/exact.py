"""6.1 독립 구현 대조 (M15). 실행: python -m gputherm.verification.exact

(1) 일정 power step(25 W → 600 W): 해석해 대 정확 이산화(numpy, torch), RK4(dt 0.001 s), torchdiffeq dopri5(rtol 1e-10).
    상태(0.1 s 끝 시각)와 출력 정의(0.5 s 구간 평균, 적분으로 계산) 둘 다 비교한다. 통과 기준: 최대 오차 1e-6 K.
(2) 측정 trace와 시간 가변 α = 1 − 0.3·u/100: 정확 이산화 대 RK4(dt 0.01 s, 입력 경계 정렬). 기준 1e-4 K.
"""
import numpy as np
import torch

from gputherm.physics.discretize import bin_average
from gputherm.physics.ladder import ladder_AB, steady_state
from gputherm.physics.simulate import analytic_constant_input, rk4, simulate_exact, simulate_exact_torch
from gputherm.verification.common import TA, representative_params, save_and_check, torchdiffeq, train_trace


def analytic_bin_average(p, P, Ta, x0, t_edges):
    """일정 입력에서 T_g의 구간 평균 해석해: ∫e^{λt}dt = (e^{λt₁} − e^{λt₀})/λ."""
    A, B = ladder_AB(p)
    xss = -np.linalg.solve(A, B @ np.array([P, Ta]))
    lam, V = np.linalg.eig(A)
    c = np.linalg.solve(V, np.asarray(x0) - xss)
    t0, t1 = t_edges[:-1], t_edges[1:]
    integ = (np.exp(np.outer(lam, t1)) - np.exp(np.outer(lam, t0))) / lam[:, None]      # (2, m)
    return (xss[0] + (V[0][:, None] * c[:, None] * integ).sum(0).real / (t1 - t0))


def step_test(p):
    dt, T_end, out_dt = 0.1, 300.0, 0.5
    n = int(round(T_end / dt))
    x0 = steady_state(p, 25.0, TA)
    t = np.arange(n + 1) * dt
    P = np.full(n, 600.0)
    ref_state = analytic_constant_input(p, 600.0, TA, x0, t)
    ref_avg = analytic_bin_average(p, 600.0, TA, x0, np.arange(0, T_end + 1e-9, out_dt))

    x_np, I_np = simulate_exact(p, P, TA, x0, dt, return_integral=True)
    _, avg_th = simulate_exact_torch(p, torch.tensor(P), TA, x0, dt, out_dt=out_dt)
    x_rk, I_rk = rk4(p, P, TA, x0, dt, substeps=100, return_integral=True)          # dt 0.001 s
    tde = torchdiffeq()
    A, B = ladder_AB(p)
    At = torch.tensor(A); bu = torch.tensor(B @ np.array([600.0, TA]))
    sol = tde.odeint(lambda as_like, x: At @ x + bu, torch.tensor(x0), torch.tensor(t),
                     method="dopri5", rtol=1e-10, atol=1e-10).numpy()
    state_err = dict(exact_numpy=np.abs(x_np - ref_state).max(), rk4_dt0p001=np.abs(x_rk - ref_state).max(),
                     torchdiffeq_dopri5=np.abs(sol - ref_state).max())
    avg_err = dict(exact_numpy=np.abs(bin_average(I_np[:, 0], dt, out_dt) - ref_avg).max(),
                   exact_torch=np.abs(avg_th.numpy() - ref_avg).max(),
                   rk4_dt0p001=np.abs(bin_average(I_rk, dt, out_dt) - ref_avg).max())
    allerr = list(state_err.values()) + list(avg_err.values())
    # 참고: 끝점 값 평균으로 구간 평균을 근사하면 생기는 차이(출력 정의를 적분으로 정한 이유)
    endpoint = x_np[1:, 0].reshape(-1, 5).mean(axis=1)
    return dict(state_max_abs_error_K={k: float(v) for k, v in state_err.items()},
                bin_average_max_abs_error_K={k: float(v) for k, v in avg_err.items()},
                endpoint_vs_integral_first_bin_K=float(abs(endpoint[0] - ref_avg[0])),
                endpoint_vs_integral_rms_first_60s_K=float(np.sqrt(np.mean((endpoint[:120] - ref_avg[:120]) ** 2))),
                endpoint_vs_integral_rms_300s_K=float(np.sqrt(np.mean((endpoint - ref_avg) ** 2))),
                criterion_K=1e-6, passed=bool(max(allerr) <= 1e-6))


def varying_alpha_test(p):
    P, u, _ = train_trace()
    alpha = 1 - 0.3 * u / 100
    x0 = steady_state(p, P[0], TA)
    ex, Iex = simulate_exact(p, P, TA, x0, 0.1, alpha_seq=alpha, return_integral=True)
    rk, Irk = rk4(p, P, TA, x0, 0.1, substeps=10, alpha_seq=alpha, return_integral=True)   # dt 0.01 s
    e = float(max(np.abs(ex - rk).max(), np.abs(bin_average(Iex[:, 0], 0.1) - bin_average(Irk, 0.1)).max()))
    return dict(n_steps=len(P), max_abs_error_K=e, criterion_K=1e-4, passed=e <= 1e-4)


def main():
    p = representative_params()
    res = dict(params=p, step=step_test(p), varying_alpha=varying_alpha_test(p))
    res["passed"] = res["step"]["passed"] and res["varying_alpha"]["passed"]
    save_and_check("exact", res)


if __name__ == "__main__":
    main()
