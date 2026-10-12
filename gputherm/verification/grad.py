"""torch 구현의 값과 미분 검사 (5.6절 추정과 8.2절 학습의 전제). 실행: python -m gputherm.verification.grad

(1) 값: torch 정확해(구간 평균)와 관측기가 numpy 구현과 같은지, 합성곱 구현이 반복 구현과 같은지. 기준 1e-9 K.
(2) 미분: 관측기 → 60 s 예측(구간 평균)으로 이어지는 손실의 C_g, C_h, R_gh, R_ha, T_a, α에 대한 자동 미분을
    5점 중심 유한차분(h = 1e-3)과 비교한다. 각 parameter는 log 공간(α는 그대로)에서 섭동한다. 기준: 상대 오차 1e-5.
"""
import numpy as np
import torch

from gputherm.physics.discretize import bin_average
from gputherm.physics.observer import observer_trajectory, observer_trajectory_torch
from gputherm.physics.simulate import simulate_exact, simulate_exact_torch, simulate_lti_torch
from gputherm.verification.common import TA, representative_params, save_and_check, train_trace


F64 = torch.float64
NAMES = ["Cg", "Ch", "Rgh", "Rha", "Ta", "alpha"]


def pipeline_loss(theta, P_obs, Tg_obs, P_hor, target, K0, dt=0.1):
    """theta = [log Cg, log Ch, log Rgh, log Rha, Ta, alpha]. 관측기로 t₀의 T_h를 구하고 60 s를 예측한다."""
    p = dict(Cg=torch.exp(theta[0]), Ch=torch.exp(theta[1]), Rgh=torch.exp(theta[2]), Rha=torch.exp(theta[3]),
             alpha=theta[5])
    Th = observer_trajectory_torch(p, P_obs, theta[4], Tg_obs, dt)[K0]
    x0 = torch.stack([Tg_obs[K0 - 1], Th])
    avg = simulate_lti_torch(p, P_hor, theta[4], x0, dt, alpha_seq=theta[5])      # fit이 쓰는 구현
    return ((avg - target) ** 2).mean()


def main():
    p = representative_params()
    P, u, T = train_trace()
    # (1) 값 대조
    K0, n_h = 3000, 600                                            # t₀ = 300 s, horizon 60 s
    x0 = np.array([T[K0 - 1], 35.0])
    _, I = simulate_exact(p, P[K0:K0 + n_h], TA, x0, 0.1, return_integral=True)
    _, avg_t = simulate_exact_torch(p, torch.tensor(P[K0:K0 + n_h]), TA, x0, 0.1)
    v_sim = float(np.abs(bin_average(I[:, 0], 0.1) - avg_t.numpy()).max())
    Th_np = observer_trajectory(p, P[:K0], TA, T[:K0], 0.1)
    Th_t = observer_trajectory_torch(p, torch.tensor(P[:K0]), TA, torch.tensor(T[:K0]), 0.1).numpy()
    v_obs = float(np.abs(Th_np - Th_t).max())
    # 관측기의 넘침 방지: 긴 run 전체(구간 연결)와, 감쇠가 한 step보다 빠른 경우(−λΔ = 800, 25, 15)
    for q in (dict(p), dict(Ch=1.0, Rgh=2.5e-4, Rha=2.5e-4), dict(Ch=1.0, Rgh=0.008, Rha=0.008), dict(Ch=1.0, Rgh=0.0133, Rha=0.0133)):
        a_np = observer_trajectory(q, P, TA, T, 0.1)
        a_t = observer_trajectory_torch(q, torch.tensor(P), TA, torch.tensor(T), 0.1).numpy()
        v_obs = max(v_obs, float(np.abs(a_np - a_t).max()) if np.isfinite(a_t).all() else float("inf"))
    # 짧은 입력 경계 사례: 감쇠 지수 20.1, 입력 길이 1–3
    q = dict(Ch=1.0, Rgh=0.0099502, Rha=0.0099502)                    # −λΔ ≈ 20.1
    for nn in (1, 2, 3):
        a_np = observer_trajectory(q, P[:nn], TA, T[:nn], 0.1)
        a_t = observer_trajectory_torch(q, torch.tensor(P[:nn]), TA, torch.tensor(T[:nn]), 0.1).numpy()
        v_obs = max(v_obs, float(np.abs(a_np - a_t).max()))
    # 합성곱 구현 대 반복 구현 (시간 가변 α, window 20개 batch)
    idx = np.arange(20) * 300 + 3000
    Pb = torch.tensor(np.stack([P[i:i + n_h] for i in idx]))
    ab = torch.tensor(np.stack([1 - 0.3 * u[i:i + n_h] / 100 for i in idx]))
    xb = torch.tensor(np.stack([[T[i - 1], 35.0] for i in idx]))
    _, a_loop = simulate_exact_torch(p, Pb, TA, xb, 0.1, alpha_seq=ab)
    v_lti = float((a_loop - simulate_lti_torch(p, Pb, TA, xb, 0.1, alpha_seq=ab)).abs().max())

    # (2) 미분 대조 (α = 0.9로 두어 α 경로도 살아 있게 한다)
    P_obs, Tg_obs = torch.tensor(P[:K0]), torch.tensor(T[:K0])
    P_hor = torch.tensor(P[K0:K0 + n_h])
    target = torch.tensor(bin_average(I[:, 0], 0.1)) + 0.3          # 0이 아닌 잔차
    theta0 = torch.tensor([np.log(p["Cg"]), np.log(p["Ch"]), np.log(p["Rgh"]), np.log(p["Rha"]), TA, 0.9], dtype=F64)
    th = theta0.clone().requires_grad_(True)
    loss = pipeline_loss(th, P_obs, Tg_obs, P_hor, target, K0)
    loss.backward()
    g_ad = th.grad.numpy()
    g_fd = np.empty(6)
    L = lambda th_: pipeline_loss(th_, P_obs, Tg_obs, P_hor, target, K0).item()
    for i in range(6):
        h = 1e-3                                                   # 5점 중심 차분, 절단 오차 O(h⁴)
        e = torch.zeros(6, dtype=F64); e[i] = h
        g_fd[i] = (-L(theta0 + 2 * e) + 8 * L(theta0 + e) - 8 * L(theta0 - e) + L(theta0 - 2 * e)) / (12 * h)
    rel = np.abs(g_ad - g_fd) / np.maximum(np.abs(g_fd), 1e-12)
    res = dict(
        value=dict(simulate_bin_average_max_abs_K=v_sim, observer_max_abs_K=v_obs, lti_vs_loop_max_abs_K=v_lti,
                   criterion_K=1e-9, passed=bool(max(v_sim, v_obs, v_lti) <= 1e-9)),
        gradient=dict(names=NAMES, autograd=g_ad.tolist(), finite_difference=g_fd.tolist(), rel_error=rel.tolist(),
                      criterion=1e-5, passed=bool(np.all(rel <= 1e-5) and np.all(np.abs(g_ad) > 0))),
    )
    res["passed"] = res["value"]["passed"] and res["gradient"]["passed"]
    save_and_check("grad", res)


if __name__ == "__main__":
    main()
