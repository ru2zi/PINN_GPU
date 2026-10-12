"""6.3 에너지 수지 진단. 실행: python -m gputherm.verification.balance

공급 ∫P dt = 저장 변화 C_g ΔT_g + C_h ΔT_h + 손실 ∫(T_h − T_a)/R_ha dt (사다리형, 보드 power 기준).
- 정확 이산화(RC, 후보 α): step마다 ∫x dt = A⁻¹(x_{k+1} − x_k − B u_k Δ)로 손실을 정확히 적분.
- RK4(후보 R_ha): 누적 공급과 누적 손실을 상태에 추가해 같은 RK4로 적분(선형 불변량 보존).
통과 기준: |잔차| / ∫P dt ≤ 1e-6. 구조상 성립하므로 A, B와 열 흐름 항의 조립을 확인하는 진단이다.
"""
import numpy as np

from gputherm.physics.discretize import step_integral
from gputherm.physics.ladder import ladder_AB, ladder_B_parts, steady_state
from gputherm.physics.simulate import rk4, simulate_exact
from gputherm.verification.common import TA, representative_params, save_and_check, train_trace
from gputherm.verification.refine import rha_decreasing


def exact_balance(p, P, alpha_seq, dt=0.1):
    x0 = steady_state(p, P[0], TA)
    x = simulate_exact(p, P, TA, x0, dt, alpha_seq=alpha_seq)
    A, _ = ladder_AB(p)
    B0, B1 = ladder_B_parts(p)
    loss = 0.0
    for k in range(len(P)):
        B = B0 + alpha_seq[k] * B1
        Ix = step_integral(A, B, x[k], x[k + 1], np.array([P[k], TA]), dt)
        loss += (Ix[1] - TA * dt) / p["Rha"]
    E_in = float(np.sum(P) * dt)
    stored = p["Cg"] * (x[-1, 0] - x[0, 0]) + p["Ch"] * (x[-1, 1] - x[0, 1])
    return dict(E_in_J=E_in, rel_residual=abs(E_in - stored - loss) / E_in)


def rk4_balance(p, P, dt=0.1):
    x0 = steady_state(p, P[0], TA)
    x, E_in, E_loss = rk4(p, P, TA, x0, dt, substeps=1, Rha_fn=rha_decreasing(p), track_energy=True)
    stored = p["Cg"] * (x[-1, 0] - x[0, 0]) + p["Ch"] * (x[-1, 1] - x[0, 1])
    return dict(E_in_J=float(E_in[-1]), rel_residual=abs(E_in[-1] - stored - E_loss[-1]) / E_in[-1])


def main():
    p = representative_params()
    P, u, _ = train_trace()
    res = dict(
        exact_alpha1=exact_balance(p, P, np.ones(len(P))),
        exact_alpha_varying=exact_balance(p, P, 1 - 0.3 * u / 100),
        rk4_rha_decreasing=rk4_balance(p, P),
        criterion=1e-6,
    )
    res["passed"] = all(v["rel_residual"] <= 1e-6 for k, v in res.items() if isinstance(v, dict))
    save_and_check("balance", res)


if __name__ == "__main__":
    main()
