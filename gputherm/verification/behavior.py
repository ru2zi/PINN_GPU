"""6.5 정상상태 거동 검증 (M18). 실행: python -m gputherm.verification.behavior

일정 power 25–600 W(25 W 간격)의 정상상태 T_g: 단조 증가, 유한, power 0이면 T_a로 복귀.
사다리형 해석 관계 T_g = T_a + P·(R_ha + α R_gh)와도 대조한다. α = 1과 α = 0.75 두 경우를 본다.
"""
import numpy as np

from gputherm.physics.ladder import steady_state
from gputherm.verification.common import TA, representative_params, save_and_check


def check(p, alpha):
    P = np.arange(25, 601, 25, dtype=float)
    Tss = np.array([steady_state(p, Pk, TA, alpha)[0] for Pk in P])
    analytic = TA + P * (p["Rha"] + alpha * p["Rgh"])
    T0 = steady_state(p, 0.0, TA, alpha)
    return dict(alpha=alpha, T_at_25W=float(Tss[0]), T_at_600W=float(Tss[-1]),
                monotone=bool(np.all(np.diff(Tss) > 0)), finite=bool(np.all(np.isfinite(Tss))),
                zero_power_returns_to_Ta=bool(np.allclose(T0, TA, atol=1e-9)),
                max_dev_from_analytic_K=float(np.abs(Tss - analytic).max()))


def main():
    p = representative_params()
    res = dict(alpha1=check(p, 1.0), alpha0p75=check(p, 0.75))
    res["passed"] = all(v["monotone"] and v["finite"] and v["zero_power_returns_to_Ta"] and v["max_dev_from_analytic_K"] < 1e-9
                        for v in res.values())
    save_and_check("behavior", res)


if __name__ == "__main__":
    main()
