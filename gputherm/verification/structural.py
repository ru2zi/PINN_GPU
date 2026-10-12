"""구조적 식별 가능성 (spec 5.1절, 6.4절 A).

두 번째 node를 측정하지 않으면 die 온도 응답은 전달함수 (b₁s + b₀)/(s² + a₁s + a₀)의 계수 4개로 정해진다.
parameter → 계수 map의 Jacobian rank로 식별 가능한 parameter 수를 구한다.
  일반형(C_g, C_2, R_ga, R_2a, R_g2, α): rank 4/6
  사다리형(R_ga → ∞, α = 1): rank 4/4, 역산 해 1개
  사다리형 + 상수 α: rank 4/5
α < 1에서는 역산 해가 두 개일 수 있음을 수치 예로 확인한다.

실행: python -m gputherm.verification.structural
"""
import numpy as np
import sympy as sp

from gputherm.physics.ladder import ladder_to_alpha, tf_coefficients
from gputherm.verification.common import save_and_check

s = sp.symbols("s")
Cg, C2, Rga, R2a, Rg2, al = sp.symbols("C_g C_2 R_ga R_2a R_g2 alpha", positive=True)
POINT = {Cg: 30, C2: 700, Rga: sp.Rational(1, 5), R2a: sp.Rational(1, 20), Rg2: sp.Rational(1, 100), al: sp.Rational(7, 10)}


def tf_coeffs(Cg_, C2_, Rga_, R2a_, Rg2_, a_):
    """일반형 2-node의 die 온도 전달함수 계수 [b₁, b₀, a₁, a₀] (분모 최고차 계수로 정규화)."""
    M = sp.Matrix([[Cg_ * s + 1 / Rga_ + 1 / Rg2_, -1 / Rg2_], [-1 / Rg2_, C2_ * s + 1 / R2a_ + 1 / Rg2_]])
    T = M.LUsolve(sp.Matrix([a_, 1 - a_]))[0]
    n, d = sp.fraction(sp.together(T))
    lc = sp.Poly(sp.expand(d), s).LC()
    num = [sp.simplify(c / lc) for c in sp.Poly(sp.expand(n), s).all_coeffs()]
    den = [sp.simplify(c / lc) for c in sp.Poly(sp.expand(d), s).all_coeffs()[1:]]
    return num + den


def rank_at(coeffs, params):
    return int(sp.Matrix(coeffs).jacobian(params).subs(POINT).rank())


def main():
    general = tf_coeffs(Cg, C2, Rga, R2a, Rg2, al)
    ladder = [sp.limit(c, Rga, sp.oo) if c.has(Rga) else c for c in tf_coeffs(Cg, C2, sp.oo, R2a, Rg2, 1)]
    ladder_a = tf_coeffs(Cg, C2, sp.oo, R2a, Rg2, al)
    target = [sp.nsimplify(c.subs(POINT)) for c in ladder]
    sols = sp.solve([c - t for c, t in zip(ladder, target)], [Cg, C2, R2a, Rg2], dict=True)

    # α < 1의 두 역산 해 (spec 5.1절의 예)
    p1 = dict(Cg=1000.0, Ch=100.0, Rgh=0.01, Rha=0.03, alpha=0.75)
    p2 = dict(Cg=1000.0, Ch=1000.0 / 11.4, Rgh=0.012, Rha=0.0285, alpha=0.75)
    two_roots_same_tf = bool(np.allclose(tf_coefficients(p1), tf_coefficients(p2)))
    # α = 1 해를 큰 근의 가지로 α = 0.75에 옮겼을 때 전달함수가 보존되는지
    q = ladder_to_alpha(dict(Cg=30.0, Ch=700.0, Rgh=0.01, Rha=0.05, alpha=1.0), 0.75)

    res = dict(
        rank_general=rank_at(general, [Cg, C2, Rga, R2a, Rg2, al]), n_general=6,
        rank_ladder=rank_at(ladder, [Cg, C2, R2a, Rg2]), n_ladder=4,
        rank_ladder_alpha=rank_at(ladder_a, [Cg, C2, R2a, Rg2, al]), n_ladder_alpha=5,
        ladder_inverse_solutions=len(sols),
        alpha_lt1_two_roots_same_tf=two_roots_same_tf,
        alpha_ref_branch_params=q,
    )
    res["passed"] = bool(res["rank_general"] == 4 and res["rank_ladder"] == 4 and res["rank_ladder_alpha"] == 4
                         and res["ladder_inverse_solutions"] == 1 and two_roots_same_tf)
    save_and_check("structural", res)


if __name__ == "__main__":
    main()
