"""사다리형 2-node RC의 정의, 전달함수, 척도 변환 (spec 5.1절).

  C_g dT_g/dt = α P − (T_g − T_h)/R_gh
  C_h dT_h/dt = (1 − α) P + (T_g − T_h)/R_gh − (T_h − T_a)/R_ha

상태 x = [T_g, T_h], 입력 u = [P, T_a]. parameter는 dict(Cg, Ch, Rgh, Rha, alpha).
"""
import numpy as np


def ladder_AB(p, alpha=None):
    """연속 시간 행렬 A (2×2)와 B (2×2, 입력 [P, T_a])."""
    a = p.get("alpha", 1.0) if alpha is None else alpha
    Cg, Ch, Rgh, Rha = p["Cg"], p["Ch"], p["Rgh"], p["Rha"]
    A = np.array([[-1 / (Cg * Rgh), 1 / (Cg * Rgh)],
                  [1 / (Ch * Rgh), -(1 / Rgh + 1 / Rha) / Ch]])
    B = np.array([[a / Cg, 0.0],
                  [(1 - a) / Ch, 1 / (Ch * Rha)]])
    return A, B


def ladder_B_parts(p):
    """B(α) = B0 + α·B1 (α에 대해 선형)."""
    Cg, Ch, Rha = p["Cg"], p["Ch"], p["Rha"]
    B0 = np.array([[0.0, 0.0], [1 / Ch, 1 / (Ch * Rha)]])
    B1 = np.array([[1 / Cg, 0.0], [-1 / Ch, 0.0]])
    return B0, B1


def tf_coefficients(p, alpha=None):
    """전달함수 (b₁s + b₀)/(s² + a₁s + a₀)의 계수 [b₁, b₀, a₁, a₀] (spec 5.1절)."""
    a = p.get("alpha", 1.0) if alpha is None else alpha
    Cg, Ch, Rgh, Rha = p["Cg"], p["Ch"], p["Rgh"], p["Rha"]
    return np.array([a / Cg, (a * Rgh + Rha) / (Cg * Ch * Rgh * Rha),
                     1 / (Ch * Rha) + 1 / (Ch * Rgh) + 1 / (Cg * Rgh), 1 / (Cg * Ch * Rgh * Rha)])


def steady_state(p, P, Ta, alpha=None):
    """일정 입력 [P, T_a]에서의 정상상태 [T_g, T_h] = −A⁻¹B u."""
    A, B = ladder_AB(p, alpha)
    return -np.linalg.solve(A, B @ np.array([P, Ta]))


def time_constants(p):
    """τ = −1/λ (극점 λ < 0), 작은 것부터."""
    A, _ = ladder_AB(p)
    lam = np.linalg.eigvals(A)
    if np.any(np.abs(lam.imag) > 1e-12) or np.any(lam.real >= 0):
        raise ValueError(f"극점이 음의 실수가 아니다: {lam}")
    return np.sort(-1 / lam.real)


def real_positive_roots(coeffs, tol=1e-9):
    """다항식의 실근 중 양수만. 허수부가 있는 근은 버리지 않고 '없음'으로 본다."""
    r = np.roots(coeffs)
    real = r[np.abs(r.imag) <= tol * np.maximum(1, np.abs(r.real))].real
    return np.sort(real[real > 0])


def ladder_from_scales(tau_fast, tau_slow, R_ss, frac_gh=0.25, alpha=1.0):
    """시간상수 두 개와 정상상태 열저항으로 사다리형 parameter를 만든다(R_gh = frac_gh·R_ss, α = 1).

    a₀ = 1/(τf τs) = 1/(Cg Ch Rgh Rha), a₁ = 1/τf + 1/τs. Cg가 작은 근(die)을 고른다.
    요청한 조합을 만들 수 없으면(복소근, 비양수 parameter) ValueError를 낸다.
    만든 뒤 time_constants()로 요청값이 복원되는지 확인한다.
    """
    if alpha != 1.0:
        raise ValueError("α = 1에서만 정의한다. α_ref로는 5.1절 역산으로 옮긴다")
    Rgh, Rha = frac_gh * R_ss, (1 - frac_gh) * R_ss
    a1, a0 = 1 / tau_fast + 1 / tau_slow, 1 / (tau_fast * tau_slow)
    K = 1 / (a0 * Rgh * Rha)                                   # Cg·Ch
    roots = real_positive_roots([(1 / Rha + 1 / Rgh) / K, -a1, 1 / Rgh])
    if len(roots) == 0:
        raise ValueError(f"τ_fast={tau_fast}, τ_slow={tau_slow}, frac_gh={frac_gh}: 양의 실근이 없다")
    Cg = roots[0]
    p = dict(Cg=Cg, Ch=K / Cg, Rgh=Rgh, Rha=Rha, alpha=1.0)
    if min(p["Cg"], p["Ch"], p["Rgh"], p["Rha"]) <= 0:
        raise ValueError(f"비양수 parameter: {p}")
    if not np.allclose(time_constants(p), [tau_fast, tau_slow], rtol=1e-8):
        raise ValueError(f"시간상수가 복원되지 않는다: {time_constants(p)}")
    return p


def ladder_to_alpha(p, alpha_ref):
    """α = 1 사다리형 parameter를 전달함수가 같은 α_ref parameter로 옮긴다. 큰 근(α = 1 해에서 이어지는 가지)을 쓴다.

    판별식이 음수이거나 parameter가 양수가 아니면 ValueError를 낸다(그 경우 α_ref = 1을 쓴다).
    """
    b1, b0, a1, a0 = tf_coefficients(p, p.get("alpha", 1.0))
    Cg = alpha_ref / b1
    roots = real_positive_roots([1.0, -(a1 - alpha_ref * b0 / b1), (1 - alpha_ref) * a0])
    if len(roots) == 0:
        raise ValueError("양의 실근이 없다")
    x = roots[-1]                                                      # 큰 근
    Rgh = 1 / (Cg * x)
    Rha = b0 / a0 - alpha_ref * Rgh
    if Rha <= 0:
        raise ValueError("R_ha ≤ 0")
    q = dict(Cg=Cg, Ch=x / (a0 * Rha), Rgh=Rgh, Rha=Rha, alpha=alpha_ref)
    if not np.allclose(tf_coefficients(q), [b1, b0, a1, a0], rtol=1e-9):
        raise ValueError("전달함수가 보존되지 않는다")
    return q
