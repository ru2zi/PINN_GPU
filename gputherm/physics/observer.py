"""두 번째 node 온도 T_h의 관측기 (spec 5.3절).

  C_h dT_h/dt = (1 − α) P − (T_h − T_a)/R_ha + (T_g,측정 − T_h)/R_gh

run 시작(T_h = T_g(0))부터 측정 T_g와 P로 적분한다. 감쇠율 (1/R_ha + 1/R_gh)/C_h > 0.
"""
import numpy as np
import torch

from gputherm.physics.discretize import F64, as_like


def observer_coeffs(p, dt, alpha=None):
    """T_h 관측기 C_h dT_h/dt = (1−α)P − (T_h−T_a)/R_ha + (T_g,측정 − T_h)/R_gh 의 ZOH 계수.

    반환 (ad, bd): T_h[k+1] = ad·T_h[k] + bd · [P_k, T_a, T_g,k].
    """
    a = p.get("alpha", 1.0) if alpha is None else alpha
    Ch, Rgh, Rha = p["Ch"], p["Rgh"], p["Rha"]
    lam = -(1 / Rha + 1 / Rgh) / Ch
    b = np.array([(1 - a) / Ch, 1 / (Ch * Rha), 1 / (Ch * Rgh)])
    ad = np.exp(lam * dt)
    return ad, b * np.expm1(lam * dt) / lam


def observer_trajectory(p, P, Ta, Tg_meas, dt, alpha_seq=None):
    """run 시작(T_h = T_g(0))부터 모든 bin 경계에서의 T_h. 입력 bin k는 [kΔ, (k+1)Δ)의 평균값.

    반환 (n+1,): 원소 k는 시각 kΔ의 T_h (bin k−1까지의 정보만 사용).
    """
    n = len(P)
    Th = np.empty(n + 1); Th[0] = Tg_meas[0]
    for k in range(n):
        ad, bd = observer_coeffs(p, dt, None if alpha_seq is None else alpha_seq[k])
        Th[k + 1] = ad * Th[k] + bd @ np.array([P[k], Ta, Tg_meas[k]])
    return Th


def observer_trajectory_torch(p, P, Ta, Tg_meas, dt, alpha=None):
    """T_h 관측기의 torch 버전(상수 α). 닫힌 형태의 누적합으로 계산해 run 길이에 대해 순차 반복이 없다.

    T_h[k] = a^k (T_h[0] + Σ_{j<k} a^{−(j+1)} c_j),  c_j = b·[P_j, T_a, T_g,j].
    반환 (n+1,).
    """
    P = torch.as_tensor(P, dtype=F64) if not torch.is_tensor(P) else P
    if P.dtype != F64:
        raise TypeError(f"입력은 float64여야 한다(받은 dtype {P.dtype})")
    Tg = as_like(Tg_meas, P)
    a_ = p.get("alpha", 1.0) if alpha is None else alpha
    Ch, Rgh, Rha, al, Ta = (as_like(v, P) for v in (p["Ch"], p["Rgh"], p["Rha"], a_, Ta))
    lam = -(1 / Rha + 1 / Rgh) / Ch
    scale = torch.expm1(lam * dt) / lam                               # (a − 1)/λ, 작은 |λΔ|에서도 정확
    c = scale * ((1 - al) / Ch * P + Ta / (Ch * Rha) + Tg / (Ch * Rgh))
    n = len(P)
    r = float(-lam.detach()) * dt                                     # 한 step의 감쇠 지수 −λΔ > 0
    if r > 20.0:
        # 감쇠가 한 step보다 훨씬 빠르면(a < e^−20) 닫힌 형태의 양의 지수가 넘친다. 이때는
        # T_h[k] = a^k T_h[0] + Σ_{i<k} a^i c_{k−1−i} 에서 a^i < 1e-18인 항을 버린 짧은 합으로 계산한다(음의 지수만 씀).
        m = min(n, int(np.ceil(41.5 / r)))                             # 입력이 짧으면 항 수를 n으로 제한
        k = torch.arange(n + 1, dtype=P.dtype, device=P.device)
        Th = torch.exp(lam * dt * k) * Tg[0]
        for i in range(m):
            Th = Th + torch.cat([torch.zeros(1 + i, dtype=P.dtype, device=P.device), torch.exp(lam * dt * i) * c[:n - i]])
        return Th
    # 구간마다 닫힌 형태로 계산하고 구간 사이를 이어 붙인다. 구간 길이는 지수 범위가 e^±40을 넘지 않게 정한다(r ≤ 20이면 L ≥ 2).
    L = max(2, int(40.0 / max(r, 1e-12)))
    out = [Tg[0].reshape(1)]
    th0 = Tg[0]
    for s in range(0, n, L):
        cc = c[s:s + L]
        mm = len(cc)
        j = torch.arange(mm, dtype=P.dtype, device=P.device)
        acc = torch.cumsum(torch.exp(-lam * dt * (j + 1)) * cc, 0)
        seg = torch.exp(lam * dt * (j + 1)) * (th0 + acc)
        out.append(seg)
        th0 = seg[-1]
    return torch.cat(out)


def observer_time_constant(p):
    """관측기 시간상수 C_h / (1/R_ha + 1/R_gh) [s]."""
    return p["Ch"] / (1 / p["Rha"] + 1 / p["Rgh"])
