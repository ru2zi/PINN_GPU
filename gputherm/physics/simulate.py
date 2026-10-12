"""사다리형 RC의 예측: 정확해(numpy, torch), 합성곱 예측, 해석해, RK4 (spec 5.6절).

모든 함수는 float64로 계산한다. 출력의 구간 평균은 상태 적분으로 계산한다.
"""
import numpy as np
import torch

from gputherm.physics.discretize import F64, as_like, discrete_parts, discrete_parts_torch, n_per
from gputherm.physics.ladder import ladder_AB


def simulate_exact(p, P, Ta, x0, dt, alpha_seq=None, return_integral=False):
    """입력이 dt마다 일정할 때의 정확해. P: (n,), alpha_seq: (n,) 또는 None.

    반환: 상태 (n+1, 2) [, step별 상태 적분 (n, 2)].
    """
    Ad, Bd, G, H = discrete_parts(p, dt)
    n = len(P)
    al = np.full(n, p.get("alpha", 1.0)) if alpha_seq is None else np.asarray(alpha_seq, float)
    x = np.empty((n + 1, 2)); x[0] = x0
    I = np.empty((n, 2))
    for k in range(n):
        v = np.array([P[k], Ta, al[k] * P[k]])
        I[k] = G @ x[k] + H @ v
        x[k + 1] = Ad @ x[k] + Bd @ v
    return (x, I) if return_integral else x


def simulate_exact_torch(p, P, Ta, x0, dt, alpha_seq=None, out_dt=0.5):
    """미분 가능한 정확해. P: (n,) 또는 (B, n), x0: (2,) 또는 (B, 2), alpha_seq: None, 스칼라, (n,) 또는 (B, n).

    T_a, parameter, 입력, 초기 상태, α의 dtype·device를 P에 맞춘다(기본 float64).
    반환: (끝 상태 (B, 2), T_g의 out_dt 구간 평균 (B, n·dt/out_dt)). 입력이 1차원이면 batch 차원을 뺀다.
    """
    P = torch.as_tensor(P, dtype=F64) if not torch.is_tensor(P) else P
    if P.dtype != F64:
        raise TypeError(f"입력은 float64여야 한다(받은 dtype {P.dtype})")
    squeeze = P.dim() == 1
    P = P[None] if squeeze else P
    Bn, n = P.shape
    m = n_per(out_dt, dt)
    Ad, Bd, G, H = discrete_parts_torch(p, dt, P)
    Ta = as_like(Ta, P).expand(Bn, n)
    a = p.get("alpha", 1.0) if alpha_seq is None else alpha_seq
    a = as_like(a, P).expand(Bn, n) if as_like(a, P).dim() <= 1 else as_like(a, P)
    V = torch.stack([P, Ta, a * P], dim=-1)                           # (B, n, 3) = [P, T_a, αP]
    x = as_like(x0, P).expand(Bn, 2) if as_like(x0, P).dim() == 1 else as_like(x0, P)
    ints = []
    for k in range(n):
        v = V[:, k]                                                   # (B, 3)
        ints.append(x @ G.T + v @ H.T)
        x = x @ Ad.T + v @ Bd.T
    I = torch.stack(ints, dim=1)[..., 0]                             # (B, n) T_g 적분
    avg = I[:, :n // m * m].reshape(Bn, -1, m).sum(-1) / out_dt
    return (x[0], avg[0]) if squeeze else (x, avg)


def simulate_lti_torch(p, P, Ta, x0, dt, alpha_seq=None, out_dt=0.5):
    """simulate_exact_torch와 같은 값을 합성곱(행렬곱)으로 계산한다. RC와 후보 α용(A가 상수인 선형 시스템).

    step k의 T_g 적분 I_k = (G A_d^k x₀)_0 + Σ_{j≤k} h_{k−j}·v_j,
      h_0 = H[0,:],  h_m = (G A_d^{m−1} B_d)[0,:] (m ≥ 1),  v = [P, T_a, αP].
    α가 시간에 따라 바뀌어도 αP를 입력 채널로 두므로 그대로 성립한다. horizon 길이만큼의 kernel을 한 번 만들고
    (N, n) × (n, n) 행렬곱으로 모든 window를 한 번에 계산한다. 반환: T_g의 out_dt 구간 평균 (N, n·dt/out_dt).
    """
    P = torch.as_tensor(P, dtype=F64) if not torch.is_tensor(P) else P
    if P.dtype != F64:
        raise TypeError(f"입력은 float64여야 한다(받은 dtype {P.dtype})")
    squeeze = P.dim() == 1
    P = P[None] if squeeze else P
    Bn, n = P.shape
    m = n_per(out_dt, dt)
    Ad, Bd, G, H = discrete_parts_torch(p, dt, P)
    # 거듭제곱 A_d^k (k = 0..n−1)
    pw = [torch.eye(2, dtype=F64, device=P.device)]
    for _ in range(n - 1):
        pw.append(pw[-1] @ Ad)
    Pw = torch.stack(pw)                                              # (n, 2, 2)
    h_free = (G[0] @ Pw)                                              # (n, 2): (G A_d^k)[0,:]
    h_forced = torch.cat([H[0:1], (G[0] @ Pw[:-1]) @ Bd], dim=0)      # (n, 3)
    k = torch.arange(n, device=P.device)
    lag = k[:, None] - k[None, :]                                     # (n, n), k − j
    Tz = torch.where((lag >= 0)[..., None], h_forced[lag.clamp(min=0)], torch.zeros((), dtype=F64))  # (n, n, 3)
    a = p.get("alpha", 1.0) if alpha_seq is None else alpha_seq
    a = as_like(a, P).expand(Bn, n) if as_like(a, P).dim() <= 1 else as_like(a, P)
    x = as_like(x0, P).expand(Bn, 2) if as_like(x0, P).dim() == 1 else as_like(x0, P)
    Ta = as_like(Ta, P)
    I = (x @ h_free.T + P @ Tz[..., 0].T + Ta * Tz[..., 1].sum(1)[None, :] + (a * P) @ Tz[..., 2].T)
    avg = I[:, :n // m * m].reshape(Bn, -1, m).sum(-1) / out_dt
    return avg[0] if squeeze else avg


def analytic_constant_input(p, P, Ta, x0, t):
    """일정 입력에서의 해석해 (eigen 분해). t: (m,) 시각 배열. 반환 (m, 2)."""
    A, B = ladder_AB(p)
    u = np.array([P, Ta])
    xss = -np.linalg.solve(A, B @ u)
    lam, V = np.linalg.eig(A)
    c = np.linalg.solve(V, np.asarray(x0) - xss)
    return (xss[None, :] + (V @ (c[:, None] * np.exp(np.outer(lam, t)))).T).real


def ladder_rhs(x, u, p, Rha_fn=None, alpha=None):
    """x: (..., 2), u: (..., 2) = [P, T_a]. Rha_fn(T_g)가 주어지면 R_ha가 온도의 함수."""
    a = p.get("alpha", 1.0) if alpha is None else alpha
    Tg, Th = x[..., 0], x[..., 1]
    P, Ta = u[..., 0], u[..., 1]
    Rha = p["Rha"] if Rha_fn is None else Rha_fn(Tg)
    q_gh = (Tg - Th) / p["Rgh"]
    q_ha = (Th - Ta) / Rha
    dTg = (a * P - q_gh) / p["Cg"]
    dTh = ((1 - a) * P + q_gh - q_ha) / p["Ch"]
    return np.stack([dTg, dTh], axis=-1), q_ha


def rk4(p, P, Ta, x0, dt_in, substeps, Rha_fn=None, alpha_seq=None, track_energy=False, return_integral=False):
    """입력이 dt_in마다 일정할 때 dt = dt_in/substeps로 RK4. 입력 경계는 항상 step 경계에 온다.

    누적 공급 에너지, 누적 손실, T_g 적분을 상태에 추가해 같은 RK4로 적분한다(선형 불변량은 정확히 보존된다).
    반환: 상태 (n+1, 2) [, E_in, E_loss (n+1,)] [, 입력 step별 T_g 적분 (n,)].
    """
    h = dt_in / substeps
    n = len(P)
    al = np.full(n, p.get("alpha", 1.0)) if alpha_seq is None else np.asarray(alpha_seq, float)

    def f(z, u, a):
        dx, q_ha = ladder_rhs(z[:2], u, p, Rha_fn, a)
        return np.r_[dx, u[0], q_ha, z[0]]

    z = np.r_[np.asarray(x0, float), 0.0, 0.0, 0.0]
    out = np.empty((n + 1, 5)); out[0] = z
    for k in range(n):
        u = np.array([P[k], Ta])
        for _ in range(substeps):
            k1 = f(z, u, al[k]); k2 = f(z + h / 2 * k1, u, al[k])
            k3 = f(z + h / 2 * k2, u, al[k]); k4 = f(z + h * k3, u, al[k])
            z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[k + 1] = z
    res = [out[:, :2]]
    if track_energy:
        res += [out[:, 2], out[:, 3]]
    if return_integral:
        res.append(np.diff(out[:, 4]))
    return res[0] if len(res) == 1 else tuple(res)
