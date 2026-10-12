"""정확 이산화와 구간 평균 (spec 5.6절).

입력 v = [P, T_a, αP]에 대해 한 step의 끝 상태와 상태 적분을 주는 행렬 A_d, B_d, G, H를
Van Loan 블록 행렬 지수로 계산한다. 출력의 0.5 s 값은 구간 시간 평균 (1/Δ)∫T_g dt다.
"""
import numpy as np
import scipy.linalg as sla
import torch

from gputherm.physics.ladder import ladder_AB, ladder_B_parts


F64 = torch.float64


def n_per(out_dt, dt):
    """out_dt가 dt의 정수배이면 그 배수를, 아니면 ValueError를 돌려준다."""
    m = out_dt / dt
    if abs(m - round(m)) > 1e-9 or round(m) < 1:
        raise ValueError(f"출력 간격 {out_dt}가 입력 간격 {dt}의 정수배가 아니다")
    return int(round(m))


def discretize(A, B, dt):
    """ZOH 정확 이산화: A_d = expm(AΔ), B_d = A⁻¹(A_d − I)B."""
    Ad = sla.expm(A * dt)
    Bd = np.linalg.solve(A, (Ad - np.eye(len(A))) @ B)
    return Ad, Bd


def _vanloan_matrix(A, Baug, lib):
    """Van Loan 블록 행렬. 상태 [x, ∫x, u]에 대해 d/dt = [[A, 0, B], [I, 0, 0], [0, 0, 0]].

    exp(MΔ)의 블록에서 A_d, B_d, G = ∫e^{As}ds, H = ∫∫e^{Ar}dr B ds를 한 번에 얻는다.
    작은 값을 큰 두 값의 차로 구하지 않으므로 값과 미분이 정확하다(Van Loan 1978).
    """
    m = Baug.shape[1]
    if lib is np:
        M = np.zeros((4 + m, 4 + m))
        M[0:2, 0:2] = A; M[0:2, 4:] = Baug; M[2:4, 0:2] = np.eye(2)
        return M
    M = torch.zeros((4 + m, 4 + m), dtype=A.dtype, device=A.device)
    M = M.index_put((torch.arange(2)[:, None], torch.arange(2)[None]), A)
    M[0:2, 4:] = Baug
    M[2:4, 0:2] = torch.eye(2, dtype=A.dtype, device=A.device)
    return M


def discrete_parts(p, dt):
    """한 step의 끝 상태와 상태 적분을 주는 행렬. 입력은 v = [P, T_a, αP]로 둔다(B(α)u = B_aug v).

    x_{k+1} = A_d x_k + B_d v_k,   ∫_step x dt = G x_k + H v_k
    """
    A, _ = ladder_AB(p)
    B0, B1 = ladder_B_parts(p)
    Baug = np.column_stack([B0[:, 0], B0[:, 1], B1[:, 0]])
    E = sla.expm(_vanloan_matrix(A, Baug, np) * dt)
    return E[0:2, 0:2], E[0:2, 4:], E[2:4, 0:2], E[2:4, 4:]


def bin_average(I, dt, out_dt=0.5):
    """step별 적분 (n, ...) → out_dt 구간 평균. out_dt는 dt의 정수배여야 한다."""
    m = n_per(out_dt, dt)
    n = I.shape[0] // m * m
    return I[:n].reshape(-1, m, *I.shape[1:]).sum(axis=1) / out_dt


def step_integral(A, B, x0, x1, u, dt):
    """한 step 동안 상태의 적분 ∫x dt = A⁻¹(x₁ − x₀ − B u Δ)."""
    return np.linalg.solve(A, x1 - x0 - B @ u * dt)


def as_like(v, ref):
    """값을 ref와 같은 dtype·device의 tensor로. 이미 tensor면 autograd 이력을 유지한다."""
    return torch.as_tensor(v, dtype=ref.dtype, device=ref.device)


def discrete_parts_torch(p, dt, ref):
    """discrete_parts의 torch 버전(Van Loan 블록 지수). p 값은 float 또는 0-dim tensor, ref는 dtype·device 기준."""
    Cg, Ch, Rgh, Rha = (as_like(p[k], ref) for k in ("Cg", "Ch", "Rgh", "Rha"))
    z = torch.zeros((), dtype=ref.dtype, device=ref.device)
    A = torch.stack([torch.stack([-1 / (Cg * Rgh), 1 / (Cg * Rgh)]),
                     torch.stack([1 / (Ch * Rgh), -(1 / Rgh + 1 / Rha) / Ch])])
    Baug = torch.stack([torch.stack([z, z, 1 / Cg]), torch.stack([1 / Ch, 1 / (Ch * Rha), -1 / Ch])])
    E = torch.linalg.matrix_exp(_vanloan_matrix(A, Baug, torch) * dt)
    return E[0:2, 0:2], E[0:2, 4:], E[2:4, 0:2], E[2:4, 4:]
