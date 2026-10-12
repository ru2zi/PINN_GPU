"""2단계: 측정 불확도 (spec 3절). train·validation run만 쓴다.

u_c = √(u_res² + u_repeat²), u_repeat는 조건별 반복 3회 표본 SD의 pooled 값이다.
u_control은 drift 진단(u_repeat² 분해)에만 쓰고 u_c에 더하지 않는다.

실행: python -m gputherm.pipeline.uncertainty
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from gputherm.config import load_cfg


def main():
    cfg = load_cfg()
    res = Path(cfg["paths"]["results"])
    inv = pd.read_csv(res / "inventory.csv")
    dev = inv[inv["split"].isin(["train", "val"])].copy()
    if len(dev) != 27 or dev.groupby("cond").size().ne(3).any():
        raise SystemExit("train·validation이 9개 조건 × 3 run이 아니다")

    u_res = 1 / np.sqrt(12)
    s_i = dev.groupby("cond")["main_mean_t"].std(ddof=1)
    dof = int((dev.groupby("cond").size() - 1).sum())                # 18
    u_repeat = float(np.sqrt((s_i ** 2).mean()))                   # 반복 수가 같으므로 Σ s_i²/9 = SS/18
    u_c = float(np.sqrt(u_res ** 2 + u_repeat ** 2))

    # drift 진단: 조건 안 반복 변동끼리 비교
    y = dev["main_mean_t"] - dev.groupby("cond")["main_mean_t"].transform("mean")
    x = dev["warmup_plateau_c"] - dev.groupby("cond")["warmup_plateau_c"].transform("mean")
    sxx, sxy, syy = float((x * x).sum()), float((x * y).sum()), float((y * y).sum())
    if sxx > 0:
        slope = sxy / sxx
        explained = slope ** 2 * sxx
        residual = float(((y - slope * x) ** 2).sum())
    else:                                                          # 조건 안 warmup 변동이 없으면 설명할 몫도 없다
        slope, explained, residual = None, 0.0, syy
    u_control = float(np.sqrt(sxx / dof))                          # plateau의 조건 안 pooled SD

    out = dict(
        note="spec 3절. train·validation 27 run(9개 조건 × 3). 단위 K. u_c는 알려진 성분만의 추정이며 NVML 기록 기준이다.",
        u_res=round(u_res, 4), u_repeat=round(u_repeat, 4), u_c=round(u_c, 4), two_u_c=round(2 * u_c, 4),
        u_c_full_precision=u_c, u_repeat_full_precision=u_repeat,     # 판정 기준 계산에는 이 값을 쓴다
        dof_repeat=dof,
        s_i={c: round(float(v), 4) for c, v in s_i.items()},
        drift_diagnostic=dict(
            u_control=round(u_control, 4), slope=None if slope is None else round(slope, 4),
            r=round(sxy / np.sqrt(sxx * syy), 4) if sxx > 0 and syy > 0 else None,
            u_repeat_sq=round(syy / dof, 5), explained_by_control_sq=round(explained / dof, 5),
            residual_sq=round(residual / dof, 5),
            note="같은 표본(27 run)과 분모(18). 해석용이며 u_c에 더하지 않는다.",
        ),
        u_timing="12절 5단계(RC 추정 뒤)에 계산",
        u_cal="unknown (u_c에 넣지 않음)",
    )
    with open(res / "uncertainty.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
