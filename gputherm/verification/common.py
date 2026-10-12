"""검증 스크립트 공통: 대표 parameter, 측정 power trace, 결과 저장.
"""
import json
import sys
from pathlib import Path

from gputherm.config import load_cfg
from gputherm.data.grid import grid
from gputherm.data.io import list_runs, load_raw
from gputherm.physics.ladder import ladder_from_scales


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "verification"
TA = 27.1                                       # train·validation cooldown 지수 2개 fit 점근값 중앙값 (spec 5.2절)


def representative_params():
    """spec 6절 서두: τ_fast 0.4 s, τ_slow 41 s, R_ss 0.038 K/W, R_gh = 0.77 R_ss (α = 1)."""
    return ladder_from_scales(0.4, 41.0, 22.9 / 600, frac_gh=0.77)


def train_trace(run_id="T02-A/r01", dt=0.1):
    """train run 하나의 0.1 s 격자 power와 사용률 (warmup–본 구간–cooldown 전체)."""
    cfg = load_cfg()
    runs = list_runs(cfg)
    r = runs[runs.run_id == run_id].iloc[0]
    assert r.split == "train"
    g = grid(load_raw(r.path), dt)
    return g["P"].values.astype(float), g["u"].values.astype(float), g["T"].values.astype(float)


def torchdiffeq():
    """대조용 독립 구현: github_code/torchdiffeq (0.2.5, commit 657943a)."""
    sys.path.insert(0, str(ROOT / "github_code" / "torchdiffeq"))
    import torchdiffeq as tde
    return tde


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=float)
    print(json.dumps(obj, indent=1, ensure_ascii=False, default=float))


def save_and_check(name, obj):
    """결과를 저장한 뒤, obj["passed"]가 거짓이면 종료 코드 1로 멈춘다(실패 뒤 다음 단계로 넘어가지 않게)."""
    save(name, obj)
    if not obj.get("passed", False):
        print(f"\n[{name}] 검증 실패")
        raise SystemExit(1)
