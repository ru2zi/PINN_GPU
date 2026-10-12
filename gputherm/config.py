"""설정 파일과 프로젝트 경로.
"""
import yaml
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "data.yaml"


def load_cfg(path=CFG_PATH):
    """`configs/data.yaml`을 읽어 dict로 돌려준다."""
    with open(path) as f:
        return yaml.safe_load(f)
