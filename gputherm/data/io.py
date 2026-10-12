"""원시 기록의 탐색과 읽기, split 소속 (spec 2.1절, 4.2절).
"""
import hashlib
import pandas as pd
from pathlib import Path


def split_of(cond, cfg):
    """조건 ID의 split 이름(train, val, test_interp, test_shift)."""
    for name in ("train", "val", "test_interp", "test_shift"):
        if cond in cfg["splits"][name]:
            return name
    raise KeyError(f"split이 정해지지 않은 조건: {cond}")


def list_runs(cfg):
    """원시 기록 폴더를 탐색해 run 목록(cond, run, run_id, split, folder, path)을 DataFrame으로 돌려준다."""
    rows = []
    for f in sorted(Path(cfg["paths"]["raw_root"]).glob("*/r*/gpu_samples.csv")):
        cond = f.parts[-3].split("_")[0]
        rows.append(dict(cond=cond, run=f.parts[-2], run_id=f"{cond}/{f.parts[-2]}", split=split_of(cond, cfg),
                         folder=f.parts[-3], path=str(f)))
    return pd.DataFrame(rows)


def md5(path, chunk=1 << 20):
    """파일의 md5 hex digest."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_raw(path):
    """원시 기록을 읽고 run 첫 행 기준 상대 시간 t [s]를 붙인다."""
    d = pd.read_csv(path)
    d["t"] = (d["timestamp_ms"] - d["timestamp_ms"].iloc[0]) / 1e3
    return d
