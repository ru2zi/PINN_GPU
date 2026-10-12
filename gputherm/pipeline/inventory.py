"""1단계: inventory, 형식·오염 검사, control, run 순서, split 고정 (spec 2.5–2.7절, 4절).

실행: python -m gputherm.pipeline.inventory
"""
import json
import pandas as pd
from pathlib import Path

from gputherm.config import ROOT, load_cfg
from gputherm.data.io import list_runs
from gputherm.data.quality import PARSE_CHECK, PARSE_TOL, exclusion_reasons, inspect_run


def main():
    cfg = load_cfg()
    res = Path(cfg["paths"]["results"]); tab = Path(cfg["paths"]["tables"])
    res.mkdir(exist_ok=True); tab.mkdir(exist_ok=True)
    runs = list_runs(cfg)
    if len(runs) != cfg["expected_runs"]:
        raise SystemExit(f"run 수 {len(runs)} ≠ 기대값 {cfg['expected_runs']}")
    inv = pd.DataFrame([inspect_run(r, cfg) for r in runs.itertuples()])
    if inv["md5"].duplicated().any():
        raise SystemExit("md5 중복 run이 있다: " + ", ".join(inv.loc[inv["md5"].duplicated(keep=False), "run_id"]))

    # 2.4절 parsing 검증 (train·validation만)
    dev = inv[inv["split"].isin(["train", "val"])].groupby("cond")[["main_mean_p", "main_mean_t"]].mean()
    bad = []
    for c, (p, tm) in PARSE_CHECK.items():
        dp, dtm = dev.loc[c, "main_mean_p"] - p, dev.loc[c, "main_mean_t"] - tm
        if abs(dp) >= PARSE_TOL[0] or abs(dtm) >= PARSE_TOL[1]:
            bad.append(f"{c}: ΔP {dp:+.1f} W, ΔT {dtm:+.2f} K")
    inv.to_csv(res / "inventory.csv", index=False, float_format="%.10g")   # 계산용 값은 반올림하지 않는다
    if bad:
        raise SystemExit("2.4절 값과 다름(parsing 오류 의심): " + "; ".join(bad))

    # 제외 규칙
    med = inv["warmup_plateau_c"].median()
    ex = []
    for r in inv.to_dict("records"):
        reasons, cand = exclusion_reasons(r, med)
        if reasons or cand:
            ex.append(dict(run_id=r["run_id"], excluded=bool(reasons), candidate_plateau=cand, reasons="; ".join(reasons)))
    pd.DataFrame(ex, columns=["run_id", "excluded", "candidate_plateau", "reasons"]).to_csv(res / "exclusions.csv", index=False)
    excluded = {e["run_id"] for e in ex if e["excluded"]}

    # control: 날짜와 반복 순서
    ctrl = inv[["run_id", "cond", "run", "split", "start_utc", "warmup_plateau_c", "warmup_plateau_p", "cooldown_end_c"]].copy()
    ctrl["plateau_dev_c"] = ctrl["warmup_plateau_c"] - med
    ctrl.sort_values("start_utc").to_csv(res / "control.csv", index=False)

    # run order: 시작 시각 순서와 직전 run과의 간격
    order = inv[["run_id", "cond", "run", "split", "start_utc", "end_utc"]].sort_values("start_utc").reset_index(drop=True)
    st, en = pd.to_datetime(order["start_utc"]), pd.to_datetime(order["end_utc"])
    order["gap_from_prev_end_min"] = ((st - en.shift()).dt.total_seconds() / 60).round(1)
    Path(ROOT / "records").mkdir(exist_ok=True)
    order.to_csv(ROOT / "records" / "run_order.csv", index=False)

    # split 고정
    splits = {name: sorted(inv.loc[(inv["split"] == name) & ~inv["run_id"].isin(excluded), "run_id"])
              for name in ("train", "val", "test_interp", "test_shift")}
    splits["unseen_subset"] = sorted(r for r in splits["test_interp"] if r.split("/")[0] in cfg["splits"]["unseen_subset"])
    splits["excluded"] = sorted(excluded)
    with open(res / "splits.json", "w") as f:
        json.dump(splits, f, indent=1, ensure_ascii=False)

    # T1: split별 크기, train·validation 범위 (test 범위는 9단계 coverage에서 계산)
    rows = []
    for name in ("train", "val", "test_interp", "test_shift"):
        g = inv[inv["run_id"].isin(splits[name])]
        row = dict(split=name, conditions=g["cond"].nunique(), runs=len(g), eval_windows=int(g["n_eval_windows"].sum()),
                   train_windows=int(g["n_train_windows"].sum()) if name == "train" else None,
                   eval_low_coverage=int(g["n_eval_low_coverage"].sum()), eval_short_history=int(g["n_eval_short_history"].sum()))
        if name in ("train", "val"):
            row.update(main_p=f"{g['main_p_min'].min():.0f}–{g['main_p_max'].max():.0f}",
                       main_t=f"{g['main_t_min'].min():.0f}–{g['main_t_max'].max():.0f}",
                       all_p=f"{g['all_p_min'].min():.0f}–{g['all_p_max'].max():.0f}",
                       all_t=f"{g['all_t_min'].min():.0f}–{g['all_t_max'].max():.0f}")
        rows.append(row)
    pd.DataFrame(rows).to_csv(tab / "T1_ranges.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nrun {len(inv)}개, 제외 {len(excluded)}개, plateau 제외 후보 {sum(e['candidate_plateau'] for e in ex)}개")


if __name__ == "__main__":
    main()
