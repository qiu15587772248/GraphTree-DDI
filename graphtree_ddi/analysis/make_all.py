"""一键：多目录合并 → 汇总表、论文图、配对 bootstrap、外部评估 demo。

冒烟:
  python -u models/analysis/make_all.py \\
    --pred_dir models/results/baselines_smoke models/results/final_smoke_pair \\
    --out_dir models/results/analysis_smoke

正式:
  --pred_dir models/results/final_v2_pair_merged models/results/baselines_v2_pair \\
             models/results/final_v2_drug models/results/baselines_v2_drug \\
    --out_dir results/analysis
"""

from __future__ import annotations

from pathlib import Path
import sys as _sys

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "graphtree_ddi" / "paths.py").is_file():
            return p
    raise RuntimeError("cannot locate repository root (graphtree_ddi/paths.py)")

_REPO_ROOT = _repo_root()
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import argparse
import sys
from pathlib import Path


from graphtree_ddi.analysis import metrics as M  # noqa: E402
from graphtree_ddi.analysis import paired_bootstrap as PB  # noqa: E402
from graphtree_ddi.analysis import plot_figures as PF  # noqa: E402
from graphtree_ddi.analysis import summarize_runs as SR  # noqa: E402
from graphtree_ddi.analysis import external_eval as EE  # noqa: E402


def run_all(
    pred_dir,
    out_dir: str | Path,
    *,
    data_dir: str | Path | None = None,
    method_a: str = "Exp-5",
    method_b: list[str] | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    external_csv: str | Path | None = None,
    skip_bootstrap: bool = False,
    skip_external: bool = False,
    labels: dict[str, str] | None = None,
) -> dict:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    labels = labels if labels is not None else M.DEFAULT_METHOD_LABELS

    print("== 1/4 summarize_runs ==")
    SR.write_tables(pred_dir, tab_dir, labels=labels)

    print("== 2/4 plot_figures ==")
    PF.plot_all(pred_dir, fig_dir, data_dir=data_dir, labels=labels)

    print("== 3/4 paired_bootstrap ==")
    if skip_bootstrap:
        print("  skipped")
    else:
        df = M.load_predictions(pred_dir)
        if df.empty:
            print("  no predictions, skip bootstrap")
        else:
            have = M.methods_in(df)
            a = method_a if method_a in have else None
            if a is None:
                # 优先 GraphTree-DDI，否则表中最后一个
                for cand in ("Exp-5",) + tuple(reversed(have)):
                    if cand in have:
                        a = cand
                        break
                print(f"[make_all] method_a 改用 {a}")
            if method_b:
                mbs = [m for m in method_b if m in have and m != a]
            else:
                mbs = [m for m in have if m != a]
            if not mbs:
                print("  无对照方法，跳过 bootstrap")
            else:
                try:
                    detail, summary = PB.run_vs_many(
                        pred_dir, a, mbs, n_boot=n_boot, seed=seed,
                        test_set="test", labels=labels, df=df,
                    )
                except SystemExit as e:
                    print(f"  holdout bootstrap 失败 ({e})，改为全部 test_set")
                    detail, summary = PB.run_vs_many(
                        pred_dir, a, mbs, n_boot=n_boot, seed=seed,
                        test_set=None, labels=labels, df=df,
                    )
                boot_csv = tab_dir / "paired_bootstrap.csv"
                detail.to_csv(boot_csv, index=False, encoding="utf-8-sig")
                summary.to_csv(
                    tab_dir / "paired_bootstrap_summary.csv",
                    index=False, encoding="utf-8-sig",
                )
                PB.write_merged_tables(summary, tab_dir / "paired_bootstrap.md")
                print(f"  → {boot_csv}  comparators={list(summary['method_b'])}")

    print("== 4/4 external_eval ==")
    if skip_external:
        print("  skipped")
    elif external_csv:
        EE.run_external(external_csv, tab_dir / "external", test_data=False)
        EE.run_external(external_csv, fig_dir / "external", test_data=False)
    else:
        demo = out_dir / "external_demo"
        demo.mkdir(parents=True, exist_ok=True)
        csv_path = EE.make_synthetic_csv(demo / "synthetic_external_200.csv", n=200, seed=0)
        (demo / "THIS_IS_TEST_DATA.txt").write_text(
            "Synthetic 200-row CSV for smoke tests only.\n"
            "Do not copy into sci_manuscript/figures.\n",
            encoding="utf-8",
        )
        EE.run_external(csv_path, demo, test_data=True)

    print(f"[make_all] done → {out_dir}")
    return dict(out_dir=out_dir, figures=fig_dir, tables=tab_dir)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="One-click analysis tables + figures")
    M.add_pred_dir_arg(p)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--data_dir", default="",
                   help="processed 目录（默认 ddi_prediction/data/processed/v2）")
    p.add_argument("--method_a", default="Exp-5")
    p.add_argument(
        "--method_b", nargs="*", default=None,
        help="对照方法（可多个）；默认=其余全部",
    )
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--external_csv", default="")
    p.add_argument("--skip_bootstrap", action="store_true")
    p.add_argument("--skip_external", action="store_true")
    M.add_method_names_arg(p)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_all(
        args.pred_dir, args.out_dir,
        data_dir=args.data_dir or None,
        method_a=args.method_a, method_b=args.method_b,
        n_boot=args.n_boot, seed=args.seed,
        external_csv=args.external_csv or None,
        skip_bootstrap=args.skip_bootstrap,
        skip_external=args.skip_external,
        labels=M.parse_method_names(args.method_names),
    )


if __name__ == "__main__":
    main()
