"""对比几组训练方案的评测结果（train/eval_bind.py --per-concept 写出的表）。

Usage:
  python -X utf8 train/compare_arms.py export/dev/arm_base_60000.csv export/dev/arm_tok_60000.csv ...
The first file is the reference the others are compared with.
"""
import csv
import sys
from pathlib import Path

import numpy as np


def load(path):
    return {r["name_en"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    tables = [load(p) for p in paths]
    names = [n for n in tables[0] if all(n in t for t in tables)]
    n_train = np.array([int(tables[0][n]["n_train"]) for n in names])
    ceiling = np.array([int(tables[0][n]["train_median"] or 0) for n in names])
    ranks = [np.array([int(t[n]["model_median"]) for n in names]) for t in tables]
    has_name = all("name_rank_median" in t[names[0]] for t in tables)
    print(f"{len(names)} concepts; ceiling (training sprites): median {int(np.median(ceiling[ceiling > 0]))}")
    groups = [("all", n_train > 0), ("10-15 sprites", (n_train >= 10) & (n_train <= 15)),
              ("16-20 sprites", (n_train >= 16) & (n_train <= 20)), ("21+ sprites", n_train >= 21),
              # 训练图自己很一致、参照组却画不对的物品：最想修的那一批
              ("data ok, reference bad", (ceiling > 0) & (ceiling <= 10) & (ranks[0] > 10))]
    for label, pick in groups:
        print(f"\n{label} ({int(pick.sum())})")
        for path, t, r in zip(paths, tables, ranks):
            line = (f"  {path.stem:22} median {int(np.median(r[pick])):4d}  top1 {np.mean(r[pick] <= 1):4.0%}  "
                    f"top5 {np.mean(r[pick] <= 5):4.0%}  top10 {np.mean(r[pick] <= 10):4.0%}  beyond 50 {np.mean(r[pick] > 50):4.0%}")
            if has_name:
                nr = np.array([int(t[n]["name_rank_median"]) for n in names])
                line += f"  | name rank median {int(np.median(nr[pick]))}"
            print(line)
    base = ranks[0]
    print("\nconcept by concept against the reference (better / worse by more than a factor of 3):")
    for path, r in zip(paths[1:], ranks[1:]):
        better, worse = int(np.sum(r * 3 < base)), int(np.sum(r > base * 3))
        fixed, broken = int(np.sum((base > 10) & (r <= 10))), int(np.sum((base <= 10) & (r > 10)))
        print(f"  {path.stem:22} better {better:4d}  worse {worse:4d}  | newly within 10: {fixed:4d}  fell out of 10: {broken:4d}")


if __name__ == "__main__":
    main()
