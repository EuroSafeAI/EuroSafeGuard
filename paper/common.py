"""Shared helpers for the reproduction scripts: loading the score bundle and checking results against the paper."""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
N_RUNS = 5   # five training runs of the input classifier (gate runs 0-4)


def parse_args(doc):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=os.path.join(HERE, "scores"), help="score bundle directory (default: ./scores)")
    return ap.parse_args()


def load(scores, name):
    return pd.read_csv(os.path.join(scores, f"{name}.csv.gz"), keep_default_na=True)


def zfit(x):
    """Mean and population standard deviation (the z-scoring used everywhere in the paper)."""
    x = np.asarray(x, dtype=float)
    return x.mean(), x.std()


def input_score(df, k, stats):
    """EuroSafeGuard (Input) score with gate run k: mean of the probe z and the input-classifier z."""
    (pm, ps), (gm, gs) = stats
    return ((df["probe"].values - pm) / ps + (df[f"gate{k}"].values - gm) / gs) / 2


def input_stats(wildchat, k):
    c = wildchat.wc_clean.values
    return zfit(wildchat.probe.values[c]), zfit(wildchat[f"gate{k}"].values[c])


class Checker:
    """Collects (label, reproduced, expected) cells, prints a markdown table, exits non-zero on any mismatch."""

    def __init__(self, title):
        self.title, self.rows, self.bad = title, [], 0

    def table(self, header, rows, expected):
        """rows: list of (name, [cell strings]); expected: {name: [cell strings]} from the paper."""
        print(f"\n### {self.title}\n")
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for name, cells in rows:
            exp = expected[name]
            marks = []
            for got, want in zip(cells, exp):
                ok = got == want
                self.bad += not ok
                marks.append(f"{got} ✓" if ok else f"{got} ✗ (paper {want})")
            print(f"| {name} | " + " | ".join(marks) + " |")
            self.rows.append(name)

    def value(self, label, got, want):
        ok = got == want
        self.bad += not ok
        print(f"- {label}: {got} {'✓' if ok else f'✗ (expected {want})'}")

    def finish(self):
        print(f"\n{self.title}: {'all cells match' if not self.bad else f'{self.bad} mismatching cell(s)'}")
        sys.exit(1 if self.bad else 0)
