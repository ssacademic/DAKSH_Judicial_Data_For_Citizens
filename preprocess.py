# preprocess.py
# Run once locally to verify, then Vercel runs it on every deploy.
# Input:  data.csv
# Output: data/summary.json, histogram.json, courts.json,
#         outcomes.json, cohorts.json

import pandas as pd
import numpy as np
import json
import os

os.makedirs("data", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data.csv", parse_dates=["DATE_FILED", "DECISION_DATE"])
df = df.dropna(subset=["DATE_FILED", "DECISION_DATE", "DISPOSALTIME_ADJ"])
df = df[df["DISPOSALTIME_ADJ"] > 0].copy()
df["filing_year"]  = df["DATE_FILED"].dt.year
df["filing_month"] = df["DATE_FILED"].dt.month
d  = df["DISPOSALTIME_ADJ"]
N  = len(df)

# ── 1. summary.json ───────────────────────────────────────────────────────────
bands = [
    ("Under 3 months",   0,   90),
    ("3–6 months",      90,  180),
    ("6–12 months",    180,  365),
    ("1–2 years",      365,  730),
    ("2–3 years",      730, 1095),
    ("3–5 years",     1095, 1825),
    ("5+ years",      1825, 9999),
]
band_data = []
for label, lo, hi in bands:
    n = int(((d > lo) & (d <= hi)).sum())
    band_data.append({"label": label, "lo": lo, "hi": hi,
                      "n": n, "pct": round(100 * n / N, 1)})

summary = {
    "total_cases":   N,
    "median_days":   int(d.median()),
    "mean_days":     int(d.mean()),
    "p10":           int(d.quantile(0.10)),
    "p25":           int(d.quantile(0.25)),
    "p75":           int(d.quantile(0.75)),
    "p90":           int(d.quantile(0.90)),
    "pct_under1yr":  round(100 * (d <= 365).mean(), 1),
    "pct_1to2yr":    round(100 * ((d > 365) & (d <= 730)).mean(), 1),
    "pct_2to3yr":    round(100 * ((d > 730) & (d <= 1095)).mean(), 1),
    "pct_over3yr":   round(100 * (d > 1095).mean(), 1),
    "bands":         band_data,
    "outcome_recorded_pct": round(
        100 * df["NATURE_OF_DISPOSAL_OUTCOME"].notna().mean(), 1),
    "outcome_missing_pct": round(
        100 * df["NATURE_OF_DISPOSAL_OUTCOME"].isna().mean(), 1),
}
with open("data/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"✓ summary.json — {N:,} cases, median {summary['median_days']}d")

# ── 2. histogram.json — 60 bins, pre-computed for D3 ─────────────────────────
bins    = np.linspace(0, min(d.max(), 2200), 61)
counts, edges = np.histogram(d.clip(upper=2200), bins=bins)
hist_data = [
    {"x0": round(float(edges[i]), 1),
     "x1": round(float(edges[i+1]), 1),
     "n":  int(counts[i])}
    for i in range(len(counts))
]
with open("data/histogram.json", "w") as f:
    json.dump(hist_data, f)
print(f"✓ histogram.json — {len(hist_data)} bins")

# ── 3. courts.json — bucketed into 5 groups, courts anonymous ────────────────
cs = (df.groupby("COURT_NUMBER")["DISPOSALTIME_ADJ"]
        .agg(n="count", median="median", mean="mean")
        .reset_index()
        .query("n >= 10")
        .sort_values("median")
        .reset_index(drop=True))

cs["quintile_rank"] = pd.qcut(cs["median"], 5,
    labels=["fastest", "fast", "middle", "slow", "slowest"])

buckets = []
for rank in ["fastest", "fast", "middle", "slow", "slowest"]:
    g = cs[cs["quintile_rank"] == rank]
    buckets.append({
        "bucket":       rank,
        "court_count":  int(len(g)),
        "case_count":   int(g["n"].sum()),
        "case_pct":     round(100 * g["n"].sum() / N, 1),
        "median_lo":    int(g["median"].min()),
        "median_hi":    int(g["median"].max()),
        "median_mid":   int(g["median"].median()),
    })

# All individual court medians anonymised — just the values, no IDs
all_medians = sorted([int(x) for x in cs["median"].tolist()])
fastest_med = int(cs["median"].min())
slowest_med = int(cs["median"].max())

courts_out = {
    "total_courts_analysed": int(len(cs)),
    "min_cases_threshold":   10,
    "fastest_median":        fastest_med,
    "slowest_median":        slowest_med,
    "spread_ratio":          round(slowest_med / fastest_med, 1),
    "buckets":               buckets,
    "all_medians":           all_medians,
}
with open("data/courts.json", "w") as f:
    json.dump(courts_out, f, indent=2)
print(f"✓ courts.json — {len(cs)} courts, {fastest_med}d–{slowest_med}d")

# ── 4. outcomes.json ──────────────────────────────────────────────────────────
outcome_map = {
    "DISPOSED":                      "Closed — manner not recorded",
    "Partly Allowed":                "Partly allowed",
    "DISMISSED":                     "Dismissed (on merits)",
    "Dismissed for Non-Prosecution": "Dropped (not pursued)",
    "ALLOWED":                       "Fully allowed",
    "ALLOWED AND REMANDED":          "Sent back to lower court",
    "REJECTED":                      "Rejected",
    "Abated":                        "Case lapsed",
}
df["outcome_label"] = df["NATURE_OF_DISPOSAL_OUTCOME"].map(outcome_map)

oc = (df["outcome_label"].value_counts(dropna=True)
        .reset_index()
        .rename(columns={"outcome_label": "label", "count": "n"}))

outcomes_out = {
    "total_cases":           N,
    "with_outcome":          int(df["outcome_label"].notna().sum()),
    "without_outcome":       int(df["outcome_label"].isna().sum()),
    "without_outcome_pct":   round(100 * df["outcome_label"].isna().mean(), 1),
    "breakdown": [
        {"label": row["label"], "n": int(row["n"]),
         "pct_of_total":   round(100 * row["n"] / N, 1),
         "pct_of_known":   round(100 * row["n"] /
                           max(df["outcome_label"].notna().sum(), 1), 1)}
        for _, row in oc.iterrows()
    ]
}
with open("data/outcomes.json", "w") as f:
    json.dump(outcomes_out, f, indent=2)
print(f"✓ outcomes.json — {outcomes_out['with_outcome']:,} with outcomes, "
      f"{outcomes_out['without_outcome_pct']}% missing")

# ── 5. cohorts.json — filing year × speed band (for investigate mode) ────────
cohort_rows = []
for yr in sorted(df["filing_year"].unique()):
    sub = df[df["filing_year"] == yr]["DISPOSALTIME_ADJ"]
    n   = int(len(sub))
    cohort_rows.append({
        "year":          int(yr),
        "n":             n,
        "median":        int(sub.median()),
        "pct_under1yr":  round(100 * (sub <= 365).mean(), 1),
        "pct_1to2yr":    round(100 * ((sub > 365) & (sub <= 730)).mean(), 1),
        "pct_2to3yr":    round(100 * ((sub > 730) & (sub <= 1095)).mean(), 1),
        "pct_over3yr":   round(100 * (sub > 1095).mean(), 1),
        "note":          ("Fewer cases visible — only fast-resolving cases "
                          "had time to close by Jan 2021")
                          if yr >= 2018 else ""
    })
with open("data/cohorts.json", "w") as f:
    json.dump(cohort_rows, f, indent=2)
print(f"✓ cohorts.json — {len(cohort_rows)} filing years")

print("\n✅ All done. data/ folder ready.")
