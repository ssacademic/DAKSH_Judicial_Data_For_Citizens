import pandas as pd
import json
import os

os.makedirs("data", exist_ok=True)

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
    "total_cases":        N,
    "median_days":        int(d.median()),
    "mean_days":          int(d.mean()),
    "p10":                int(d.quantile(0.10)),
    "p25":                int(d.quantile(0.25)),
    "p75":                int(d.quantile(0.75)),
    "p90":                int(d.quantile(0.90)),
    "pct_under1yr":       round(100 * (d <= 365).mean(), 1),
    "pct_1to2yr":         round(100 * ((d > 365) & (d <= 730)).mean(), 1),
    "pct_2to3yr":         round(100 * ((d > 730) & (d <= 1095)).mean(), 1),
    "pct_over3yr":        round(100 * (d > 1095).mean(), 1),
    "bands":              band_data,
    "outcome_recorded_pct": round(
        100 * df["NATURE_OF_DISPOSAL_OUTCOME"].notna().mean(), 1),
    "outcome_missing_pct":  round(
        100 * df["NATURE_OF_DISPOSAL_OUTCOME"].isna().mean(), 1),
}
with open("data/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"✓ summary.json — {N:,} cases, median {summary['median_days']}d")

# ── 2. histogram.json — pure pandas, no numpy ─────────────────────────────────
d_clipped = d.clip(upper=2200)
bin_width  = 2200 / 60
hist_data  = []
for i in range(60):
    lo = i * bin_width
    hi = (i + 1) * bin_width
    count = int(((d_clipped >= lo) & (d_clipped < hi)).sum())
    hist_data.append({"x0": round(lo, 1), "x1": round(hi, 1), "n": count})
# last bin includes upper edge
hist_data[-1]["n"] += int((d_clipped == 2200).sum())

with open("data/histogram.json", "w") as f:
    json.dump(hist_data, f)
print(f"✓ histogram.json — {len(hist_data)} bins")

# ── 3. courts.json ────────────────────────────────────────────────────────────
cs = (df.groupby("COURT_NUMBER")["DISPOSALTIME_ADJ"]
        .agg(n="count", median="median")
        .reset_index()
        .query("n >= 10")
        .sort_values("median")
        .reset_index(drop=True))

cs["quintile_rank"] = pd.qcut(cs["median"], 5,
    labels=["fastest","fast","middle","slow","slowest"])

buckets = []
for rank in ["fastest","fast","middle","slow","slowest"]:
    g = cs[cs["quintile_rank"] == rank]
    buckets.append({
        "bucket":      rank,
        "court_count": int(len(g)),
        "case_count":  int(g["n"].sum()),
        "case_pct":    round(100 * g["n"].sum() / N, 1),
        "median_lo":   int(g["median"].min()),
        "median_hi":   int(g["median"].max()),
        "median_mid":  int(g["median"].median()),
    })

courts_out = {
    "total_courts_analysed": int(len(cs)),
    "min_cases_threshold":   10,
    "fastest_median":        int(cs["median"].min()),
    "slowest_median":        int(cs["median"].max()),
    "spread_ratio":          round(cs["median"].max() / cs["median"].min(), 1),
    "buckets":               buckets,
    "all_medians":           sorted([int(x) for x in cs["median"].tolist()]),
}
with open("data/courts.json", "w") as f:
    json.dump(courts_out, f, indent=2)
print(f"✓ courts.json — {len(cs)} courts, "
      f"{courts_out['fastest_median']}d–{courts_out['slowest_median']}d")

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
        .rename(columns={"outcome_label":"label","count":"n"}))
known = int(df["outcome_label"].notna().sum())

outcomes_out = {
    "total_cases":         N,
    "with_outcome":        known,
    "without_outcome":     int(df["outcome_label"].isna().sum()),
    "without_outcome_pct": round(100 * df["outcome_label"].isna().mean(), 1),
    "breakdown": [
        {"label": row["label"],
         "n":     int(row["n"]),
         "pct_of_total": round(100 * row["n"] / N, 1),
         "pct_of_known": round(100 * row["n"] / max(known, 1), 1)}
        for _, row in oc.iterrows()
    ]
}
with open("data/outcomes.json", "w") as f:
    json.dump(outcomes_out, f, indent=2)
print(f"✓ outcomes.json — {known:,} with outcomes, "
      f"{outcomes_out['without_outcome_pct']}% missing")

# ── 5. cohorts.json ───────────────────────────────────────────────────────────
cohort_rows = []
for yr in sorted(df["filing_year"].dropna().unique()):
    yr  = int(yr)
    sub = df[df["filing_year"] == yr]["DISPOSALTIME_ADJ"]
    cohort_rows.append({
        "year":         yr,
        "n":            int(len(sub)),
        "median":       int(sub.median()),
        "pct_under1yr": round(100 * (sub <= 365).mean(), 1),
        "pct_1to2yr":   round(100 * ((sub > 365) & (sub <= 730)).mean(), 1),
        "pct_2to3yr":   round(100 * ((sub > 730) & (sub <= 1095)).mean(), 1),
        "pct_over3yr":  round(100 * (sub > 1095).mean(), 1),
        "note": ("Only fast-resolving cases had time to close by Jan 2021"
                 if yr >= 2018 else "")
    })
with open("data/cohorts.json", "w") as f:
    json.dump(cohort_rows, f, indent=2)
print(f"✓ cohorts.json — {len(cohort_rows)} filing years")

print("\n✅ All done. data/ folder ready.")
