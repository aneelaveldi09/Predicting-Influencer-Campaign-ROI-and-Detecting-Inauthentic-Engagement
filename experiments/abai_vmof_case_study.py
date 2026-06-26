"""
Real-data case study: @aneela_veldi Instagram Insights
Computes ABAI and VMOF using first-party Instagram Insights export.

Run standalone:  python3 data/ground_truth/abai_vmof_case_study.py
Or import:       from data.ground_truth.abai_vmof_case_study import run_case_study
"""
import csv
import json
import os


# ── Brand scenario definitions ────────────────────────────────────────────────

BRAND_SCENARIOS = {
    "us_beauty_brand": {
        "name": "US Beauty Brand (18–34 Female)",
        "target_country": "United States",
        "target_gender": "female",
        "target_age_brackets": ["18–24", "25–34"],
        "cpm_low": 15.0,   # USD per 1000 impressions
        "cpm_high": 35.0,
        "campaign_budget_usd": 5000.0,
    },
    "india_tech_brand": {
        "name": "India Tech Brand (25–44 Male)",
        "target_country": "India",
        "target_gender": "male",
        "target_age_brackets": ["25–34", "35–44"],
        "cpm_low": 5.0,
        "cpm_high": 12.0,
        "campaign_budget_usd": 2000.0,
    },
}


def load_insights(csv_path: str) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return next(reader)


def compute_abai(row: dict, scenario: dict) -> dict:
    """
    ABAI = 0.5 * geo_overlap + 0.35 * gender_match + 0.15 * age_overlap

    geo_overlap  : fraction of audience in target country
    gender_match : fraction of audience matching target gender
    age_overlap  : sum of fractions in target age brackets
    """
    # geo_overlap
    target_country = scenario["target_country"]
    geo_overlap = 0.0
    for i in range(1, 6):
        country = row.get(f"country_{i}", "")
        pct = float(row.get(f"country_{i}_pct", 0) or 0)
        if country == target_country:
            geo_overlap = pct / 100.0
            break

    # gender_match
    if scenario["target_gender"] == "female":
        gender_match = float(row.get("gender_female_pct", 0) or 0) / 100.0
    else:
        gender_match = float(row.get("gender_male_pct", 0) or 0) / 100.0

    # age_overlap — map bracket strings to CSV column names
    AGE_COL_MAP = {
        "13–17": "age_13_17_pct",
        "18–24": "age_18_24_pct",
        "25–34": "age_25_34_pct",
        "35–44": "age_35_44_pct",
        "45–54": "age_45_54_pct",
        "55–64": "age_55_64_pct",
        "65+":   "age_65plus_pct",
    }
    age_overlap = 0.0
    for bracket in scenario["target_age_brackets"]:
        col = AGE_COL_MAP.get(bracket, "")
        if col:
            age_overlap += float(row.get(col, 0) or 0) / 100.0

    abai = 0.5 * geo_overlap + 0.35 * gender_match + 0.15 * age_overlap

    return {
        "geo_overlap": round(geo_overlap, 4),
        "gender_match": round(gender_match, 4),
        "age_overlap": round(age_overlap, 4),
        "abai": round(abai, 4),
    }


def compute_vmof(row: dict, abai_score: float, scenario: dict) -> dict:
    """
    Vanity Metric Overestimation Factor (VMOF)

    naive_value   = reel_views * cpm / 1000  (what a brand pays using raw reach)
    adjusted_value = naive_value * abai_score (what the brand SHOULD pay)
    vmof          = naive_value / adjusted_value  = 1 / abai_score
    dollar_gap    = naive_value - adjusted_value

    For campaign_budget:
    naive_impressions   = budget / cpm * 1000
    adjusted_impressions = naive_impressions * abai_score
    impressions_gap     = naive_impressions - adjusted_impressions
    """
    reel_views = float(row.get("reel_views", 0) or 0)
    cpm_mid = (scenario["cpm_low"] + scenario["cpm_high"]) / 2.0
    budget = scenario["campaign_budget_usd"]

    naive_value_low = reel_views * scenario["cpm_low"] / 1000.0
    naive_value_mid = reel_views * cpm_mid / 1000.0
    naive_value_high = reel_views * scenario["cpm_high"] / 1000.0

    adjusted_value_low = naive_value_low * abai_score
    adjusted_value_mid = naive_value_mid * abai_score
    adjusted_value_high = naive_value_high * abai_score

    vmof = (1.0 / abai_score) if abai_score > 0 else float("inf")

    dollar_gap_low = naive_value_low - adjusted_value_low
    dollar_gap_mid = naive_value_mid - adjusted_value_mid
    dollar_gap_high = naive_value_high - adjusted_value_high

    # Per-budget analysis: how many aligned impressions does a $5K budget buy?
    naive_impressions = budget / cpm_mid * 1000.0
    aligned_impressions = naive_impressions * abai_score

    return {
        "reel_views": reel_views,
        "cpm_range": [scenario["cpm_low"], scenario["cpm_high"]],
        "naive_campaign_value_usd": {
            "low": round(naive_value_low, 2),
            "mid": round(naive_value_mid, 2),
            "high": round(naive_value_high, 2),
        },
        "abai_adjusted_value_usd": {
            "low": round(adjusted_value_low, 2),
            "mid": round(adjusted_value_mid, 2),
            "high": round(adjusted_value_high, 2),
        },
        "vmof": round(vmof, 2),
        "dollar_overestimation_usd": {
            "low": round(dollar_gap_low, 2),
            "mid": round(dollar_gap_mid, 2),
            "high": round(dollar_gap_high, 2),
        },
        "budget_analysis": {
            "campaign_budget_usd": budget,
            "naive_impressions_bought": round(naive_impressions),
            "brand_aligned_impressions": round(aligned_impressions),
            "wasted_impressions": round(naive_impressions - aligned_impressions),
            "waste_pct": round((1.0 - abai_score) * 100.0, 1),
        },
    }


def run_case_study(csv_path: str | None = None) -> dict:
    if csv_path is None:
        # Resolve relative to this file's directory
        here = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(here, "aneela_veldi_instagram_insights.csv")

    row = load_insights(csv_path)

    results = {
        "creator": row["handle"],
        "period": f"{row['period_start']} to {row['period_end']} ({row['period_days']} days)",
        "raw_metrics": {
            "followers": int(row["followers"]),
            "reel_views": int(row["reel_views"]),
            "accounts_reached": int(row["accounts_reached"]),
            "engagement_rate_on_reach_pct": float(row["engagement_rate_on_reach"]),
            "virality_ratio": float(row["virality_ratio"]),
            "audience_top_country": row["country_1"],
            "audience_top_country_pct": float(row["country_1_pct"]),
            "gender_female_pct": float(row["gender_female_pct"]),
            "gender_male_pct": float(row["gender_male_pct"]),
            "age_25_34_pct": float(row["age_25_34_pct"]),
        },
        "brand_scenarios": {},
    }

    for scenario_key, scenario in BRAND_SCENARIOS.items():
        abai_result = compute_abai(row, scenario)
        vmof_result = compute_vmof(row, abai_result["abai"], scenario)
        results["brand_scenarios"][scenario_key] = {
            "scenario": scenario["name"],
            "abai": abai_result,
            "vmof": vmof_result,
        }

    return results


def print_summary(results: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  ABAI / VMOF Case Study — @{results['creator']}")
    print(f"  Period: {results['period']}")
    print(f"{'='*70}")
    m = results["raw_metrics"]
    print(f"\nRAW METRICS")
    print(f"  Followers:          {m['followers']:,}")
    print(f"  Reel views:         {m['reel_views']:,}   (virality {m['virality_ratio']}x followers)")
    print(f"  Accounts reached:   {m['accounts_reached']:,}")
    print(f"  Engagement rate:    {m['engagement_rate_on_reach_pct']}% on reach")
    print(f"  Top country:        {m['audience_top_country']} ({m['audience_top_country_pct']}%)")
    print(f"  Gender split:       {m['gender_female_pct']}% F / {m['gender_male_pct']}% M")
    print(f"  Age 25–34:          {m['age_25_34_pct']}%")

    for key, sc in results["brand_scenarios"].items():
        a = sc["abai"]
        v = sc["vmof"]
        b = v["budget_analysis"]
        print(f"\n{'─'*70}")
        print(f"  Scenario: {sc['scenario']}")
        print(f"{'─'*70}")
        print(f"  ABAI Components:")
        print(f"    geo_overlap   = {a['geo_overlap']:.4f}")
        print(f"    gender_match  = {a['gender_match']:.4f}")
        print(f"    age_overlap   = {a['age_overlap']:.4f}")
        print(f"    ABAI          = {a['abai']:.4f}  (0.5·{a['geo_overlap']}  +  0.35·{a['gender_match']}  +  0.15·{a['age_overlap']})")
        print(f"\n  VMOF (Vanity Metric Overestimation Factor):")
        print(f"    Naive campaign value (CPM ${v['cpm_range'][0]}–${v['cpm_range'][1]}): ${v['naive_campaign_value_usd']['low']:,.0f}–${v['naive_campaign_value_usd']['high']:,.0f}")
        print(f"    ABAI-adjusted value:                            ${v['abai_adjusted_value_usd']['low']:,.0f}–${v['abai_adjusted_value_usd']['high']:,.0f}")
        print(f"    Dollar overestimation:                          ${v['dollar_overestimation_usd']['low']:,.0f}–${v['dollar_overestimation_usd']['high']:,.0f}")
        print(f"    VMOF multiplier:                                {v['vmof']}x")
        print(f"\n  ${b['campaign_budget_usd']:,.0f} Budget Analysis:")
        print(f"    Naive impressions purchased:  {b['naive_impressions_bought']:,}")
        print(f"    Brand-aligned impressions:    {b['brand_aligned_impressions']:,}")
        print(f"    Wasted impressions:           {b['wasted_impressions']:,}  ({b['waste_pct']}% waste)")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    results = run_case_study()
    print_summary(results)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "abai_vmof_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out}")
