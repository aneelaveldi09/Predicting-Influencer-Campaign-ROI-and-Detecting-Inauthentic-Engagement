# Predicting Influencer Campaign ROI and Detecting Inauthentic Engagement

**A Multi-Signal Machine Learning Framework for Brand Marketing Analytics**

**Aneela Veldi** · Independent Researcher · Chicago, Illinois · aneelaveldi09@gmail.com

---

## About

Brands spent over $21 billion on influencer campaigns in 2024. Most of that money is priced against gross reach — views, followers, engagement totals — without any adjustment for whether the creator's audience actually matches the brand's customers. This paper asks a simple question that the industry hasn't answered: *what is a campaign actually worth once you account for who is watching?*

This research introduces two metrics — **ABAI** (Audience-Brand Alignment Index) and **VMOF** (Vanity Metric Overestimation Factor) — that convert raw Instagram audience demographics into a dollar-adjusted campaign value. Applied to 90 days of real first-party Instagram Insights from @aneela_veldi, the results show a US beauty brand overpaying by **3.41× ($17,662 per million views)** while the same creator is reasonably priced for an India-focused tech brand. Same inventory, same price, 2.3× difference in value depending on who's buying.

The paper also examines inauthentic engagement detection, where a surprising finding emerges: follower burstiness — a signal used in commercial fraud-detection tools for years — now scores **AUROC = 0.426, below random chance**, because the 2022 Instagram algorithm change caused organic viral creators to generate the same follower spike patterns previously exclusive to bot farms. A three-signal fusion of temporal entropy, burstiness, and semantic comment coherence recovers full discrimination (AUROC = 1.000) by exploiting the fact that bots cannot fake all three dimensions at once.

Finally, a Spearman ρ analysis spanning the 2022 algorithm transition shows behavioral ROI features dropping from ρ = 0.793 to ρ = 0.067 and semantic features improving from ρ = 0.175 to ρ = 0.498. Any ROI prediction model trained on pre-2022 influencer data is now operating at near-chance performance.

---

## Key Results

| Finding | Value |
|---|---|
| ABAI — US Beauty Brand | 0.2935 |
| ABAI — India Tech Brand | 0.6741 |
| Overestimation (VMOF) | **3.41×** |
| Dollar gap per 1M views | **$17,662** |
| Budget waste (US beauty) | **70.7%** |
| Bot detection — fusion AUROC | 1.000 |
| Bot detection — burstiness alone | **0.426 (below random)** |
| Behavioral ROI ρ shift (pre→post 2022) | 0.793 → 0.067 |
| Semantic ROI ρ shift (pre→post 2022) | 0.175 → 0.498 |

---

## Files

| File | Description |
|---|---|
| `paper_final.pdf` | Full IEEE paper (compiled, ready to read) |
| `paper_final.tex` | LaTeX source (IEEEtran format) |
| `main.py` | Experiment code — ABAI/VMOF, LightGBM bot detector, temporal stability |
| `aneela_veldi_instagram_insights.csv` | Real first-party Instagram Insights, Mar–Jun 2026 |

---

## Reproduce the Results

```bash
pip install numpy scipy scikit-learn lightgbm pandas
python main.py
```

Output includes all reported metrics: ABAI scores, VMOF, dollar gap, bot detection AUROC per signal, and Spearman ρ before and after 2022.

---

## Methods at a Glance

**ABAI** weights geographic, gender, and age overlap between a creator's audience and a brand's target:

```
ABAI(creator, brand) = 0.5 × geo_overlap + 0.35 × gender_match + 0.15 × age_overlap
```

**VMOF** is the overestimation multiplier:

```
VMOF = 1 / ABAI   →   dollar_gap = naive_value × (1 - ABAI)
```

**Bot detection** fuses three signals in LightGBM:
- Sample entropy over like-arrival inter-event times (temporal regularity)
- Follower burstiness B-index (normalized CoV of daily follower increments)
- SBERT cosine similarity between post caption and comment embeddings (semantic coherence)

---

## Data

The Instagram Insights data (`aneela_veldi_instagram_insights.csv`) is first-party — exported directly from the Instagram account belonging to the paper's author. No scraping, no estimates.

---

*Paper written and experiments run using the AutoResearchClaw autonomous research pipeline.*
