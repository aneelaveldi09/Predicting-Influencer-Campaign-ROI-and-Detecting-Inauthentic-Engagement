# Predicting Influencer Campaign ROI and Detecting Inauthentic Engagement
### A Multi-Signal Machine Learning Framework for Brand Marketing Analytics

<p align="center">
  <img src="https://img.shields.io/badge/Status-Published-brightgreen" />
  <img src="https://img.shields.io/badge/Format-IEEE-blue" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

<p align="center">
  <strong>Aneela Veldi</strong><br>
  Independent Researcher · Chicago, Illinois<br>
  <a href="mailto:aneelaveldi09@gmail.com">aneelaveldi09@gmail.com</a>
</p>

---

## Overview

Influencer marketing is priced on reach. But reach is not value — not unless the audience watching actually matches the brand's customers. A creator with one million views and a 7.73% engagement rate can simultaneously be a terrible investment for one brand and an underpriced one for another, depending entirely on who those viewers are.

This paper formalizes that gap. It introduces two metrics — **ABAI** (Audience-Brand Alignment Index) and **VMOF** (Vanity Metric Overestimation Factor) — derived from real first-party Instagram Insights and designed to convert gross reach figures into brand-adjusted campaign valuations. The analysis also surfaces a previously undocumented failure in bot detection: follower-growth velocity, a signal used across commercial fraud tools, now performs below random chance on post-2022 data. And a temporal stability experiment shows that ROI prediction models built on pre-2022 behavioral features have effectively collapsed in predictive power after Instagram's 2022 algorithm change.

---

## Key Findings

| Metric | Value |
|---|---|
| ABAI — US Beauty Brand vs. @aneela_veldi | 0.2935 |
| ABAI — India Tech Brand vs. @aneela_veldi | 0.6741 |
| Vanity Metric Overestimation (VMOF) | **3.41×** |
| Dollar gap per 1M views (US beauty) | **$17,662** |
| Effective budget waste | **70.7%** |
| Bot detection — 3-signal fusion AUROC | **1.000** |
| Bot detection — burstiness alone AUROC | **0.426 (below random)** |
| Behavioral ROI ρ before → after 2022 | 0.793 → 0.067 |
| Semantic ROI ρ before → after 2022 | 0.175 → 0.498 |

---

## Repository Contents

```
├── paper_final.pdf                      # Compiled IEEE paper
├── paper_final.tex                      # LaTeX source (IEEEtran)
├── main.py                              # Full experiment code
└── aneela_veldi_instagram_insights.csv  # First-party Instagram Insights data
```

---

## Methods

### 1. Audience-Brand Alignment Index (ABAI)

ABAI scores the fit between a creator's audience and a brand's target market across three dimensions:

```
ABAI(creator, brand) = 0.5 · geo_overlap + 0.35 · gender_match + 0.15 · age_overlap
```

Where each term is the share of the creator's audience that falls within the brand's target geography, gender, and age range. ABAI ∈ [0, 1]; a score of 1 is perfect alignment.

The **Vanity Metric Overestimation Factor** follows directly:

```
VMOF = 1 / ABAI
Dollar gap = naive_campaign_value × (1 − ABAI)
```

Applied to 90 days of real Instagram Insights from @aneela_veldi: a US beauty brand at standard CPM overpays by 3.41× relative to the audience it is actually buying. An India tech brand targeting the same inventory is getting a fair deal.

### 2. Multi-Signal Bot Detection

Three signals are extracted per engagement sequence and fused in a LightGBM classifier:

- **Temporal entropy (SampEn)** — sample entropy of like-arrival inter-event times. Bot activity has rigid cadence and low entropy; organic engagement is irregular.
- **Follower burstiness (B-index)** — normalized coefficient of variation of daily follower increments. Historically high for bot farms; now also high for creators with viral organic content after 2022.
- **Cross-modal semantic coherence** — cosine similarity between SBERT embeddings of post captions and comments. Bot comments are generic and off-topic; genuine audience comments reference the post.

The fusion reaches AUROC = 1.000 on the controlled corpus. The critical finding is the ablation: burstiness alone scores AUROC = 0.426, below random, because the 2022 Reels-first algorithm now causes viral organic creators to generate the same growth spikes that once identified bot farms.

### 3. ROI Temporal Stability

Spearman ρ between predicted and actual ROI is computed separately on a pre-2022 cohort and a post-2023 cohort, for behavioral and semantic feature families independently. The results show a near-complete inversion after the algorithm change — behavioral features collapse from ρ = 0.793 to ρ = 0.067; semantic features improve from ρ = 0.175 to ρ = 0.498.

---

## Data

`aneela_veldi_instagram_insights.csv` is a first-party export from Instagram's native Insights dashboard, covering the 90-day window March 25 – June 23, 2026. The data belongs to the paper's author. No scraping or third-party estimation was involved.

Key figures: 5,112 followers · 1,000,000 reel views · 628,000 accounts reached · 7.73% engagement rate on reach · 68.2% India · 13.7% United States · 68.5% male · 62.2% aged 25–34.

---

## Reproducing the Results

```bash
pip install numpy scipy scikit-learn lightgbm pandas
python main.py
```

The script prints all reported metrics to stdout: ABAI scores, VMOF, dollar gap, per-signal AUROC values, and Spearman ρ for each feature family and period.

---

## Citation

If you use the ABAI/VMOF framework or reference the burstiness inversion finding, please cite:

```bibtex
@article{veldi2026influencer,
  title   = {Predicting Influencer Campaign ROI and Detecting Inauthentic Engagement:
             A Multi-Signal Machine Learning Framework for Brand Marketing Analytics},
  author  = {Veldi, Aneela},
  year    = {2026},
  note    = {Independent Researcher, Chicago, Illinois}
}
```

---

## License

MIT — free to use, adapt, and build on with attribution.
