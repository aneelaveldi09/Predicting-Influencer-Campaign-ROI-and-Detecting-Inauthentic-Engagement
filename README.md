# Predicting Influencer Campaign ROI and Detecting Inauthentic Engagement

**A Multi-Signal Machine Learning Framework for Brand Marketing Analytics**

**Author:** Aneela Veldi — Independent Researcher, Chicago, Illinois, USA  
**Contact:** aneelaveldi09@gmail.com

---

## Overview

This repository contains the paper, experiment code, and real Instagram Insights data for the study examining two core failures in influencer marketing analytics: audience-brand mismatch and inauthentic engagement detection.

**Key findings:**
- A US beauty brand overpays by **3.41× ($17,662 per 1M views)** when contracting a creator whose audience is 68% Indian and 68% male
- Follower burstiness — a widely used fraud signal — now scores **AUROC = 0.426** (below random) because the 2022 Instagram algorithm change made organic viral creators look identical to bot farms on that metric
- A three-signal LightGBM fusion (temporal entropy + burstiness + semantic coherence) reaches **AUROC = 1.000** by combining signals with independent failure modes
- Behavioral ROI features collapsed from **ρ = 0.793 to ρ = 0.067** after the 2022 algorithm shift; semantic features improved from **ρ = 0.175 to ρ = 0.498**

---

## Files

| File | Description |
|---|---|
| `paper_final.pdf` | Compiled IEEE paper (IEEEtran format) |
| `paper_final.tex` | LaTeX source |
| `main.py` | Experiment code — ABAI/VMOF, LightGBM bot detection, temporal stability |
| `aneela_veldi_instagram_insights.csv` | Real first-party Instagram Insights, Mar–Jun 2026 |

---

## Reproducing the Results

```bash
pip install numpy scipy scikit-learn lightgbm pandas
python main.py
```

Key metrics printed: `abai_us_beauty_brand`, `vmof_overestimation_factor`, `dollar_gap_usd`, `auroc_fusion`, `auroc_burstiness_only`, `spearman_rho_semantic_pre2022`, `spearman_rho_semantic_post2023`, `spearman_rho_behavioral_pre2022`, `spearman_rho_behavioral_post2023`.

---

## Metrics (ABAI/VMOF)

**ABAI(c, b) = 0.5 · geo\_overlap + 0.35 · gender\_match + 0.15 · age\_overlap**

Applied to @aneela\_veldi audience data (Mar–Jun 2026):

| Brand | ABAI | Adjusted Value | VMOF | Gap |
|---|---|---|---|---|
| US Beauty Brand | 0.2935 | $8,805 | **3.41×** | **$17,662** |
| India Tech Brand | 0.6741 | $20,223 | 1.48× | $9,777 |

*(Naive value = $30,000 at CPM $30 for 1M views)*
