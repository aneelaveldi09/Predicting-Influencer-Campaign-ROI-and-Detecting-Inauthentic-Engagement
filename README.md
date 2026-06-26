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
  Independent Researcher, Chicago, Illinois<br>
  <a href="mailto:aneelaveldi09@gmail.com">aneelaveldi09@gmail.com</a>
</p>

---

## About

Influencer marketing is priced on reach. But reach is not value, not unless the audience watching actually matches the brand's customers. A creator with one million views and a 7.73% engagement rate can simultaneously be a terrible investment for one brand and an underpriced one for another, depending entirely on who those viewers are.

This paper formalizes that gap. It introduces two metrics, **ABAI** (Audience-Brand Alignment Index) and **VMOF** (Vanity Metric Overestimation Factor), derived from real first-party Instagram Insights and designed to convert gross reach figures into brand-adjusted campaign valuations. The analysis also surfaces a previously undocumented failure in bot detection: follower-growth velocity, a signal used across commercial fraud tools, now performs below random chance on post-2022 data. And a temporal stability experiment shows that ROI prediction models built on pre-2022 behavioral features have effectively collapsed in predictive power after Instagram's 2022 algorithm change.

All experiments are reproducible from the code and data in this repository.

---

## Key Results

| Metric | Value |
|---|---|
| ABAI, US Beauty Brand vs. @aneela_veldi | 0.2935 |
| ABAI, India Tech Brand vs. @aneela_veldi | 0.6741 |
| Vanity Metric Overestimation (VMOF) | **3.41x** |
| Dollar gap per 1M views (US beauty scenario) | **$17,662** |
| Effective budget waste | **70.7%** |
| Bot detection, 3-signal fusion AUROC | **1.000** |
| Bot detection, burstiness alone AUROC | **0.426 (below random)** |
| Behavioral ROI, Spearman rho before 2022 | 0.793 |
| Behavioral ROI, Spearman rho after 2022 | 0.067 |
| Semantic ROI, Spearman rho before 2022 | 0.175 |
| Semantic ROI, Spearman rho after 2022 | 0.498 |

---

## Repository Structure

```
.
├── paper_final.pdf                        # Compiled IEEE paper (read this first)
├── paper_final.tex                        # LaTeX source (IEEEtran format)
├── requirements.txt                       # Python dependencies
│
├── data/
│   ├── aneela_veldi_instagram_insights.csv  # Real first-party Instagram Insights
│   └── DATA_DESCRIPTION.md                  # Field definitions and data notes
│
├── experiments/
│   ├── main.py                            # Full experiment code (all 3 experiments)
│   └── abai_vmof_case_study.py            # Standalone ABAI/VMOF module
│
└── results/
    ├── experiment_run.json                # Complete run output with stdout
    └── metrics.json                       # Aggregated metrics across all seeds
```

---

## Reproducing the Results

```bash
git clone https://github.com/aneelaveldi09/Predicting-Influencer-Campaign-ROI-and-Detecting-Inauthentic-Engagement.git
cd Predicting-Influencer-Campaign-ROI-and-Detecting-Inauthentic-Engagement

pip install -r requirements.txt
python experiments/main.py
```

The script prints all reported metrics to stdout and saves results to `results/`. Expected runtime is under 30 seconds on any modern CPU. No GPU required.

To run just the ABAI/VMOF case study:

```bash
python experiments/abai_vmof_case_study.py
```

---

## Methods Summary

**ABAI** scores the fit between a creator's audience and a brand's target market:

```
ABAI(creator, brand) = 0.5 x geo_overlap + 0.35 x gender_match + 0.15 x age_overlap
```

**VMOF** is the overestimation multiplier and dollar gap:

```
VMOF = 1 / ABAI
Dollar gap = naive_campaign_value x (1 - ABAI)
```

**Bot detection** fuses three signals in LightGBM:

- Sample entropy over like-arrival inter-event times (temporal regularity)
- Follower burstiness B-index (normalized coefficient of variation of daily follower increments)
- SBERT cosine similarity between post captions and comments (semantic coherence)

**Temporal stability** measures Spearman rho between predicted and actual ROI for behavioral vs. semantic feature families, separately on pre-2022 and post-2023 campaign cohorts.

---

## Data

`data/aneela_veldi_instagram_insights.csv` is a first-party export from Instagram's native Insights dashboard covering March 25 to June 23, 2026. The data belongs to the paper's author. No scraping or third-party estimation was used.

Summary: 5,112 followers, 1,000,000 reel views, 628,000 accounts reached, 7.73% engagement rate on reach, 68.2% India, 13.7% United States, 68.5% male, 62.2% aged 25 to 34.

See `data/DATA_DESCRIPTION.md` for full field definitions.

---

## Verified Output

The `results/experiment_run.json` file contains the complete stdout from the experiment run, including per-seed results across 5 random seeds. The aggregated values match what is reported in the paper:

```
Fusion AUROC (commercial bots):  1.0000 +/- 0.0000
Burstiness-only AUROC:           0.4258 +/- 0.0175
Temporal entropy AUROC:          0.9481 +/- 0.0039
Behavioral rho degradation:      0.7260 +/- 0.0190
Semantic rho improvement:        0.3233 +/- 0.0809
ABAI (US beauty brand):          0.2935
VMOF:                            3.41x
Dollar gap:                      $17,662
```

---

## Citation

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

MIT. Free to use, adapt, and build on with attribution.
