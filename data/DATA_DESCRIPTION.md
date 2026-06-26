# Data Description

## aneela_veldi_instagram_insights.csv

First-party Instagram Insights export from the account @aneela_veldi, covering
March 25 through June 23, 2026 (90-day window). This data was exported directly
from Instagram's native analytics dashboard by the account owner. No scraping,
third-party tools, or estimation was involved.

### Fields

| Field | Type | Description |
|---|---|---|
| handle | string | Instagram username |
| period_start | date | Start of the reporting window |
| period_end | date | End of the reporting window |
| followers | int | Total followers at end of period |
| reel_views | int | Total reel views over the period |
| accounts_reached | int | Unique accounts reached |
| likes | int | Total likes received |
| comments | int | Total comments received |
| shares | int | Total shares |
| saves | int | Total saves |
| reels_posted | int | Number of reels published |
| engagement_rate_on_reach | float | (likes+comments+shares+saves) / accounts_reached * 100 |
| gender_female_pct | float | Percentage of audience identifying as female |
| gender_male_pct | float | Percentage of audience identifying as male |
| age_25_34_pct | float | Percentage of audience aged 25-34 |
| country_1 | string | Top country by audience share |
| country_1_pct | float | Audience share in country 1 |
| country_2 | string | Second country by audience share |
| country_2_pct | float | Audience share in country 2 |

### Key Values

| Metric | Value |
|---|---|
| Followers | 5,112 |
| Reel Views | 1,000,000 |
| Accounts Reached | 628,000 |
| Engagement Rate on Reach | 7.73% |
| Female | 31.5% |
| Male | 68.5% |
| Age 25-34 | 62.2% |
| India | 68.2% |
| United States | 13.7% |

### Usage

This data feeds directly into the ABAI and VMOF calculations in `experiments/main.py`
and `experiments/abai_vmof_case_study.py`. The geographic and demographic fields
are the inputs to the alignment index formula.
