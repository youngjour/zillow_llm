# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a real estate NLP research project analyzing geographic variation in property marketing language across Chicago, New York, and Los Angeles. The core finding is that cities share only **3-9% of discriminative vocabulary**, demonstrating extreme geographic divergence in how properties are marketed.

**Research Status:** Core analyses complete. Target publication: *Real Estate Economics* or *Journal of Urban Economics* (4-6 months to submission).

## Data Structure

**Primary dataset:** `dataset/raw/2. zillow_cleaned.geojson` (10,111 properties)
- **Format:** GeoJSON (not CSV!)
- **Cities:** CH (Chicago), NY (New York), LA (Los Angeles)
- **Property types:** 0 = Single-family, 1 = Condo/Townhouse
- **Target variable:** `duration` (time-on-market in days)
- **Features:** city, single, submarket, address, parking, bathroom, bedroom, age, living, description

**Discriminative words:** `dataset/word_counts/{threshold}/{city}_{type}_group_{speed}_zscore.csv`
- Extracted using log-odds ratio with Dirichlet prior (Monroe et al. 2008)
- Thresholds: 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 (25% is primary)
- Groups: 0 = fast-selling, 2 = slow-selling

**City-specific thresholds:** Defined in `const.py` as `sales_speed` dict
- Fast/slow cutoffs vary by city and property type
- Based on percentile splits (e.g., 25th/75th for 0.25 threshold)

## Running Analyses

### Interactive Analysis (Recommended)
```bash
jupyter notebook
# Open and run in order:
# 1. 1_Geographic_Language_Analysis.ipynb (2-3 min)
# 2. 2_Luxury_Language_Paradox.ipynb (3-5 min)
# 3. 3_City_Specific_Models.ipynb (10-15 min)
```

### Core Analysis Scripts
```bash
# Extract discriminative words (slow, ~30 min)
python extract_words.py

# Geographic language analysis (main results)
python geographic_language_analysis.py

# City-specific vs pooled models
python city_specific_models.py

# Luxury language paradox test
python luxury_language_analysis.py
```

### Full Pipeline (LLM + ML experiments)
```bash
# Requires OpenAI API key in .env
python main.py  # Runs all thresholds (0.05-0.30), takes hours
```

## Key Architecture

### Two Research Pipelines

**Pipeline 1: LLM Experiments** (main.py → chain.py)
- **Purpose:** Test if LLMs can predict time-on-market
- **Status:** Complete but underperforming (29-41% accuracy)
- **Architecture:**
  - Three prompt strategies: basic (attributes only), words (+ discriminative words), full (+ semantic meanings)
  - Uses GPT-4o-mini via LangChain
  - Structured output with Pydantic (MultiTOM class)
- **Key files:**
  - `chain.py`: Prompt building, LLM chains
  - `meanings.py`: GPT-generated semantic meanings of discriminative words
  - `main.py`: Orchestrates experiments at multiple thresholds

**Pipeline 2: Geographic Variation Analysis** (geographic_language_analysis.py + notebooks)
- **Purpose:** Main research focus - quantify geographic language differences
- **Status:** Core analysis complete, paper-ready
- **Architecture:**
  - Jaccard similarity for vocabulary overlap
  - Thematic categorization (9 categories)
  - City-specific model comparison
  - Luxury language paradox testing
- **Key files:**
  - `geographic_language_analysis.py`: Main analysis script
  - Jupyter notebooks: Interactive versions with explanations

### Core Utilities

**`extract_words.py`**: Discriminative word extraction
- Implements Monroe et al. (2008) log-odds with Dirichlet prior
- Function: `calculate_log_odds_idp()` in `smt203util.py`
- Splits data by percentiles, counts words, calculates z-scores
- Outputs to `dataset/word_counts/`

**`sampling.py`**: Train/test split
- 80/20 split with `random_state=42` (consistent across all experiments)
- Returns: X_train, X_test, y_train, y_test, descriptions, zpids, df_words

**`classifier.py`**: ML model wrapper
- Supports: logistic, rf, xgb (with/without class balancing)
- Auto grid search (5-fold CV)
- Used in main.py for baseline comparisons

**`const.py`**: City-specific thresholds
- `sales_speed` dict defines fast/slow cutoffs
- Example: Chicago single-family fast-selling at 25% threshold = 37 days

## Critical Implementation Details

### Data Path Issues
⚠️ **IMPORTANT:** The dataset is `.geojson` format, not `.csv`
```python
# WRONG (in some old scripts):
df = pd.read_csv("dataset/2. zillow_cleaned.csv")

# CORRECT:
import geopandas as gpd
df = gpd.read_file("dataset/raw/2. zillow_cleaned.geojson")
```

### City/Type Encoding
- Cities: "CH", "NY", "LA" (string codes in data)
- Property types: 0 or 1 (integer in data)
- Display names in `chain.py`: {"CH": "Chicago (IL)", ...}

### Multi-class Classification
All analyses use **3-class** targets (not binary):
- "fast": duration ≤ fast threshold
- "slow": duration ≥ slow threshold
- "moderate": between fast and slow

Thresholds are city/type-specific and threshold-specific:
```python
fast_cutoff = sales_speed[city][single]["fast"][th_idx]
slow_cutoff = sales_speed[city][single]["slow"][th_idx]
```

### Text Preprocessing
Chain: raw description → `text_preprocess()` → remove stopwords (`clean_text_round2()`)
- NLTK stopwords + length filter (>2 chars)
- Applied in `extract_words.py` before word counting

## Research Documentation

**Start here:** `result/geographic_analysis/ACCOMPLISHMENTS_AND_NEXT_STEPS.md`
- Complete project status
- What's done vs. remaining
- Timeline to submission

**Paper drafts:**
- `PAPER_INTRODUCTION_DRAFT.md` (~1,800 words)
- `PAPER_LITERATURE_REVIEW_DRAFT.md` (~2,500 words)

**Methodology reference:** `DETAILED_METHODOLOGY.md` (60+ pages)
- Mathematical derivations
- Complete methods section for paper

**Paper planning:** `RESEARCH_ROADMAP.md` (75+ pages)
- Full outline intro → conclusion
- Target journals and submission strategy

**Results interpretation:** `market_interpretation.md`
- City-specific findings narratives
- Ready to adapt for Results section

**Quick reference:** `FILE_INDEX.md`
- Status of all outputs
- Key findings summary

## Common Tasks

### Add a New Analysis
1. Load data correctly (use geojson path)
2. Apply city-specific thresholds from `const.py`
3. Follow train/test split from `sampling.py` (random_state=42)
4. Save outputs to `result/geographic_analysis/`
5. Update `FILE_INDEX.md` with new outputs

### Modify Discriminative Word Extraction
- Edit thresholds: `extract_words.py` line 150-151 (percentages list)
- Change word count: `find_discriminative_words()` num_i/num_j parameters
- Algorithm change: Modify `calculate_log_odds_idp()` in `smt203util.py`

### Create New Visualizations
- Use existing data in `result/geographic_analysis/*.csv` or `*.json`
- Matplotlib/Seaborn recommended for publication quality
- Save as PNG to `result/geographic_analysis/`
- DPI ≥ 300 for publication

### Update Paper Sections
- Introduction/Literature: Edit markdown files directly
- Results: Draft using outputs + `market_interpretation.md` narratives
- Methods: Adapt from `DETAILED_METHODOLOGY.md`

## Dependencies

**Core packages:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn geopandas scipy jupyter
```

**For LLM experiments (optional):**
```bash
pip install langchain-openai python-dotenv pydantic
# Requires .env with OPENAI_API_KEY
```

**For luxury regression (optional):**
```bash
pip install statsmodels  # Or use sklearn LinearRegression in notebooks
```

**Already installed:** nltk, xgboost, tqdm

## Key Findings Reference

**Finding 1:** Vocabulary overlap = 3-9% (Jaccard similarity)
- Evidence: `result/geographic_analysis/vocabulary_overlap.csv`
- Visualization: `vocabulary_overlap_heatmap.png`

**Finding 2:** City-specific market narratives
- Chicago: Transit (16% of words), renovation language
- LA: Outdoor (25%), ADU/income opportunities
- NY: Luxury restrictions (20%), suburban access
- Evidence: `category_distribution.csv`, `unique_city_words.json`

**Finding 3 (expected):** Luxury language paradox
- Hypothesis: More luxury words → LONGER time-on-market
- Expected coefficient: +8-12 days per luxury word
- Test in: `2_Luxury_Language_Paradox.ipynb`

**Finding 4 (expected):** City-specific models outperform pooled
- Expected improvement: 7-15% accuracy gain
- Test in: `3_City_Specific_Models.ipynb`

## Notes for Future Work

- The "disappointing" LLM results (29-41% accuracy) led to the BETTER research question (geographic variation)
- City-specific models failing to improve would actually strengthen the geographic variation argument
- All analyses use consistent train/test split (random_state=42) for comparability
- Robustness checks needed: alternative thresholds (15th/85th, 30th/70th percentiles)
- Consider temporal analysis if multi-year data becomes available

## File Naming Conventions

- Analysis scripts: `{analysis_name}.py` in root
- Notebooks: `{number}_{Title_Case_Name}.ipynb` in root
- Results: `result/geographic_analysis/{descriptive_name}.{csv|json|png|md}`
- Word counts: `dataset/word_counts/{threshold}/{CITY}_{TYPE}_group_{SPEED}_zscore.csv`
- Documentation: `{UPPERCASE_NAME}.md` in result/geographic_analysis/
