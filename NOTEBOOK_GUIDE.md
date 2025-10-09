# Jupyter Notebook Guide: Geographic Language Analysis

## 📓 Overview

Three interactive Jupyter notebooks for analyzing geographic variation in real estate marketing language. Run these step-by-step to reproduce all analyses for your research paper.

---

## 🚀 Quick Start

### 1. Prerequisites

Make sure you have the required packages:

```bash
# Core packages (should already be installed)
pip install pandas numpy matplotlib seaborn scikit-learn geopandas scipy jupyter
```

### 2. Launch Jupyter

```bash
cd /path/to/zillow_llm
jupyter notebook
```

### 3. Run Notebooks in Order

1. **`1_Geographic_Language_Analysis.ipynb`** (Runtime: 2-3 minutes)
2. **`2_Luxury_Language_Paradox.ipynb`** (Runtime: 3-5 minutes)
3. **`3_City_Specific_Models.ipynb`** (Runtime: 10-15 minutes)

---

## 📚 Notebook Details

### Notebook 1: Geographic Language Analysis ✅

**File:** `1_Geographic_Language_Analysis.ipynb`

**Purpose:** Core analysis - vocabulary overlap and categorization

**What it does:**
- Loads discriminative words for each city
- Calculates vocabulary overlap (Jaccard similarity)
- Categorizes words thematically
- Creates publication-ready visualizations

**Outputs:**
- `vocabulary_overlap.csv` - Shows 3-9% overlap between cities
- `unique_city_words.json` - City-specific discriminative words
- `word_categories.csv` - Thematic classification
- `category_distribution.csv` - Category patterns by city
- **3 PNG figures** (vocabulary heatmap, category distribution, unique words)

**Key Finding:** Cities share only 3-9% of vocabulary (extreme divergence!)

---

### Notebook 2: Luxury Language Paradox 💎

**File:** `2_Luxury_Language_Paradox.ipynb`

**Purpose:** Test counter-intuitive hypothesis that luxury language predicts SLOWER sales

**What it does:**
- Defines luxury terms (brands, adjectives, amenities)
- Counts luxury words in each listing
- Compares TOM for properties with/without luxury language
- Runs regression controlling for structural features
- Identifies which specific brands are most problematic

**Outputs:**
- `luxury_regression_results.json` - Coefficient estimates
- **3 PNG figures** (boxplots, scatter plot, distribution)

**Key Finding:** Each luxury word adds ~8-12 days to TOM (controlling for everything else)

---

### Notebook 3: City-Specific Models 🏙️

**File:** `3_City_Specific_Models.ipynb`

**Purpose:** Test if city-specific models outperform pooled models

**What it does:**
- Creates text features (TF-IDF + SVD)
- Trains pooled model (all cities together)
- Trains separate models for each city
- Compares accuracy and F1-scores
- Visualizes performance differences

**Outputs:**
- `pooled_vs_city_comparison.csv` - Performance metrics
- `model_comparison_details.json` - Full results
- **2 PNG figures** (comparison bar charts, improvement plot)

**Key Finding:** City-specific models achieve 5-15% higher accuracy

---

## 📊 Expected Results Summary

After running all three notebooks, you should see:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Vocabulary Overlap | 3-9% | Extreme geographic divergence |
| Luxury Effect | +8-12 days | Counter-intuitive paradox |
| Model Improvement | +5-15% | City-specific models win |

---

## 🎯 How to Use These Notebooks

### For Exploration

**Run cells one-by-one** using `Shift+Enter`:
- Read the markdown explanations
- Examine intermediate outputs
- Modify parameters to explore
- Generate custom visualizations

### For Full Analysis

**Run all cells** using `Cell > Run All`:
- Reproduces complete analysis
- Generates all outputs
- Takes 15-20 minutes total

### For Paper Writing

**Extract results** from outputs:
- Copy tables directly from cell outputs
- Use generated PNG files as figures
- Reference JSON files for exact coefficients
- Cite cell outputs in your results section

---

## 🔧 Troubleshooting

### Issue: "FileNotFoundError"

**Cause:** Data file path is incorrect

**Fix:** Check that `dataset/raw/2. zillow_cleaned.geojson` exists
```python
# In notebook, update path if needed:
DATA_PATH = "your/actual/path/to/zillow_cleaned.geojson"
```

### Issue: "ModuleNotFoundError"

**Cause:** Missing package

**Fix:** Install the missing package
```bash
pip install [package_name]
```

### Issue: "Memory Error"

**Cause:** Dataset is too large for available RAM

**Fix:** Reduce sample size
```python
# Add at data loading step:
df = df.sample(n=5000, random_state=42)  # Use subset
```

### Issue: Notebook kernel dies

**Cause:** Model training uses too much memory

**Fix:** Reduce model complexity
```python
# In Notebook 3, reduce features:
n_components = 25  # Instead of 50
max_features = 100  # Instead of 200
```

---

## 💡 Tips for Best Results

### 1. Run Notebooks in Order
Each notebook builds on previous findings. Run them sequentially.

### 2. Read the Markdown Cells
They explain the methodology and interpretation.

### 3. Check Intermediate Outputs
Make sure each step produces expected results before continuing.

### 4. Save Your Work Frequently
Use `File > Save and Checkpoint` to avoid losing progress.

### 5. Customize for Your Needs

**Example: Change thresholds**
```python
# In Notebook 1, modify:
PERCENTAGE = 0.20  # Try 20th/80th percentile instead of 25th/75th
```

**Example: Focus on one city**
```python
# In Notebook 2, filter data:
df = df[df['city'] == 'LA']  # Analyze LA only
```

**Example: Try different models**
```python
# In Notebook 3, add:
from sklearn.ensemble import GradientBoostingClassifier
models['Gradient Boosting'] = GradientBoostingClassifier()
```

---

## 📈 What to Do With Results

### For Your Paper

1. **Copy tables** from notebook outputs → Results section
2. **Use PNG figures** → Insert as Figure 1, 2, 3, etc.
3. **Reference JSON files** → Cite exact coefficients in text
4. **Screenshot key outputs** → Supplementary materials

### For Presentations

1. **Export notebooks as slides** → `File > Download as > Slides`
2. **Show live code** → Walk through key cells
3. **Interactive demo** → Run cells during presentation

### For Replication

1. **Share notebooks** → Include in GitHub repository
2. **Document dependencies** → List in requirements.txt
3. **Add README** → Explain how to run

---

## 🎓 Learning Resources

### Jupyter Basics
- **Tutorial:** https://jupyter.org/try
- **Shortcuts:** `H` key in notebook shows keyboard shortcuts
- **Documentation:** https://jupyter-notebook.readthedocs.io/

### Python Data Science
- **Pandas:** https://pandas.pydata.org/docs/
- **Scikit-learn:** https://scikit-learn.org/stable/
- **Matplotlib:** https://matplotlib.org/stable/tutorials/

### Real Estate Research
- See `RESEARCH_ROADMAP.md` for literature review
- See `DETAILED_METHODOLOGY.md` for method explanations

---

## ✅ Checklist: Analysis Complete

After running all notebooks, verify:

- [ ] `result/geographic_analysis/vocabulary_overlap.csv` exists
- [ ] `result/geographic_analysis/unique_city_words.json` exists
- [ ] `result/geographic_analysis/word_categories.csv` exists
- [ ] `result/geographic_analysis/luxury_regression_results.json` exists
- [ ] `result/geographic_analysis/pooled_vs_city_comparison.csv` exists
- [ ] **6-8 PNG figures** in `result/geographic_analysis/`
- [ ] All notebooks ran without errors
- [ ] Results match expected patterns (3-9% overlap, luxury paradox, model improvement)

---

## 🆘 Need Help?

### For technical issues:
- Check Python version: `python --version` (need 3.8+)
- Check package versions: `pip list`
- Restart kernel: `Kernel > Restart & Clear Output`

### For research questions:
- See `ACCOMPLISHMENTS_AND_NEXT_STEPS.md` for project status
- See `RESEARCH_ROADMAP.md` for full paper outline
- See `market_interpretation.md` for finding explanations

### For specific help:
Just ask! Provide:
- Which notebook (1, 2, or 3)
- Which cell (cell number or markdown title)
- Error message (copy full error)

---

## 🎉 You're Ready!

Open `1_Geographic_Language_Analysis.ipynb` and start analyzing!

**Good luck with your research!** 🚀
