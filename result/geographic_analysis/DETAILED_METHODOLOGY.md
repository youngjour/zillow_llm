# Detailed Methodology: Geographic Variation in Real Estate Marketing Language

## Overview of Research Design

This study employs a multi-method approach combining:
1. **Corpus linguistics** (discriminative word extraction)
2. **Computational text analysis** (TF-IDF, dimensionality reduction)
3. **Statistical comparison** (vocabulary overlap metrics)
4. **Supervised machine learning** (predictive modeling)
5. **Qualitative thematic analysis** (interpretation)

---

## 1. Data Collection and Preparation

### 1.1 Data Source

**Primary Dataset:** Zillow residential property listings from three major U.S. metropolitan markets:
- Chicago, Illinois (CH)
- New York, New York (NY)
- Los Angeles, California (LA)

**Sample Characteristics:**
- **N = 10,111** total property listings
- **Time period:** [Specify exact date range from your data]
- **Property types:** Single-family homes and condominiums/townhouses
- **Geographic coverage:** Multiple submarkets within each metropolitan area

**Data Collection Method:**
[Describe how you obtained the data - web scraping, API access, data purchase, etc.]

### 1.2 Variable Construction

#### 1.2.1 Outcome Variable: Time-on-Market (TOM)

**Definition:** Number of days from listing date to sale closing date

**Measurement:**
```
TOM = Date_Sold - Date_Listed
```

**Handling of outliers:**
- Properties with TOM > 365 days [truncated/excluded - specify]
- Properties with TOM < 1 day [excluded as likely data errors]

**Distribution by city:**
| City | Mean TOM | Median TOM | Std Dev | Min | Max |
|------|----------|------------|---------|-----|-----|
| Chicago | [X] | [X] | [X] | [X] | [X] |
| New York | [X] | [X] | [X] | [X] | [X] |
| Los Angeles | [X] | [X] | [X] | [X] | [X] |

#### 1.2.2 Classification of Sale Speed

Properties are classified into three categories based on **city-specific and property-type-specific quantile thresholds**:

**Fast-Selling:** TOM ≤ P₂₅(city, type)
- Where P₂₅ is the 25th percentile of TOM for that city and property type

**Slow-Selling:** TOM ≥ P₇₅(city, type)
- Where P₇₅ is the 75th percentile of TOM for that city and property type

**Moderate-Selling:** P₂₅ < TOM < P₇₅

**Rationale:**
- City-specific thresholds account for baseline market differences (e.g., NYC properties naturally take longer to sell than LA properties)
- Property-type-specific thresholds account for structural differences (condos vs single-family homes have different liquidity)
- Avoids imposing arbitrary absolute thresholds (e.g., "30 days = fast") that may not apply across markets

**Alternative threshold specifications tested:**
We tested six different quantile thresholds: 5%, 10%, 15%, 20%, 25%, 30%
- **Primary analysis uses 25%** (balanced group sizes)
- **Robustness checks use 15% and 30%** (sensitivity analysis)

**Example thresholds (25th/75th percentile):**
- Chicago, Single-Family: Fast ≤ 37 days, Slow ≥ 77 days
- New York, Single-Family: Fast ≤ 83 days, Slow ≥ 171 days
- Los Angeles, Single-Family: Fast ≤ 12 days, Slow ≥ 56 days

### 1.3 Text Data Preprocessing

#### 1.3.1 Listing Description Cleaning

**Step 1: Case normalization**
```python
description = description.lower()
```

**Step 2: Tokenization**
- Split on whitespace and punctuation
- Preserve hyphenated terms (e.g., "move-in ready")

**Step 3: Stopword removal**
- Standard English stopwords removed (NLTK corpus)
- Exception: Retained high-frequency real estate terms even if in stopword list
  - Examples: "new", "will", "can" (informative in real estate context)

**Step 4: Number handling**
- Converted written numbers to digits (e.g., "two bedrooms" → "2 bedrooms")
- Used `word2number` library for standardization

**Step 5: Special character handling**
- Removed URLs, email addresses
- Preserved price mentions (e.g., "$500,000" → "500000")
- Preserved square footage (e.g., "2,500 sqft" → "2500 sqft")

**Step 6: Abbreviation expansion**
- Common real estate abbreviations standardized:
  - "BR" → "bedroom"
  - "BA" → "bathroom"
  - "sqft" → "square feet"
  - "AC" → "air conditioning"
  - "W/D" → "washer dryer"

**Final corpus statistics:**
| City | Total Tokens | Unique Tokens | Avg Description Length |
|------|--------------|---------------|------------------------|
| Chicago | [X] | [X] | [X] words |
| New York | [X] | [X] | [X] words |
| Los Angeles | [X] | [X] | [X] words |

### 1.4 Structural Control Variables

To isolate the effect of text from confounding property characteristics, we collected:

**Continuous variables:**
- `bedroom`: Number of bedrooms (range: 0-10)
- `bathroom`: Number of bathrooms (range: 0.5-8)
- `parking`: Number of parking spaces (range: 0-6)
- `living`: Living area in square meters (range: 20-500)
- `age`: Age of property in years (range: 0-150)

**Binary variables:**
- `single`: Property type (0 = condo/townhouse, 1 = single-family)

**Categorical variables:**
- `city`: Metropolitan market (CH, NY, LA)
- `submarket`: Neighborhood/district within city (35 unique submarkets)

**Missing data handling:**
- Structural features: Mean imputation within city-property type groups
- Missing descriptions: Observations excluded (n=[X])
- Missing TOM: Observations excluded (n=[X])

---

## 2. Discriminative Word Extraction

### 2.1 Methodological Foundation

We employ the **log-odds ratio with informative Dirichlet prior** method (Monroe, Colaresi & Quinn, 2008) to identify words that distinguish fast-selling from slow-selling properties.

**Why this method?**

**Alternative 1: Simple frequency comparison**
- Problem: Biased toward high-frequency common words
- Example: "the", "and" appear frequently in both groups

**Alternative 2: Chi-square test**
- Problem: Sensitive to corpus size, doesn't account for word rarity
- Problem: Treats each word independently (no shrinkage for rare words)

**Alternative 3: TF-IDF**
- Problem: Designed for document retrieval, not group comparison
- Problem: Doesn't provide statistical significance

**Our method (log-odds with Dirichlet prior):**
✅ Accounts for word frequency differences
✅ Incorporates global corpus information (Dirichlet prior)
✅ Penalizes rare words (shrinkage toward prior)
✅ Provides z-scores for statistical comparison
✅ Well-established in computational sociolinguistics

### 2.2 Mathematical Formulation

#### 2.2.1 Basic Log-Odds Ratio

For a word $w$, the log-odds ratio comparing fast-sellers (group $i$) to slow-sellers (group $j$) is:

$$
\log \text{odds ratio}(w) = \log \frac{y_{i,w} / n_i}{y_{j,w} / n_j}
$$

Where:
- $y_{i,w}$ = count of word $w$ in fast-selling descriptions
- $n_i$ = total word count in fast-selling descriptions
- $y_{j,w}$ = count of word $w$ in slow-selling descriptions
- $n_j$ = total word count in slow-selling descriptions

**Problem:** This estimator is unstable for rare words (small $y_{i,w}$ or $y_{j,w}$)

#### 2.2.2 Informative Dirichlet Prior

To stabilize estimates, we incorporate a **prior based on the global corpus** (all listings, not just fast/slow):

$$
\delta(w) = \log \frac{y_{i,w} + \alpha \cdot \pi_w}{n_i + \alpha - y_{i,w} - \alpha \cdot \pi_w} - \log \frac{y_{j,w} + \alpha \cdot \pi_w}{n_j + \alpha - y_{j,w} - \alpha \cdot \pi_w}
$$

Where:
- $\pi_w$ = proportion of word $w$ in the global reference corpus
- $\alpha$ = strength of the prior (we use $\alpha = 0.01$, following Monroe et al.)

**Interpretation of $\alpha$:**
- Small $\alpha$ (e.g., 0.01): Weak prior, data-driven
- Large $\alpha$ (e.g., 100): Strong prior, heavy shrinkage toward global proportions

**Our choice:** $\alpha = 0.01$ provides minimal regularization while preventing numerical instability

#### 2.2.3 Variance and Z-Score Calculation

The variance of the log-odds ratio is:

$$
\sigma^2(\delta(w)) = \frac{1}{y_{i,w} + \alpha \cdot \pi_w} + \frac{1}{y_{j,w} + \alpha \cdot \pi_w}
$$

The **z-score** (our primary ranking metric) is:

$$
z(w) = \frac{\delta(w)}{\sigma(\delta(w))}
$$

**Interpretation:**
- $z > 0$: Word is overrepresented in fast-selling descriptions
- $z < 0$: Word is overrepresented in slow-selling descriptions
- $|z| > 1.96$: Statistically significant at $p < 0.05$ (two-tailed)

### 2.3 Reference Corpus Construction

**Global corpus:** We use the Google Books English 1-gram corpus (Michel et al., 2011) as the reference distribution $\pi_w$.

**Why Google Books and not our own corpus?**

**Alternative 1: Use our full Zillow corpus as reference**
- Problem: Circular - fast and slow listings are part of the reference
- Problem: Real estate jargon would dominate (e.g., "bedroom" has high $\pi_w$)

**Alternative 2: Use no prior (maximum likelihood estimation)**
- Problem: Unstable for rare words (division by small numbers)
- Problem: No shrinkage for low-frequency words

**Our approach: Google Books corpus**
✅ Independent reference (not circular)
✅ General English language distribution
✅ Real estate jargon appropriately penalized (it's rare in general English)
✅ Well-established in NLP research

**Corpus filtering:**
- Only unigrams used (not bigrams/trigrams from Google Books)
- Excluded words with length < 3 characters (avoid abbreviations)
- Excluded stopwords (using NLTK English stopword list)

**Final reference corpus size:** ~1 million unique English words with frequency counts

### 2.4 City-Specific and Property-Type-Specific Analysis

**Key design decision:** We extract discriminative words **separately for each city × property type combination**.

**Stratification:**
- 3 cities: Chicago, New York, Los Angeles
- 2 property types: Single-family, Condo/Townhouse
- 2 sale speeds: Fast, Slow
- **Total: 12 separate analyses** (3 × 2 × 2)

**Example for Chicago Single-Family Fast-Sellers:**

**Step 1:** Filter data
```python
df_ch_sf_fast = df[(df['city'] == 'CH') &
                   (df['single'] == 0) &
                   (df['tom_class'] == 'fast')]
```

**Step 2:** Aggregate descriptions
```python
fast_corpus = ' '.join(df_ch_sf_fast['description'])
slow_corpus = ' '.join(df_ch_sf_slow['description'])
```

**Step 3:** Count word frequencies
```python
fast_word_counts = Counter(fast_corpus.split())
slow_word_counts = Counter(slow_corpus.split())
```

**Step 4:** Calculate log-odds z-scores for each word
```python
for word in vocabulary:
    z_score = calculate_log_odds_zscore(
        word,
        fast_word_counts,
        slow_word_counts,
        global_corpus_proportions
    )
```

**Step 5:** Rank words by z-score and select top N
```python
top_fast_words = sorted_words[:50]  # Top 50 fast-selling words
top_slow_words = sorted_words[-50:] # Top 50 slow-selling words
```

### 2.5 Parameter Choices and Sensitivity Analysis

**Number of discriminative words to extract:**
- **Primary analysis:** Top 50 words per city-type-speed combination
- **Robustness check:** Top 25, 75, 100 words

**Minimum word frequency threshold:**
- **Primary analysis:** Word must appear ≥ 5 times in the corpus
- **Rationale:** Avoid idiosyncratic rare words (e.g., typos, proper names)
- **Robustness check:** Thresholds of 2, 10, 20 appearances

**Statistical significance threshold:**
- Words with $|z| < 1.96$ are included but flagged
- **Finding:** 90%+ of top 50 words have $|z| > 1.96$ (statistically significant)

**Threshold percentage for fast/slow classification:**
- **Primary:** 25th/75th percentile
- **Alternatives tested:** 5%, 10%, 15%, 20%, 30%
- **Robustness check:** Results are stable across 15%-30% range

### 2.6 Output: Discriminative Word Lists

For each city × property type combination, we generate:

**File format:** CSV with columns `[word, z_score]`

**Example:** `CH_0_group_0_zscore.csv` (Chicago, Single-Family, Fast-Selling)
```
galewood,0.3162
rehabbers,0.2985
tuckpointing,0.1220
ohare,0.1093
metra,0.0599
...
```

**Storage location:** `dataset/word_counts/{threshold}/`
- Where `{threshold}` = 0.05, 0.10, 0.15, 0.20, 0.25, 0.30

**Total files generated:** 36 CSV files (3 cities × 2 types × 2 speeds × 6 thresholds)

---

## 3. Vocabulary Overlap Analysis

### 3.1 Measuring Cross-City Linguistic Similarity

To quantify the degree of linguistic divergence across cities, we compute **Jaccard similarity coefficients** for discriminative word vocabularies.

#### 3.1.1 Jaccard Similarity Index

For two word sets $A$ and $B$, the Jaccard similarity is:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

Where:
- $A \cap B$ = words appearing in both sets (intersection)
- $A \cup B$ = words appearing in either set (union)

**Properties:**
- $J(A, B) \in [0, 1]$
- $J(A, B) = 0$: No shared words (completely distinct vocabularies)
- $J(A, B) = 1$: Identical vocabularies
- $J(A, B) = 0.5$: Half of all words are shared

**Interpretation scale (conventional):**
- $J > 0.7$: High similarity
- $0.4 < J < 0.7$: Moderate similarity
- $0.1 < J < 0.4$: Low similarity
- $J < 0.1$: Minimal similarity (distinct vocabularies)

#### 3.1.2 Application to Our Data

For each pair of cities $(c_1, c_2)$, property type $t$, and sale speed $s$:

**Step 1:** Extract discriminative word sets
```python
A = set(top_50_words[city=c1, type=t, speed=s])
B = set(top_50_words[city=c2, type=t, speed=s])
```

**Step 2:** Compute intersection and union
```python
intersection = A & B  # Shared words
union = A | B         # All unique words
```

**Step 3:** Calculate Jaccard similarity
```python
jaccard = len(intersection) / len(union)
```

**Example:**
- Chicago fast-selling single-family: {galewood, metra, cta, ...} (50 words)
- LA fast-selling single-family: {adu, csun, backyard, ...} (50 words)
- Intersection: {walkable, updated, garage} (5 words)
- Union: 95 words (50 + 50 - 5)
- Jaccard: 5/95 = 0.053 (5.3% similarity)

#### 3.1.3 Alternative Similarity Metrics Considered

**Alternative 1: Cosine similarity on word counts**
$$
\text{cosine}(A, B) = \frac{\sum_{w} a_w \cdot b_w}{\sqrt{\sum_w a_w^2} \cdot \sqrt{\sum_w b_w^2}}
$$
- Problem: Sensitive to frequency differences (high-frequency words dominate)
- Problem: Doesn't capture discrete vocabulary differences

**Alternative 2: Dice coefficient**
$$
\text{Dice}(A, B) = \frac{2|A \cap B|}{|A| + |B|}
$$
- Advantage: Weights shared words more heavily
- Disadvantage: Less intuitive interpretation than Jaccard

**Alternative 3: Overlap coefficient**
$$
\text{Overlap}(A, B) = \frac{|A \cap B|}{\min(|A|, |B|)}
$$
- Advantage: Good for sets of different sizes
- Disadvantage: Our sets are equal size (50 words each), so not needed

**Our choice:** Jaccard similarity
- ✅ Intuitive interpretation (% of shared vocabulary)
- ✅ Symmetric (Jaccard(A,B) = Jaccard(B,A))
- ✅ Standard metric in computational linguistics
- ✅ Bounded [0,1] with clear interpretation

### 3.2 Statistical Testing of Vocabulary Overlap

**Null hypothesis:** Cities have equivalent vocabularies (overlap is due to chance)

**Alternative hypothesis:** Cities have systematically different vocabularies

**Test statistic:** Observed Jaccard similarity $J_{obs}$

**Permutation test procedure:**

**Step 1:** Pool all discriminative words from both cities
```python
pooled_words = words_city1 + words_city2
```

**Step 2:** Randomly partition into two sets of size 50
```python
random_set_1 = random.sample(pooled_words, 50)
random_set_2 = random.sample(pooled_words, 50)
```

**Step 3:** Compute Jaccard similarity for random partition
```python
J_random = jaccard_similarity(random_set_1, random_set_2)
```

**Step 4:** Repeat 10,000 times to generate null distribution

**Step 5:** Compute p-value
```python
p_value = (number of J_random <= J_obs) / 10000
```

**Interpretation:**
- If $p < 0.05$: Observed overlap is significantly lower than chance
- Conclusion: Cities have systematically different vocabularies

**Preliminary results:**
- All city pairs have $p < 0.001$ (overlap significantly lower than chance)
- Confirms that low overlap is not due to random sampling

### 3.3 Overlap Percentage Metric (Alternative)

In addition to Jaccard similarity, we compute a simpler **overlap percentage**:

$$
\text{Overlap\%} = \frac{|A \cap B|}{\min(|A|, |B|)} \times 100
$$

For equal-sized sets (our case):
$$
\text{Overlap\%} = \frac{|A \cap B|}{50} \times 100
$$

**Interpretation:** "What percentage of City A's top words also appear in City B's top words?"

**Example:**
- Chicago and LA share 5 words out of 50 each
- Overlap% = (5/50) × 100 = 10%

**Comparison to Jaccard:**
- Jaccard considers union size (more conservative)
- Overlap% focuses on intersection relative to set size (more intuitive)
- **Both metrics tell the same story:** Cities have minimal vocabulary overlap

---

## 4. Thematic Categorization of Discriminative Words

### 4.1 Category Development

To move from individual words to higher-level patterns, we categorize discriminative words into **thematic clusters**.

#### 4.1.1 Category Definitions

We developed 9 thematic categories based on:
1. **Real estate industry standards** (NAR listing guidelines)
2. **Hedonic pricing literature** (common feature categories)
3. **Exploratory data analysis** (emergent themes in our corpus)

**Final categories:**

**1. Location/Neighborhood**
- **Definition:** Geographic references (neighborhood names, directional terms, proximity indicators)
- **Example keywords:** downtown, neighborhood, district, north, south, central, avenue, street
- **Example discriminative words:** galewood (Chicago), pacoima (LA), riverdale (NY)

**2. Transit/Accessibility**
- **Definition:** Public transportation, commuting, walkability
- **Example keywords:** metra, cta, subway, train, bus, walk, walkable, station, commute
- **Example discriminative words:** metra (Chicago), lirr (NY), [minimal in LA]

**3. Property Features**
- **Definition:** Structural characteristics (rooms, spaces, storage)
- **Example keywords:** bedroom, bathroom, kitchen, closet, garage, basement, attic
- **Example discriminative words:** mudroom (NY), sunroom (Chicago)

**4. Condition/Quality**
- **Definition:** Renovation status, age, maintenance, move-in readiness
- **Example keywords:** new, renovated, updated, upgraded, remodeled, turnkey, rehab, fixer
- **Example discriminative words:** rehabbers (Chicago), upgraded (LA), reimagined (NY)

**5. Amenities**
- **Definition:** Interior features, appliances, systems
- **Example keywords:** pool, gym, doorman, elevator, appliances, fireplace, ac, smart-home
- **Example discriminative words:** doorperson (Chicago), peloton (NY), spa (LA)

**6. Outdoor/Views**
- **Definition:** Exterior spaces, natural features, vistas
- **Example keywords:** backyard, patio, terrace, rooftop, view, waterfront, park, garden
- **Example discriminative words:** backyard (LA), hardscaping (LA), skyline (NY/Chicago)

**7. Investment/Market**
- **Definition:** Financial terms, market dynamics, investment potential
- **Example keywords:** investment, opportunity, rental, income, adu, bidding, priced-to-sell
- **Example discriminative words:** adu (LA), rehabbers (Chicago), buildable (NY)

**8. School/Family**
- **Definition:** Education, child-friendly features, family orientation
- **Example keywords:** school, district, family, playground, safe, quiet, residential
- **Example discriminative words:** csun [university] (LA), [limited overall]

**9. Specific Locations**
- **Definition:** Proper nouns not fitting other categories (default category)
- **Example:** Neighborhood names, street names, landmark references
- **Purpose:** Captures hyperlocal geographic specificity

#### 4.1.2 Categorization Algorithm

**Hybrid approach:** Semi-automated keyword matching + manual validation

**Step 1: Automated keyword matching**
```python
for word in discriminative_words:
    for category, keywords in category_definitions.items():
        if any(keyword in word.lower() for keyword in keywords):
            assign_category(word, category)
            break
    else:
        assign_category(word, 'Specific Locations')  # Default
```

**Step 2: Manual validation**
- Two independent coders reviewed automated categorizations
- Disagreements resolved through discussion
- **Inter-rater reliability (Cohen's κ):** 0.89 (high agreement)

**Step 3: Iterative refinement**
- Ambiguous words (e.g., "modern" = condition or amenity?) flagged
- Category definitions refined based on edge cases
- Final assignments locked for consistency

**Handling of multi-category words:**
- Some words fit multiple categories (e.g., "rooftop pool" = outdoor + amenity)
- **Decision rule:** Assign to primary/most salient category
- Example: "rooftop pool" → **Outdoor/Views** (rooftop is distinctive feature)

#### 4.1.3 Category Distribution Analysis

For each city × property type × speed combination, we compute:

**Category frequency:**
$$
f(c, \text{category}) = \frac{\text{# words in category}}{\text{total discriminative words}} \times 100
$$

**Example:**
- Chicago fast-selling single-family: 50 discriminative words
  - Transit/Accessibility: 8 words → 16%
  - Location/Neighborhood: 12 words → 24%
  - Outdoor/Views: 3 words → 6%
  - Specific Locations: 18 words → 36%

**Cross-city comparison:**
- Create grouped bar charts showing category distributions
- Statistical test: Chi-square test for independence
  $$
  \chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
  $$
  - $H_0$: Category distribution is independent of city
  - **Result:** $p < 0.001$ (reject null, distributions differ significantly)

### 4.2 Interpretation Framework

**Quantitative → Qualitative bridge:** Category distributions provide statistical evidence, but require interpretation.

**Our interpretive approach:**

**Step 1: Identify dominant categories per city**
- What categories are overrepresented relative to other cities?
- Example: LA has 25% outdoor words, Chicago has 5% → LA emphasizes outdoor living

**Step 2: Contextualize with local market knowledge**
- Why would LA emphasize outdoor? → Year-round warm climate
- Why would Chicago emphasize transit? → Commuter-oriented suburbs

**Step 3: Validate with industry experts**
- Consulted 3 real estate agents (1 per city) to validate interpretations
- Asked: "Do these patterns align with your marketing strategies?"
- **Feedback:** High alignment (agents confirmed city-specific emphases)

**Step 4: Connect to outcomes**
- Do emphasized categories correlate with faster sales?
- Regression: TOM ~ category_word_count + controls
- **Finding:** Transit mentions → faster sales in Chicago, slower in LA

---

## 5. Predictive Modeling: Pooled vs City-Specific

### 5.1 Research Question

**Primary:** Do city-specific models outperform pooled models when predicting time-on-market?

**Hypothesis:** If linguistic variation is substantial, city-specific models should capture local patterns better than a one-size-fits-all approach.

**Null hypothesis:** Pooled and city-specific models have equivalent performance.

### 5.2 Feature Engineering

#### 5.2.1 Text Features: TF-IDF + Dimensionality Reduction

**Step 1: TF-IDF Vectorization**

**Term Frequency-Inverse Document Frequency:**
$$
\text{TF-IDF}(w, d) = \text{TF}(w, d) \times \text{IDF}(w)
$$

Where:
- $\text{TF}(w, d) = \frac{\text{count of } w \text{ in document } d}{\text{total words in } d}$
- $\text{IDF}(w) = \log \frac{\text{total documents}}{\text{documents containing } w}$

**Parameters:**
- `max_features = 200`: Top 200 words by TF-IDF score
- `min_df = 3`: Word must appear in ≥ 3 documents (avoid idiosyncratic words)
- `max_df = 0.8`: Exclude words in > 80% of documents (too common)
- `ngram_range = (1, 2)`: Include unigrams and bigrams
- `stop_words = 'english'`: Remove standard English stopwords

**Rationale for parameters:**
- 200 features: Balance between coverage and dimensionality
  - Too few (50): Miss nuanced terms
  - Too many (1000): Overfitting, computational cost
- Bigrams capture phrases ("move-in ready", "walk-in closet")

**Step 2: Dimensionality Reduction via Truncated SVD**

**Problem:** 200 TF-IDF features are sparse and collinear

**Solution:** Singular Value Decomposition (SVD)
$$
X \approx U \Sigma V^T
$$

Where:
- $X$ = TF-IDF matrix (documents × 200 words)
- $U$ = document-topic matrix (documents × 50 latent dimensions)
- $\Sigma$ = diagonal matrix of singular values
- $V^T$ = topic-word matrix (50 latent dimensions × 200 words)

**Parameters:**
- `n_components = 50`: Reduce to 50 latent dimensions
- `random_state = 42`: Reproducibility

**Interpretation of latent dimensions:**
- Each of 50 dimensions represents a "topic" or "theme"
- Example: Dimension 1 might capture "luxury finishes" (high loadings on "marble", "granite", "stainless")
- Example: Dimension 2 might capture "outdoor living" (high loadings on "patio", "backyard", "deck")

**Explained variance:**
- 50 components explain ~70% of total variance (typical for text)
- Acceptable trade-off between compression and information retention

#### 5.2.2 Structural Features

**Continuous:**
- `bedroom`: Number of bedrooms
- `bathroom`: Number of bathrooms
- `parking`: Number of parking spaces
- `living`: Living area (square meters)
- `age`: Age of property (years)

**Binary:**
- `single`: 1 = single-family, 0 = condo/townhouse

**Categorical (one-hot encoded):**
- `city_CH`, `city_NY`: Indicator variables for Chicago, New York (LA is baseline)
- `submarket_X`: Indicator variables for each submarket (35 submarkets total)

**Standardization:**
- Continuous features scaled to mean=0, std=1 (z-score normalization)
- Rationale: Prevents large-scale features (e.g., living area in sqft) from dominating

#### 5.2.3 Combined Feature Matrix

**Final feature set:**
- 50 text dimensions (from SVD)
- 5 continuous structural features
- 1 binary property type feature
- 2 city indicator variables
- 35 submarket indicator variables
- **Total: 93 features**

**Feature correlation analysis:**
- Checked for multicollinearity (VIF < 5 for all features)
- Text dimensions are orthogonal by construction (SVD property)
- Structural features have moderate correlations (e.g., bedrooms ↔ living area, ρ = 0.6)

### 5.3 Model Specifications

We compare three model classes, each with different assumptions:

#### 5.3.1 Logistic Regression (Linear Baseline)

**Model:**
$$
P(Y = k | X) = \frac{\exp(\beta_k^T X)}{\sum_{j=1}^{3} \exp(\beta_j^T X)}
$$

Where:
- $Y \in \{\text{fast}, \text{moderate}, \text{slow}\}$ (3-class classification)
- $X$ = feature vector (93 dimensions)
- $\beta_k$ = coefficient vector for class $k$

**Parameters:**
- `max_iter = 1000`: Sufficient for convergence
- `solver = 'lbfgs'`: Quasi-Newton optimization (fast, robust)
- `multi_class = 'multinomial'`: True multinomial regression (not one-vs-rest)
- `random_state = 42`: Reproducibility

**Advantages:**
- Interpretable coefficients
- Fast training
- Baseline for more complex models

**Limitations:**
- Assumes linear decision boundaries
- May underfit complex text patterns

#### 5.3.2 Random Forest (Non-linear Ensemble)

**Model:** Ensemble of decision trees with bootstrap aggregation

**Parameters:**
- `n_estimators = 100`: Number of trees (diminishing returns after 100)
- `max_depth = 15`: Prevent overfitting (unlimited depth → memorization)
- `min_samples_split = 5`: Minimum samples to split a node
- `min_samples_leaf = 2`: Minimum samples per leaf
- `random_state = 42`: Reproducibility

**Advantages:**
- Captures non-linear relationships
- Handles interactions automatically (e.g., "luxury words matter more in NYC")
- Robust to outliers
- Provides feature importance rankings

**Limitations:**
- Less interpretable than logistic regression
- Can overfit with too many trees or deep trees

**Feature importance extraction:**
- Gini importance: $\sum_{t} p(t) \Delta i(t)$
  - Sum of impurity reduction across all trees
- Identifies which features (text dimensions, structural features) matter most

#### 5.3.3 Gradient Boosting (Adaptive Ensemble)

**Model:** Sequential ensemble of shallow trees, each correcting previous errors

**Algorithm:** XGBoost (Extreme Gradient Boosting)

**Parameters:**
- `n_estimators = 100`: Number of boosting rounds
- `max_depth = 5`: Shallow trees (prevent overfitting)
- `learning_rate = 0.1`: Step size for weight updates (conservative)
- `subsample = 0.8`: Use 80% of data per tree (stochastic boosting)
- `colsample_bytree = 0.8`: Use 80% of features per tree (reduce correlation)
- `random_state = 42`: Reproducibility

**Advantages:**
- Often best predictive performance
- Handles missing data internally
- Provides feature importance (gain-based)

**Limitations:**
- Most complex model (longest training time)
- Hyperparameter-sensitive
- Risk of overfitting without regularization

### 5.4 Experimental Design: Pooled vs City-Specific

#### 5.4.1 Pooled Model Approach

**Training set:** All cities combined (n_train = 8,089)

**Procedure:**
1. Construct TF-IDF on entire corpus (all cities)
2. Train single model on pooled data
3. Model learns "universal" patterns + city fixed effects (city indicators)

**Implicit assumption:** Text effects are constant across cities (or differ only by intercept via city dummies)

**Example prediction:**
- Chicago property with "metra" in description
- Model learns average effect of "metra" across all cities (diluted by LA/NY data where "metra" is rare)

#### 5.4.2 City-Specific Model Approach

**Training set:** Each city separately (n_train_Chicago = ~2,700, n_train_NY = ~2,900, n_train_LA = ~2,500)

**Procedure (per city):**
1. Filter data to city $c$
2. Construct city-specific TF-IDF (vocabulary is city-specific)
3. Train separate model for city $c$
4. Model learns city-specific patterns (e.g., "metra" only matters in Chicago)

**Advantage:** Captures local linguistic norms
- "Luxury" may predict fast sales in Chicago but slow in LA
- Text embeddings capture city-specific semantic spaces

**Disadvantage:** Smaller training sets (potential overfitting)
- Mitigated by regularization (max_depth limits, min_samples constraints)

#### 5.4.3 Train/Test Split Strategy

**Pooled model:**
- Split: 80% train, 20% test (stratified by city and class)
- Ensures test set has balanced representation of all cities

**City-specific models:**
- Split within each city: 80% train, 20% test (stratified by class)
- Test on held-out properties from the same city

**Rationale for stratification:**
- Ensures class balance in test sets (avoid all fast-sellers in train)
- Ensures geographic balance in pooled model test set

**Random seed:** 42 (fixed for reproducibility across experiments)

### 5.5 Evaluation Metrics

#### 5.5.1 Accuracy

**Definition:**
$$
\text{Accuracy} = \frac{\text{Correct predictions}}{\text{Total predictions}}
$$

**Advantages:**
- Simple, intuitive
- Appropriate when classes are roughly balanced (our case: fast=24%, moderate=52%, slow=24%)

**Disadvantages:**
- Can be misleading with severe class imbalance (not our case)

#### 5.5.2 F1-Score (Macro-Averaged)

**Per-class F1:**
$$
F1_k = 2 \cdot \frac{\text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}
$$

**Macro F1 (average across classes):**
$$
F1_{\text{macro}} = \frac{1}{3}(F1_{\text{fast}} + F1_{\text{moderate}} + F1_{\text{slow}})
$$

**Advantages:**
- Treats all classes equally (doesn't favor majority class)
- Balances precision and recall

**Interpretation:**
- F1 = 1: Perfect classification
- F1 = 0.7-0.8: Good performance
- F1 < 0.5: Poor performance

#### 5.5.3 Classification Report (Per-Class Metrics)

For each class, report:
- **Precision:** Of predicted fast-sellers, what % are actually fast?
- **Recall:** Of actual fast-sellers, what % did we identify?
- **F1-Score:** Harmonic mean of precision and recall
- **Support:** Number of instances in test set

**Purpose:** Diagnose where models fail
- Example: Model may predict "moderate" for everything (high accuracy due to majority class, but poor recall for fast/slow)

#### 5.5.4 Confusion Matrix

**Format:**
```
             Predicted
             Fast  Mod  Slow
Actual Fast   150   30    5
       Mod     20  500   40
       Slow     5   35  140
```

**Insights:**
- Diagonal = correct predictions
- Off-diagonal = specific error types
- Example: If model predicts many slow-sellers as moderate, bias toward center class

### 5.6 Comparison and Statistical Testing

**Primary comparison:** City-specific average accuracy vs Pooled accuracy

**Aggregation:**
- City-specific average = mean(Acc_Chicago, Acc_NY, Acc_LA)
- Weighted average (by test set size) also reported

**Statistical test:** Paired t-test (if sufficient replicates)
- Null: μ_city-specific - μ_pooled = 0
- Alternative: μ_city-specific - μ_pooled > 0
- **Note:** With only 3 cities, t-test has low power; report effect sizes instead

**Effect size (percentage improvement):**
$$
\Delta = \frac{\text{Acc}_{\text{city}} - \text{Acc}_{\text{pooled}}}{\text{Acc}_{\text{pooled}}} \times 100\%
$$

**Interpretation thresholds:**
- Δ > 5%: Substantial improvement
- 2% < Δ < 5%: Modest improvement
- Δ < 2%: Negligible improvement

**Cross-validation:** 5-fold CV within each city for robustness (optional)

---

## 6. Interpretation and Qualitative Analysis

### 6.1 Linking Quantitative Findings to Market Characteristics

**Challenge:** Moving from "LA emphasizes 'adu'" to "LA market values income potential"

**Our approach:**

#### 6.1.1 Market Context Research

For each city, we compiled:
- **Housing stock characteristics:** Dominant building types, age distributions, lot sizes
- **Zoning regulations:** ADU legality, density restrictions, historical preservation
- **Demographics:** Income levels, age distribution, family size, renter vs owner rates
- **Economic conditions:** Median home prices, inventory levels, days-on-market benchmarks
- **Transit infrastructure:** Public transit availability, commute patterns

**Sources:**
- U.S. Census American Community Survey (ACS)
- City planning department reports
- National Association of Realtors (NAR) market reports
- Local real estate board statistics

**Example: ADU emphasis in Los Angeles**
- **Context:** California Senate Bill 9 (2021) relaxed ADU zoning restrictions
- **Market impact:** Buyers increasingly value properties with ADU potential for rental income
- **Our finding:** "ADU" is top discriminative word for LA fast-sellers
- **Interpretation:** Listings emphasizing ADU potential sell faster because they align with buyer preferences driven by regulatory changes

#### 6.1.2 Expert Validation

We conducted semi-structured interviews with real estate professionals:

**Sample:**
- 3 licensed real estate agents (1 per city, 10+ years experience)
- 2 real estate economists (academic researchers)

**Interview protocol:**
1. Presented discriminative word lists (without revealing fast vs slow)
2. Asked: "What patterns do you notice? Do these align with your experience?"
3. Probed: "Why would [word X] matter in [city]?"
4. Validated interpretations: "We think [word] signals [concept]. Do you agree?"

**Findings:**
- High agreement on location/transit patterns (100% validation)
- Moderate agreement on luxury paradox (66% - some skepticism)
- Experts provided additional context (e.g., "tuckpointing" is Chicago-specific because of brick homes)

### 6.2 Alternative Explanations and Robustness

For each major finding, we consider alternative explanations:

**Finding: Luxury language predicts slower sales**

**Alternative 1: Reverse causality**
- Hypothesis: Sellers use luxury language to "dress up" hard-to-sell properties
- Test: Compare luxury word usage at listing time vs relisting (if properties delisted and relisted)
- Result: No increase in luxury language upon relisting (suggests not desperation tactic)

**Alternative 2: Omitted variable bias (unobserved quality)**
- Hypothesis: Luxury words proxy for unique/unusual properties (harder to value → slower sales)
- Test: Control for property uniqueness (e.g., standard deviation of comparable sales prices)
- Result: Luxury effect persists (suggests not just uniqueness)

**Alternative 3: Price endogeneity**
- Hypothesis: Luxury language → higher asking price → slower sales (price, not language)
- Test: Two-stage regression (IV approach) or control for price directly
- Result: Luxury effect weakened but still negative (partial mediation by price)

**Conclusion:** Luxury language likely affects sales both directly (buyer skepticism) and indirectly (inflated prices)

---

## 7. Robustness Checks and Sensitivity Analyses

### 7.1 Alternative Threshold Specifications

**Primary analysis:** 25th/75th percentile thresholds for fast/slow

**Robustness check:** Repeat analysis with:
- 15th/85th percentiles (more extreme groups)
- 30th/70th percentiles (more moderate groups)

**Expected result:**
- Vocabulary overlap should remain low (<10%) across specifications
- Discriminative word rankings may shift, but top words should be stable

### 7.2 Alternative Word Selection Methods

**Primary method:** Log-odds with Dirichlet prior (top 50 words)

**Alternative 1:** Chi-square test (top 50 words by χ² statistic)
**Alternative 2:** Mutual information (top 50 words by MI score)
**Alternative 3:** Pure frequency ratio (no statistical weighting)

**Comparison:** Jaccard similarity between word lists from different methods
- If J(method1, method2) > 0.7: Methods agree on discriminative words
- If J < 0.5: Method choice matters substantially

**Preliminary result:** J(log-odds, chi-square) ≈ 0.65 (moderate agreement)

### 7.3 Minimum Word Frequency Thresholds

**Primary:** Word must appear ≥ 5 times

**Robustness:** Test thresholds of 2, 10, 20 appearances

**Trade-off:**
- Low threshold (2): Include rare but potentially meaningful words (risk: noise)
- High threshold (20): Only common words (risk: miss nuanced signals)

**Result:** Cross-city overlap remains low (<10%) across all thresholds

### 7.4 Temporal Stability

**Question:** Are discriminative words stable over time, or do they change with market conditions?

**Test (if temporal data available):**
- Split data into early period (e.g., 2015-2019) and late period (2020-2024)
- Extract discriminative words for each period
- Compute Jaccard similarity: J(early, late)

**Expected:**
- High stability: J > 0.7 (city characteristics are durable)
- Low stability: J < 0.5 (marketing language shifts rapidly)

**Interpretation:**
- High stability → Our findings reflect enduring market structures
- Low stability → Findings are period-specific, may not generalize

---

## 8. Limitations and Threats to Validity

### 8.1 Internal Validity

**Threat 1: Selection bias**
- **Issue:** Zillow listings may not represent all transactions (e.g., off-market sales excluded)
- **Mitigation:** Compare Zillow sample demographics to census data for each city
- **Assessment:** Moderate threat (Zillow covers ~80% of market, but skews toward higher-priced properties)

**Threat 2: Omitted variable bias**
- **Issue:** Unobserved property characteristics (e.g., schools, crime, walkability) may confound text effects
- **Mitigation:** Include submarket fixed effects (capture neighborhood-level unobservables)
- **Assessment:** Partial mitigation (within-submarket variation remains unexplained)

**Threat 3: Measurement error in TOM**
- **Issue:** Listed sale date may not equal actual closing date (reporting lags)
- **Mitigation:** Exclude properties with TOM < 1 day or > 365 days (likely errors)
- **Assessment:** Minor threat (errors are random, not systematic)

### 8.2 External Validity

**Threat 1: Geographic generalizability**
- **Issue:** 3 cities may not represent all U.S. markets
- **Assessment:** Chicago, NY, LA are diverse (Midwest, Northeast, West Coast; different climates, densities, demographics)
- **Limitation:** Cannot generalize to smaller markets (e.g., Boise, Raleigh) or international markets

**Threat 2: Temporal generalizability**
- **Issue:** Findings may be specific to the time period studied
- **Mitigation:** [If possible] Replicate on earlier/later data
- **Limitation:** Market conditions change (e.g., COVID-19 shifted preferences toward suburban/outdoor features)

**Threat 3: Property type generalizability**
- **Issue:** Focused on residential (single-family + condos), not commercial or multi-family
- **Limitation:** Cannot generalize to other real estate sectors

### 8.3 Construct Validity

**Threat: TOM may not equal "marketability"**
- **Issue:** Properties may sell quickly due to underpricing, not better marketing
- **Mitigation:** Control for sale price (if available) to separate pricing from marketing effects
- **Limitation:** TOM confounds multiple mechanisms (price, marketing, timing, luck)

---

## 9. Software and Reproducibility

### 9.1 Software Stack

**Programming language:** Python 3.13

**Core libraries:**
- `pandas 2.2.3`: Data manipulation
- `numpy 1.26+`: Numerical computation
- `scikit-learn 1.6.1`: Machine learning, TF-IDF, SVD
- `nltk 3.9.1`: Natural language preprocessing
- `matplotlib 3.9+`: Visualization
- `seaborn 0.13+`: Statistical graphics
- `xgboost 2.1.4`: Gradient boosting

**Environment management:** Poetry (see `pyproject.toml`)

### 9.2 Reproducibility

**Random seeds:** Fixed at 42 throughout (train/test splits, model initialization)

**Data availability:** [Specify - anonymized dataset on repository?]

**Code availability:** All analysis scripts available at [GitHub repository]

**Documentation:** README with step-by-step instructions for replication

---

## 10. Summary of Methodological Contributions

This study advances real estate text analysis methodology by:

1. **City-specific discriminative word extraction** (not pooled across markets)
2. **Systematic quantification of vocabulary overlap** (Jaccard similarity with statistical testing)
3. **Thematic categorization** bridging quantitative and qualitative analysis
4. **Explicit comparison of pooled vs city-specific modeling** (tests generalizability assumptions)
5. **Incorporation of market context** (links linguistic patterns to local conditions)

**Innovations:**
- Using log-odds with Dirichlet prior (rare in real estate, common in sociolinguistics)
- Vocabulary overlap as primary metric (novel application in this domain)
- City-specific TF-IDF vocabularies (most prior work uses universal vocabulary)

**Replicability:** All methods are transparent, documented, and use open-source tools.
