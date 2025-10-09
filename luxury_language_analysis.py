"""
Luxury Language Paradox Analysis

This script tests the counter-intuitive finding that luxury brand names
and high-end language correlate with SLOWER sales across all cities.

Research Question: Does luxury language predict longer time-on-market,
controlling for structural features and location?
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import json


class LuxuryLanguageAnalyzer:
    """Analyze the relationship between luxury language and time-on-market."""

    def __init__(self):
        # Define luxury brand names and high-end terms
        self.luxury_brands = [
            # Appliance brands
            'bertazzoni', 'miele', 'thermador', 'subzero', 'wolf', 'viking',
            'gaggenau', 'bosch', 'asko', 'liebherr',

            # Fixture/hardware brands
            'lutron', 'poggenpohl', 'dornbracht', 'grohe', 'duravit',
            'kallista', 'waterworks', 'brizo', 'kohler',

            # Technology brands
            'savant', 'crestron', 'control4', 'nest', 'sonos',

            # Materials/finishes
            'carrara', 'calacatta', 'statuary', 'calcutta', 'caesarstone',
            'silestone', 'dekton',

            # Design terms
            'reimagined', 'curated', 'bespoke', 'artisan', 'handcrafted',
            'seamlessly', 'meticulously', 'exquisite', 'pristine',

            # Architect/designer names
            'heatherwick', 'gehry', 'pei', 'meier', 'adjmi'
        ]

        self.luxury_adjectives = [
            'luxury', 'luxurious', 'upscale', 'high-end', 'highend',
            'premium', 'exclusive', 'prestigious', 'elite', 'sophisticated',
            'opulent', 'lavish', 'sumptuous'
        ]

        self.luxury_amenities = [
            'concierge', 'doorman', 'doorperson', 'valet', 'butler',
            'wine cellar', 'wine room', 'home theater', 'theatre',
            'infinity pool', 'lap pool', 'spa', 'sauna', 'steam room',
            'gym', 'fitness center', 'elevator', 'private elevator'
        ]

    def load_data(self, file_path="dataset/raw/2. zillow_cleaned.geojson"):
        """Load zillow data from geojson."""

        print(f"Loading data from {file_path}...")
        df = gpd.read_file(file_path)

        # Convert to regular DataFrame (drop geometry for analysis)
        df = pd.DataFrame(df.drop(columns='geometry'))

        print(f"Loaded {len(df)} properties")
        print(f"Cities: {df['city'].unique()}")
        print(f"\nColumns: {df.columns.tolist()}")

        return df

    def create_luxury_features(self, df):
        """Create luxury language count features."""

        print("\nCreating luxury language features...")

        # Ensure description is string
        df['description'] = df['description'].fillna('').astype(str)
        df['description_lower'] = df['description'].str.lower()

        # Count luxury brands
        df['luxury_brand_count'] = df['description_lower'].apply(
            lambda x: sum(1 for brand in self.luxury_brands if brand in x)
        )

        # Count luxury adjectives
        df['luxury_adj_count'] = df['description_lower'].apply(
            lambda x: sum(1 for adj in self.luxury_adjectives if adj in x)
        )

        # Count luxury amenities
        df['luxury_amenity_count'] = df['description_lower'].apply(
            lambda x: sum(1 for amenity in self.luxury_amenities if amenity in x)
        )

        # Total luxury word count
        df['luxury_total_count'] = (
            df['luxury_brand_count'] +
            df['luxury_adj_count'] +
            df['luxury_amenity_count']
        )

        # Binary indicator: has any luxury language
        df['has_luxury_language'] = (df['luxury_total_count'] > 0).astype(int)

        print(f"Properties with luxury language: {df['has_luxury_language'].sum()} ({df['has_luxury_language'].mean()*100:.1f}%)")
        print(f"\nLuxury word count distribution:")
        print(df['luxury_total_count'].describe())

        return df

    def descriptive_analysis(self, df, output_dir="result/geographic_analysis"):
        """Descriptive statistics and visualizations."""

        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*70)
        print("DESCRIPTIVE ANALYSIS: Luxury Language and TOM")
        print("="*70)

        # 1. TOM by luxury language presence
        print("\nMean TOM by luxury language presence:")
        tom_by_luxury = df.groupby('has_luxury_language')['duration'].agg([
            'count', 'mean', 'median', 'std'
        ])
        tom_by_luxury.index = ['No Luxury Language', 'Has Luxury Language']
        print(tom_by_luxury)

        # T-test
        no_luxury_tom = df[df['has_luxury_language'] == 0]['duration']
        luxury_tom = df[df['has_luxury_language'] == 1]['duration']
        t_stat, p_value = stats.ttest_ind(luxury_tom, no_luxury_tom)
        print(f"\nT-test: t={t_stat:.3f}, p={p_value:.4f}")

        if p_value < 0.05:
            print("✅ Difference is statistically significant (p < 0.05)")

        # 2. By city
        print("\n" + "-"*70)
        print("Mean TOM by city and luxury language:")
        city_luxury = df.groupby(['city', 'has_luxury_language'])['duration'].mean().unstack()
        city_luxury.columns = ['No Luxury', 'Has Luxury']
        city_luxury['Difference'] = city_luxury['Has Luxury'] - city_luxury['No Luxury']
        print(city_luxury)

        # 3. Correlation between luxury count and TOM
        print("\n" + "-"*70)
        print("Correlation: Luxury word count vs TOM")
        correlation = df['luxury_total_count'].corr(df['duration'])
        print(f"Pearson r = {correlation:.3f}")

        # Spearman (robust to outliers)
        spearman_r, spearman_p = stats.spearmanr(df['luxury_total_count'], df['duration'])
        print(f"Spearman ρ = {spearman_r:.3f}, p = {spearman_p:.4f}")

        # 4. Visualizations
        self._create_visualizations(df, output_dir)

        # 5. Examples of luxury listings
        print("\n" + "-"*70)
        print("Examples of high luxury language listings (top 5):")
        top_luxury = df.nlargest(5, 'luxury_total_count')[
            ['zpid', 'city', 'duration', 'luxury_total_count', 'description']
        ]
        for idx, row in top_luxury.iterrows():
            print(f"\nZPID: {row['zpid']}, City: {row['city']}, TOM: {row['duration']} days")
            print(f"Luxury word count: {row['luxury_total_count']}")
            print(f"Description (first 200 chars): {row['description'][:200]}...")

    def _create_visualizations(self, df, output_dir):
        """Create visualizations for luxury language analysis."""

        # 1. Box plot: TOM by luxury language
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Overall
        df['Luxury Language'] = df['has_luxury_language'].map({0: 'No', 1: 'Yes'})
        sns.boxplot(data=df, x='Luxury Language', y='duration', ax=axes[0])
        axes[0].set_title('Time-on-Market by Luxury Language Presence', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Days on Market')
        axes[0].set_ylim(0, 300)  # Cap outliers for visibility

        # By city
        sns.boxplot(data=df, x='city', y='duration', hue='Luxury Language', ax=axes[1])
        axes[1].set_title('Time-on-Market by City and Luxury Language', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Days on Market')
        axes[1].set_xlabel('City')
        axes[1].set_ylim(0, 300)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/luxury_language_boxplot.png", dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: {output_dir}/luxury_language_boxplot.png")
        plt.close()

        # 2. Scatter plot: Luxury word count vs TOM
        fig, ax = plt.subplots(figsize=(10, 6))

        for city in df['city'].unique():
            city_data = df[df['city'] == city]
            ax.scatter(
                city_data['luxury_total_count'],
                city_data['duration'],
                alpha=0.3,
                label=city,
                s=20
            )

        # Add regression line (overall)
        x = df['luxury_total_count']
        y = df['duration']
        mask = ~(np.isnan(x) | np.isnan(y))
        z = np.polyfit(x[mask], y[mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Trend (slope={z[0]:.1f})')

        ax.set_xlabel('Luxury Word Count', fontsize=11)
        ax.set_ylabel('Days on Market', fontsize=11)
        ax.set_title('Relationship: Luxury Language and Time-on-Market', fontweight='bold', fontsize=13)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 350)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/luxury_language_scatter.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_dir}/luxury_language_scatter.png")
        plt.close()

        # 3. Distribution of luxury word counts
        fig, ax = plt.subplots(figsize=(10, 6))

        counts = df['luxury_total_count'].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color='steelblue', edgecolor='black')
        ax.set_xlabel('Number of Luxury Words in Listing', fontsize=11)
        ax.set_ylabel('Number of Properties', fontsize=11)
        ax.set_title('Distribution of Luxury Language Usage', fontweight='bold', fontsize=13)
        ax.set_xlim(-0.5, min(15, counts.index.max()) + 0.5)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/luxury_language_distribution.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_dir}/luxury_language_distribution.png")
        plt.close()

    def regression_analysis(self, df, output_dir="result/geographic_analysis"):
        """Regression analysis: Luxury language → TOM, controlling for structural features."""

        print("\n" + "="*70)
        print("REGRESSION ANALYSIS")
        print("="*70)

        # Prepare data
        df_reg = df.copy()

        # Create city dummies (LA is baseline)
        df_reg = pd.get_dummies(df_reg, columns=['city'], prefix='city', drop_first=False)

        # Ensure we have the necessary columns
        required_cols = ['duration', 'luxury_total_count', 'bedroom', 'bathroom',
                        'parking', 'age', 'living', 'single']

        # Check for missing required columns
        missing_cols = [col for col in required_cols if col not in df_reg.columns]
        if missing_cols:
            print(f"⚠️  Warning: Missing columns: {missing_cols}")
            print("Available columns:", df_reg.columns.tolist())
            return

        # Remove rows with missing values
        df_reg = df_reg.dropna(subset=required_cols + ['city_CH', 'city_NY', 'city_LA'])

        print(f"\nSample size: {len(df_reg)} properties")

        # Model 1: Bivariate (luxury count only)
        print("\n" + "-"*70)
        print("MODEL 1: Bivariate (Luxury Language Only)")
        print("-"*70)

        model1 = ols('duration ~ luxury_total_count', data=df_reg).fit()
        print(model1.summary())

        # Model 2: + Structural controls
        print("\n" + "-"*70)
        print("MODEL 2: + Structural Controls")
        print("-"*70)

        model2 = ols(
            'duration ~ luxury_total_count + bedroom + bathroom + parking + age + living + single',
            data=df_reg
        ).fit()
        print(model2.summary())

        # Model 3: + City fixed effects
        print("\n" + "-"*70)
        print("MODEL 3: + City Fixed Effects")
        print("-"*70)

        model3 = ols(
            'duration ~ luxury_total_count + bedroom + bathroom + parking + age + living + single + city_CH + city_NY',
            data=df_reg
        ).fit()
        print(model3.summary())

        # Model 4: City × Luxury interactions
        print("\n" + "-"*70)
        print("MODEL 4: City × Luxury Interactions")
        print("-"*70)

        # Create interaction terms
        df_reg['luxury_x_CH'] = df_reg['luxury_total_count'] * df_reg['city_CH']
        df_reg['luxury_x_NY'] = df_reg['luxury_total_count'] * df_reg['city_NY']

        model4 = ols(
            'duration ~ luxury_total_count + luxury_x_CH + luxury_x_NY + bedroom + bathroom + parking + age + living + single + city_CH + city_NY',
            data=df_reg
        ).fit()
        print(model4.summary())

        # Interpretation
        print("\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)

        coef1 = model1.params['luxury_total_count']
        pval1 = model1.pvalues['luxury_total_count']

        coef3 = model3.params['luxury_total_count']
        pval3 = model3.pvalues['luxury_total_count']

        print(f"\nBivariate effect: Each luxury word adds {coef1:.2f} days (p={pval1:.4f})")
        print(f"Controlled effect: Each luxury word adds {coef3:.2f} days (p={pval3:.4f})")

        if coef3 > 0 and pval3 < 0.05:
            print("\n✅ LUXURY LANGUAGE PARADOX CONFIRMED:")
            print(f"   Luxury language predicts LONGER time-on-market (+{coef3:.1f} days per word)")
            print(f"   This effect persists even after controlling for structural features and city")
        elif coef3 > 0 and pval3 >= 0.05:
            print("\n⚠️  LUXURY LANGUAGE PARADOX (Weak Evidence):")
            print(f"   Luxury language shows positive association (+{coef3:.1f} days)")
            print(f"   But effect is not statistically significant (p={pval3:.4f})")
        else:
            print("\n❌ LUXURY LANGUAGE PARADOX NOT SUPPORTED:")
            print(f"   Luxury language does not predict longer TOM")

        # City-specific effects (from Model 4)
        print("\n" + "-"*70)
        print("City-Specific Luxury Effects (Model 4):")
        print("-"*70)

        la_effect = model4.params['luxury_total_count']
        ch_effect = la_effect + model4.params['luxury_x_CH']
        ny_effect = la_effect + model4.params['luxury_x_NY']

        print(f"Los Angeles: {la_effect:.2f} days per luxury word")
        print(f"Chicago: {ch_effect:.2f} days per luxury word")
        print(f"New York: {ny_effect:.2f} days per luxury word")

        # Save results
        results_summary = {
            'model1_coef': float(coef1),
            'model1_pval': float(pval1),
            'model3_coef': float(coef3),
            'model3_pval': float(pval3),
            'city_effects': {
                'LA': float(la_effect),
                'CH': float(ch_effect),
                'NY': float(ny_effect)
            }
        }

        with open(f"{output_dir}/luxury_regression_results.json", 'w') as f:
            json.dump(results_summary, f, indent=2)

        print(f"\n✅ Saved results to {output_dir}/luxury_regression_results.json")

        return model1, model2, model3, model4

    def brand_specific_analysis(self, df):
        """Analyze which specific luxury brands/terms are most problematic."""

        print("\n" + "="*70)
        print("BRAND-SPECIFIC ANALYSIS")
        print("="*70)

        # For each luxury term, compute mean TOM difference
        term_effects = []

        all_terms = self.luxury_brands + self.luxury_adjectives + self.luxury_amenities

        for term in all_terms:
            has_term = df['description_lower'].str.contains(term, na=False)
            count = has_term.sum()

            if count < 10:  # Skip rare terms
                continue

            tom_with = df[has_term]['duration'].mean()
            tom_without = df[~has_term]['duration'].mean()
            difference = tom_with - tom_without

            # T-test
            _, p_value = stats.ttest_ind(
                df[has_term]['duration'],
                df[~has_term]['duration']
            )

            term_effects.append({
                'term': term,
                'count': count,
                'tom_with_term': tom_with,
                'tom_without_term': tom_without,
                'difference': difference,
                'p_value': p_value
            })

        # Create DataFrame and sort by difference
        term_df = pd.DataFrame(term_effects).sort_values('difference', ascending=False)

        print("\nTop 10 terms that INCREASE TOM the most:")
        print(term_df.head(10)[['term', 'count', 'difference', 'p_value']].to_string(index=False))

        print("\nTop 10 terms that DECREASE TOM the most:")
        print(term_df.tail(10)[['term', 'count', 'difference', 'p_value']].to_string(index=False))

        return term_df


if __name__ == "__main__":
    print("="*70)
    print("LUXURY LANGUAGE PARADOX ANALYSIS")
    print("="*70)
    print()

    analyzer = LuxuryLanguageAnalyzer()

    # Load data
    df = analyzer.load_data()

    # Create luxury features
    df = analyzer.create_luxury_features(df)

    # Descriptive analysis
    analyzer.descriptive_analysis(df)

    # Regression analysis
    models = analyzer.regression_analysis(df)

    # Brand-specific analysis
    term_effects = analyzer.brand_specific_analysis(df)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey outputs:")
    print("- result/geographic_analysis/luxury_language_boxplot.png")
    print("- result/geographic_analysis/luxury_language_scatter.png")
    print("- result/geographic_analysis/luxury_language_distribution.png")
    print("- result/geographic_analysis/luxury_regression_results.json")
