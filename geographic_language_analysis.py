"""
Geographic Variation in Real Estate Marketing Language Analysis

This script analyzes city-specific discriminative words to understand
how real estate marketing language varies across different markets.
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Set
import json


class GeographicLanguageAnalyzer:
    """Analyze geographic variation in real estate marketing language."""

    def __init__(self, word_counts_dir: str = "dataset/word_counts"):
        self.word_counts_dir = word_counts_dir
        self.cities = ["CH", "NY", "LA"]
        self.city_names = {
            "CH": "Chicago",
            "NY": "New York",
            "LA": "Los Angeles"
        }
        self.property_types = {0: "Single Family", 1: "Condo/Townhouse"}
        self.percentages = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

        # Storage for loaded data
        self.fast_words = {}  # {(city, type, pct): [(word, zscore), ...]}
        self.slow_words = {}  # {(city, type, pct): [(word, zscore), ...]}

    def load_discriminative_words(self, n_words: int = 50, percentage: float = 0.25):
        """Load discriminative words for all cities and property types."""

        for city in self.cities:
            for prop_type in [0, 1]:
                # Load fast-selling words (group_0)
                fast_file = os.path.join(
                    self.word_counts_dir,
                    str(percentage),
                    f"{city}_{prop_type}_group_0_zscore.csv"
                )

                # Load slow-selling words (group_2)
                slow_file = os.path.join(
                    self.word_counts_dir,
                    str(percentage),
                    f"{city}_{prop_type}_group_2_zscore.csv"
                )

                try:
                    fast_df = pd.read_csv(fast_file, header=None, names=["word", "zscore"])
                    self.fast_words[(city, prop_type, percentage)] = list(
                        zip(fast_df['word'].head(n_words), fast_df['zscore'].head(n_words))
                    )

                    slow_df = pd.read_csv(slow_file, header=None, names=["word", "zscore"])
                    self.slow_words[(city, prop_type, percentage)] = list(
                        zip(slow_df['word'].head(n_words), slow_df['zscore'].head(n_words))
                    )
                except FileNotFoundError as e:
                    print(f"Warning: Could not load file: {e}")

        print(f"Loaded discriminative words for {len(self.fast_words)} city/type combinations")

    def calculate_vocabulary_overlap(self, percentage: float = 0.25) -> pd.DataFrame:
        """Calculate Jaccard similarity between cities' discriminative vocabularies."""

        results = []

        for prop_type in [0, 1]:
            for sale_type in ['fast', 'slow']:
                words_dict = self.fast_words if sale_type == 'fast' else self.slow_words

                # Get word sets for each city
                city_words = {}
                for city in self.cities:
                    key = (city, prop_type, percentage)
                    if key in words_dict:
                        city_words[city] = set([w for w, _ in words_dict[key]])

                # Calculate pairwise Jaccard similarity
                for i, city1 in enumerate(self.cities):
                    for city2 in self.cities[i+1:]:
                        if city1 in city_words and city2 in city_words:
                            intersection = len(city_words[city1] & city_words[city2])
                            union = len(city_words[city1] | city_words[city2])
                            jaccard = intersection / union if union > 0 else 0

                            results.append({
                                'Property Type': self.property_types[prop_type],
                                'Sale Speed': sale_type.capitalize(),
                                'City 1': self.city_names[city1],
                                'City 2': self.city_names[city2],
                                'Jaccard Similarity': jaccard,
                                'Shared Words': intersection,
                                'Total Unique Words': union,
                                'Overlap %': (intersection / min(
                                    len(city_words[city1]),
                                    len(city_words[city2])
                                ) * 100) if min(len(city_words[city1]), len(city_words[city2])) > 0 else 0
                            })

        return pd.DataFrame(results)

    def get_unique_city_words(self, percentage: float = 0.25, top_n: int = 20) -> Dict:
        """Identify words unique to each city (appearing in one city but not others)."""

        unique_words = {}

        for prop_type in [0, 1]:
            for sale_type in ['fast', 'slow']:
                words_dict = self.fast_words if sale_type == 'fast' else self.slow_words

                # Get word sets for each city
                city_word_sets = {}
                city_word_scores = {}

                for city in self.cities:
                    key = (city, prop_type, percentage)
                    if key in words_dict:
                        city_word_sets[city] = set([w for w, _ in words_dict[key]])
                        city_word_scores[city] = {w: s for w, s in words_dict[key]}

                # Find unique words for each city
                for city in self.cities:
                    if city not in city_word_sets:
                        continue

                    other_cities = [c for c in self.cities if c != city]
                    other_words = set()
                    for other_city in other_cities:
                        if other_city in city_word_sets:
                            other_words |= city_word_sets[other_city]

                    # Words in this city but not in others
                    unique = city_word_sets[city] - other_words

                    # Sort by z-score
                    unique_scored = [
                        (w, city_word_scores[city][w]) for w in unique
                        if w in city_word_scores[city]
                    ]
                    unique_scored.sort(key=lambda x: x[1], reverse=True)

                    key_label = f"{self.city_names[city]}_{self.property_types[prop_type]}_{sale_type}"
                    unique_words[key_label] = unique_scored[:top_n]

        return unique_words

    def categorize_words_thematically(self, percentage: float = 0.25) -> pd.DataFrame:
        """Categorize discriminative words into thematic categories."""

        # Define category keywords (you can expand these)
        categories = {
            'Location/Neighborhood': [
                'downtown', 'neighborhood', 'district', 'area', 'block', 'street',
                'ave', 'avenue', 'road', 'drive', 'place', 'way', 'north', 'south',
                'east', 'west', 'central', 'uptown', 'midtown'
            ],
            'Transit/Accessibility': [
                'metra', 'cta', 'subway', 'train', 'bus', 'transit', 'transportation',
                'walk', 'walkable', 'walkability', 'bike', 'commute', 'accessible',
                'station', 'stop', 'line'
            ],
            'Property Features': [
                'bedroom', 'bathroom', 'kitchen', 'living', 'dining', 'room', 'space',
                'floor', 'ceiling', 'window', 'door', 'closet', 'storage', 'garage',
                'parking', 'basement', 'attic', 'patio', 'deck', 'balcony', 'yard',
                'garden', 'backyard', 'frontyard'
            ],
            'Condition/Quality': [
                'new', 'renovated', 'updated', 'upgraded', 'remodeled', 'modern',
                'contemporary', 'luxury', 'pristine', 'immaculate', 'mint', 'turnkey',
                'move-in', 'ready', 'finished', 'polished', 'maintained', 'restored',
                'refurbished', 'rehab', 'fixer', 'as-is', 'tlc', 'potential'
            ],
            'Amenities': [
                'pool', 'spa', 'gym', 'fitness', 'doorman', 'concierge', 'elevator',
                'laundry', 'washer', 'dryer', 'dishwasher', 'ac', 'heating', 'hvac',
                'central-air', 'fireplace', 'hardwood', 'carpet', 'tile', 'granite',
                'marble', 'stainless', 'appliances', 'wifi', 'smart-home'
            ],
            'Outdoor/Views': [
                'view', 'views', 'skyline', 'waterfront', 'lakefront', 'river',
                'ocean', 'beach', 'mountain', 'park', 'green', 'trees', 'nature',
                'outdoor', 'patio', 'terrace', 'rooftop', 'deck', 'sunroom'
            ],
            'Investment/Market': [
                'investment', 'opportunity', 'potential', 'investor', 'rental',
                'income', 'cash-flow', 'roi', 'appreciation', 'equity', 'bidding',
                'offer', 'price', 'value', 'deal', 'motivated', 'must-see',
                'won\'t-last', 'hot', 'priced-to-sell'
            ],
            'School/Family': [
                'school', 'schools', 'district', 'rated', 'family', 'kid',
                'children', 'playground', 'park', 'safe', 'quiet', 'residential',
                'neighborhood', 'community'
            ],
            'Specific Locations': [],  # Will be populated with specific neighborhood names
        }

        results = []

        for prop_type in [0, 1]:
            for sale_type in ['fast', 'slow']:
                words_dict = self.fast_words if sale_type == 'fast' else self.slow_words

                for city in self.cities:
                    key = (city, prop_type, percentage)
                    if key not in words_dict:
                        continue

                    words_with_scores = words_dict[key]

                    for word, zscore in words_with_scores:
                        # Find category
                        word_lower = word.lower().replace('_', '-')
                        category = 'Specific Locations'  # Default

                        for cat_name, keywords in categories.items():
                            if any(kw in word_lower for kw in keywords):
                                category = cat_name
                                break

                        results.append({
                            'City': self.city_names[city],
                            'Property Type': self.property_types[prop_type],
                            'Sale Speed': sale_type.capitalize(),
                            'Word': word,
                            'Z-Score': zscore,
                            'Category': category
                        })

        return pd.DataFrame(results)

    def analyze_category_distribution(self, categorized_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze distribution of word categories across cities."""

        distribution = categorized_df.groupby([
            'City', 'Property Type', 'Sale Speed', 'Category'
        ]).size().reset_index(name='Word Count')

        # Calculate percentages within each city/type/speed group
        total_by_group = categorized_df.groupby([
            'City', 'Property Type', 'Sale Speed'
        ]).size().reset_index(name='Total')

        distribution = distribution.merge(
            total_by_group,
            on=['City', 'Property Type', 'Sale Speed']
        )
        distribution['Percentage'] = (distribution['Word Count'] / distribution['Total'] * 100).round(2)

        return distribution

    def export_analysis_summary(self, output_dir: str = "result/geographic_analysis"):
        """Export comprehensive analysis summary."""

        os.makedirs(output_dir, exist_ok=True)

        # 1. Vocabulary overlap analysis
        overlap_df = self.calculate_vocabulary_overlap()
        overlap_df.to_csv(f"{output_dir}/vocabulary_overlap.csv", index=False)
        print(f"Saved vocabulary overlap analysis to {output_dir}/vocabulary_overlap.csv")

        # 2. Unique words per city
        unique_words = self.get_unique_city_words()
        with open(f"{output_dir}/unique_city_words.json", 'w') as f:
            # Convert to serializable format
            unique_serializable = {
                k: [(w, float(s)) for w, s in v] for k, v in unique_words.items()
            }
            json.dump(unique_serializable, f, indent=2)
        print(f"Saved unique city words to {output_dir}/unique_city_words.json")

        # 3. Thematic categorization
        categorized_df = self.categorize_words_thematically()
        categorized_df.to_csv(f"{output_dir}/word_categories.csv", index=False)
        print(f"Saved word categorization to {output_dir}/word_categories.csv")

        # 4. Category distribution
        distribution_df = self.analyze_category_distribution(categorized_df)
        distribution_df.to_csv(f"{output_dir}/category_distribution.csv", index=False)
        print(f"Saved category distribution to {output_dir}/category_distribution.csv")

        return {
            'overlap': overlap_df,
            'unique_words': unique_words,
            'categorized': categorized_df,
            'distribution': distribution_df
        }


def create_visualizations(analyzer: GeographicLanguageAnalyzer, output_dir: str = "result/geographic_analysis"):
    """Create visualizations for geographic language analysis."""

    os.makedirs(output_dir, exist_ok=True)

    # 1. Vocabulary overlap heatmap
    overlap_df = analyzer.calculate_vocabulary_overlap()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Vocabulary Overlap Across Cities (Jaccard Similarity)', fontsize=16, fontweight='bold')

    for idx, (prop_type, sale_type) in enumerate([
        ('Single Family', 'Fast'),
        ('Single Family', 'Slow'),
        ('Condo/Townhouse', 'Fast'),
        ('Condo/Townhouse', 'Slow')
    ]):
        ax = axes[idx // 2, idx % 2]

        subset = overlap_df[
            (overlap_df['Property Type'] == prop_type) &
            (overlap_df['Sale Speed'] == sale_type)
        ]

        if not subset.empty:
            # Create matrix for heatmap
            cities = ['Chicago', 'New York', 'Los Angeles']
            matrix = np.zeros((3, 3))

            for _, row in subset.iterrows():
                i = cities.index(row['City 1'])
                j = cities.index(row['City 2'])
                matrix[i, j] = row['Jaccard Similarity']
                matrix[j, i] = row['Jaccard Similarity']

            # Set diagonal to 1 (self-similarity)
            np.fill_diagonal(matrix, 1.0)

            sns.heatmap(
                matrix,
                annot=True,
                fmt='.3f',
                xticklabels=cities,
                yticklabels=cities,
                cmap='RdYlGn',
                vmin=0,
                vmax=1,
                ax=ax,
                cbar_kws={'label': 'Jaccard Similarity'}
            )
            ax.set_title(f'{prop_type} - {sale_type}-Selling', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/vocabulary_overlap_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_dir}/vocabulary_overlap_heatmap.png")
    plt.close()

    # 2. Category distribution by city
    categorized_df = analyzer.categorize_words_thematically()
    distribution_df = analyzer.analyze_category_distribution(categorized_df)

    # Focus on fast-selling single family homes for clearer visualization
    subset = distribution_df[
        (distribution_df['Property Type'] == 'Single Family') &
        (distribution_df['Sale Speed'] == 'Fast')
    ]

    if not subset.empty:
        fig, ax = plt.subplots(figsize=(14, 8))

        # Pivot for grouped bar chart
        pivot_data = subset.pivot(
            index='Category',
            columns='City',
            values='Percentage'
        ).fillna(0)

        pivot_data.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Category Distribution: Fast-Selling Single Family Homes by City',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Word Category', fontsize=12)
        ax.set_ylabel('Percentage of Discriminative Words', fontsize=12)
        ax.legend(title='City', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/category_distribution_by_city.png", dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_dir}/category_distribution_by_city.png")
        plt.close()

    # 3. Top unique words per city
    unique_words = analyzer.get_unique_city_words(top_n=15)

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('Top City-Specific Words (Fast-Selling Single Family)', fontsize=16, fontweight='bold')

    for idx, city_code in enumerate(['CH', 'NY', 'LA']):
        city_name = analyzer.city_names[city_code]
        key = f"{city_name}_Single Family_fast"

        if key in unique_words and unique_words[key]:
            words, scores = zip(*unique_words[key][:15])

            axes[idx].barh(range(len(words)), scores, color=f'C{idx}')
            axes[idx].set_yticks(range(len(words)))
            axes[idx].set_yticklabels(words)
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel('Z-Score', fontsize=11)
            axes[idx].set_title(city_name, fontsize=13, fontweight='bold')
            axes[idx].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/unique_words_by_city.png", dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_dir}/unique_words_by_city.png")
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("Geographic Variation in Real Estate Marketing Language Analysis")
    print("=" * 70)
    print()

    # Initialize analyzer
    analyzer = GeographicLanguageAnalyzer()

    # Load discriminative words
    print("Loading discriminative words...")
    analyzer.load_discriminative_words(n_words=50, percentage=0.25)
    print()

    # Export comprehensive analysis
    print("Generating analysis summaries...")
    results = analyzer.export_analysis_summary()
    print()

    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(analyzer)
    print()

    # Print summary statistics
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print()

    print("Vocabulary Overlap (Jaccard Similarity):")
    print(results['overlap'].groupby(['Property Type', 'Sale Speed'])['Jaccard Similarity'].agg([
        ('Mean', 'mean'),
        ('Std', 'std'),
        ('Min', 'min'),
        ('Max', 'max')
    ]).round(3))
    print()

    print("Category Distribution Summary:")
    print(results['distribution'].groupby('Category')['Word Count'].sum().sort_values(ascending=False))
    print()

    print("Analysis complete! Check the 'result/geographic_analysis' directory for outputs.")
