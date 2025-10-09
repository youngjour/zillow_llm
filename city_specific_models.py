"""
City-Specific ML Models for Time-on-Market Prediction

This script tests whether city-specific models outperform pooled models,
demonstrating the value of accounting for geographic language variation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import os
import json

from sampling import split_data
from const import sales_speed


class CitySpecificPredictor:
    """Compare pooled vs city-specific models for TOM prediction."""

    def __init__(self, threshold_pct: float = 0.25, threshold_idx: int = 4):
        self.threshold_pct = threshold_pct
        self.threshold_idx = threshold_idx
        self.results = {}

    def load_and_prepare_data(self):
        """Load data and create city-specific text features."""

        # Load zillow data
        zillow = pd.read_csv("dataset/2. zillow_cleaned.csv")

        # Create target variable
        zillow['tom_class'] = zillow.apply(
            lambda row: self._classify_tom(row), axis=1
        )

        # Remove any rows without class
        zillow = zillow.dropna(subset=['tom_class', 'description'])

        print(f"Total samples: {len(zillow)}")
        print(f"Class distribution:\n{zillow['tom_class'].value_counts()}")

        return zillow

    def _classify_tom(self, row):
        """Classify TOM into fast/moderate/slow based on city-specific thresholds."""
        try:
            city = row['city']
            single = row['single']
            duration = row['duration']

            fast_threshold = sales_speed[city][single]["fast"][self.threshold_idx]
            slow_threshold = sales_speed[city][single]["slow"][self.threshold_idx]

            if duration <= fast_threshold:
                return 'fast'
            elif duration >= slow_threshold:
                return 'slow'
            else:
                return 'moderate'
        except (KeyError, IndexError):
            return None

    def create_text_features(self, df, city=None):
        """Create TF-IDF text features."""

        # Use city-specific corpus if specified
        if city:
            text_data = df[df['city'] == city]['description']
        else:
            text_data = df['description']

        # TF-IDF with moderate parameters
        vectorizer = TfidfVectorizer(
            max_features=200,
            min_df=3,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words='english'
        )

        tfidf_matrix = vectorizer.fit_transform(text_data)

        # Dimensionality reduction
        svd = TruncatedSVD(n_components=50, random_state=42)
        text_features = svd.fit_transform(tfidf_matrix)

        # Create DataFrame
        text_df = pd.DataFrame(
            text_features,
            index=text_data.index,
            columns=[f'text_dim_{i}' for i in range(50)]
        )

        return text_df, vectorizer, svd

    def create_combined_features(self, df, text_features):
        """Combine structural and text features."""

        # Structural features
        structural_cols = ['bedroom', 'bathroom', 'parking', 'age', 'living', 'single']

        # One-hot encode categorical
        df_encoded = pd.get_dummies(df[['city', 'submarket']], prefix=['city', 'submarket'])

        # Combine
        X = pd.concat([
            df[structural_cols].reset_index(drop=True),
            df_encoded.reset_index(drop=True),
            text_features.reset_index(drop=True)
        ], axis=1)

        return X

    def train_pooled_model(self, X_train, y_train, X_test, y_test):
        """Train a model on pooled data from all cities."""

        print("\n" + "="*70)
        print("POOLED MODEL (All Cities Combined)")
        print("="*70)

        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        }

        results = {}

        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')

            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score (macro): {f1_macro:.4f}")
            print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

            results[model_name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'predictions': y_pred
            }

        return results

    def train_city_specific_models(self, df):
        """Train separate models for each city."""

        print("\n" + "="*70)
        print("CITY-SPECIFIC MODELS")
        print("="*70)

        cities = df['city'].unique()
        city_results = {}

        for city in cities:
            print(f"\n{'='*70}")
            print(f"CITY: {city}")
            print('='*70)

            # Filter data for this city
            city_df = df[df['city'] == city].copy()

            print(f"Samples for {city}: {len(city_df)}")
            print(f"Class distribution:\n{city_df['tom_class'].value_counts()}")

            # Skip if too few samples
            if len(city_df) < 100:
                print(f"Skipping {city} - insufficient data")
                continue

            # Create city-specific text features
            text_features, vectorizer, svd = self.create_text_features(city_df, city=city)

            # Combine features
            X = self.create_combined_features(city_df, text_features)
            y = city_df['tom_class']

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train models
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
            }

            city_model_results = {}

            for model_name, model in models.items():
                print(f"\nTraining {model_name}...")

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                f1_macro = f1_score(y_test, y_pred, average='macro')

                print(f"Accuracy: {accuracy:.4f}")
                print(f"F1-Score (macro): {f1_macro:.4f}")

                city_model_results[model_name] = {
                    'accuracy': accuracy,
                    'f1_macro': f1_macro
                }

            city_results[city] = city_model_results

        return city_results

    def compare_approaches(self, df):
        """Compare pooled vs city-specific modeling approaches."""

        print("\n" + "="*70)
        print("EXPERIMENT: Pooled vs City-Specific Models")
        print("="*70)

        # Prepare data
        print("\nPreparing data...")

        # Create text features on full dataset (for pooled model)
        text_features_full, _, _ = self.create_text_features(df)

        # Combine features
        X_full = self.create_combined_features(df, text_features_full)
        y_full = df['tom_class']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
        )

        # Also keep track of which city each test sample is from
        test_indices = X_test.index
        test_cities = df.loc[test_indices, 'city']

        # 1. Train pooled model
        pooled_results = self.train_pooled_model(X_train, y_train, X_test, y_test)

        # 2. Train city-specific models
        city_results = self.train_city_specific_models(df)

        # 3. Compare results
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)

        comparison_df = pd.DataFrame({
            'Model': [],
            'Pooled Accuracy': [],
            'Pooled F1 (macro)': [],
            'City-Specific Avg Accuracy': [],
            'City-Specific Avg F1': [],
            'Improvement (Accuracy)': [],
            'Improvement (F1)': []
        })

        for model_name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting']:
            pooled_acc = pooled_results[model_name]['accuracy']
            pooled_f1 = pooled_results[model_name]['f1_macro']

            city_accs = [city_results[city][model_name]['accuracy']
                         for city in city_results.keys()]
            city_f1s = [city_results[city][model_name]['f1_macro']
                        for city in city_results.keys()]

            avg_city_acc = np.mean(city_accs)
            avg_city_f1 = np.mean(city_f1s)

            improvement_acc = ((avg_city_acc - pooled_acc) / pooled_acc) * 100
            improvement_f1 = ((avg_city_f1 - pooled_f1) / pooled_f1) * 100

            comparison_df = pd.concat([comparison_df, pd.DataFrame({
                'Model': [model_name],
                'Pooled Accuracy': [pooled_acc],
                'Pooled F1 (macro)': [pooled_f1],
                'City-Specific Avg Accuracy': [avg_city_acc],
                'City-Specific Avg F1': [avg_city_f1],
                'Improvement (Accuracy)': [improvement_acc],
                'Improvement (F1)': [improvement_f1]
            })], ignore_index=True)

        print("\n", comparison_df.to_string(index=False))

        # Save results
        output_dir = "result/geographic_analysis"
        os.makedirs(output_dir, exist_ok=True)

        comparison_df.to_csv(f"{output_dir}/pooled_vs_city_comparison.csv", index=False)
        print(f"\nSaved comparison to {output_dir}/pooled_vs_city_comparison.csv")

        # Save detailed results
        detailed_results = {
            'pooled': {k: {'accuracy': float(v['accuracy']), 'f1_macro': float(v['f1_macro'])}
                      for k, v in pooled_results.items()},
            'city_specific': city_results
        }

        with open(f"{output_dir}/model_comparison_details.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)

        print(f"Saved detailed results to {output_dir}/model_comparison_details.json")

        return comparison_df, pooled_results, city_results


if __name__ == "__main__":
    print("="*70)
    print("City-Specific vs Pooled Model Comparison")
    print("="*70)

    predictor = CitySpecificPredictor(threshold_pct=0.25, threshold_idx=4)

    # Load data
    print("\nLoading and preparing data...")
    df = predictor.load_and_prepare_data()

    # Run comparison
    comparison_df, pooled_results, city_results = predictor.compare_approaches(df)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")

    # Analyze improvements
    avg_acc_improvement = comparison_df['Improvement (Accuracy)'].mean()
    avg_f1_improvement = comparison_df['Improvement (F1)'].mean()

    print(f"- Average accuracy improvement from city-specific models: {avg_acc_improvement:+.2f}%")
    print(f"- Average F1 improvement from city-specific models: {avg_f1_improvement:+.2f}%")

    if avg_acc_improvement > 2:
        print("\n✅ City-specific models show substantial improvement!")
        print("   → Geographic variation in language matters for prediction")
    elif avg_acc_improvement > 0:
        print("\n⚠️  City-specific models show modest improvement")
        print("   → Some benefit to geographic segmentation")
    else:
        print("\n❌ City-specific models do not improve performance")
        print("   → Text features may not capture geographic variation effectively")

    print("\nCheck 'result/geographic_analysis/' for detailed outputs.")
