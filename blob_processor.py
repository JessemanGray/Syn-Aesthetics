# blob_processor.py
import pandas as pd
import numpy as np
import requests
import io
from pathlib import Path
import sys
import json
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class ChillsBlobProcessor:
    def __init__(self):
        self.df_responses = None
        self.df_stimuli = None
        self.df_aggregated = None
        self.df_merged = None
        self.df_traits = None
        self.df_header = None
        self.debug = True
        self.models = {}

    def log(self, msg):
        if self.debug:
            print(f"[DEBUG] {msg}")

    def load_from_figshare(self):
        """Load all ChillsDB 2.0 files from Figshare."""
        print("📥 Fetching ChillsDB 2.0 from Figshare...")
        article_id = 23935611
        api_url = f"https://api.figshare.com/v2/articles/{article_id}/files"
        r = requests.get(api_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        files = r.json()
        print(f"✅ Found {len(files)} files")

        for f in files:
            if not f['name'].endswith('.csv'):
                continue
            print(f"   Downloading: {f['name']}")
            try:
                dl = requests.get(f['download_url'], headers={"User-Agent": "Mozilla/5.0"})
                dl.raise_for_status()
                df = pd.read_csv(io.StringIO(dl.content.decode('utf-8')))
                name = f['name'].lower()

                # Explicitly identify files by their exact names
                if 'chillsdb 2 - chillsdb 2.csv' == name:
                    self.df_responses = df
                    print(f"   📊 Identified as main responses file")
                elif 'stimuli.csv' == name:
                    self.df_stimuli = df
                    print(f"   🎬 Identified as stimuli metadata file")
                elif 'header coding.csv' in name:
                    self.df_header = df
                    print(f"   📋 Identified as header coding file")
                elif 'traits.csv' == name:
                    self.df_traits = df
                    print(f"   🧬 Identified as traits file")
                else:
                    print(f"   ⚠️ Unknown file: {f['name']}")
            except Exception as e:
                print(f"   ❌ Failed to load {f['name']}: {e}")

        if self.df_responses is None:
            raise ValueError("No main responses DataFrame loaded! Please check the file names on Figshare.")
        print(f"\n✅ Loaded {len(self.df_responses)} responses")
        if self.df_stimuli is not None:
            print(f"✅ Loaded {len(self.df_stimuli)} stimuli")
        return self.df_responses, self.df_stimuli

    def aggregate_responses(self):
        """Aggregate responses by stimulus with robust column handling."""
        if self.df_responses is None:
            raise ValueError("No responses data. Run load_from_figshare() first.")

        df = self.df_responses.copy()
        print(f"\n🔧 Aggregating {len(df)} responses...")

        # --- Convert numeric columns ---
        numeric_cols = ['Chills intensity', 'Valence Post', 'Arousal Post', 'Liking', 'Valence Pre', 'Arousal pre']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                self.log(f"Converted {col} to numeric ({(df[col].isna().sum())} NaNs)")

        # --- Encode Yes/No columns ---
        if 'Chills?' in df.columns:
            df['chills_encoded'] = df['Chills?'].map({'Yes': 1, 'No': 0}).fillna(0)
            self.log("Encoded Chills? → chills_encoded")

        # --- Find the stimulus column ---
        # The main data file should have a column named 'Stimulus'
        stim_col = 'Stimulus' if 'Stimulus' in df.columns else None
        if stim_col is None:
            # Fallback: case-insensitive search
            stim_col = next((c for c in df.columns if 'stimulus' in c.lower()), None)

        if stim_col is None:
            raise ValueError(f"No stimulus column found! Available columns: {list(df.columns)}")
        self.log(f"Using stimulus column: '{stim_col}'")

        # --- Build aggregation dictionary ---
        agg_dict = {}
        if 'Chills intensity' in df.columns:
            agg_dict['mean_intensity'] = ('Chills intensity', 'mean')
            agg_dict['max_intensity'] = ('Chills intensity', 'max')
            agg_dict['std_intensity'] = ('Chills intensity', 'std')
        if 'chills_encoded' in df.columns:
            agg_dict['chills_ratio'] = ('chills_encoded', 'mean')
        if 'Valence Post' in df.columns:
            agg_dict['mean_valence'] = ('Valence Post', 'mean')
        if 'Arousal Post' in df.columns:
            agg_dict['mean_arousal'] = ('Arousal Post', 'mean')
        if 'Liking' in df.columns:
            agg_dict['mean_liking'] = ('Liking', 'mean')

        self.log(f"Aggregation dict: {agg_dict}")

        # --- Perform aggregation ---
        try:
            self.df_aggregated = df.groupby(stim_col).agg(**agg_dict).reset_index()
        except Exception as e:
            print(f"❌ Aggregation failed: {e}")
            self.log(f"DataFrame head:\n{df.head()}")
            raise

        # --- Add Polarity ---
        if 'mean_valence' in self.df_aggregated.columns:
            def classify(valence):
                if valence > 0.2:
                    return 'beneficial'
                elif valence < -0.2:
                    return 'detrimental'
                return 'neutral'
            self.df_aggregated['polarity'] = self.df_aggregated['mean_valence'].apply(classify)

        print(f"✅ Aggregated into {len(self.df_aggregated)} stimuli")
        self.log(f"Columns after aggregation: {list(self.df_aggregated.columns)}")
        return self.df_aggregated

    def merge_stimuli_metadata(self):
        """Merge the Stimuli.csv metadata with aggregated response data."""
        if self.df_aggregated is None:
            self.aggregate_responses()
        if self.df_stimuli is None:
            print("⚠️ No stimuli metadata to merge. Returning aggregated data.")
            self.df_merged = self.df_aggregated
            return self.df_merged

        # --- Find the stimulus identifier column in both DataFrames ---
        # In aggregated data
        stim_col_agg = 'Stimulus' if 'Stimulus' in self.df_aggregated.columns else self.df_aggregated.columns[0]
        # In stimuli metadata
        stim_col_meta = next((c for c in self.df_stimuli.columns if 'stimulus' in c.lower()), None)
        if stim_col_meta is None:
            stim_col_meta = self.df_stimuli.columns[0]

        self.log(f"Merging aggregated data on '{stim_col_agg}' with stimuli metadata on '{stim_col_meta}'")

        # --- Perform merge ---
        try:
            self.df_merged = self.df_aggregated.merge(
                self.df_stimuli,
                left_on=stim_col_agg,
                right_on=stim_col_meta,
                how='left',
                suffixes=('', '_stim')
            )
            print(f"✅ Merged {len(self.df_merged)} records with stimuli metadata")
        except Exception as e:
            print(f"❌ Merge failed: {e}")
            self.df_merged = self.df_aggregated

        return self.df_merged

    def run_linear_regression(self, target_col='mean_intensity', feature_cols=None):
        """Run linear regression to predict intensity from features."""
        df = self.df_merged if self.df_merged is not None else self.df_aggregated
        if feature_cols is None:
            feature_cols = ['chills_ratio', 'mean_valence', 'mean_arousal']
            feature_cols = [c for c in feature_cols if c in df.columns]

        # Prepare data
        X = df[feature_cols].dropna()
        y = df[target_col].loc[X.index]

        if len(X) == 0:
            print("⚠️ Not enough data for regression.")
            return None, 0, 0

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit model
        model = LinearRegression()
        model.fit(X_scaled, y)

        # Evaluate
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        # Store
        self.models['linear_regression'] = model
        self.models['linear_regression_scaler'] = scaler
        self.models['linear_regression_features'] = feature_cols

        print(f"📊 Linear Regression — R²: {r2:.3f}, RMSE: {rmse:.2f}")
        self.log(f"Coefficients: {dict(zip(feature_cols, model.coef_))}")
        return model, r2, rmse

    def run_decision_tree(self, target_col='mean_intensity', feature_cols=None, max_depth=5):
        """Run decision tree regression."""
        df = self.df_merged if self.df_merged is not None else self.df_aggregated
        if feature_cols is None:
            feature_cols = ['chills_ratio', 'mean_valence', 'mean_arousal']
            feature_cols = [c for c in feature_cols if c in df.columns]

        X = df[feature_cols].dropna()
        y = df[target_col].loc[X.index]

        if len(X) == 0:
            print("⚠️ Not enough data for decision tree.")
            return None, 0, 0

        model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        model.fit(X, y)

        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        self.models['decision_tree'] = model
        self.models['decision_tree_features'] = feature_cols

        print(f"🌳 Decision Tree — R²: {r2:.3f}, RMSE: {rmse:.2f}")
        self.log(f"Feature importances: {dict(zip(feature_cols, model.feature_importances_))}")
        return model, r2, rmse

    def correlation_matrix(self):
        """Compute and display correlation matrix of key features."""
        df = self.df_merged if self.df_merged is not None else self.df_aggregated
        corr_cols = ['mean_intensity', 'chills_ratio', 'mean_valence', 'mean_arousal', 'mean_liking']
        corr_cols = [c for c in corr_cols if c in df.columns]

        if len(corr_cols) < 2:
            print("⚠️ Not enough columns for correlation matrix.")
            return pd.DataFrame()

        corr_matrix = df[corr_cols].corr()
        print("\n📈 Correlation Matrix:")
        print(corr_matrix.round(3))
        return corr_matrix

    def run_statistical_analysis(self):
        """Run all statistical analyses and print results."""
        print("\n" + "="*60)
        print("📊 STATISTICAL ANALYSIS")
        print("="*60)

        # Correlation matrix
        corr = self.correlation_matrix()

        # Linear regression
        if self.df_merged is not None and 'mean_intensity' in self.df_merged.columns:
            print("\n🔹 Linear Regression (predicting intensity):")
            self.run_linear_regression()

            print("\n🔹 Decision Tree (predicting intensity):")
            self.run_decision_tree()

        # Summary stats
        print("\n📋 Summary Statistics:")
        df = self.df_merged if self.df_merged is not None else self.df_aggregated
        if df is not None:
            stats_cols = ['mean_intensity', 'chills_ratio', 'mean_valence']
            stats_cols = [c for c in stats_cols if c in df.columns]
            if stats_cols:
                print(df[stats_cols].describe().round(3))
            else:
                print(df.describe().round(3))

        return corr

    def build_rag_documents(self):
        """Generate RAG-ready documents from merged data."""
        df = self.df_merged if self.df_merged is not None else self.df_aggregated
        if df is None:
            print("⚠️ No data to build documents from.")
            return []

        docs = []
        for _, row in df.iterrows():
            # Build a comprehensive text description
            text_parts = []
            if 'Stimulus' in row:
                text_parts.append(f"Stimulus: {row['Stimulus']}")
            elif 'name' in row:
                text_parts.append(f"Stimulus: {row['name']}")

            if 'polarity' in row:
                text_parts.append(f"Polarity: {row['polarity']}")
            if 'mean_intensity' in row:
                text_parts.append(f"Mean intensity: {row['mean_intensity']:.1f}")
            if 'mean_valence' in row:
                text_parts.append(f"Mean valence: {row['mean_valence']:.2f}")
            if 'chills_ratio' in row:
                text_parts.append(f"Chills ratio: {row['chills_ratio']:.2f}")

            text = ". ".join(text_parts) + "."
            docs.append({'text': text, 'metadata': row.to_dict()})

        print(f"✅ Built {len(docs)} RAG documents")
        return docs

    def save_merged_data(self, path="stimuli_merged.csv"):
        """Save the merged dataset."""
        if self.df_merged is not None:
            self.df_merged.to_csv(path, index=False)
            print(f"💾 Saved merged data to {path}")
        else:
            print("⚠️ No merged data to save.")

    def run_pipeline(self):
        """Full pipeline: load → aggregate → merge → analyze → RAG docs."""
        print("🚀 Starting ChillsDB Blob Processor Pipeline")
        print("="*60)

        self.load_from_figshare()
        self.aggregate_responses()
        self.merge_stimuli_metadata()
        self.run_statistical_analysis()
        docs = self.build_rag_documents()

        print("\n" + "="*60)
        print("✅ Pipeline complete.")
        return self.df_merged, docs


if __name__ == "__main__":
    processor = ChillsBlobProcessor()
    df_merged, docs = processor.run_pipeline()
    processor.save_merged_data()
    print(f"🎯 {len(docs)} RAG documents ready.")
    