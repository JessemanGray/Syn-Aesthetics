# src/rag_agent/data_processor.py
import pandas as pd
import numpy as np
import requests
import io
import networkx as nx
from scipy.spatial import KDTree
from sklearn.preprocessing import MinMaxScaler

class SynAestheticsDataProcessor:
    def __init__(self):
        self.df = None
        self.df_processed = None
        self.graph = None
        self.feature_summary = {}

    def load_chills_data(self):
        article_id = 23935611
        api_url = f"https://api.figshare.com/v2/articles/{article_id}/files"
        print("📥 Fetching ChillsDB 2.0 from Figshare...")
        r = requests.get(api_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        for f in r.json():
            if 'ChillsDB 2' in f['name'] and 'Header' not in f['name']:
                dl = requests.get(f['download_url'], headers={"User-Agent": "Mozilla/5.0"})
                self.df = pd.read_csv(io.StringIO(dl.content.decode('utf-8')))
                break
        if self.df is None:
            raise ValueError("No data file found on Figshare")
        print(f"✅ Loaded {len(self.df)} records")
        return self.df

    def process_chills_data(self):
        df = self.df.copy()
        print(f"🔧 Processing {len(df)} records with robust EDA...")
        
        # Column mapping
        col_map = {
            'intensity': next((c for c in df.columns if 'intensity' in c.lower()), None),
            'valence_pre': next((c for c in df.columns if 'valence pre' in c.lower()), None),
            'valence_post': next((c for c in df.columns if 'valence post' in c.lower()), None),
            'arousal_pre': next((c for c in df.columns if 'arousal pre' in c.lower()), None),
            'arousal_post': next((c for c in df.columns if 'arousal post' in c.lower()), None),
            'liking': next((c for c in df.columns if 'liking' in c.lower()), None),
            'stimulus': next((c for c in df.columns if 'stimulus' in c.lower()), None),
            'modality': next((c for c in df.columns if 'modality' in c.lower()), None),
            'participant': next((c for c in df.columns if 'participant' in c.lower()), None),
            'chills_binary': next((c for c in df.columns if 'chills' in c.lower() and 'binary' not in c.lower()), None),
        }
        
        # 1. Core metrics
        if col_map['intensity']:
            df['Intensity'] = df[col_map['intensity']].fillna(0)
            df['Chill_Rating'] = (df['Intensity'] / 10).round(1).clip(0, 10)
        else:
            df['Intensity'] = 0
            df['Chill_Rating'] = 0
        
        if col_map['valence_pre'] and col_map['valence_post']:
            df['Valence'] = df[col_map['valence_post']] - df[col_map['valence_pre']]
            df['Valence'] = df['Valence'].fillna(0)
        else:
            df['Valence'] = 0
        
        if col_map['arousal_pre'] and col_map['arousal_post']:
            df['Arousal'] = df[col_map['arousal_post']] - df[col_map['arousal_pre']]
            df['Arousal'] = df['Arousal'].fillna(0)
        else:
            df['Arousal'] = 0
        
        df['Liking'] = df[col_map['liking']].fillna(0) if col_map['liking'] else 0
        
        # 2. Polarity
        def get_polarity(row):
            i = row['Intensity']
            v = row['Valence']
            if i == 0:
                return 'Neutral'
            if v > 0.2:
                return 'Beneficial' if i > 30 else 'Mildly Beneficial'
            if v < -0.2:
                return 'Detrimental' if i > 30 else 'Mildly Detrimental'
            return 'Neutral'
        df['Polarity'] = df.apply(get_polarity, axis=1)
        df['Is_Beneficial'] = df['Polarity'].str.contains('Beneficial')
        df['Is_Detrimental'] = df['Polarity'].str.contains('Detrimental')
        
        # 3. Intensity clusters
        bins = [0, 20, 40, 60, 80, 100]
        labels = ['None', 'Low', 'Medium', 'High', 'Very High']
        df['Intensity_Cluster'] = pd.cut(df['Intensity'], bins=bins, labels=labels, right=False)
        
        # 4. Media labels
        if col_map['stimulus']:
            def make_label(x):
                if pd.isna(x):
                    return 'Unknown'
                try:
                    return f"S{int(x):03d}"
                except:
                    return str(x)[:20]
            df['Media_Label'] = df[col_map['stimulus']].apply(make_label)
        else:
            df['Media_Label'] = df.index.astype(str)
        
        # 5. Modality
        df['Modality'] = df[col_map['modality']].fillna('Unknown') if col_map['modality'] else 'Unknown'
        
        # 6. Response patterns (EDA features)
        # In process_chills_data, replace the entire Chills_Occurred block with:
        if col_map['chills_binary']:
            col = col_map['chills_binary']
            # Handle different possible formats
            if df[col].dtype == 'object' or df[col].dtype == 'string':
                # Convert 'Yes'/'No', 'True'/'False', etc.
                df['Chills_Occurred'] = df[col].astype(str).str.lower().map({
                    'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0
                }).fillna(0).astype(int)
            else:
                df['Chills_Occurred'] = df[col].fillna(0).astype(int)
        else:
            df['Chills_Occurred'] = 0
        
        if col_map['participant']:
            df['Chills_Rate'] = df.groupby(col_map['participant'])['Chills_Occurred'].transform('mean').round(2)
            df['Is_High_Responder'] = df['Chills_Rate'] > 0.5
        else:
            df['Chills_Rate'] = np.random.uniform(0, 0.8, len(df)).round(2)
            df['Is_High_Responder'] = False
        
        if col_map['stimulus']:
            df['Stimulus_Popularity'] = df.groupby(col_map['stimulus'])['Intensity'].transform('mean').round(1)
            df['Stimulus_Hit_Rate'] = df.groupby(col_map['stimulus'])['Chills_Occurred'].transform('mean').round(2)
        
        # 7. Response categories
        def classify_response(row):
            i = row['Intensity']
            v = row['Valence']
            if i == 0:
                return 'No Response'
            if v > 0.3 and i > 50:
                return 'Strong Positive'
            elif v > 0.3:
                return 'Mild Positive'
            elif v < -0.3 and i > 50:
                return 'Strong Negative'
            elif v < -0.3:
                return 'Mild Negative'
            else:
                return 'Mixed/Neutral'
        df['Response_Category'] = df.apply(classify_response, axis=1)
        
        # 8. Context richness
        df['Context_Richness'] = (
            (df['Chill_Rating'] > 5).astype(int) +
            (df['Valence'] > 0.3).astype(int) +
            (df['Arousal'] > 0.3).astype(int) +
            (df['Liking'] > 4).astype(int)
        ).clip(0, 4)
        
        # 9. Consistency flag
        if col_map['participant']:
            df['Is_Consistent'] = df.groupby(col_map['participant'])['Polarity'].transform(
                lambda x: x.value_counts().max() / len(x) > 0.6
            )
        else:
            df['Is_Consistent'] = False
        
        # 10. EDA summary
        self.feature_summary = {
            'total_records': len(df),
            'polarity': df['Polarity'].value_counts().to_dict(),
            'intensity_clusters': df['Intensity_Cluster'].value_counts().to_dict(),
            'response_categories': df['Response_Category'].value_counts().to_dict(),
            'avg_intensity': df['Intensity'].mean(),
            'avg_valence': df['Valence'].mean(),
            'context_richness': df['Context_Richness'].mean(),
            'high_responders': df['Is_High_Responder'].sum(),
            'unique_stimuli': df['Media_Label'].nunique()
        }
        
        self.df_processed = df
        print(f"✅ Processed {len(df)} records with {len(df.columns)} features")
        print(f"   Polarity: {df['Polarity'].value_counts().to_dict()}")
        print(f"   Response Categories: {df['Response_Category'].value_counts().to_dict()}")
        return self.df_processed

    def build_phyllotactic_kg(self):
        df = self.df_processed.sample(min(len(self.df_processed), 5000), random_state=42).reset_index(drop=True)
        
        scaler = MinMaxScaler()
        features = scaler.fit_transform(df[['Chill_Rating', 'Valence']].fillna(0).abs())
        
        golden_angle = np.pi * (3 - np.sqrt(5))
        radius = 25
        x_vals, y_vals, z_vals = [], [], []
        for i in range(len(df)):
            theta = i * golden_angle
            phi = np.arccos(1 - 2 * (i + 0.5) / len(df))
            r = radius * (0.4 + 0.6 * features[i, 0])
            x_vals.append(r * np.sin(phi) * np.cos(theta))
            y_vals.append(r * np.sin(phi) * np.sin(theta))
            z_vals.append(r * np.cos(phi) * 0.8)
        df['kg_x'], df['kg_y'], df['kg_z'] = x_vals, y_vals, z_vals
        
        def get_color(row):
            v = (row['Valence'] + 1) / 2
            v = max(0, min(1, v))
            intensity_factor = max(0, min(1, row['Intensity'] / 100))
            r = int(255 * (1 - v))
            b = int(255 * v)
            g = int(80 * intensity_factor * (1 - abs(v - 0.5) * 2))
            brightness = 0.6 + 0.4 * intensity_factor
            r = max(0, min(255, int(r * brightness)))
            g = max(0, min(255, int(g * brightness)))
            b = max(0, min(255, int(b * brightness)))
            return f'rgb({r}, {g}, {b})'
        
        df['color'] = df.apply(get_color, axis=1)
        
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_node(
                str(row.name),
                pos=(row['kg_x'], row['kg_y'], row['kg_z']),
                color=row['color'],
                rating=row['Chill_Rating'],
                polarity=row['Polarity'],
                intensity=row['Intensity'],
                valence=row['Valence'],
                arousal=row['Arousal'],
                media=row['Media_Label'],
                modality=row['Modality'],
                response=row['Response_Category'],
                context=row['Context_Richness']
            )
        
        positions = df[['kg_x', 'kg_y', 'kg_z']].values
        tree = KDTree(positions)
        for i in range(len(df)):
            dist, idx = tree.query(positions[i], k=6)
            for j in idx[1:]:
                if i != j and dist[1] < 14:
                    G.add_edge(str(i), str(j))
        
        self.graph = G
        print(f"✅ KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def get_eda_summary(self):
        return self.feature_summary

    def run_pipeline(self):
        self.load_chills_data()
        self.process_chills_data()
        self.build_phyllotactic_kg()
        return self.df_processed, self.graph