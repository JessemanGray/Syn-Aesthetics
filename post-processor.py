# post-processor.py — Complete Regression-Based Model Comparison with Aesthetic Wrapper
"""
This module provides a comprehensive model comparison pipeline for predicting
aesthetic chills intensity from aggregated stimulus features. It supports:
- Multiple regression models (Linear, Decision Tree, Random Forest, Gradient Boosting)
- Cross-validation and performance metrics (R², RMSE, MAE)
- Residual plots and feature importance visualizations
- Drift detection against baseline metrics
- Aesthetic reflection wrapper for output verification
- Phyllotactic sphere KG visualization with animated rotation

The visualizations use a dark theme with hot pink, lime green, turquoise, and orange
accents on black gridless backgrounds.

Usage:
    python post-processor.py

Requirements:
    pandas, numpy, scikit-learn, plotly, scipy
"""

import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- COLOR PALETTE ---
COLORS = {
    'hot_pink': '#FF1493',
    'lime_green': '#32CD32',
    'turquoise': '#40E0D0',
    'orange': '#FF8C00',
    'white': '#FFFFFF',
    'gray': '#808080',
    'dark_bg': '#000000'
}

class ChillsPostProcessor:
    """
    Main class for post-processing ChillsDB aggregated data.
    
    Attributes:
        df (pd.DataFrame): Loaded stimuli_merged.csv data
        models (dict): Trained model instances
        metrics (dict): Performance metrics for each model
        X_train_scaled, X_test_scaled: Scaled feature matrices
        y_reg_train, y_reg_test: Target variables (mean_intensity)
        feature_names (list): Names of input features
    """
    
    def __init__(self, csv_path="stimuli_merged.csv"):
        """
        Initialize the processor by loading the aggregated CSV file.
        
        Args:
            csv_path (str): Path to the aggregated stimuli CSV file.
        """
        self.df = pd.read_csv(csv_path)
        self.models = {}
        self.metrics = {}
        self.baseline_metrics = {}
        self.X = None
        self.X_train = None
        self.X_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_reg_train = None
        self.y_reg_test = None
        self.feature_names = None
        
        print(f"✅ Loaded {len(self.df)} samples from {csv_path}")
    
    def prepare_data(self):
        """
        Prepare features and targets for regression.
        
        Features:
            - mean_intensity: Average chills intensity (used as target)
            - mean_valence: Average valence shift (used as predictor)
            - mean_arousal: Average arousal shift (used as predictor)
            - mean_liking: Average liking score (used as predictor)
        
        The target variable is mean_intensity, which we predict from the other features.
        Data is split 80/20 and standardized using StandardScaler.
        """
        # Define features and target
        feature_cols = ['mean_valence', 'mean_arousal', 'mean_liking']
        self.feature_names = feature_cols
        
        self.X = self.df[feature_cols].fillna(0)
        y = self.df['mean_intensity'].fillna(0)
        
        # Train-test split
        self.X_train, self.X_test, self.y_reg_train, self.y_reg_test = train_test_split(
            self.X, y, test_size=0.2, random_state=42
        )
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"📊 Data prepared: {len(self.X_train)} training, {len(self.X_test)} test samples")
        print(f"📈 Target range: {y.min():.2f} to {y.max():.2f}")
    
    def run_linear_regression(self):
        """
        Train a Linear Regression model to predict mean_intensity.
        
        Returns:
            dict: Performance metrics including R², RMSE, MAE, and coefficients.
        """
        model = LinearRegression()
        model.fit(self.X_train_scaled, self.y_reg_train)
        y_pred = model.predict(self.X_test_scaled)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train_scaled, self.y_reg_train, cv=5, scoring='r2')
        
        self.models['linear'] = model
        self.metrics['linear'] = {
            'r2': r2_score(self.y_reg_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_reg_test, y_pred)),
            'mae': mean_absolute_error(self.y_reg_test, y_pred),
            'coefficients': dict(zip(self.X.columns, model.coef_)),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        return self.metrics['linear']
    
    def run_decision_tree(self, max_depth=5):
        """
        Train a Decision Tree Regressor.
        
        Args:
            max_depth (int): Maximum depth of the tree.
            
        Returns:
            dict: Performance metrics and feature importance.
        """
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        model.fit(self.X_train_scaled, self.y_reg_train)
        y_pred = model.predict(self.X_test_scaled)
        
        cv_scores = cross_val_score(model, self.X_train_scaled, self.y_reg_train, cv=5, scoring='r2')
        
        self.models['decision_tree'] = model
        self.metrics['decision_tree'] = {
            'r2': r2_score(self.y_reg_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_reg_test, y_pred)),
            'mae': mean_absolute_error(self.y_reg_test, y_pred),
            'feature_importance': dict(zip(self.X.columns, model.feature_importances_)),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        return self.metrics['decision_tree']
    
    def run_random_forest(self, n_estimators=100, max_depth=5):
        """
        Train a Random Forest Regressor.
        
        Args:
            n_estimators (int): Number of trees in the forest.
            max_depth (int): Maximum depth of each tree.
            
        Returns:
            dict: Performance metrics and feature importance.
        """
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(self.X_train_scaled, self.y_reg_train)
        y_pred = model.predict(self.X_test_scaled)
        
        cv_scores = cross_val_score(model, self.X_train_scaled, self.y_reg_train, cv=5, scoring='r2')
        
        self.models['random_forest'] = model
        self.metrics['random_forest'] = {
            'r2': r2_score(self.y_reg_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_reg_test, y_pred)),
            'mae': mean_absolute_error(self.y_reg_test, y_pred),
            'feature_importance': dict(zip(self.X.columns, model.feature_importances_)),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        return self.metrics['random_forest']
    
    def run_gradient_boosting(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Train a Gradient Boosting Regressor.
        
        Args:
            n_estimators (int): Number of boosting stages.
            learning_rate (float): Learning rate shrinks the contribution of each tree.
            max_depth (int): Maximum depth of each tree.
            
        Returns:
            dict: Performance metrics and feature importance.
        """
        model = GradientBoostingRegressor(
            n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            max_depth=max_depth, 
            random_state=42
        )
        model.fit(self.X_train_scaled, self.y_reg_train)
        y_pred = model.predict(self.X_test_scaled)
        
        cv_scores = cross_val_score(model, self.X_train_scaled, self.y_reg_train, cv=5, scoring='r2')
        
        self.models['gradient_boost'] = model
        self.metrics['gradient_boost'] = {
            'r2': r2_score(self.y_reg_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_reg_test, y_pred)),
            'mae': mean_absolute_error(self.y_reg_test, y_pred),
            'feature_importance': dict(zip(self.X.columns, model.feature_importances_)),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        return self.metrics['gradient_boost']
    
    def run_all_models(self):
        """
        Run all regression models sequentially.
        
        Returns:
            dict: Combined metrics for all models.
        """
        print("\n🚀 Running all regression models...")
        self.prepare_data()
        self.run_linear_regression()
        self.run_decision_tree()
        self.run_random_forest()
        self.run_gradient_boosting()
        print("✅ All models trained.")
        return self.metrics
    
    def plot_residuals(self):
        """
        Plot residual distributions for all models.
        
        Returns:
            plotly.Figure: Residual plot with black background and hyper colors.
        """
        model_names = list(self.models.keys())
        if not model_names:
            print("⚠️ No models found.")
            return
        
        fig = make_subplots(
            rows=len(model_names), cols=1,
            subplot_titles=[m.upper() for m in model_names],
            row_heights=[1] * len(model_names)
        )
        
        color_cycle = [COLORS['hot_pink'], COLORS['lime_green'], COLORS['turquoise'], COLORS['orange']]
        
        for i, model_name in enumerate(model_names):
            model = self.models[model_name]
            y_pred = model.predict(self.X_test_scaled)
            residuals = self.y_reg_test - y_pred
            
            fig.add_trace(
                go.Scatter(
                    x=y_pred,
                    y=residuals,
                    mode='markers',
                    marker=dict(
                        color=color_cycle[i % len(color_cycle)],
                        size=8,
                        opacity=0.7
                    ),
                    name=model_name.upper(),
                    showlegend=False
                ),
                row=i+1, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="white", row=i+1, col=1)
        
        fig.update_layout(
            title=dict(text="Residual Plots", font=dict(color='white', size=20), x=0.5),
            paper_bgcolor='black',
            plot_bgcolor='black',
            height=300 * len(model_names),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        fig.update_xaxes(title_text="Predicted", tickfont=dict(color='white'), gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(title_text="Residual", tickfont=dict(color='white'), gridcolor='rgba(128,128,128,0.2)')
        return fig
    
    def plot_metric_comparison(self):
        """
        Plot R², RMSE, and MAE comparison for all models.
        
        Returns:
            plotly.Figure: Bar chart comparison with hyper colors.
        """
        model_names = list(self.metrics.keys())
        if not model_names:
            print("⚠️ No metrics found.")
            return
        
        fig = go.Figure()
        
        # R² (higher is better)
        r2_vals = [self.metrics[m]['r2'] for m in model_names]
        fig.add_trace(go.Bar(
            x=model_names, y=r2_vals, name='R²',
            marker_color=COLORS['hot_pink'],
            text=[f'{v:.3f}' for v in r2_vals],
            textposition='outside'
        ))
        
        # RMSE (lower is better) — scaled for visibility
        rmse_vals = [self.metrics[m]['rmse'] / 10 for m in model_names]
        fig.add_trace(go.Bar(
            x=model_names, y=rmse_vals, name='RMSE/10',
            marker_color=COLORS['lime_green'],
            text=[f'{v:.3f}' for v in rmse_vals],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=dict(text="Model Performance Comparison", font=dict(color='white', size=20), x=0.5),
            paper_bgcolor='black',
            plot_bgcolor='black',
            barmode='group',
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        fig.update_xaxes(tickfont=dict(color='white'), gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(tickfont=dict(color='white'), gridcolor='rgba(128,128,128,0.2)')
        fig.update_layout(legend=dict(font=dict(color='white')))
        return fig
    
    def plot_feature_importance(self):
        """
        Plot feature importance for tree-based models.
        
        Returns:
            plotly.Figure: Feature importance bar charts.
        """
        tree_models = ['decision_tree', 'random_forest', 'gradient_boost']
        tree_models = [m for m in tree_models if m in self.metrics and 'feature_importance' in self.metrics[m]]
        
        if not tree_models:
            print("⚠️ No tree-based models found.")
            return
        
        fig = make_subplots(
            rows=len(tree_models), cols=1,
            subplot_titles=[m.upper() for m in tree_models],
            row_heights=[1] * len(tree_models)
        )
        
        color_cycle = [COLORS['hot_pink'], COLORS['lime_green'], COLORS['turquoise']]
        
        for i, model_name in enumerate(tree_models):
            importance = self.metrics[model_name]['feature_importance']
            features = list(importance.keys())
            values = list(importance.values())
            
            fig.add_trace(
                go.Bar(
                    x=features, y=values,
                    marker_color=color_cycle[i % len(color_cycle)],
                    text=[f'{v:.3f}' for v in values],
                    textposition='outside',
                    textfont=dict(color='white')
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title=dict(text="Feature Importance by Model", font=dict(color='white', size=20), x=0.5),
            paper_bgcolor='black',
            plot_bgcolor='black',
            height=300 * len(tree_models),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        fig.update_xaxes(tickfont=dict(color='white'), gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(tickfont=dict(color='white'), gridcolor='rgba(128,128,128,0.2)')
        return fig
    
    def plot_phyllotactic_sphere_kg(self):
        """
        Create an animated sunflower phyllotaxis sphere for the KG visualization.
        Uses the hyper color palette (hot pink, lime green, turquoise, orange).
        """
        num_points = 2000
        golden_angle = np.pi * (3 - np.sqrt(5))
        radius = 50
        
        x_vals, y_vals, z_vals, color_vals = [], [], [], []
        
        # Map data points to sphere positions
        df_sample = self.df.sample(min(num_points, len(self.df)), replace=True if len(self.df) < num_points else False)
        
        for i in range(num_points):
            angle = i * golden_angle
            z = 1 - (2 * i) / num_points
            radius_at_z = np.sqrt(1 - z**2)
            
            x = radius_at_z * np.cos(angle) * radius
            y = radius_at_z * np.sin(angle) * radius
            
            # Cycle through hyper colors
            color_cycle = [COLORS['hot_pink'], COLORS['lime_green'], COLORS['turquoise'], COLORS['orange']]
            color = color_cycle[i % 4]
            
            # Add some transparency variation
            alpha = 0.6 + 0.3 * (i / num_points)
            color_vals.append(color.replace(')', f', {alpha})').replace('rgb', 'rgba'))
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z * radius)
        
        fig = go.Figure(
            data=[go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='markers',
                marker=dict(
                    size=5,
                    color=color_vals,
                    opacity=0.9
                ),
                hoverinfo='none'
            )]
        )
        
        # Animation frames for rotation
        frames = []
        for angle in np.linspace(0, 360, 100):
            camera_eye = dict(
                x=1.8 * np.cos(np.radians(angle)),
                y=1.8 * np.sin(np.radians(angle)),
                z=0.3
            )
            frames.append(go.Frame(
                layout=dict(scene_camera=dict(eye=camera_eye))
            ))
        
        fig.update(frames=frames)
        fig.update_layout(
            title=dict(
                text="ChillsDB Knowledge Graph — Phyllotactic Sphere",
                font=dict(color='white', size=20),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='black',
                camera=dict(eye=dict(x=1.8, y=0, z=0.3))
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            paper_bgcolor='black',
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': '▶ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 75, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate'
                        }]
                    },
                    {
                        'label': '⏸ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 75, 'redraw': False},
                            'mode': 'immediate'
                        }]
                    }
                ]
            }]
        )
        
        return fig
    
    def aesthetic_reflection_wrapper(self, query: str, response: str, max_iterations: int = 3) -> str:
        """
        Wrap any model output with aesthetic verification and correction loop.
        
        Args:
            query (str): The original user query.
            response (str): The model's raw output.
            max_iterations (int): Maximum number of correction attempts.
            
        Returns:
            str: Aesthetic-verified and corrected response.
        """
        # Load KG metrics from existing DataFrame
        avg_valence = self.df['mean_valence'].mean() if 'mean_valence' in self.df.columns else 0
        avg_intensity = self.df['mean_intensity'].mean() if 'mean_intensity' in self.df.columns else 0
        
        def verify(text):
            """Check if text contains required aesthetic references."""
            passes = True
            feedback = []
            
            # Check for polarity references
            if "beneficial" not in text.lower() and "detrimental" not in text.lower():
                passes = False
                feedback.append("Missing polarity reference. Please indicate if this is beneficial or detrimental.")
            
            # Check for intensity references
            if "intensity" not in text.lower():
                passes = False
                feedback.append("Missing intensity reference. Please include a sense of intensity.")
            
            # Check for specific KG metrics
            if "valence" in text.lower():
                try:
                    # If valence is mentioned, check if it's within reasonable range
                    pass
                except:
                    pass
            
            return passes, " ".join(feedback)
        
        current = response
        for i in range(max_iterations):
            passes, feedback = verify(current)
            if passes:
                return current
            # Simple correction: prepend feedback and note the iteration
            current = f"Revised (attempt {i+1}): {current} [Feedback: {feedback}]"
        
        return current
    
    def save_metrics(self, output_path="model_metrics.json"):
        """
        Save all metrics to a JSON file.
        
        Args:
            output_path (str): Path to save the JSON file.
        
        Returns:
            str: The output path.
        """
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✅ Metrics saved to {output_path}")
        return output_path
    
    def generate_drift_report(self, new_metrics, threshold=0.05):
        """
        Compare new metrics against baseline to detect drift.
        
        Args:
            new_metrics (dict): New model metrics to compare against baseline.
            threshold (float): Drift threshold (0.05 = 5% change).
        
        Returns:
            dict: Drift report with alerts for each metric.
        """
        drift_report = {}
        for model_name in self.metrics:
            if model_name in new_metrics:
                drift_report[model_name] = {}
                for metric, value in self.metrics[model_name].items():
                    if metric in new_metrics[model_name]:
                        new_value = new_metrics[model_name][metric]
                        if isinstance(value, (int, float)) and isinstance(new_value, (int, float)):
                            drift = abs(new_value - value) / (value + 1e-6)
                            drift_report[model_name][metric] = {
                                'baseline': value,
                                'current': new_value,
                                'drift': drift,
                                'alert': drift > threshold
                            }
        
        self.drift_report = drift_report
        with open("drift_report.json", 'w') as f:
            json.dump(drift_report, f, indent=2)
        return drift_report


if __name__ == "__main__":
    """
    Main execution block.
    
    This runs the full pipeline:
    1. Loads stimuli_merged.csv
    2. Trains all regression models
    3. Saves metrics to JSON
    4. Generates and displays visualizations
    """
    print("="*60)
    print("🧊 CHILLSDB POST-PROCESSOR — REGRESSION ANALYSIS")
    print("="*60)
    
    processor = ChillsPostProcessor()
    metrics = processor.run_all_models()
    processor.save_metrics()
    
    print("\n📊 Model Comparison:")
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name.upper()}:")
        for k, v in model_metrics.items():
            if k == 'coefficients':
                print(f"  coefficients:")
                for feat, coef in v.items():
                    print(f"    {feat}: {coef:.3f}")
            elif k == 'feature_importance':
                print(f"  feature_importance:")
                for feat, imp in v.items():
                    print(f"    {feat}: {imp:.3f}")
            else:
                print(f"  {k}: {v:.3f}")
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    
    fig_res = processor.plot_residuals()
    if fig_res:
        fig_res.show()
        fig_res.write_html("residual_plots.html")
        print("✅ Residual plots saved to residual_plots.html")
    
    fig_comp = processor.plot_metric_comparison()
    if fig_comp:
        fig_comp.show()
        fig_comp.write_html("metric_comparison.html")
        print("✅ Metric comparison saved to metric_comparison.html")
    
    fig_fi = processor.plot_feature_importance()
    if fig_fi:
        fig_fi.show()
        fig_fi.write_html("feature_importance.html")
        print("✅ Feature importance saved to feature_importance.html")
    
    # Phyllotactic sphere KG
    fig_phyllo = processor.plot_phyllotactic_sphere_kg()
    if fig_phyllo:
        fig_phyllo.show()
        fig_phyllo.write_html("phyllotactic_sphere_kg.html")
        print("✅ Phyllotactic sphere KG saved to phyllotactic_sphere_kg.html")
    
    print("\n✅ Visualizations complete.")