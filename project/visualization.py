"""
Visualization: Create plots for decision tree and feature importance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
import pickle
import json
import os
import lightgbm as lgb

class Visualizer:
    """Create visualizations for model analysis"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.load_metadata()
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def load_metadata(self):
        """Load metadata"""
        with open(os.path.join(self.model_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
    
    def plot_sklearn_tree(self, max_depth=3):
        """Plot sklearn decision tree"""
        print("Creating sklearn tree plot...")
        
        with open(os.path.join(self.model_dir, 'sklearn_tree.pkl'), 'rb') as f:
            clf = pickle.load(f)
        
        plt.figure(figsize=(25, 15))
        plot_tree(clf, 
                 feature_names=self.metadata['feature_names'],
                 class_names=self.metadata['classes'],
                 filled=True,
                 max_depth=max_depth,
                 fontsize=8)
        plt.tight_layout()
        
        output_path = 'reports/decision_tree_plot.png'
        os.makedirs('reports', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {output_path}")
        plt.close()
    
    def plot_sklearn_feature_importance(self):
        """Plot feature importance from sklearn tree"""
        print("Creating sklearn feature importance plot...")
        
        with open(os.path.join(self.model_dir, 'sklearn_tree.pkl'), 'rb') as f:
            clf = pickle.load(f)
        
        feature_importance = pd.DataFrame({
            'feature': self.metadata['feature_names'],
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
        plt.xlabel('Importance')
        plt.title('Top 15 Features - Sklearn Decision Tree')
        plt.tight_layout()
        
        output_path = 'reports/sklearn_feature_importance.png'
        os.makedirs('reports', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {output_path}")
        plt.close()
        
        return feature_importance
    
    def plot_lgb_feature_importance(self):
        """Plot feature importance from LightGBM"""
        print("Creating LightGBM feature importance plot...")
        
        lgb_model = lgb.Booster(model_file=os.path.join(self.model_dir, 'lgb_model.pkl'))
        
        feature_importance = pd.DataFrame({
            'feature': self.metadata['feature_names'],
            'importance': lgb_model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
        plt.xlabel('Importance')
        plt.title('Top 15 Features - LightGBM')
        plt.tight_layout()
        
        output_path = 'reports/lgb_feature_importance.png'
        os.makedirs('reports', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {output_path}")
        plt.close()
        
        return feature_importance
    
    def plot_class_distribution(self, data_csv):
        """Plot class distribution"""
        print("Creating class distribution plot...")
        
        df = pd.read_csv(data_csv)
        if 'language' not in df.columns:
            print("  (Skipping - 'language' column not found)")
            return
        
        class_counts = df['language'].value_counts().sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        class_counts.plot(kind='barh')
        plt.xlabel('Number of Samples')
        plt.title('Class Distribution')
        plt.tight_layout()
        
        output_path = 'reports/class_distribution.png'
        os.makedirs('reports', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {output_path}")
        plt.close()
    
    def plot_feature_distributions(self, features_csv):
        """Plot distributions of selected features"""
        print("Creating feature distribution plots...")
        
        df = pd.read_csv(features_csv)
        
        # Select numeric features
        numeric_features = [col for col in df.select_dtypes(include=[np.number]).columns][:8]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for idx, feature in enumerate(numeric_features):
            axes[idx].hist(df[feature].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'Distribution: {feature}')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        output_path = 'reports/feature_distributions.png'
        os.makedirs('reports', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {output_path}")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name='sklearn'):
        """Plot confusion matrix"""
        print(f"Creating confusion matrix plot for {model_name}...")
        
        from sklearn.metrics import confusion_matrix as calc_confusion_matrix
        
        cm = calc_confusion_matrix(y_true, y_pred)
        
        # Only plot if reasonable size
        n_classes = len(self.metadata['classes'])
        if n_classes > 20:
            print(f"  (Skipping - too many classes ({n_classes}))")
            return
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.metadata['classes'],
                   yticklabels=self.metadata['classes'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = f'reports/confusion_matrix_{model_name}.png'
        os.makedirs('reports', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {output_path}")
        plt.close()

def main():
    """Main visualization pipeline"""
    print("Starting visualization pipeline...\n")
    
    visualizer = Visualizer()
    
    # Create all visualizations
    visualizer.plot_sklearn_tree(max_depth=4)
    visualizer.plot_sklearn_feature_importance()
    visualizer.plot_lgb_feature_importance()
    visualizer.plot_class_distribution('data/raw_dataset.csv')
    visualizer.plot_feature_distributions('data/features.csv')
    
    print("\n✓ All visualizations created successfully!")

if __name__ == "__main__":
    main()
