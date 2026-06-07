"""
Visualization: Create plots for decision tree, feature importance,
class distribution, feature distributions, and threshold calibration curve.

FIX: Added plot_threshold_curve() — required by the assignment to visualise
     the confidence threshold calibration for UNKNOWN detection.
"""

import matplotlib
matplotlib.use('Agg')
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
    """Create visualizations for model analysis."""

    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.load_metadata()

    def load_metadata(self):
        with open(os.path.join(self.model_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)

    # ------------------------------------------------------------------ #
    # Decision tree
    # ------------------------------------------------------------------ #
    def plot_sklearn_tree(self, max_depth=3):
        """Plot first N levels of the sklearn decision tree."""
        print("Creating sklearn tree plot...")

        with open(os.path.join(self.model_dir, 'sklearn_tree.pkl'), 'rb') as f:
            clf = pickle.load(f)

        plt.figure(figsize=(25, 12))
        plot_tree(
            clf,
            feature_names=self.metadata['feature_names'],
            class_names=self.metadata['classes'],
            filled=True,
            max_depth=max_depth,
            fontsize=7,
            rounded=True,
        )
        plt.title(
            f"Decision Tree (depth {max_depth}) — "
            "feature at root node is the most discriminative",
            fontsize=12,
        )
        plt.tight_layout()

        output_path = 'reports/decision_tree_plot.png'
        os.makedirs('reports', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {output_path}")
        plt.close()

    # ------------------------------------------------------------------ #
    # Feature importance
    # ------------------------------------------------------------------ #
    def plot_sklearn_feature_importance(self):
        """Bar chart of top-15 feature importances from sklearn tree."""
        print("Creating sklearn feature importance plot...")

        with open(os.path.join(self.model_dir, 'sklearn_tree.pkl'), 'rb') as f:
            clf = pickle.load(f)

        fi = pd.DataFrame({
            'feature':    self.metadata['feature_names'],
            'importance': clf.feature_importances_,
        }).sort_values('importance', ascending=True).tail(15)

        plt.figure(figsize=(10, 7))
        plt.barh(fi['feature'], fi['importance'], color='steelblue')
        plt.xlabel('Importance (Gini)')
        plt.title('Top 15 Features — sklearn Decision Tree')
        plt.tight_layout()

        output_path = 'reports/sklearn_feature_importance.png'
        os.makedirs('reports', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {output_path}")
        plt.close()

        print("\nTop-15 features:")
        for _, row in fi.iloc[::-1].iterrows():
            print(f"  {row['feature']:25s}  {row['importance']:.4f}")

        return fi

    def plot_lgb_feature_importance(self):
        """Bar chart of top-15 LightGBM feature importances (gain)."""
        print("Creating LightGBM feature importance plot...")

        lgb_model = lgb.Booster(
            model_file=os.path.join(self.model_dir, 'lgb_model.pkl'))

        fi = pd.DataFrame({
            'feature':    self.metadata['feature_names'],
            'importance': lgb_model.feature_importance(importance_type='gain'),
        }).sort_values('importance', ascending=True).tail(15)

        plt.figure(figsize=(10, 7))
        plt.barh(fi['feature'], fi['importance'], color='darkorange')
        plt.xlabel('Importance (gain)')
        plt.title('Top 15 Features — LightGBM')
        plt.tight_layout()

        output_path = 'reports/lgb_feature_importance.png'
        os.makedirs('reports', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {output_path}")
        plt.close()
        return fi

    # ------------------------------------------------------------------ #
    # Threshold calibration curve  (FIX: new method, required by assignment)
    # ------------------------------------------------------------------ #
    def plot_threshold_curve(self,
                             threshold_data=None,
                             threshold_csv='reports/threshold_table.csv'):
        """
        Plot the confidence-threshold calibration curve showing:
          - known_accuracy vs threshold
          - unknown_reject_rate vs threshold
          - combined balance score vs threshold
          - vertical line at best threshold

        Parameters
        ----------
        threshold_data : dict with keys 'thresholds', 'known_acc',
                         'unknown_reject', 'balance', 'best_threshold'
                         (passed from model_evaluation if available)
        threshold_csv  : fallback — load data from this CSV if threshold_data is None
        """
        print("Creating threshold calibration curve...")

        if threshold_data is not None:
            thr     = threshold_data['thresholds']
            k_acc   = threshold_data['known_acc']
            u_rej   = threshold_data['unknown_reject']
            balance = threshold_data['balance']
            best    = threshold_data['best_threshold']
        elif os.path.exists(threshold_csv):
            df  = pd.read_csv(threshold_csv)
            thr     = df['threshold'].values
            k_acc   = df['known_acc'].values
            u_rej   = df['unknown_reject'].values
            balance = df['balance'].values
            best    = df.loc[df['balance'].idxmax(), 'threshold']
        else:
            print("  (No threshold data — skipping threshold curve)")
            return

        plt.figure(figsize=(9, 5))
        plt.plot(thr, k_acc,   'o-', label='Known accuracy',      color='steelblue')
        plt.plot(thr, u_rej,   's-', label='Unknown reject rate',  color='tomato')
        plt.plot(thr, balance, '^--', label='Balance (combined)',  color='seagreen')
        plt.axvline(best, color='gray', ls=':', lw=1.5,
                    label=f'Best threshold = {best:.2f}')
        plt.xlabel('Confidence threshold')
        plt.ylabel('Rate')
        plt.title('Unknown-detection threshold calibration')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        output_path = 'reports/threshold_curve.png'
        os.makedirs('reports', exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"✓ Saved to {output_path}")
        plt.close()

    # ------------------------------------------------------------------ #
    # Auxiliary plots
    # ------------------------------------------------------------------ #
    def plot_class_distribution(self, data_csv):
        """Horizontal bar chart of samples per class."""
        print("Creating class distribution plot...")

        df = pd.read_csv(data_csv)
        if 'language' not in df.columns:
            print("  (Skipping — 'language' column not found)")
            return

        counts = df['language'].value_counts().sort_values(ascending=True)
        plt.figure(figsize=(12, max(6, len(counts) * 0.3)))
        counts.plot(kind='barh', color='steelblue')
        plt.xlabel('Number of Samples')
        plt.title('Class Distribution')
        plt.tight_layout()

        output_path = 'reports/class_distribution.png'
        os.makedirs('reports', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {output_path}")
        plt.close()

    def plot_feature_distributions(self, features_csv):
        """Histograms of selected numeric features."""
        print("Creating feature distribution plots...")

        df = pd.read_csv(features_csv)
        numeric = [c for c in df.select_dtypes(include=[np.number]).columns][:8]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        for idx, feat in enumerate(numeric):
            axes[idx].hist(df[feat].dropna(), bins=30, alpha=0.7,
                           edgecolor='black', color='steelblue')
            axes[idx].set_title(feat, fontsize=9)
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')

        plt.tight_layout()
        output_path = 'reports/feature_distributions.png'
        os.makedirs('reports', exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"✓ Saved to {output_path}")
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, model_name='sklearn'):
        """Heatmap confusion matrix (skipped if > 20 classes)."""
        from sklearn.metrics import confusion_matrix as _cm

        n_classes = len(self.metadata['classes'])
        if n_classes > 20:
            print(f"  (Skipping confusion matrix — {n_classes} classes)")
            return

        print(f"Creating confusion matrix for {model_name}...")
        cm = _cm(y_true, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.metadata['classes'],
                    yticklabels=self.metadata['classes'])
        plt.title(f'Confusion Matrix — {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_path = f'reports/confusion_matrix_{model_name}.png'
        os.makedirs('reports', exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"✓ Saved to {output_path}")
        plt.close()


def main():
    """Standalone visualization pipeline."""
    print("Starting visualization pipeline...\n")
    visualizer = Visualizer()
    visualizer.plot_sklearn_tree(max_depth=4)
    visualizer.plot_sklearn_feature_importance()
    visualizer.plot_lgb_feature_importance()
    visualizer.plot_class_distribution('data/raw_dataset.csv')
    visualizer.plot_feature_distributions('data/features.csv')
    visualizer.plot_threshold_curve()
    print("\n✓ All visualizations created successfully!")


if __name__ == "__main__":
    main()
