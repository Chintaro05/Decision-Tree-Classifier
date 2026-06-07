"""
Evaluation: Evaluate models with comprehensive metrics and unknown format handling
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (classification_report, accuracy_score, 
                             f1_score, precision_score, recall_score,
                             confusion_matrix)
import pickle
import json
import os
import lightgbm as lgb

class ModelEvaluator:
    """Evaluate models with various metrics"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        with open(os.path.join(self.model_dir, 'sklearn_tree.pkl'), 'rb') as f:
            self.sklearn_model = pickle.load(f)
        
        self.lgb_model = lgb.Booster(model_file=os.path.join(self.model_dir, 'lgb_model.pkl'))
        
        with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        with open(os.path.join(self.model_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        print("✓ Models loaded successfully")

    def calibrate_threshold(self, model, X_known_val, y_known_val, X_unknown_val):
        """Search for the best confidence threshold using held-out unknown data."""
        print("\nCalibrating unknown detection threshold...")
        if (
            X_known_val is None or
            X_unknown_val is None or
            X_known_val.shape[0] == 0 or
            X_unknown_val.shape[0] == 0
        ):
            print("  No validation data available for threshold calibration.")
            return 0.6, {}

        # if model is Sklearn 
        if hasattr(model, 'predict_proba'):
            proba_known = model.predict_proba(X_known_val)
            proba_unknown = model.predict_proba(X_unknown_val)
        # if model is LightGBM 
        else:
            proba_known = model.predict(X_known_val)
            proba_unknown = model.predict(X_unknown_val)
        known_targets = self.label_encoder.inverse_transform(y_known_val)

        thresholds = np.linspace(0.2, 0.95, 76)
        best_score = -1.0
        best_threshold = 0.6
        best_metrics = {}

        for threshold in thresholds:
            known_preds = []
            for p in proba_known:
                if p.max() < threshold:
                    known_preds.append('UNKNOWN')
                else:
                    known_preds.append(self.label_encoder.classes_[p.argmax()])

            unknown_preds = []
            for p in proba_unknown:
                if p.max() < threshold:
                    unknown_preds.append('UNKNOWN')
                else:
                    unknown_preds.append(self.label_encoder.classes_[p.argmax()])

            known_acc = np.mean([pred == truth for pred, truth in zip(known_preds, known_targets)])
            true_unknown = np.array([pred == 'UNKNOWN' for pred in unknown_preds])
            unknown_recall = true_unknown.mean() if len(true_unknown) > 0 else 0.0
            false_unknown = np.sum([pred == 'UNKNOWN' for pred in known_preds])
            precision_denom = false_unknown + np.sum(true_unknown)
            unknown_precision = np.sum(true_unknown) / precision_denom if precision_denom > 0 else 0.0
            combined_score = 0.6 * known_acc + 0.4 * unknown_recall

            if combined_score > best_score:
                best_score = combined_score
                best_threshold = threshold
                best_metrics = {
                    'known_acc': known_acc,
                    'unknown_recall': unknown_recall,
                    'unknown_precision': unknown_precision,
                    'combined_score': combined_score,
                }

        print(f"  Best threshold: {best_threshold:.2f}")
        print(f"  Known accuracy at threshold: {best_metrics['known_acc']:.4f}")
        print(f"  Unknown recall at threshold: {best_metrics['unknown_recall']:.4f}")
        print(f"  Unknown precision at threshold: {best_metrics['unknown_precision']:.4f}")
        return best_threshold, best_metrics

    def _apply_threshold(self, proba, threshold):
        predictions = []
        for p in proba:
            if p.max() < threshold:
                predictions.append('UNKNOWN')
            else:
                predictions.append(self.label_encoder.classes_[p.argmax()])
        return np.array(predictions)

    def _compute_unknown_metrics(self, known_preds, unknown_preds, known_targets):
        true_unknown = np.array([pred == 'UNKNOWN' for pred in unknown_preds])
        predicted_unknown_known = np.array([pred == 'UNKNOWN' for pred in known_preds])
        tp = true_unknown.sum()
        fp = predicted_unknown_known.sum()
        fn = len(unknown_preds) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        known_acc = np.mean([pred == truth for pred, truth in zip(known_preds, known_targets)])
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'known_accuracy': known_acc,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }

    def _fillna_matrix(self, X):
        if X is None or X.shape[0] == 0:
            return X
        df = pd.DataFrame(X)
        return df.fillna(df.mean()).values
    
    def evaluate_sklearn(
        self,
        X_known_test,
        y_known_test,
        X_unknown_test,
        threshold=None,
        X_known_val=None,
        y_known_val=None,
        X_unknown_val=None,
    ):
        """Evaluate sklearn model and unknown detection."""
        print("\n" + "="*60)
        print("Sklearn DecisionTreeClassifier Evaluation")
        print("="*60)

        X_known_test = self._fillna_matrix(X_known_test)
        X_unknown_test = self._fillna_matrix(X_unknown_test)
        X_known_val = self._fillna_matrix(X_known_val)
        X_unknown_val = self._fillna_matrix(X_unknown_val)

        if threshold is None:
            threshold, metrics = self.calibrate_threshold(
                self.sklearn_model,
                X_known_val,
                y_known_val,
                X_unknown_val,
            )
        else:
            metrics = {}

        proba_known = self.sklearn_model.predict_proba(X_known_test)
        y_pred_raw = np.argmax(proba_known, axis=1)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred_raw)

        proba_unknown = self.sklearn_model.predict_proba(X_unknown_test) if X_unknown_test.shape[0] > 0 else np.empty((0, len(self.label_encoder.classes_)))
        y_pred_threshold_known = self._apply_threshold(proba_known, threshold)
        y_pred_threshold_unknown = self._apply_threshold(proba_unknown, threshold)

        unknown_metrics = self._compute_unknown_metrics(
            y_pred_threshold_known,
            y_pred_threshold_unknown,
            self.label_encoder.inverse_transform(y_known_test),
        )

        accuracy = accuracy_score(y_known_test, y_pred_raw)
        report = classification_report(
            y_known_test,
            y_pred_raw,
            target_names=list(self.label_encoder.classes_),
            digits=4,
            zero_division=0,
        )
        macro_f1 = f1_score(y_known_test, y_pred_raw, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_known_test, y_pred_raw, average='weighted', zero_division=0)

        print(f"\nThreshold used: {threshold:.2f}")
        print(f"Accuracy (known classes): {accuracy:.4f}")
        print(f"Unknown detection precision: {unknown_metrics['precision']:.4f}")
        print(f"Unknown detection recall: {unknown_metrics['recall']:.4f}")
        print(f"Unknown detection F1: {unknown_metrics['f1']:.4f}")
        print("\nDetailed Classification Report:")
        print(report)
        print(f"Macro F1-Score: {macro_f1:.4f}")
        print(f"Weighted F1-Score: {weighted_f1:.4f}")

        return {
            'model': 'sklearn_tree',
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'threshold': threshold,
            'unknown_metrics': unknown_metrics,
            'unknown_count': X_unknown_test.shape[0] if X_unknown_test is not None else 0,
            'report': report,
        }
    
    def evaluate_lightgbm(
        self,
        X_known_test,
        y_known_test,
        X_unknown_test,
        threshold=None,
        X_known_val=None,
        y_known_val=None,
        X_unknown_val=None,
    ):
        """Evaluate LightGBM model and unknown detection."""
        print("\n" + "="*60)
        print("LightGBM Classifier Evaluation")
        print("="*60)

        X_known_test = self._fillna_matrix(X_known_test)
        X_unknown_test = self._fillna_matrix(X_unknown_test)
        X_known_val = self._fillna_matrix(X_known_val)
        X_unknown_val = self._fillna_matrix(X_unknown_val)

        if threshold is None:
            threshold, metrics = self.calibrate_threshold(
                self.lgb_model,
                X_known_val,
                y_known_val,
                X_unknown_val,
            )
        else:
            metrics = {}

        proba_known = self.lgb_model.predict(X_known_test)
        y_pred_raw = np.argmax(proba_known, axis=1)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred_raw)

        proba_unknown = self.lgb_model.predict(X_unknown_test) if X_unknown_test.shape[0] > 0 else np.empty((0, len(self.label_encoder.classes_)))
        y_pred_threshold_known = self._apply_threshold(proba_known, threshold)
        y_pred_threshold_unknown = self._apply_threshold(proba_unknown, threshold)

        unknown_metrics = self._compute_unknown_metrics(
            y_pred_threshold_known,
            y_pred_threshold_unknown,
            self.label_encoder.inverse_transform(y_known_test),
        )

        accuracy = accuracy_score(y_known_test, y_pred_raw)
        report = classification_report(
            y_known_test,
            y_pred_raw,
            target_names=list(self.label_encoder.classes_),
            digits=4,
            zero_division=0,
        )
        macro_f1 = f1_score(y_known_test, y_pred_raw, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_known_test, y_pred_raw, average='weighted', zero_division=0)

        print(f"\nThreshold used: {threshold:.2f}")
        print(f"Accuracy (known classes): {accuracy:.4f}")
        print(f"Unknown detection precision: {unknown_metrics['precision']:.4f}")
        print(f"Unknown detection recall: {unknown_metrics['recall']:.4f}")
        print(f"Unknown detection F1: {unknown_metrics['f1']:.4f}")
        print("\nDetailed Classification Report:")
        print(report)
        print(f"Macro F1-Score: {macro_f1:.4f}")
        print(f"Weighted F1-Score: {weighted_f1:.4f}")

        return {
            'model': 'lightgbm',
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'threshold': threshold,
            'unknown_metrics': unknown_metrics,
            'unknown_count': X_unknown_test.shape[0] if X_unknown_test is not None else 0,
            'report': report,
        }
    
    def compare_models(self, results_sklearn, results_lgb):
        """Compare model performance"""
        print("\n" + "="*60)
        print("Model Comparison")
        print("="*60)
        
        comparison = pd.DataFrame({
            'Model': ['Sklearn Tree', 'LightGBM'],
            'Accuracy': [results_sklearn['accuracy'], results_lgb['accuracy']],
            'Macro F1': [results_sklearn['macro_f1'], results_lgb['macro_f1']],
            'Weighted F1': [results_sklearn['weighted_f1'], results_lgb['weighted_f1']],
            'Unknown Samples': [results_sklearn['unknown_count'], results_lgb['unknown_count']]
        })
        
        print("\n" + comparison.to_string(index=False))
        
        winner = 'LightGBM' if results_lgb['macro_f1'] > results_sklearn['macro_f1'] else 'Sklearn Tree'
        print(f"\n🏆 Winner (by Macro F1-Score): {winner}")
        
        return comparison

def main():
    """Main evaluation pipeline"""
    print("Starting model evaluation pipeline...\n")
    
    # Load known/unknown test data
    known_df = pd.read_csv('data/test_known.csv')
    unknown_df = pd.read_csv('data/test_unknown.csv')
    
    feature_columns = [col for col in known_df.columns if col not in ['language', 'y_true']]
    
    X_known_test = known_df[feature_columns].values
    y_known_test = known_df['y_true'].values
    X_unknown_test = unknown_df[feature_columns].values
    y_unknown_test = unknown_df['y_true'].values
    
    # Load validation data if present
    val_known_path = 'data/val_known.csv'
    val_unknown_path = 'data/val_unknown.csv'
    if os.path.exists(val_known_path) and os.path.exists(val_unknown_path):
        val_known_df = pd.read_csv(val_known_path)
        val_unknown_df = pd.read_csv(val_unknown_path)
        X_known_val = val_known_df[feature_columns].values
        y_known_val = val_known_df['y_true'].values
        X_unknown_val = val_unknown_df[feature_columns].values
        y_unknown_val = val_unknown_df['y_true'].values
    else:
        X_known_val = None
        y_known_val = None
        X_unknown_val = None
        y_unknown_val = None
    
    # Evaluate models
    evaluator = ModelEvaluator()
    
    results_sklearn = evaluator.evaluate_sklearn(
        X_known_test=X_known_test,
        y_known_test=y_known_test,
        X_unknown_test=X_unknown_test,
        X_known_val=X_known_val,
        y_known_val=y_known_val,
        X_unknown_val=X_unknown_val,
    )
    results_lgb = evaluator.evaluate_lightgbm(
        X_known_test=X_known_test,
        y_known_test=y_known_test,
        X_unknown_test=X_unknown_test,
        X_known_val=X_known_val,
        y_known_val=y_known_val,
        X_unknown_val=X_unknown_val,
    )
    
    comparison = evaluator.compare_models(results_sklearn, results_lgb)
    
    # Save results
    os.makedirs('reports', exist_ok=True)
    comparison.to_csv('reports/model_comparison.csv', index=False)
    print(f"\n✓ Saved comparison to reports/model_comparison.csv")
    
    # Save detailed results
    with open('reports/sklearn_evaluation.json', 'w') as f:
        json.dump({
            'accuracy': float(results_sklearn['accuracy']),
            'macro_f1': float(results_sklearn['macro_f1']),
            'weighted_f1': float(results_sklearn['weighted_f1']),
            'unknown_samples': int(results_sklearn['unknown_count']),
            'unknown_precision': float(results_sklearn['unknown_metrics']['precision']),
            'unknown_recall': float(results_sklearn['unknown_metrics']['recall']),
            'unknown_f1': float(results_sklearn['unknown_metrics']['f1'])
        }, f, indent=2)
    
    with open('reports/lgb_evaluation.json', 'w') as f:
        json.dump({
            'accuracy': float(results_lgb['accuracy']),
            'macro_f1': float(results_lgb['macro_f1']),
            'weighted_f1': float(results_lgb['weighted_f1']),
            'unknown_samples': int(results_lgb['unknown_count']),
            'unknown_precision': float(results_lgb['unknown_metrics']['precision']),
            'unknown_recall': float(results_lgb['unknown_metrics']['recall']),
            'unknown_f1': float(results_lgb['unknown_metrics']['f1'])
        }, f, indent=2)
    
    print(f"✓ Saved detailed results to reports/")
    
    return evaluator, results_sklearn, results_lgb

if __name__ == "__main__":
    main()
