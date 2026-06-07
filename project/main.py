"""
Main Pipeline: Orchestrate the complete decision tree classification workflow.

Steps:
  1. Data Preparation   — download + clean from The Stack
  2. Feature Engineering — extract structural/textual features
  3. Model Training      — sklearn DecisionTree + LightGBM
  4. Model Evaluation    — Precision / Recall / Accuracy / F1-macro +
                           UNKNOWN detection with calibrated threshold
  5. Visualization       — tree plot, feature importance, threshold curve
"""

import sys
import os
import json
import numpy as np
import pandas as pd

from data_preparation   import download_dataset
from feature_engineering import process_features
from model_training     import ModelTrainer
from model_evaluation   import ModelEvaluator
from visualization      import Visualizer


def main():
    print("=" * 70)
    print("DECISION TREE TEXT CLASSIFIER — COMPLETE PIPELINE")
    print("=" * 70)

    os.makedirs('data',    exist_ok=True)
    os.makedirs('models',  exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    # ------------------------------------------------------------------ #
    # STEP 1: Data Preparation
    # ------------------------------------------------------------------ #
    print("\n[STEP 1/5] Data Preparation")
    print("-" * 70)
    download_dataset(samples_per_class=1000)

    # ------------------------------------------------------------------ #
    # STEP 2: Feature Engineering
    # ------------------------------------------------------------------ #
    print("\n[STEP 2/5] Feature Engineering")
    print("-" * 70)
    process_features('data/raw_dataset.csv', 'data/features.csv')

    # ------------------------------------------------------------------ #
    # STEP 3: Model Training
    # ------------------------------------------------------------------ #
    print("\n[STEP 3/5] Model Training")
    print("-" * 70)
    trainer = ModelTrainer('data/features.csv', samples_per_class=1000)
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     X_unknown_val, X_unknown_test,
     y_unknown_val, y_unknown_test) = trainer.prepare_data()

    print(f"\nData shapes:")
    print(f"  X_train:       {X_train.shape}")
    print(f"  X_val:         {X_val.shape}")
    print(f"  X_test:        {X_test.shape}")
    print(f"  X_unknown_val: {X_unknown_val.shape}")
    print(f"  X_unknown_test:{X_unknown_test.shape}")
    print(f"  Classes:       {len(trainer.class_labels)}")

    sklearn_clf = trainer.train_sklearn_tree(X_train, y_train, max_depth=15)
    lgb_model   = trainer.train_lightgbm(X_train, y_train, X_val, y_val)
    trainer.save_models(sklearn_clf, lgb_model)
    trainer.save_root_split(sklearn_clf)

    # Save splits for evaluation
    def _save(X, y_labels, y_enc, path):
        df = pd.DataFrame(X, columns=trainer.feature_columns)
        df['language'] = y_labels
        df['y_true']   = y_enc
        df.to_csv(path, index=False)

    _save(X_val,          trainer.label_encoder.inverse_transform(y_val),  y_val,          'data/val_known.csv')
    _save(X_test,         trainer.label_encoder.inverse_transform(y_test), y_test,         'data/test_known.csv')
    _save(X_unknown_val,  y_unknown_val,                                   y_unknown_val,  'data/val_unknown.csv')
    _save(X_unknown_test, y_unknown_test,                                  y_unknown_test, 'data/test_unknown.csv')

    print("\n✓ Splits saved.")

    # ------------------------------------------------------------------ #
    # STEP 4: Model Evaluation
    # ------------------------------------------------------------------ #
    print("\n[STEP 4/5] Model Evaluation")
    print("-" * 70)
    evaluator = ModelEvaluator()

    results_sklearn = evaluator.evaluate_sklearn(
        X_known_test=X_test,
        y_known_test=y_test,
        X_unknown_test=X_unknown_test,
        X_known_val=X_val,
        y_known_val=y_val,
        X_unknown_val=X_unknown_val,
    )
    results_lgb = evaluator.evaluate_lightgbm(
        X_known_test=X_test,
        y_known_test=y_test,
        X_unknown_test=X_unknown_test,
        X_known_val=X_val,
        y_known_val=y_val,
        X_unknown_val=X_unknown_val,
    )

    comparison = evaluator.compare_models(results_sklearn, results_lgb)
    comparison.to_csv('reports/model_comparison.csv', index=False)

    # Persist threshold table for the curve plot
    _save_threshold_table(evaluator, results_lgb,
                          X_val, y_val, X_unknown_val)

    # ------------------------------------------------------------------ #
    # STEP 5: Visualization
    # ------------------------------------------------------------------ #
    print("\n[STEP 5/5] Visualization")
    print("-" * 70)
    visualizer = Visualizer()
    visualizer.plot_sklearn_tree(max_depth=4)
    visualizer.plot_sklearn_feature_importance()
    visualizer.plot_lgb_feature_importance()
    visualizer.plot_class_distribution('data/raw_dataset.csv')
    visualizer.plot_feature_distributions('data/features.csv')
    visualizer.plot_threshold_curve()            # threshold calibration curve

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated artifacts:")
    print("  models/  : sklearn_tree.pkl, lgb_model.pkl, metadata.json, label_encoder.pkl")
    print("  data/    : raw_dataset.csv, features.csv, test_known.csv, test_unknown.csv")
    print("  reports/ : model_comparison.csv, threshold_table.csv,")
    print("             decision_tree_plot.png, sklearn_feature_importance.png,")
    print("             lgb_feature_importance.png, threshold_curve.png,")
    print("             class_distribution.png, feature_distributions.png")
    print("=" * 70)


def _save_threshold_table(evaluator, results_lgb, X_val, y_val, X_unknown_val):
    """Sweep thresholds and save table + find best for the curve plot."""
    try:
        from model_evaluation import ModelEvaluator as ME
        thresholds = np.linspace(0.2, 0.95, 76)
        rows = []

        xkv = ME._fillna_matrix(evaluator, X_val)
        xuk = ME._fillna_matrix(evaluator, X_unknown_val)

        if xkv is None or xuk is None or xkv.shape[0] == 0 or xuk.shape[0] == 0:
            return

        proba_known   = evaluator.lgb_model.predict(xkv)
        proba_unknown = evaluator.lgb_model.predict(xuk)
        known_targets = evaluator.label_encoder.inverse_transform(y_val)

        for thr in thresholds:
            kp = ['UNKNOWN' if p.max() < thr
                  else evaluator.label_encoder.classes_[p.argmax()]
                  for p in proba_known]
            up = ['UNKNOWN' if p.max() < thr
                  else evaluator.label_encoder.classes_[p.argmax()]
                  for p in proba_unknown]

            known_acc   = np.mean([a == b for a, b in zip(kp, known_targets)])
            unk_reject  = np.mean([p == 'UNKNOWN' for p in up])
            denom = known_acc + unk_reject
            balance = (2 * known_acc * unk_reject / denom) if denom > 0 else 0.0

            rows.append({'threshold': round(float(thr), 3),
                         'known_acc': round(known_acc, 4),
                         'unknown_reject': round(unk_reject, 4),
                         'balance': round(balance, 4)})

        df = pd.DataFrame(rows)
        df.to_csv('reports/threshold_table.csv', index=False)
        print("✓ Saved threshold table -> reports/threshold_table.csv")
    except Exception as e:
        print(f"  (threshold table skipped: {e})")


if __name__ == "__main__":
    try:
        import pandas, numpy, sklearn, lightgbm, matplotlib, seaborn, datasets
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    main()
