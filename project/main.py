"""
Main Pipeline: Orchestrate entire decision tree classification workflow
"""

import sys
import os
import pandas as pd
from data_preparation import download_dataset
from feature_engineering import process_features
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from visualization import Visualizer

def main():
    """Run complete pipeline"""
    
    print("="*70)
    print("DECISION TREE TEXT CLASSIFIER - COMPLETE PIPELINE")
    print("="*70)
    
    # Step 1: Data Preparation
    print("\n[STEP 1/5] Data Preparation")
    print("-" * 70)
    print("Downloading/preparing dataset...")
    df = download_dataset(samples_per_class=1000)
    
    # Step 2: Feature Engineering
    print("\n[STEP 2/5] Feature Engineering")
    print("-" * 70)
    print("Extracting features from raw content...")
    features_df = process_features('data/raw_dataset.csv', 'data/features.csv')
    
    # Step 3: Model Training
    print("\n[STEP 3/5] Model Training")
    print("-" * 70)
    print("Training sklearn and LightGBM models...")
    trainer = ModelTrainer('data/features.csv', samples_per_class=1000)
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_unknown_val,
        X_unknown_test,
        y_unknown_val,
        y_unknown_test,
    ) = trainer.prepare_data(known_test_size=0.2, val_size=0.2)
    sklearn_clf = trainer.train_sklearn_tree(X_train, y_train, max_depth=15)
    lgb_model = trainer.train_lightgbm(X_train, y_train, X_val, y_val)
    trainer.save_models(sklearn_clf, lgb_model)
    
    # Save validation and test splits for evaluation
    val_known = pd.DataFrame(X_val, columns=trainer.feature_columns)
    val_known['language'] = trainer.label_encoder.inverse_transform(y_val)
    val_known['y_true'] = y_val
    val_known.to_csv('data/val_known.csv', index=False)
    
    val_unknown = pd.DataFrame(X_unknown_val, columns=trainer.feature_columns)
    val_unknown['language'] = 'UNKNOWN'
    val_unknown['y_true'] = y_unknown_val
    val_unknown.to_csv('data/val_unknown.csv', index=False)
    
    known_test = pd.DataFrame(X_test, columns=trainer.feature_columns)
    known_test['language'] = trainer.label_encoder.inverse_transform(y_test)
    known_test['y_true'] = y_test
    known_test.to_csv('data/test_known.csv', index=False)
    
    unknown_test = pd.DataFrame(X_unknown_test, columns=trainer.feature_columns)
    unknown_test['language'] = 'UNKNOWN'
    unknown_test['y_true'] = y_unknown_test
    unknown_test.to_csv('data/test_unknown.csv', index=False)
    
    test_data = pd.concat([known_test, unknown_test], ignore_index=True)
    test_data.to_csv('data/test_set.csv', index=False)
    print(f"\n✓ Saved known/unknown test sets to data/test_known.csv and data/test_unknown.csv")
    print(f"✓ Saved combined test set to data/test_set.csv")
    
    # Step 4: Model Evaluation
    print("\n[STEP 4/5] Model Evaluation")
    print("-" * 70)
    print("Evaluating models...")
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
    
    # Step 5: Visualization
    print("\n[STEP 5/5] Visualization")
    print("-" * 70)
    print("Creating visualizations...")
    visualizer = Visualizer()
    visualizer.plot_sklearn_tree(max_depth=4)
    visualizer.plot_sklearn_feature_importance()
    visualizer.plot_lgb_feature_importance()
    visualizer.plot_class_distribution('data/raw_dataset.csv')
    visualizer.plot_feature_distributions('data/features.csv')
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated artifacts:")
    print("  Models: models/")
    print("    - sklearn_tree.pkl")
    print("    - lgb_model.pkl")
    print("    - metadata.json")
    print("    - label_encoder.pkl")
    print("  Data: data/")
    print("    - raw_dataset.csv")
    print("    - features.csv")
    print("    - test_set.csv")
    print("  Reports: reports/")
    print("    - model_comparison.csv")
    print("    - sklearn_evaluation.json")
    print("    - lgb_evaluation.json")
    print("    - decision_tree_plot.png")
    print("    - sklearn_feature_importance.png")
    print("    - lgb_feature_importance.png")
    print("    - class_distribution.png")
    print("    - feature_distributions.png")
    print("\n" + "="*70)

if __name__ == "__main__":
    # Check if all dependencies are available
    try:
        import pandas
        import numpy
        import sklearn
        import lightgbm
        import matplotlib
        import seaborn
        import datasets
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install requirements.txt first")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    main()
