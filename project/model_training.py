"""
Model Training: Train both sklearn DecisionTreeClassifier and LightGBM
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import pickle
import os
import json
from data_preparation import KNOWN_LANGUAGES, UNKNOWN_LANGUAGES

class ModelTrainer:
    """Train and save models"""
    
    def __init__(self, features_csv, samples_per_class=1000):
        self.df = pd.read_csv(features_csv)
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.class_labels = None
        self.samples_per_class = samples_per_class
        
    def prepare_data(self, known_test_size=0.2, val_size=0.2):
        """Prepare training, validation, and held-out unknown data"""
        # Feature columns (all except language and file_id)
        self.feature_columns = [col for col in self.df.columns 
                               if col not in ['language', 'file_id']]
        
        known_df = self.df[self.df['language'].isin(KNOWN_LANGUAGES)].copy()
        unknown_df = self.df[self.df['language'].isin(UNKNOWN_LANGUAGES)].copy()
        
        # Ensure a consistent sample size per known class
        known_df = known_df.groupby('language', group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), self.samples_per_class), random_state=42)
        ).reset_index(drop=True)
        
        self.class_labels = sorted(known_df['language'].unique())
        self.label_encoder.fit(self.class_labels)
        
        X_known = known_df[self.feature_columns]
        y_known = self.label_encoder.transform(known_df['language'])
        
        # Split known classes into train / validation / test
        X_train, X_test, y_train, y_test = train_test_split(
            X_known, y_known, test_size=known_test_size, stratify=y_known, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, stratify=y_train, random_state=42
        )
        
        # Hold out unseen formats for true unknown detection evaluation
        if len(unknown_df) > 0:
            unknown_df = unknown_df.groupby('language', group_keys=False).apply(
                lambda g: g.sample(n=min(len(g), self.samples_per_class), random_state=42)
            ).reset_index(drop=True)
            X_unknown_val, X_unknown_test = train_test_split(
                unknown_df[self.feature_columns],
                test_size=0.5,
                stratify=unknown_df['language'],
                random_state=42
            )
            y_unknown_val = np.array(['UNKNOWN'] * len(X_unknown_val))
            y_unknown_test = np.array(['UNKNOWN'] * len(X_unknown_test))
        else:
            X_unknown_val = pd.DataFrame(columns=self.feature_columns)
            X_unknown_test = pd.DataFrame(columns=self.feature_columns)
            y_unknown_val = np.array([])
            y_unknown_test = np.array([])
        
        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            X_unknown_val,
            X_unknown_test,
            y_unknown_val,
            y_unknown_test
        )
    
    def train_sklearn_tree(self, X_train, y_train, max_depth=10):
        """Train sklearn DecisionTreeClassifier"""
        print("\n" + "="*60)
        print("Training sklearn DecisionTreeClassifier")
        print("="*60)
        
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=42
        )
        
        X_train_filled = self._fillna_for_sklearn(X_train)
        clf.fit(X_train_filled, y_train)
        
        print(f"✓ Tree depth: {clf.get_depth()}")
        print(f"✓ Leaves: {clf.get_n_leaves()}")
        
        return clf
    
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None):
        """Train LightGBM classifier"""
        print("\n" + "="*60)
        print("Training LightGBM Classifier")
        print("="*60)

        if X_val is not None and len(X_val) > 0:
            model = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=8,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=30,
                verbose=False
            )
        else:
            model = lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=8,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_train, y_train)

        rounds = model.best_iteration_ if hasattr(model, 'best_iteration_') else model.n_estimators_
        print(f"✓ LightGBM trained with {rounds} boosting rounds")

        return model
    
    def save_models(self, sklearn_model, lgb_model, output_dir='models'):
        """Save trained models"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sklearn model
        sklearn_path = os.path.join(output_dir, 'sklearn_tree.pkl')
        with open(sklearn_path, 'wb') as f:
            pickle.dump(sklearn_model, f)
        print(f"✓ Saved sklearn model to {sklearn_path}")
        
        # Save LightGBM model
        lgb_txt_path = os.path.join(output_dir, 'lgb_model.txt')
        if hasattr(lgb_model, 'booster_'):
            lgb_model.booster_.save_model(lgb_txt_path)
        else:
            lgb_model.save_model(lgb_txt_path)
        print(f"✓ Saved LightGBM booster text to {lgb_txt_path}")

        lgb_pickle_path = os.path.join(output_dir, 'lgb_model.pkl')
        with open(lgb_pickle_path, 'wb') as f:
            pickle.dump({'model': lgb_model, 'features': self.feature_columns}, f)
        print(f"✓ Saved LightGBM wrapper to {lgb_pickle_path}")
        
        # Save encoder and metadata
        metadata = {
            'classes': list(self.class_labels),
            'feature_names': self.feature_columns,
            'n_features': len(self.feature_columns),
            'n_classes': len(self.class_labels)
        }
        
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata to {metadata_path}")
        
        # Save label encoder
        encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✓ Saved label encoder to {encoder_path}")

    def save_root_split(self, clf, output_dir='reports'):
        os.makedirs(output_dir, exist_ok=True)
        root_index = clf.tree_.feature[0]
        root_feature = self.feature_columns[root_index] if root_index >= 0 else 'leaf'
        root_threshold = clf.tree_.threshold[0] if root_index >= 0 else None
        report_path = os.path.join(output_dir, 'root_split.txt')

        with open(report_path, 'w') as f:
            f.write(f"Root split feature: {root_feature}\n")
            f.write(f"Root split threshold: {root_threshold}\n")

        print(f"✓ Saved root split info to {report_path}")

    def _fillna_for_sklearn(self, X):
        return X.fillna(X.mean())

def main():
    """Main training pipeline"""
    print("Starting model training pipeline...")
    
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
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  Unknown val: {X_unknown_val.shape}")
    print(f"  Unknown test: {X_unknown_test.shape}")
    print(f"  Classes: {len(trainer.class_labels)}")
    
    # Train sklearn tree
    sklearn_clf = trainer.train_sklearn_tree(X_train, y_train, max_depth=15)
    
    # Train LightGBM with validation
    lgb_model = trainer.train_lightgbm(X_train, y_train, X_val, y_val)
    
    # Save models and metadata
    trainer.save_models(sklearn_clf, lgb_model)
    trainer.save_root_split(sklearn_clf, output_dir='reports')
    
    # Save known and unknown test sets for evaluation
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

    # Save validation sets for threshold calibration and held-out unknown evaluation
    if X_val.shape[0] > 0:
        val_known = pd.DataFrame(X_val, columns=trainer.feature_columns)
        val_known['language'] = trainer.label_encoder.inverse_transform(y_val)
        val_known['y_true'] = y_val
        val_known.to_csv('data/val_known.csv', index=False)

    if X_unknown_val.shape[0] > 0:
        val_unknown = pd.DataFrame(X_unknown_val, columns=trainer.feature_columns)
        val_unknown['language'] = 'UNKNOWN'
        val_unknown['y_true'] = y_unknown_val
        val_unknown.to_csv('data/val_unknown.csv', index=False)

    if X_val.shape[0] > 0 or X_unknown_val.shape[0] > 0:
        print(f"✓ Saved validation sets to data/val_known.csv and data/val_unknown.csv")
    
    return (
        trainer,
        sklearn_clf,
        lgb_model,
        X_test,
        y_test,
        X_unknown_test,
        y_unknown_test,
        X_unknown_val,
        y_unknown_val,
    )

if __name__ == "__main__":
    main()
