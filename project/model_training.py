"""
Model Training: Train both sklearn DecisionTreeClassifier and LightGBM

FIX: Updated LightGBM API — early_stopping_rounds and verbose_eval
     are deprecated in LightGBM 4.x; replaced with callbacks=[].
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
    """Train and save models."""

    def __init__(self, features_csv, samples_per_class=1000):
        self.df = pd.read_csv(features_csv)
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.class_labels = None
        self.samples_per_class = samples_per_class

    def prepare_data(self, known_test_size=0.2, val_size=0.2):
        """Prepare training, validation, and held-out unknown data."""
        self.feature_columns = [
            col for col in self.df.columns
            if col not in ['language', 'file_id']
        ]

        known_df  = self.df[self.df['language'].isin(KNOWN_LANGUAGES)].copy()
        unknown_df = self.df[self.df['language'].isin(UNKNOWN_LANGUAGES)].copy()

        # Balance known classes — use explicit loop for pandas 2.x compatibility
        balanced_parts = []
        for lang in known_df['language'].unique():
            ldf = known_df[known_df['language'] == lang]
            n   = min(len(ldf), self.samples_per_class)
            balanced_parts.append(ldf.sample(n=n, random_state=42))
        known_df = pd.concat(balanced_parts, ignore_index=True)

        self.class_labels = sorted(known_df['language'].unique())
        self.label_encoder.fit(self.class_labels)

        X_known = known_df[self.feature_columns].fillna(0)
        y_known = self.label_encoder.transform(known_df['language'])

        # Split known → train / val / test
        X_train, X_test, y_train, y_test = train_test_split(
            X_known, y_known,
            test_size=known_test_size, stratify=y_known, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size, stratify=y_train, random_state=42
        )

        # Unknown sets
        if len(unknown_df) > 0:
            unk_parts = []
            for lang in unknown_df['language'].unique():
                ldf = unknown_df[unknown_df['language'] == lang]
                n   = min(len(ldf), self.samples_per_class)
                unk_parts.append(ldf.sample(n=n, random_state=42))
            unknown_df = pd.concat(unk_parts, ignore_index=True)
            X_unk = unknown_df[self.feature_columns].fillna(0)
            X_unknown_val, X_unknown_test = train_test_split(
                X_unk,
                test_size=0.5,
                stratify=unknown_df['language'],
                random_state=42
            )
            y_unknown_val  = np.array(['UNKNOWN'] * len(X_unknown_val))
            y_unknown_test = np.array(['UNKNOWN'] * len(X_unknown_test))
        else:
            X_unknown_val  = pd.DataFrame(columns=self.feature_columns)
            X_unknown_test = pd.DataFrame(columns=self.feature_columns)
            y_unknown_val  = np.array([])
            y_unknown_test = np.array([])

        return (X_train, X_val, X_test,
                y_train, y_val, y_test,
                X_unknown_val, X_unknown_test,
                y_unknown_val, y_unknown_test)

    def train_sklearn_tree(self, X_train, y_train, max_depth=15):
        """Train sklearn DecisionTreeClassifier."""
        print("\n" + "=" * 60)
        print("Training sklearn DecisionTreeClassifier")
        print("=" * 60)

        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=42
        )
        clf.fit(self._fillna(X_train), y_train)

        print(f"✓ Tree depth: {clf.get_depth()}")
        print(f"✓ Leaves: {clf.get_n_leaves()}")
        return clf

    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None):
        """Train LightGBM classifier."""
        print("\n" + "=" * 60)
        print("Training LightGBM Classifier")
        print("=" * 60)

        X_train_filled = self._fillna(X_train)
        train_data = lgb.Dataset(X_train_filled, label=y_train)

        params = {
            'objective':       'multiclass',
            'num_class':       len(self.class_labels),
            'boosting_type':   'gbdt',
            'num_leaves':      31,
            'learning_rate':   0.05,
            'n_estimators':    300,
            'feature_fraction':0.8,
            'bagging_fraction':0.8,
            'bagging_freq':    5,
            'verbose':        -1,
            'metric':          'multi_logloss',
        }

        # FIX: use callbacks instead of deprecated early_stopping_rounds / verbose_eval
        callbacks = [lgb.log_evaluation(period=50)]

        if X_val is not None and len(X_val) > 0:
            val_data = lgb.Dataset(
                self._fillna(X_val), label=y_val, reference=train_data)
            callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))
            model = lgb.train(
                params,
                train_data,
                num_boost_round=300,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=callbacks,
            )
        else:
            model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                callbacks=callbacks,
            )

        rounds = model.current_iteration()
        print(f"✓ LightGBM trained with {rounds} boosting rounds")
        return model

    def save_models(self, sklearn_model, lgb_model, output_dir='models'):
        """Save trained models and metadata."""
        os.makedirs(output_dir, exist_ok=True)

        sklearn_path = os.path.join(output_dir, 'sklearn_tree.pkl')
        with open(sklearn_path, 'wb') as f:
            pickle.dump(sklearn_model, f)
        print(f"✓ Saved sklearn model  -> {sklearn_path}")

        lgb_path = os.path.join(output_dir, 'lgb_model.pkl')
        lgb_model.save_model(lgb_path)
        print(f"✓ Saved LightGBM model -> {lgb_path}")

        metadata = {
            'classes':       list(self.class_labels),
            'feature_names': self.feature_columns,
            'n_features':    len(self.feature_columns),
            'n_classes':     len(self.class_labels),
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)

        print(f"✓ Saved metadata + encoder -> {output_dir}/")

    def save_root_split(self, clf, output_dir='reports'):
        """Save info about the root split for reporting."""
        os.makedirs(output_dir, exist_ok=True)
        root_idx = clf.tree_.feature[0]
        root_feature = (self.feature_columns[root_idx]
                        if root_idx >= 0 else 'leaf')
        root_threshold = clf.tree_.threshold[0] if root_idx >= 0 else None
        path = os.path.join(output_dir, 'root_split.txt')
        with open(path, 'w') as f:
            f.write(f"Root split feature : {root_feature}\n")
            f.write(f"Root split threshold: {root_threshold}\n")
        print(f"✓ Saved root split info -> {path}")

    @staticmethod
    def _fillna(X):
        """Fill NaN with column mean (safe for DataFrame and ndarray)."""
        if isinstance(X, pd.DataFrame):
            return X.fillna(X.mean())
        df = pd.DataFrame(X)
        return df.fillna(df.mean()).values


def main():
    """Main training pipeline."""
    trainer = ModelTrainer('data/features.csv', samples_per_class=1000)
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     X_unknown_val, X_unknown_test,
     y_unknown_val, y_unknown_test) = trainer.prepare_data()

    print(f"\nData shapes:")
    print(f"  X_train      : {X_train.shape}")
    print(f"  X_val        : {X_val.shape}")
    print(f"  X_test       : {X_test.shape}")
    print(f"  X_unknown_val : {X_unknown_val.shape}")
    print(f"  X_unknown_test: {X_unknown_test.shape}")
    print(f"  Classes: {len(trainer.class_labels)}")

    sklearn_clf = trainer.train_sklearn_tree(X_train, y_train, max_depth=15)
    lgb_model   = trainer.train_lightgbm(X_train, y_train, X_val, y_val)

    trainer.save_models(sklearn_clf, lgb_model)
    trainer.save_root_split(sklearn_clf)

    # Save splits for evaluation
    os.makedirs('data', exist_ok=True)

    def _save_split(X, y_enc, label_col, path):
        df = pd.DataFrame(X, columns=trainer.feature_columns)
        df['language'] = label_col
        df['y_true']   = y_enc
        df.to_csv(path, index=False)

    _save_split(X_val,          y_val,          trainer.label_encoder.inverse_transform(y_val),         'data/val_known.csv')
    _save_split(X_test,         y_test,         trainer.label_encoder.inverse_transform(y_test),        'data/test_known.csv')
    _save_split(X_unknown_val,  y_unknown_val,  y_unknown_val,                                          'data/val_unknown.csv')
    _save_split(X_unknown_test, y_unknown_test, y_unknown_test,                                         'data/test_unknown.csv')

    print("\n✓ All splits saved to data/")


if __name__ == "__main__":
    main()
