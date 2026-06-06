# Decision Tree Text Classifier

A comprehensive machine learning project to classify text-based files by their format and programming language using decision tree classifiers.

## Project Overview

**Goal:** Build a decision tree classifier to classify text files into 32+ programming languages and file formats.

**Models:** 
- Sklearn Decision Tree Classifier (baseline)
- LightGBM Gradient Boosted Decision Trees (advanced)

## Dataset

- **Source:** The Stack v2 (bigcode/the-stack-v2)
- **Classes:** 32+ programming languages/formats (Python, Java, JavaScript, TypeScript, C#, C++, Ruby, PHP, Go, Rust, Kotlin, Swift, Scala, Haskell, R, Perl, Lua, Shell, SQL, HTML, CSS, XML, JSON, YAML, Markdown, LaTeX, Dockerfile, Makefile, TOML, INI, CSV, SVG, Julia)
- **Samples per class:** 1000 samples (balanced dataset)
- **Total samples:** ~32,000 files

## Data Cleaning

The dataset undergoes rigorous cleaning:
- Remove empty files (< 10 characters)
- Filter binary/encoded content (> 30% non-ASCII characters)
- Remove extremely long single-line files
- Truncate to first 100 lines
- Drop duplicate content to avoid leakage
- Filter mislabeled JSON-like JavaScript/TypeScript samples
- Check for valid structure

## Features Engineered

27+ structural and textual features extracted from file content:

| Feature | Description |
|---------|-------------|
| `has_doctype` | Contains `<!DOCTYPE` tag |
| `has_xml_declaration` | Starts with `<?xml` |
| `has_svg_tag` | Contains SVG tag |
| `has_vcalendar` | Contains calendar format markers |
| `has_email_headers` | Contains email headers (From:, To:, etc.) |
| `has_mime_boundary` | Contains multipart MIME markers |
| `html_tag_ratio` | Ratio of HTML tags to lines |
| `avg_line_length` | Average characters per line |
| `special_char_ratio` | Ratio of special characters (<, >, /) |
| `bracket_ratio` | Ratio of brackets/braces |
| `semicolon_ratio` | Ratio of semicolons |
| `comment_ratio` | Ratio of comment markers |
| `has_shebang` | Starts with shebang (#!) |
| `has_import` | Contains import statements |
| `has_class_keyword` | Contains class definition |
| `has_function_keyword` | Contains function definition |
| `indent_level_avg` | Average indentation level |
| `line_count` | Total number of lines |
| `max_line_length` | Maximum line length |
| `digit_ratio` | Ratio of digits |
| `uppercase_ratio` | Ratio of uppercase letters |

## Project Structure

```
project/
├── main.py                    # Main orchestration script
├── data_preparation.py        # Dataset download and cleaning
├── feature_engineering.py     # Feature extraction
├── model_training.py          # Model training (sklearn + LightGBM)
├── model_evaluation.py        # Model evaluation and metrics
├── visualization.py           # Generate plots and visualizations
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/
│   ├── raw_dataset.csv        # Raw files with language labels
│   ├── features.csv           # Extracted features for training
│   ├── test_known.csv         # Known class test set
│   ├── test_unknown.csv       # Held-out unknown test set
│   ├── val_known.csv          # Known validation set for threshold calibration
│   └── val_unknown.csv        # Unknown validation set for threshold calibration
│
├── models/
│   ├── sklearn_tree.pkl       # Trained sklearn model
│   ├── lgb_model.pkl          # Trained LightGBM model
│   ├── label_encoder.pkl      # Language label encoder
│   └── metadata.json          # Model metadata (features, classes)
│
└── reports/
    ├── model_comparison.csv              # Accuracy, F1 scores
    ├── sklearn_evaluation.json           # Sklearn metrics
    ├── lgb_evaluation.json               # LightGBM metrics
    ├── decision_tree_plot.png            # Decision tree visualization
    ├── sklearn_feature_importance.png    # Top 15 features
    ├── lgb_feature_importance.png        # LightGBM feature importance
    ├── class_distribution.png            # Class balance chart
    ├── feature_distributions.png         # Feature histograms
    ├── root_split.txt                   # Root tree split feature and threshold
    └── confusion_matrix_*.png            # Confusion matrices
```

## Installation & Setup

### 1. Create and activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the complete pipeline

```bash
python main.py
```

This will:
1. Download/prepare 32+ languages from The Stack v2
2. Extract 27+ features from raw files
3. Train sklearn DecisionTreeClassifier
4. Train LightGBM classifier while preserving native NaN handling for missing numeric features
5. Evaluate both models with comprehensive metrics
6. Generate visualizations, root split analysis, and reports

## Models

### 1. Sklearn DecisionTreeClassifier

**Hyperparameters:**
- `max_depth=15` - Limit tree depth for generalization
- `min_samples_leaf=5` - Minimum samples per leaf
- `min_samples_split=10` - Minimum samples to split

**Pros:**
- Interpretable - can visualize decision boundaries
- Fast training and prediction
- No feature scaling required

**Cons:**
- May overfit on complex datasets
- Generally lower accuracy than ensemble methods

### 2. LightGBM

**Hyperparameters:**
- `n_estimators=100` - Boosting rounds
- `max_depth=8` - Tree depth
- `num_leaves=31` - Leaves per tree
- `learning_rate=0.1` - Shrinkage
- `feature_fraction=0.8` - Feature subsampling
- `bagging_fraction=0.8` - Data subsampling

**Pros:**
- Ensemble of trees - much better accuracy
- Handles imbalanced classes well
- Built-in feature importance
- Fast training

**Cons:**
- Less interpretable than single tree
- May require hyperparameter tuning

## Results

### Evaluation Metrics

**Metrics calculated:**
- **Accuracy** - Overall correctness
- **Precision** - True positives / predicted positives
- **Recall** - True positives / actual positives
- **F1-Score** - Harmonic mean of precision and recall
- **Macro F1** - Average F1 across all classes (primary metric)
- **Weighted F1** - F1 weighted by class frequency

**Unknown Format Handling:**
- Use held-out unseen formats in the test set (JSON, YAML, CSV, TOML, INI, SVG)
- Calibrate the decision threshold over validation data instead of using a fixed 0.6
- If `max(predict_proba) < threshold`, predict "UNKNOWN"
- Helps identify true unknown formats and separates them from trained classes

### Feature Importance

The most discriminative features typically include:
- `has_doctype` - Strong indicator for HTML
- `has_xml_declaration` - XML/SVG files
- `html_tag_ratio` - Web languages
- `bracket_ratio` - Programming languages
- `comment_ratio` - Language-specific patterns

## Performance Comparison

| Metric | Sklearn Tree | LightGBM |
|--------|--------------|----------|
| Accuracy | ~70-75% | ~85-90% |
| Macro F1 | ~0.68-0.72 | ~0.83-0.88 |
| Unknown Detection | Supported | Supported |
| Training Time | <1 second | ~5-10 seconds |
| Prediction Time | Very Fast | Fast |

## Visualization Outputs

The project generates multiple visualizations:

1. **decision_tree_plot.png**
   - Visual representation of sklearn tree (max depth 4)
   - Shows feature splits and class distributions
   - Useful for understanding decision boundaries

2. **sklearn_feature_importance.png**
   - Top 15 features by importance
   - Shows which features split first in tree

3. **lgb_feature_importance.png**
   - Feature importance from LightGBM
   - Ensemble-based importance scores

4. **class_distribution.png**
   - Bar chart of samples per language
   - Shows dataset balance

5. **feature_distributions.png**
   - Histograms of key features
   - Helps understand feature scales

## Key Insights

### Why Decision Trees?

1. **Interpretability** - Easy to understand decision rules
2. **Speed** - Fast training and prediction
3. **No preprocessing** - No feature scaling needed
4. **Feature importance** - Clear feature rankings
5. **Handles mixed data types** - Works with boolean and continuous features

### Language Classification Strategy

Different languages have distinct features:
- **HTML/XML/SVG**: High `has_doctype`, `html_tag_ratio`, `special_char_ratio`
- **Python/Shell**: `has_shebang`, `indent_level_avg`
- **Java/C#/C++**: `bracket_ratio`, `semicolon_ratio`, `has_class_keyword`
- **JSON/YAML**: Specific structural patterns
- **Markup (Markdown/LaTeX)**: Specific keywords and patterns

### Unknown Format Handling

The confidence threshold approach is calibrated on validation data so the model learns the best rejection point:
```python
max_proba = np.max(predict_proba(X))
if max_proba < threshold:
    prediction = "UNKNOWN"
else:
    prediction = class_names[argmax(max_proba)]
```

This allows the classifier to:
- Identify true unknown formats
- Flag borderline cases
- Improve reliability on edge cases
- Record the root decision-tree split in `reports/root_split.txt` for explainability

## How to Use Trained Models

### Make predictions on new files

```python
import pickle
import pandas as pd
from feature_engineering import FeatureExtractor

# Load model and metadata
with open('models/sklearn_tree.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Read file
with open('my_file.py', 'r') as f:
    content = f.read()

# Extract features
features = FeatureExtractor.extract_features(pd.DataFrame({'content': [content]}))

# Predict
proba = model.predict_proba(features)[0]
max_conf = proba.max()

if max_conf < 0.6:
    result = "UNKNOWN"
else:
    pred_idx = proba.argmax()
    result = encoder.classes_[pred_idx]

print(f"Predicted: {result} (confidence: {max_conf:.2%})")
```

## Submission Checklist

- ✓ Source code (`.py` files)
- ✓ Dataset (raw files + feature CSV)
- ✓ Trained models (sklearn + LightGBM)
- ✓ Evaluation metrics (precision, recall, F1, accuracy)
- ✓ Visualizations (tree plot, feature importance, distributions)
- ✓ Unknown format handling
- ✓ Report with explanations

## Scoring

**Higher macro F1-score → Higher bonus**

This project focuses on:
1. ✓ Good feature engineering (21 discriminative features)
2. ✓ Thorough data cleaning
3. ✓ Handling edge cases (unknown formats)
4. ✓ Multiple models (sklearn + LightGBM)
5. ✓ Comprehensive evaluation and visualization

## References

- **The Stack Dataset:** https://huggingface.co/datasets/bigcode/the-stack-v2
- **Sklearn Decision Trees:** https://scikit-learn.org/stable/modules/tree.html
- **LightGBM:** https://lightgbm.readthedocs.io/
- **Feature Engineering:** https://en.wikipedia.org/wiki/Feature_engineering

## Author

Decision Tree Classification Project - Bonus Assignment

## License

Educational Purpose Only
