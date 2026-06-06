# Self-practice 2: Decision Tree Classifier

## Goal

Build a decision tree classifier to classify text-based files by their format/programming language.

**Choose your own classes** — pick at least **32 classes** (file formats or programming languages). Examples: Python, Java, C, JavaScript, HTML, XML, SVG, Markdown, JSON, YAML, SQL, CSS, Shell, Ruby, Go, Rust, PHP, TypeScript, Kotlin, Swift, R, Lua, Perl, Haskell, Scala, TOML, INI, CSV, LaTeX, Dockerfile, Makefile, etc.

---

## Approach

### 1. Dataset Preparation

**Recommended source: [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) or [The Stack v1](https://huggingface.co/datasets/bigcode/the-stack)**

These datasets contain deduplicated source code from GitHub, organized by programming language — ideal for this task.

- Select at least **32 classes** (languages/formats)
- Sample **1,000 files per class** (balanced dataset)
- Use `datasets` library to stream/download specific languages

```python
from datasets import load_dataset

# Example: stream specific languages from The Stack v2
ds = load_dataset("bigcode/the-stack-v2", data_dir="data/python", split="train", streaming=True)
samples = list(ds.take(1000))
```

> **⚠️ Data Cleaning:** The Stack can contain noise — empty files, binary content misclassified as text, auto-generated code, or files with wrong language labels. **You must clean your data:**
> - Remove empty or near-empty files (< 10 characters)
> - Filter out binary/encoded content (base64 blobs, minified bundles)
> - Check for mislabeled samples (e.g., JSON labeled as JavaScript)
> - Remove duplicates or near-duplicates
> - Consider truncating very long files (keep first N lines)

**Feature engineering** — extract structural/textual features from raw file content:

| Feature | Description |
|---------|-------------|
| `has_doctype` | Contains `<!DOCTYPE` |
| `has_xml_declaration` | Starts with `<?xml` |
| `has_svg_tag` | Contains `<svg` |
| `has_vcalendar` | Contains `BEGIN:VCALENDAR` |
| `has_from_header` | Contains `From:` or `To:` email headers |
| `has_mime_boundary` | Contains `Content-Type: multipart` |
| `html_tag_ratio` | Ratio of HTML tags to total lines |
| `avg_line_length` | Average characters per line |
| `special_char_ratio` | Ratio of `<`, `>`, `/` characters |
| `first_line_pattern` | Encoded pattern of the first non-empty line |

> **Tip:** Read only the first 50–100 lines. File headers are the most discriminative.

### 2. Model Training

#### Baseline: sklearn DecisionTreeClassifier

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

clf = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42)
clf.fit(X_train, y_train)
```

#### Advanced: LightGBM (Gradient Boosted Decision Trees)

LightGBM builds an ensemble of decision trees using gradient boosting — it's still a tree-based model but significantly more powerful.

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=8,
    num_leaves=31,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)
```

**Why LightGBM?**
- Handles imbalanced classes well
- Built-in feature importance
- Fast training, strong generalization
- Still interpretable as a tree ensemble

### 3. Evaluation

```python
from sklearn.metrics import classification_report, accuracy_score

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))  # precision, recall, F1 per class
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

**Required metrics:** Precision, Recall, Accuracy, F1-score (macro-averaged).

### 4. Unknown Format Handling

The test set must include files of **unseen formats** (e.g., JSON, CSV, YAML). Strategy:

- **Confidence thresholding** — if `max(predict_proba) < threshold`, label as "unknown"
- **One-vs-rest with rejection** — train a binary classifier per class; reject if none fires

```python
proba = clf.predict_proba(X_test)
predictions = []
for p in proba:
    if p.max() < 0.6:  # confidence threshold
        predictions.append("UNKNOWN")
    else:
        predictions.append(clf.classes_[p.argmax()])
```

### 5. Visualization

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=feature_names, class_names=class_names,
          filled=True, max_depth=3, fontsize=8)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)
```

For LightGBM:
```python
lgb.plot_importance(model, max_num_features=15)
lgb.plot_tree(model, tree_index=0, figsize=(20, 10))
```

**Explain:** Which features split first? Why do those features distinguish formats?

---

## Submission Checklist

| Item | Details |
|------|---------|
| Source code | `.py` files |
| Dataset | Raw files + extracted feature CSV |
| Model data | Saved model (`.pkl` or `.txt` for LightGBM) |
| Report | Metrics table, tree plot, feature importance, explanation of unknown handling |

---

## Scoring

Higher **macro F1-score** → higher bonus. Focus on:
- Good features > complex models
- Thorough data cleaning (garbage in → garbage out)
- Handling edge cases between similar formats
- Robust unknown detection
