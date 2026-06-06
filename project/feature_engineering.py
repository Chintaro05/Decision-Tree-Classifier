"""
Feature Engineering: Extract structural and textual features from files
"""

import pandas as pd
import numpy as np
import re

_KEYWORDS = {
    "def": r"\bdef\b", "class": r"\bclass\b", "function": r"\bfunction\b",
    "import": r"\bimport\b", "include": r"#\s*include", "package": r"\bpackage\b",
    "public": r"\bpublic\b", "private": r"\bprivate\b", "fn": r"\bfn\b",
    "func": r"\bfunc\b", "let": r"\blet\b", "const": r"\bconst\b",
    "var": r"\bvar\b", "use": r"\buse\b", "echo": r"\becho\b",
    "select": r"\bselect\b", "from": r"\bfrom\b", "end": r"\bend\b",
    "begin": r"\bbegin\b", "module": r"\bmodule\b", "return": r"\breturn\b",
    "void": r"\bvoid\b", "namespace": r"\bnamespace\b", "struct": r"\bstruct\b",
}


def _normalize_text(text: str) -> str:
    return text if isinstance(text, str) else ""


class FeatureExtractor:
    """Extract features from code/text content"""

    @staticmethod
    def extract_features(df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame()
        features['has_doctype'] = df['content'].apply(FeatureExtractor.has_doctype)
        features['has_xml_declaration'] = df['content'].apply(FeatureExtractor.has_xml_declaration)
        features['has_svg'] = df['content'].apply(FeatureExtractor.has_svg)
        features['has_php_tag'] = df['content'].apply(FeatureExtractor.has_php_tag)
        features['has_shebang'] = df['content'].apply(FeatureExtractor.has_shebang)
        features['sb_bash'] = df['content'].apply(FeatureExtractor.sb_bash)
        features['sb_python'] = df['content'].apply(FeatureExtractor.sb_python)
        features['has_docker_from'] = df['content'].apply(FeatureExtractor.has_docker_from)
        features['has_html_tag'] = df['content'].apply(FeatureExtractor.has_html_tag)
        features['md_header'] = df['content'].apply(FeatureExtractor.md_header)

        for keyword in _KEYWORDS.keys():
            features[f'kw_{keyword}'] = df['content'].apply(
                lambda text, kw=keyword: FeatureExtractor.keyword_count(text, kw)
            )

        features['cmt_slash'] = df['content'].apply(FeatureExtractor.cmt_slash)
        features['cmt_hash'] = df['content'].apply(FeatureExtractor.cmt_hash)
        features['cmt_block'] = df['content'].apply(FeatureExtractor.cmt_block)
        features['cmt_sql'] = df['content'].apply(FeatureExtractor.cmt_sql)
        features['cmt_html'] = df['content'].apply(FeatureExtractor.cmt_html)
        features['cmt_semi'] = df['content'].apply(FeatureExtractor.cmt_semi)
        features['cmt_pct'] = df['content'].apply(FeatureExtractor.cmt_pct)
        features['cmt_excl'] = df['content'].apply(FeatureExtractor.cmt_excl)

        for ch, key in [('{', 'brace'), ('}', 'brace_c'), (';', 'semi'),
                        ('<', 'lt'), ('>', 'gt'), (':', 'colon'), ('=', 'eq'),
                        ('$', 'dollar'), ('@', 'at'), ('(', 'paren'),
                        ('[', 'bracket'), (',', 'comma'), ('"', 'dquote'),
                        ("'", 'squote'), ('|', 'pipe')]:
            features[f'r_{key}'] = df['content'].apply(
                lambda text, ch=ch: FeatureExtractor.punctuation_ratio(text, ch)
            )

        features['avg_len'] = df['content'].apply(FeatureExtractor.avg_len)
        features['max_len'] = df['content'].apply(FeatureExtractor.max_len)
        features['blank_r'] = df['content'].apply(FeatureExtractor.blank_ratio)
        features['tab_r'] = df['content'].apply(FeatureExtractor.tab_ratio)
        features['space4_r'] = df['content'].apply(FeatureExtractor.space4_ratio)
        features['arrow_fat'] = df['content'].apply(FeatureExtractor.arrow_fat)
        features['arrow_thin'] = df['content'].apply(FeatureExtractor.arrow_thin)
        features['scope_op'] = df['content'].apply(FeatureExtractor.scope_op)
        features['tag_r'] = df['content'].apply(FeatureExtractor.tag_ratio)
        features['triple_q'] = df['content'].apply(FeatureExtractor.triple_q)
        features['backtick'] = df['content'].apply(FeatureExtractor.backtick)
        features['special_r'] = df['content'].apply(FeatureExtractor.special_ratio)
        features['digit_r'] = df['content'].apply(FeatureExtractor.digit_ratio)
        features['upper_r'] = df['content'].apply(FeatureExtractor.upper_ratio)
        features['dot_r'] = df['content'].apply(FeatureExtractor.dot_ratio)
        features['end_brace'] = df['content'].apply(FeatureExtractor.end_brace)
        features['end_colon'] = df['content'].apply(FeatureExtractor.end_colon)
        features['line_count'] = df['content'].apply(lambda text: len(str(text).split('\n')))

        return features

    @staticmethod
    def has_doctype(text: str) -> int:
        text = _normalize_text(text)
        return int('<!doctype' in text.lower())

    @staticmethod
    def has_xml_declaration(text: str) -> int:
        text = _normalize_text(text)
        return int(text.lstrip().startswith('<?xml'))

    @staticmethod
    def has_svg(text: str) -> int:
        text = _normalize_text(text)
        return int('<svg' in text.lower())

    @staticmethod
    def has_php_tag(text: str) -> int:
        text = _normalize_text(text)
        return int('<?php' in text.lower())

    @staticmethod
    def has_shebang(text: str) -> int:
        text = _normalize_text(text)
        first = text.split('\n')[0] if text else ''
        return int(first.startswith('#!'))

    @staticmethod
    def sb_bash(text: str) -> int:
        text = _normalize_text(text)
        first = text.split('\n')[0] if text else ''
        return int(bool(re.match(r'#!.*\b(sh|bash|zsh|ksh)\b', first)))

    @staticmethod
    def sb_python(text: str) -> int:
        text = _normalize_text(text)
        first = text.split('\n')[0] if text else ''
        return int(bool(re.match(r'#!.*python', first)))

    @staticmethod
    def has_docker_from(text: str) -> int:
        text = _normalize_text(text)
        return int(bool(re.search(r'(?im)^\s*FROM\s+\S+', text)))

    @staticmethod
    def has_html_tag(text: str) -> int:
        text = _normalize_text(text)
        return int('<html' in text.lower())

    @staticmethod
    def md_header(text: str) -> int:
        text = _normalize_text(text)
        return int(bool(re.search(r'(?m)^#{1,6}\s', text)))

    @staticmethod
    def keyword_count(text: str, keyword: str) -> int:
        text = _normalize_text(text)
        return len(re.findall(_KEYWORDS[keyword], text, re.I))

    @staticmethod
    def cmt_slash(text: str) -> int:
        text = _normalize_text(text)
        return text.count('//')

    @staticmethod
    def cmt_hash(text: str) -> int:
        text = _normalize_text(text)
        return sum(1 for line in text.split('\n') if line.lstrip().startswith('#'))

    @staticmethod
    def cmt_block(text: str) -> int:
        text = _normalize_text(text)
        return text.count('/*')

    @staticmethod
    def cmt_sql(text: str) -> int:
        text = _normalize_text(text)
        return sum(1 for line in text.split('\n') if line.lstrip().startswith('--'))

    @staticmethod
    def cmt_html(text: str) -> int:
        text = _normalize_text(text)
        return text.count('<!--')

    @staticmethod
    def cmt_semi(text: str) -> int:
        text = _normalize_text(text)
        return sum(1 for line in text.split('\n') if line.lstrip().startswith(';'))

    @staticmethod
    def cmt_pct(text: str) -> int:
        text = _normalize_text(text)
        return sum(1 for line in text.split('\n') if line.lstrip().startswith('%'))

    @staticmethod
    def cmt_excl(text: str) -> int:
        text = _normalize_text(text)
        return sum(1 for line in text.split('\n') if line.lstrip().startswith('!'))

    @staticmethod
    def punctuation_ratio(text: str, ch: str) -> float:
        text = _normalize_text(text)
        return text.count(ch) / max(len(text), 1)

    @staticmethod
    def avg_len(text: str) -> float:
        text = _normalize_text(text)
        lines = text.split('\n')
        return float(np.mean([len(l) for l in lines])) if lines else 0.0

    @staticmethod
    def max_len(text: str) -> int:
        text = _normalize_text(text)
        return max((len(l) for l in text.split('\n')), default=0)

    @staticmethod
    def blank_ratio(text: str) -> float:
        text = _normalize_text(text)
        lines = text.split('\n')
        non_empty = [l for l in lines if l.strip()]
        return (len(lines) - len(non_empty)) / max(len(lines), 1)

    @staticmethod
    def tab_ratio(text: str) -> float:
        text = _normalize_text(text)
        lines = text.split('\n')
        return sum(1 for l in lines if l.startswith('\t')) / max(len(lines), 1)

    @staticmethod
    def space4_ratio(text: str) -> float:
        text = _normalize_text(text)
        lines = text.split('\n')
        return sum(1 for l in lines if l.startswith('    ')) / max(len(lines), 1)

    @staticmethod
    def arrow_fat(text: str) -> int:
        text = _normalize_text(text)
        return text.count('=>')

    @staticmethod
    def arrow_thin(text: str) -> int:
        text = _normalize_text(text)
        return text.count('->')

    @staticmethod
    def scope_op(text: str) -> int:
        text = _normalize_text(text)
        return text.count('::')

    @staticmethod
    def tag_ratio(text: str) -> float:
        text = _normalize_text(text)
        lines = max(len(text.split('\n')), 1)
        return len(re.findall(r'<[a-zA-Z/!?][^>]*>', text)) / lines

    @staticmethod
    def triple_q(text: str) -> int:
        text = _normalize_text(text)
        return text.count('"""') + text.count("'''")

    @staticmethod
    def backtick(text: str) -> int:
        text = _normalize_text(text)
        return text.count('`')

    @staticmethod
    def special_ratio(text: str) -> float:
        text = _normalize_text(text)
        return sum(1 for c in text if c in '<>/{}[]();') / max(len(text), 1)

    @staticmethod
    def digit_ratio(text: str) -> float:
        text = _normalize_text(text)
        return sum(1 for c in text if c.isdigit()) / max(len(text), 1)

    @staticmethod
    def upper_ratio(text: str) -> float:
        text = _normalize_text(text)
        return sum(1 for c in text if c.isupper()) / max(len(text), 1)

    @staticmethod
    def dot_ratio(text: str) -> float:
        text = _normalize_text(text)
        return text.count('.') / max(len(text), 1)

    @staticmethod
    def end_brace(text: str) -> int:
        text = _normalize_text(text)
        non_empty = [l for l in text.split('\n') if l.strip()]
        return sum(1 for l in non_empty if l.rstrip().endswith(('{', ';', '}')))

    @staticmethod
    def end_colon(text: str) -> int:
        text = _normalize_text(text)
        non_empty = [l for l in text.split('\n') if l.strip()]
        return sum(1 for l in non_empty if l.rstrip().endswith(':'))


def process_features(input_csv, output_csv):
    """Load dataset, extract features, save to CSV"""
    print("Loading raw dataset...")
    df = pd.read_csv(input_csv)
    
    print(f"Total samples: {len(df)}")
    print(f"Classes: {df['language'].nunique()}")
    
    print("\nExtracting features...")
    features = FeatureExtractor.extract_features(df)
    
    # Combine with labels
    result = pd.concat([
        df[['language', 'file_id']],
        features
    ], axis=1)
    
    print(f"Features extracted: {features.shape[1]}")
    print(f"Final dataset shape: {result.shape}")
    
    result.to_csv(output_csv, index=False)
    print(f"\n✓ Saved to {output_csv}")
    
    return result

if __name__ == "__main__":
    result = process_features('data/raw_dataset.csv', 'data/features.csv')
    print("\nFeature Statistics:")
    print(result.describe())
