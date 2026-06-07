"""
Feature Engineering: Extract structural and textual features from files

FIX: Added missing static methods that were called but not defined:
     json_like_score, yaml_like_score, xml_like_score,
     keyword_count, keyword_density, hash_ratio, colon_ratio,
     equals_ratio, comma_ratio, quote_ratio,
     line_comment_ratio, block_comment_ratio
"""

import pandas as pd
import numpy as np
import re
import json as _json

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

# All keyword names in a fixed order
_ALL_KEYWORDS = list(_KEYWORDS.keys())


def _normalize_text(text: str) -> str:
    return text if isinstance(text, str) else ""


class FeatureExtractor:
    """Extract features from code/text content."""

    @staticmethod
    def extract_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from the 'content' column of df."""
        features = pd.DataFrame()

        # ----- 1) SIGNATURE features -----
        features['has_doctype']      = df['content'].apply(FeatureExtractor.has_doctype)
        features['has_xml_declaration'] = df['content'].apply(FeatureExtractor.has_xml_declaration)
        features['has_svg_tag']      = df['content'].apply(FeatureExtractor.has_svg)
        features['has_php_tag']      = df['content'].apply(FeatureExtractor.has_php_tag)
        features['has_shebang']      = df['content'].apply(FeatureExtractor.has_shebang)
        features['sb_bash']          = df['content'].apply(FeatureExtractor.sb_bash)
        features['sb_python']        = df['content'].apply(FeatureExtractor.sb_python)
        features['has_docker_from']  = df['content'].apply(FeatureExtractor.has_docker_from)
        features['has_html_tag']     = df['content'].apply(FeatureExtractor.has_html_tag)
        features['md_header']        = df['content'].apply(FeatureExtractor.md_header)
        features['has_vcalendar']    = df['content'].apply(FeatureExtractor.has_vcalendar)
        features['has_email_headers']= df['content'].apply(FeatureExtractor.has_email_headers)
        features['has_mime_boundary']= df['content'].apply(FeatureExtractor.has_mime_boundary)

        # ----- 2) KEYWORD count features -----
        for keyword in _ALL_KEYWORDS:
            features[f'kw_{keyword}'] = df['content'].apply(
                lambda text, kw=keyword: FeatureExtractor.keyword_count(text, kw)
            )
        features['keyword_count']   = df['content'].apply(FeatureExtractor.total_keyword_count)
        features['keyword_density'] = df['content'].apply(FeatureExtractor.keyword_density)

        # ----- 3) COMMENT style features -----
        features['cmt_slash']          = df['content'].apply(FeatureExtractor.cmt_slash)
        features['cmt_hash']           = df['content'].apply(FeatureExtractor.cmt_hash)
        features['cmt_block']          = df['content'].apply(FeatureExtractor.cmt_block)
        features['cmt_sql']            = df['content'].apply(FeatureExtractor.cmt_sql)
        features['cmt_html']           = df['content'].apply(FeatureExtractor.cmt_html)
        features['cmt_semi']           = df['content'].apply(FeatureExtractor.cmt_semi)
        features['cmt_pct']            = df['content'].apply(FeatureExtractor.cmt_pct)
        features['cmt_excl']           = df['content'].apply(FeatureExtractor.cmt_excl)
        features['line_comment_ratio'] = df['content'].apply(FeatureExtractor.line_comment_ratio)
        features['block_comment_ratio']= df['content'].apply(FeatureExtractor.block_comment_ratio)

        # ----- 4) PUNCTUATION ratio features -----
        for ch, key in [('{', 'brace'), ('}', 'brace_c'), (';', 'semi'),
                        ('<', 'lt'), ('>', 'gt'), (':', 'colon'), ('=', 'eq'),
                        ('$', 'dollar'), ('@', 'at'), ('(', 'paren'),
                        ('[', 'bracket'), (',', 'comma'), ('"', 'dquote'),
                        ("'", 'squote'), ('|', 'pipe')]:
            features[f'r_{key}'] = df['content'].apply(
                lambda text, c=ch: FeatureExtractor.punctuation_ratio(text, c)
            )
        features['hash_ratio']   = df['content'].apply(
            lambda t: FeatureExtractor.punctuation_ratio(t, '#'))
        features['colon_ratio']  = features['r_colon']
        features['equals_ratio'] = features['r_eq']
        features['comma_ratio']  = features['r_comma']
        features['quote_ratio']  = df['content'].apply(FeatureExtractor.quote_ratio)

        # ----- 5) STRUCTURE features -----
        features['avg_len']     = df['content'].apply(FeatureExtractor.avg_len)
        features['max_len']     = df['content'].apply(FeatureExtractor.max_len)
        features['blank_r']     = df['content'].apply(FeatureExtractor.blank_ratio)
        features['tab_r']       = df['content'].apply(FeatureExtractor.tab_ratio)
        features['space4_r']    = df['content'].apply(FeatureExtractor.space4_ratio)
        features['arrow_fat']   = df['content'].apply(FeatureExtractor.arrow_fat)
        features['arrow_thin']  = df['content'].apply(FeatureExtractor.arrow_thin)
        features['scope_op']    = df['content'].apply(FeatureExtractor.scope_op)
        features['tag_r']       = df['content'].apply(FeatureExtractor.tag_ratio)
        features['triple_q']    = df['content'].apply(FeatureExtractor.triple_q)
        features['backtick']    = df['content'].apply(FeatureExtractor.backtick)
        features['special_r']   = df['content'].apply(FeatureExtractor.special_ratio)
        features['digit_r']     = df['content'].apply(FeatureExtractor.digit_ratio)
        features['upper_r']     = df['content'].apply(FeatureExtractor.upper_ratio)
        features['dot_r']       = df['content'].apply(FeatureExtractor.dot_ratio)
        features['end_brace']   = df['content'].apply(FeatureExtractor.end_brace)
        features['end_colon']   = df['content'].apply(FeatureExtractor.end_colon)
        features['indent_level_avg'] = df['content'].apply(FeatureExtractor.indent_level_avg)
        features['line_count']  = df['content'].apply(
            lambda t: len(_normalize_text(t).split('\n')))

        # ----- 6) FORMAT-SPECIFIC scores -----
        features['json_like_score'] = df['content'].apply(FeatureExtractor.json_like_score)
        features['yaml_like_score'] = df['content'].apply(FeatureExtractor.yaml_like_score)
        features['xml_like_score']  = df['content'].apply(FeatureExtractor.xml_like_score)

        return features

    # ------------------------------------------------------------------ #
    # SIGNATURE
    # ------------------------------------------------------------------ #
    @staticmethod
    def has_doctype(text: str) -> int:
        return int('<!doctype' in _normalize_text(text).lower())

    @staticmethod
    def has_xml_declaration(text: str) -> int:
        return int(_normalize_text(text).lstrip().startswith('<?xml'))

    @staticmethod
    def has_svg(text: str) -> int:
        return int('<svg' in _normalize_text(text).lower())

    @staticmethod
    def has_php_tag(text: str) -> int:
        return int('<?php' in _normalize_text(text).lower())

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
        return int(bool(re.search(r'(?im)^\s*FROM\s+\S+', _normalize_text(text))))

    @staticmethod
    def has_html_tag(text: str) -> int:
        return int('<html' in _normalize_text(text).lower())

    @staticmethod
    def md_header(text: str) -> int:
        return int(bool(re.search(r'(?m)^#{1,6}\s', _normalize_text(text))))

    @staticmethod
    def has_vcalendar(text: str) -> int:
        return int('BEGIN:VCALENDAR' in _normalize_text(text).upper())

    @staticmethod
    def has_email_headers(text: str) -> int:
        headers = ['From:', 'To:', 'Subject:', 'Date:']
        return int(any(h in _normalize_text(text) for h in headers))

    @staticmethod
    def has_mime_boundary(text: str) -> int:
        return int('multipart' in _normalize_text(text).lower())

    # ------------------------------------------------------------------ #
    # KEYWORD
    # ------------------------------------------------------------------ #
    @staticmethod
    def keyword_count(text: str, keyword: str) -> int:
        text = _normalize_text(text)
        return len(re.findall(_KEYWORDS[keyword], text, re.I))

    @staticmethod
    def total_keyword_count(text: str) -> int:
        text = _normalize_text(text)
        return sum(len(re.findall(p, text, re.I)) for p in _KEYWORDS.values())

    @staticmethod
    def keyword_density(text: str) -> float:
        text = _normalize_text(text)
        n_words = max(len(text.split()), 1)
        total = sum(len(re.findall(p, text, re.I)) for p in _KEYWORDS.values())
        return total / n_words

    # ------------------------------------------------------------------ #
    # COMMENT STYLE
    # ------------------------------------------------------------------ #
    @staticmethod
    def cmt_slash(text: str) -> int:
        return _normalize_text(text).count('//')

    @staticmethod
    def cmt_hash(text: str) -> int:
        return sum(1 for l in _normalize_text(text).split('\n')
                   if l.lstrip().startswith('#'))

    @staticmethod
    def cmt_block(text: str) -> int:
        return _normalize_text(text).count('/*')

    @staticmethod
    def cmt_sql(text: str) -> int:
        return sum(1 for l in _normalize_text(text).split('\n')
                   if l.lstrip().startswith('--'))

    @staticmethod
    def cmt_html(text: str) -> int:
        return _normalize_text(text).count('<!--')

    @staticmethod
    def cmt_semi(text: str) -> int:
        return sum(1 for l in _normalize_text(text).split('\n')
                   if l.lstrip().startswith(';'))

    @staticmethod
    def cmt_pct(text: str) -> int:
        return sum(1 for l in _normalize_text(text).split('\n')
                   if l.lstrip().startswith('%'))

    @staticmethod
    def cmt_excl(text: str) -> int:
        return sum(1 for l in _normalize_text(text).split('\n')
                   if l.lstrip().startswith('!'))

    @staticmethod
    def line_comment_ratio(text: str) -> float:
        """Ratio of lines that start with a line-comment marker."""
        lines = _normalize_text(text).split('\n')
        if not lines:
            return 0.0
        markers = ('//', '#', '--', ';', '%', '!')
        n_comment = sum(1 for l in lines if l.lstrip().startswith(markers))
        return n_comment / len(lines)

    @staticmethod
    def block_comment_ratio(text: str) -> float:
        """Ratio of block-comment openers to total lines."""
        text = _normalize_text(text)
        lines = text.split('\n')
        n_lines = max(len(lines), 1)
        return (text.count('/*') + text.count('<!--')) / n_lines

    # ------------------------------------------------------------------ #
    # PUNCTUATION RATIO
    # ------------------------------------------------------------------ #
    @staticmethod
    def punctuation_ratio(text: str, ch: str) -> float:
        text = _normalize_text(text)
        return text.count(ch) / max(len(text), 1)

    @staticmethod
    def quote_ratio(text: str) -> float:
        """Combined ratio of single + double quotes."""
        text = _normalize_text(text)
        return (text.count('"') + text.count("'")) / max(len(text), 1)

    # ------------------------------------------------------------------ #
    # STRUCTURE
    # ------------------------------------------------------------------ #
    @staticmethod
    def avg_len(text: str) -> float:
        lines = _normalize_text(text).split('\n')
        return float(np.mean([len(l) for l in lines])) if lines else 0.0

    @staticmethod
    def max_len(text: str) -> int:
        return max((len(l) for l in _normalize_text(text).split('\n')), default=0)

    @staticmethod
    def blank_ratio(text: str) -> float:
        lines = _normalize_text(text).split('\n')
        non_empty = [l for l in lines if l.strip()]
        return (len(lines) - len(non_empty)) / max(len(lines), 1)

    @staticmethod
    def tab_ratio(text: str) -> float:
        lines = _normalize_text(text).split('\n')
        return sum(1 for l in lines if l.startswith('\t')) / max(len(lines), 1)

    @staticmethod
    def space4_ratio(text: str) -> float:
        lines = _normalize_text(text).split('\n')
        return sum(1 for l in lines if l.startswith('    ')) / max(len(lines), 1)

    @staticmethod
    def arrow_fat(text: str) -> int:
        return _normalize_text(text).count('=>')

    @staticmethod
    def arrow_thin(text: str) -> int:
        return _normalize_text(text).count('->')

    @staticmethod
    def scope_op(text: str) -> int:
        return _normalize_text(text).count('::')

    @staticmethod
    def tag_ratio(text: str) -> float:
        text = _normalize_text(text)
        n_lines = max(len(text.split('\n')), 1)
        return len(re.findall(r'<[a-zA-Z/!?][^>]*>', text)) / n_lines

    @staticmethod
    def triple_q(text: str) -> int:
        text = _normalize_text(text)
        return text.count('"""') + text.count("'''")

    @staticmethod
    def backtick(text: str) -> int:
        return _normalize_text(text).count('`')

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
        non_empty = [l for l in _normalize_text(text).split('\n') if l.strip()]
        return sum(1 for l in non_empty if l.rstrip().endswith(('{', ';', '}')))

    @staticmethod
    def end_colon(text: str) -> int:
        non_empty = [l for l in _normalize_text(text).split('\n') if l.strip()]
        return sum(1 for l in non_empty if l.rstrip().endswith(':'))

    @staticmethod
    def indent_level_avg(text: str) -> float:
        lines = _normalize_text(text).split('\n')
        indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
        return float(np.mean(indents)) if indents else 0.0

    # ------------------------------------------------------------------ #
    # FORMAT-SPECIFIC SCORES  (FIX: these were missing, causing crash)
    # ------------------------------------------------------------------ #
    @staticmethod
    def json_like_score(text: str) -> float:
        """
        Score 0–1 for how much the text resembles JSON.
        1.0 = valid JSON; partial scores for JSON-like structure.
        Used to filter mislabeled JS/TS samples that are actually JSON.
        """
        text = _normalize_text(text).strip()
        if not text:
            return 0.0
        # Try strict parse first
        try:
            _json.loads(text)
            return 1.0
        except Exception:
            pass
        # Heuristic: starts with { or [ and has "key": patterns
        score = 0.0
        if text.startswith(('{', '[')):
            score += 0.4
        kv_pairs = len(re.findall(r'"[^"]+"\s*:', text))
        n_lines = max(len(text.split('\n')), 1)
        score += min(kv_pairs / n_lines * 0.3, 0.4)
        # High ratio of { } : " chars suggests JSON
        json_chars = sum(text.count(c) for c in '{}[]:,"')
        score += min(json_chars / max(len(text), 1) * 2, 0.2)
        return min(score, 1.0)

    @staticmethod
    def yaml_like_score(text: str) -> float:
        """Score 0–1 for how much the text resembles YAML."""
        text = _normalize_text(text).strip()
        if not text:
            return 0.0
        lines = text.split('\n')
        n_lines = max(len(lines), 1)
        # YAML key-value pattern: "key: value" at start of line
        kv_lines = sum(1 for l in lines if re.match(r'^\s*[\w-]+\s*:', l))
        # YAML list items: "  - item"
        list_lines = sum(1 for l in lines if re.match(r'^\s*-\s+\S', l))
        score = min(kv_lines / n_lines * 0.7 + list_lines / n_lines * 0.3, 1.0)
        # Penalise if it has { } which suggest JSON or code
        brace_ratio = (text.count('{') + text.count('}')) / max(len(text), 1)
        score *= max(1.0 - brace_ratio * 10, 0.0)
        return float(score)

    @staticmethod
    def xml_like_score(text: str) -> float:
        """Score 0–1 for how much the text resembles XML/HTML."""
        text = _normalize_text(text).strip()
        if not text:
            return 0.0
        n_tags = len(re.findall(r'<[a-zA-Z/!?][^>]*>', text))
        n_lines = max(len(text.split('\n')), 1)
        tag_density = n_tags / n_lines
        score = min(tag_density * 0.5, 0.8)
        if text.lstrip().startswith('<?xml') or '<!DOCTYPE' in text.upper():
            score = min(score + 0.2, 1.0)
        return float(score)


def process_features(input_csv, output_csv):
    """Load raw dataset, extract features, save to CSV."""
    print("Loading raw dataset...")
    df = pd.read_csv(input_csv)
    print(f"Total samples: {len(df)}")
    print(f"Classes: {df['language'].nunique()}")

    print("\nExtracting features...")
    features = FeatureExtractor.extract_features(df)

    result = pd.concat([df[['language', 'file_id']], features], axis=1)
    print(f"Features extracted: {features.shape[1]}")
    print(f"Final dataset shape: {result.shape}")

    result.to_csv(output_csv, index=False)
    print(f"\n✓ Saved to {output_csv}")
    return result


if __name__ == "__main__":
    result = process_features('data/raw_dataset.csv', 'data/features.csv')
    print("\nFeature Statistics:")
    print(result.describe())
