"""
Feature Engineering: Extract structural and textual features from files
"""

import pandas as pd
import numpy as np
import re
from collections import Counter

class FeatureExtractor:
    """Extract features from code/text content"""
    
    @staticmethod
    def extract_features(df):
        """Extract all features from content column"""
        features = pd.DataFrame()
        features['has_doctype'] = df['content'].apply(FeatureExtractor.has_doctype)
        features['has_xml_declaration'] = df['content'].apply(FeatureExtractor.has_xml_declaration)
        features['has_svg_tag'] = df['content'].apply(FeatureExtractor.has_svg_tag)
        features['has_vcalendar'] = df['content'].apply(FeatureExtractor.has_vcalendar)
        features['has_email_headers'] = df['content'].apply(FeatureExtractor.has_email_headers)
        features['has_mime_boundary'] = df['content'].apply(FeatureExtractor.has_mime_boundary)
        features['html_tag_ratio'] = df['content'].apply(FeatureExtractor.html_tag_ratio)
        features['avg_line_length'] = df['content'].apply(FeatureExtractor.avg_line_length)
        features['special_char_ratio'] = df['content'].apply(FeatureExtractor.special_char_ratio)
        features['bracket_ratio'] = df['content'].apply(FeatureExtractor.bracket_ratio)
        features['semicolon_ratio'] = df['content'].apply(FeatureExtractor.semicolon_ratio)
        features['comment_ratio'] = df['content'].apply(FeatureExtractor.comment_ratio)
        features['line_comment_ratio'] = df['content'].apply(FeatureExtractor.line_comment_ratio)
        features['block_comment_ratio'] = df['content'].apply(FeatureExtractor.block_comment_ratio)
        features['has_shebang'] = df['content'].apply(FeatureExtractor.has_shebang)
        features['has_import'] = df['content'].apply(FeatureExtractor.has_import)
        features['has_class_keyword'] = df['content'].apply(FeatureExtractor.has_class_keyword)
        features['has_function_keyword'] = df['content'].apply(FeatureExtractor.has_function_keyword)
        features['keyword_count'] = df['content'].apply(FeatureExtractor.keyword_count)
        features['keyword_density'] = df['content'].apply(FeatureExtractor.keyword_density)
        features['hash_ratio'] = df['content'].apply(FeatureExtractor.hash_ratio)
        features['colon_ratio'] = df['content'].apply(FeatureExtractor.colon_ratio)
        features['equals_ratio'] = df['content'].apply(FeatureExtractor.equals_ratio)
        features['comma_ratio'] = df['content'].apply(FeatureExtractor.comma_ratio)
        features['quote_ratio'] = df['content'].apply(FeatureExtractor.quote_ratio)
        features['json_like_score'] = df['content'].apply(FeatureExtractor.json_like_score)
        features['yaml_like_score'] = df['content'].apply(FeatureExtractor.yaml_like_score)
        features['xml_like_score'] = df['content'].apply(FeatureExtractor.xml_like_score)
        features['indent_level_avg'] = df['content'].apply(FeatureExtractor.indent_level_avg)
        features['line_count'] = df['content'].apply(lambda x: len(x.split('\n')))
        features['max_line_length'] = df['content'].apply(FeatureExtractor.max_line_length)
        features['digit_ratio'] = df['content'].apply(FeatureExtractor.digit_ratio)
        features['uppercase_ratio'] = df['content'].apply(FeatureExtractor.uppercase_ratio)
        
        return features
    
    @staticmethod
    def has_doctype(text):
        return 1 if '<!DOCTYPE' in text.upper() else 0
    
    @staticmethod
    def has_xml_declaration(text):
        return 1 if text.strip().startswith('<?xml') else 0
    
    @staticmethod
    def has_svg_tag(text):
        return 1 if '<svg' in text.lower() else 0
    
    @staticmethod
    def has_vcalendar(text):
        return 1 if 'BEGIN:VCALENDAR' in text.upper() else 0
    
    @staticmethod
    def has_email_headers(text):
        headers = ['From:', 'To:', 'Subject:', 'Date:']
        return 1 if any(h in text for h in headers) else 0
    
    @staticmethod
    def has_mime_boundary(text):
        return 1 if 'multipart' in text.lower() else 0
    
    @staticmethod
    def html_tag_ratio(text):
        tags = len(re.findall(r'<[^>]+>', text))
        lines = len(text.split('\n'))
        return tags / lines if lines > 0 else 0
    
    @staticmethod
    def avg_line_length(text):
        lines = [l for l in text.split('\n') if l.strip()]
        return np.mean([len(l) for l in lines]) if lines else 0
    
    @staticmethod
    def special_char_ratio(text):
        special_chars = len(re.findall(r'[<>/]', text))
        return special_chars / len(text) if len(text) > 0 else 0
    
    @staticmethod
    def bracket_ratio(text):
        brackets = len(re.findall(r'[{}\[\]()]', text))
        return brackets / len(text) if len(text) > 0 else 0
    
    @staticmethod
    def semicolon_ratio(text):
        semicolons = text.count(';')
        return semicolons / len(text.split('\n')) if len(text.split('\n')) > 0 else 0
    
    @staticmethod
    def comment_ratio(text):
        comments = len(re.findall(r'(//|#|/\*|\*|--)', text))
        lines = len(text.split('\n'))
        return comments / lines if lines > 0 else 0
    
    @staticmethod
    def has_shebang(text):
        return 1 if text.split('\n')[0].startswith('#!') else 0
    
    @staticmethod
    def has_import(text):
        import_keywords = ['import ', 'require(', 'from ', 'using ', 'include ']
        return 1 if any(kw in text for kw in import_keywords) else 0
    
    @staticmethod
    def has_class_keyword(text):
        keywords = ['class ', 'struct ', 'interface ']
        return 1 if any(kw in text for kw in keywords) else 0
    
    @staticmethod
    def has_function_keyword(text):
        keywords = ['def ', 'function ', 'fn ', 'func ', 'method ']
        return 1 if any(kw in text for kw in keywords) else 0
    
    @staticmethod
    def indent_level_avg(text):
        lines = text.split('\n')
        indents = []
        for line in lines:
            if line.strip():
                spaces = len(line) - len(line.lstrip())
                indents.append(spaces)
        return np.mean(indents) if indents else 0
    
    @staticmethod
    def max_line_length(text):
        lines = text.split('\n')
        return max([len(l) for l in lines]) if lines else 0
    
    @staticmethod
    def digit_ratio(text):
        digits = sum(1 for c in text if c.isdigit())
        return digits / len(text) if len(text) > 0 else 0
    
    @staticmethod
    def uppercase_ratio(text):
        upper = sum(1 for c in text if c.isupper())
        return upper / len(text) if len(text) > 0 else 0

    @staticmethod
    def line_comment_ratio(text):
        lines = text.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith(('#', '//', '--')))
        return comment_lines / len(lines) if len(lines) > 0 else 0

    @staticmethod
    def block_comment_ratio(text):
        blocks = len(re.findall(r'/\*.*?\*/', text, flags=re.DOTALL))
        lines = len(text.split('\n'))
        return blocks / lines if lines > 0 else 0

    @staticmethod
    def keyword_count(text):
        keywords = [
            'def', 'class', 'function', 'fn', 'func', 'import', 'from', 'require',
            'package', 'namespace', 'public', 'private', 'protected', 'const', 'let',
            'var', 'await', 'async', 'return', 'if', 'else', 'elif', 'switch', 'case',
            'select', 'insert', 'update', 'delete', 'struct', 'interface', 'extends',
            'implements', 'module', 'using', 'include', 'begin', 'end'
        ]
        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text.lower())
        return sum(tokens.count(k) for k in keywords)

    @staticmethod
    def keyword_density(text):
        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text.lower())
        keyword_count = FeatureExtractor.keyword_count(text)
        return keyword_count / len(tokens) if len(tokens) > 0 else 0

    @staticmethod
    def hash_ratio(text):
        return text.count('#') / len(text) if len(text) > 0 else 0

    @staticmethod
    def colon_ratio(text):
        return text.count(':') / len(text) if len(text) > 0 else 0

    @staticmethod
    def equals_ratio(text):
        return text.count('=') / len(text) if len(text) > 0 else 0

    @staticmethod
    def comma_ratio(text):
        return text.count(',') / len(text) if len(text) > 0 else 0

    @staticmethod
    def quote_ratio(text):
        quotes = text.count('"') + text.count("'")
        return quotes / len(text) if len(text) > 0 else 0

    @staticmethod
    def json_like_score(text):
        score = text.count('{') + text.count('}') + text.count(':') + text.count('[') + text.count(']')
        return score / len(text) if len(text) > 0 else 0

    @staticmethod
    def yaml_like_score(text):
        score = text.count('\n- ') + text.count(': ') + text.count('  - ')
        return score / len(text) if len(text) > 0 else 0

    @staticmethod
    def xml_like_score(text):
        tags = len(re.findall(r'<[^>]+>', text))
        lines = len(text.split('\n'))
        return tags / lines if lines > 0 else 0


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
