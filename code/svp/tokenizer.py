"""Helper functions for BoW Tokenization"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_code(file_content):
    """Remove comments and empty lines from the content of a file."""
    file_content = file_content.encode("utf-8", "replace").decode("utf-8")
    # Remove multi-line comments
    pattern = r"^\s*/\*(.*?)\*/"
    file_content = re.sub(pattern, "", file_content, flags=re.DOTALL | re.MULTILINE)
    # Remove inline comments within /* and */
    pattern = r"/\*(.*?)\*/"
    file_content = re.sub(pattern, "", file_content)
    # Remove single-line comments
    pattern = r"\s*//.*"
    file_content = re.sub(pattern, "", file_content, flags=re.MULTILINE)
    # Remove empty lines
    pattern = r"^\s*[\r\n]"
    file_content = re.sub(pattern, "", file_content, flags=re.MULTILINE)

    return file_content


def replace_tokens(file_content):
    """Replace literals with generic tokens."""
    # String literals
    pattern = r'"[^"]*"'
    file_content = re.sub(pattern, "TSTRING", file_content)
    # Char literals
    pattern = r'\'[^\']*\''
    file_content = re.sub(pattern, "TCHAR", file_content)
    # Numeric literals
    pattern = r"\b((?:0x[a-zA-Z_]+)|(?:0[a-zA-Z\d]+)|(?:\d*\.?\d*E\-?\d*)|(?:\d+\.(?:\d{0,3}_?)*)|(?:\d+)|(?:\d*\.\d+))\b"
    file_content = re.sub(pattern, "TNUM", file_content)

    return file_content


def gen_tok_pattern():
    """Generate a Regex C/C++ tokenization pattern."""
    single_toks = ['<=', '>=', '<', '>', '\\?', '\\/=', '\\+=', '\\-=',
                   '\\+\\+', '--', '\\*=', '\\+', '-', '\\*', '\\/', '!=',
                   '==', '=', '!', '&=', '&', '\\%', '\\|\\|', '\\|=', '\\|',
                   '\\$', '\\:']
    single_toks = '(?:' + '|'.join(single_toks) + ')'
    word_toks = '(?:[a-zA-Z0-9]+)'
    return single_toks + '|' + word_toks


def get_vectorizer(min_df=1, max_features=None, token_pattern=None, vocabulary=None):
    """Get a BoW Vectorizer (use l2 normalization)."""
    code_token_pattern = gen_tok_pattern()
    return TfidfVectorizer(stop_words=None, ngram_range=(1, 1), min_df=min_df, max_df=1.0, max_features=max_features, use_idf=False, norm='l2', smooth_idf=False, lowercase=False, token_pattern=code_token_pattern, vocabulary=vocabulary)


def encode_bow(sentences, vectorizer, train=False):
    """
    Encode sentences using a BoW vectorizer.
    @param: train - specify whether to first fit the vectorizer.
    """
    if train:
        vectorizer.fit(sentences)
    features = vectorizer.transform(sentences)
    if train:
        return features, vectorizer
    return features
