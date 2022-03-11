"""
Generate text encodings.
Uses either simple BoW encoding (one-hot).
Or contextual embeddings.

Usage: python code/svp/text_encoder.py <encoder> <release>
@param encoder: 'bow' or 'bert'
@param release: [63-84]
"""
import sys
import pandas as pd
from codeBERT import CodeBERT
from data_utils import Data
from tokenizer import clean_code, replace_tokens, get_vectorizer, encode_bow

# Read system parameters
encoding, release = 'bow', 63
if len(sys.argv) < 3:
    print('DEFAULT CONFIG: BoW release_63')
else:
    encoding, release = sys.argv[1], sys.argv[2]


dataset = 'Mozilla_Code'
release = int(release)
# Release 80 has no testing data so skip
if release == 78:
    test_release = 81
elif release == 79:
    val_release, test_release = 81, 82
else:
    val_release, test_release = release+1, release+2

# Load a dataset
df = Data(dataset).get_dataset()
# Extract sentences
sentences = df[df.release == release].code.fillna('').tolist()
sentences = [clean_code(s) for s in sentences]
print(len(sentences))

if encoding == 'bow':
    vectorizer = get_vectorizer(max_features=4000)
    if dataset == 'Mozilla_Code':
        # Test partitions require same encoding
        val_sentences = df[df.release == val_release].code.fillna('').tolist()
        test_sentences = df[df.release == test_release].code.fillna('').tolist()
        # Test partitions require same encoding
        val_sentences = df[df.release == val_release].code.fillna('').tolist()
        test_sentences = df[df.release == test_release].code.fillna('').tolist()
    # Clean
    features, vectorizer = encode_bow([replace_tokens(clean_code(s)) for s in sentences], vectorizer, train=True)
    if dataset == 'Mozilla_Code':
        val_features = encode_bow([replace_tokens(clean_code(s)) for s in val_sentences], vectorizer)
        test_features = encode_bow([replace_tokens(clean_code(s)) for s in test_sentences], vectorizer)
    features = [x for x in features]
elif encoding == 'bert':
    cb = CodeBERT()
    features = cb.encode(sentences)
    features = [x.tolist() for x in features]

# Convert to structured format
df_features = pd.DataFrame()
if dataset == 'Mozilla_Code':
    df_features['file'] = df[df.release == release]['file']
df_features['features'] = features
# Save as .pkl
df_features.to_pickle(f"code/svp/feature_sets/{encoding}_release_{release}.pkl")
if encoding == 'bow' and dataset == 'Mozilla_Code':
    df_val_features, test_features = pd.DataFrame(), pd.DataFrame()
    df_val_features['file'] = df[df.release == val_release]['file']
    df_test_features['file'] = df[df.release == test_release]['file']
    df_val_features['features'] = [x for x in val_features]
    df_test_features['features'] = [x for x in test_features]
    df_val_features.to_pickle(f"code/svp/feature_sets/{encoding}_release_{release}_val.pkl")
    df_test_features.to_pickle(f"code/svp/feature_sets/{encoding}_release_{release}_test.pkl")
