# Preprocessa as descrições dos jogos para tokenização e vetorização.
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text


def vectorize_texts(texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def concatenate_all_fields(row):
    fields = [
        str(row.get('title', '')),
        str(row.get('description', '')),
        ' '.join(str(row.get('genres', []))),
        ' '.join(str(row.get('platforms', []))),
        ' '.join(str(row.get('tags', []))),
        str(row.get('rating', '')),
        str(row.get('price', '')),
        str(row.get('developers', '')),
        str(row.get('release_date', ''))
    ]
    return ' '.join(fields)
