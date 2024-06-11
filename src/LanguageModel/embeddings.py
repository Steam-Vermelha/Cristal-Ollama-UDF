# Gera embeddings das descrições dos jogos.

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib

from DataBusiness import BaseConnection


def generate_embeddings(df, model):
    embeddings = model.encode(df['full_text'].tolist(), show_progress_bar=True)
    metadata = df.to_dict(orient='records')
    return embeddings, metadata


def save_embeddings_to_file(embeddings, metadata, embeddings_file, metadata_file):
    np.save(embeddings_file, embeddings)
    joblib.dump(metadata, metadata_file)


def process_embeddings():
    BaseConnection.charging_collections_in_dataframe()
    df_games = BaseConnection.df_games

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings, metadata = generate_embeddings(df_games, model)

    save_embeddings_to_file(embeddings, metadata, 'embeddings.npy', 'metadata.pkl')
    print("Embeddings and metadata saved successfully.")