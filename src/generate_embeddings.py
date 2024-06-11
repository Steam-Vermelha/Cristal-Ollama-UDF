import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

from DataBusiness import BaseConnection

# OPÇÃO PARA GERAR EMBEDDINGS COM SENTENCE TRANSFORMER

def load_games_data():
    BaseConnection.charging_collections_in_dataframe()
    df_games = BaseConnection.df_games
    return df_games

def process_embeddings(df, model):
    df['full_text'] = df.apply(lambda row: f"{row['title']} {row['description']} {' '.join(map(str, row['genres']))} {' '.join(map(str, row['plataforms']))} {' '.join(map(str, row['tags']))} {row['rating']} {row['price']} {row['developers']} {row['release_date']}", axis=1)
    embeddings = model.encode(df['full_text'].tolist(), show_progress_bar=True)
    return embeddings

def generate_and_save_embeddings():
    model = SentenceTransformer('custom_model')
    df_games = load_games_data()
    embeddings = process_embeddings(df_games, model)
    np.save('embeddings.npy', embeddings)
    joblib.dump(df_games.to_dict(orient='records'), 'metadata.pkl')

if __name__ == "__main__":
    generate_and_save_embeddings()
