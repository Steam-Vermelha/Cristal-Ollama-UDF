import numpy as np
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

from DataBusiness import BaseConnection

#   OPÇÃO PARA GERAR EMBEDDINGS COM O OLLAMA


def load_games_data_from_mongodb():
    connection = BaseConnection()
    embedding_llm = connection.embeddings_llm
    documents = list(embedding_llm.find())

    embeddings = [doc['embedding'] for doc in documents]
    metadatas = [{key: value for key, value in doc.items() if key != 'embedding'} for doc in documents]

    df_games = pd.DataFrame(metadatas)
    return np.array(embeddings), df_games


# Função para salvar embeddings e metadados em arquivos
def save_embeddings_and_metadata(embeddings, df, embeddings_filepath, metadata_filepath):
    np.save(embeddings_filepath, embeddings)
    joblib.dump(df.to_dict(orient='records'), metadata_filepath)


def generate_and_save_embeddings():
    # Carregar embeddings e metadados do MongoDB
    embeddings, df_games = load_games_data_from_mongodb()

    # Salvar os embeddings e metadados
    save_embeddings_and_metadata(embeddings, df_games, 'embeddings_llm.npy', 'metadata_llm.pkl')


if __name__ == "__main__":
    generate_and_save_embeddings()
