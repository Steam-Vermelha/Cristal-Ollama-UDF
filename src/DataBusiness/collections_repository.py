from .index import MongoDBConnectionFactory
import pandas as pd

class BaseConnection():
    df_games = pd.DataFrame()
    def __init__(self):
        self.db = MongoDBConnectionFactory.get_db()
        self.games = self.db['Games']  # Conectando à coleção 'Games'
        self.embeddings = self.db['Embeddings']
        self.embeddings_llm = self.db['Embeddings-Ollama']



    @classmethod
    def charging_collections_in_dataframe(cls):
        try:
            games = list(cls().games.find())
            df_games = pd.DataFrame(games)

            # Substituir valores NaN por listas vazias onde necessário
            df_games['genres'] = df_games['genres'].apply(lambda x: x if isinstance(x, list) else [])
            df_games['plataforms'] = df_games['plataforms'].apply(lambda x: x if isinstance(x, list) else [])
            df_games['tags'] = df_games['tags'].apply(lambda x: x if isinstance(x, list) else [])

            # Criar a coluna full_text concatenando as informações relevantes
            df_games['full_text'] = df_games.apply(
                lambda row: f"{row['title']}: {row['description']} \n Gêneros: {' '.join(map(str, row['genres']))} \n"
                            f"{' '.join(map(str, row['plataforms']))} \n Tags: {' '.join(map(str, row['tags']))} "
                            f"\n Rating: {row['rating']}\n Preço: {row['price']} \n Desenvolvedores: {row['developers']} \n Data de Lançamento: {row['release_date']}", axis=1)

            cls.df_games = df_games
            print(f"{len(games)} documentos carregados no DataFrame.")
        except Exception as e:
            print(f"Erro ao carregar os dados no DataFrame: {e}")


def load_game_data():
    BaseConnection.charging_collections_in_dataframe()
    return BaseConnection.df_games
