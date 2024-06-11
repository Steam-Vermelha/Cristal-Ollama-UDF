from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import numpy as np
import joblib
from DataBusiness import BaseConnection

# Função para carregar os dados de jogos
def load_games_data():
    BaseConnection.charging_collections_in_dataframe()
    df_games = BaseConnection.df_games
    return df_games

# Função para processar e gerar embeddings
def process_embeddings(df, model):
    df['full_text'] = df.apply(lambda row: f"{row['title']} {row['description']} {' '.join(map(str, row['genres']))} {' '.join(map(str, row['plataforms']))} {' '.join(map(str, row['tags']))} {row['rating']} {row['price']} {row['developers']} {row['release_date']}", axis=1)
    embeddings = model.encode(df['full_text'].tolist(), show_progress_bar=True)
    return embeddings

# Função para treinar o modelo
def train_model():
    # Passo 1: Preparação dos Dados

    # Passo 2: Carregar o Modelo Base
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Passo 3: Configuração do Treinamento
    train_examples = [
        # Exemplos Corretos
        InputExample(texts=['jogo de culinária', 'Overcooked'], label=1),
        InputExample(texts=['jogo de culinária', 'Cook, Serve, Delicious!'], label=1),
        InputExample(texts=['jogo de culinária', 'Cooking Mama'], label=1),
        InputExample(texts=['jogo de aventura', 'The Legend of Zelda: Breath of the Wild'], label=1),
        InputExample(texts=['jogo de aventura', 'Uncharted 4: A Thief\'s End'], label=1),
        InputExample(texts=['jogo de aventura', 'Tomb Raider'], label=1),
        InputExample(texts=['jogo de ação e tiro', 'Call of Duty: Modern Warfare'], label=1),
        InputExample(texts=['jogo de ação e tiro', 'Battlefield V'], label=1),
        InputExample(texts=['jogo de ação e tiro', 'DOOM Eternal'], label=1),
        InputExample(texts=['jogo de ação e tiro', 'Destiny 2'], label=1),
        InputExample(texts=['soulslike', 'Elden Ring'], label=1),
        InputExample(texts=['soulslike', 'Dark Souls'], label=1),
        InputExample(texts=['soulslike', 'Bloodborne'], label=1),
        InputExample(texts=['soulslike', 'Nioh'], label=1),
        InputExample(texts=['jogo de RPG', 'The Witcher 3: Wild Hunt'], label=1),
        InputExample(texts=['jogo de RPG', 'Final Fantasy XV'], label=1),
        InputExample(texts=['jogo de RPG', 'Persona 5'], label=1),
        InputExample(texts=['fantasia', 'The Legend of Zelda: Breath of the Wild'], label=1),
        InputExample(texts=['história', 'Red Dead Redemption 2'], label=1),
        InputExample(texts=['sombrio', 'Salt and Sanctuary'], label=1),
        InputExample(texts=['sombrio', 'Bloodborne'], label=1),
        InputExample(texts=['sombrio', 'Batman: Arkham Knight'], label=1),
        InputExample(texts=['sombrio', 'Limbo'], label=1),
        InputExample(texts=['sobrevivência', 'The Forest'], label=1),
        InputExample(texts=['sobrevivência', "Don't Starve"], label=1),
        InputExample(texts=['sobrevivência', "Subnáutica"], label=1),
        InputExample(texts=['difícil', "Super Meat Boy"], label=1),
        InputExample(texts=['difícil', 'Elden Ring'], label=1),
        InputExample(texts=['difícil', 'Dark Souls'], label=1),
        InputExample(texts=['difícil', 'Bloodborne'], label=1),
        InputExample(texts=['difícil', 'Keep Talking and Nobody Explodes'], label=1),
        InputExample(texts=['Co-op', 'Keep Talking and Nobody Explodes'], label=1),
        InputExample(texts=['quebra-cabeça', 'Donut County'], label=1),
        InputExample(texts=['quebra-cabeça', 'Inside'], label=1),
        InputExample(texts=['quebra-cabeça', 'Limbo'], label=1),
        InputExample(texts=['quebra-cabeça', 'Gunpoint'], label=1),
        InputExample(texts=['quebra-cabeça', 'The Witness'], label=1),
        InputExample(texts=['Puzzle', 'Donut County'], label=1),
        InputExample(texts=['Puzzle', 'Inside'], label=1),
        InputExample(texts=['Puzzle', 'Limbo'], label=1),
        InputExample(texts=['Puzzle', 'Gunpoint'], label=1),
        InputExample(texts=['Puzzle', 'The Witness'], label=1),
        InputExample(texts=['luta', 'Mortal Kombat X'], label=1),
        InputExample(texts=['luta', 'Dragon Ball FighterZ'], label=1),
        InputExample(texts=['batalha', 'Pokémon Sun and Moon'], label=1),

        # Exemplos Incorretos
        InputExample(texts=['jogo de culinária', 'Call of Duty: Modern Warfare'], label=0),
        InputExample(texts=['jogo de culinária', 'Donut County'], label=0),
        InputExample(texts=['jogo de culinária', 'Gunpoint'], label=0),
        InputExample(texts=['jogo de culinária', 'Limbo'], label=0),
        InputExample(texts=['jogo de culinária', 'Hotline Miami'], label=0),
        InputExample(texts=['jogo de culinária', 'Inside'], label=0),
        InputExample(texts=['jogo de culinária', 'The Legend of Zelda: Breath of the Wild'], label=0),
        InputExample(texts=['jogo de culinária', 'Dark Souls'], label=0),
        InputExample(texts=['jogo de aventura', 'Overcooked'], label=0),
        InputExample(texts=['jogo de aventura', 'DOOM Eternal'], label=0),
        InputExample(texts=['jogo de aventura', 'Call of Duty: Modern Warfare'], label=0),
        InputExample(texts=['jogo de ação e tiro', 'The Witcher 3: Wild Hunt'], label=0),
        InputExample(texts=['jogo de ação e tiro', 'Donut County'], label=0),
        InputExample(texts=['jogo de ação e tiro', 'Horizon Zero Dawn'], label=0),
        InputExample(texts=['jogo de ação e tiro', 'Nioh'], label=0),
        InputExample(texts=['jogo de ação e tiro', 'Overcooked'], label=0),
        InputExample(texts=['jogo de ação e tiro', 'Cooking Mama'], label=0),
        InputExample(texts=['soulslike', 'Overcooked'], label=0),
        InputExample(texts=['soulslike', 'The Legend of Zelda: Breath of the Wild'], label=0),
        InputExample(texts=['soulslike', 'Call of Duty: Modern Warfare'], label=0),
        InputExample(texts=['jogo de RPG', 'Overcooked'], label=0),
        InputExample(texts=['jogo de RPG', 'DOOM Eternal'], label=0),
        InputExample(texts=['jogo de RPG', 'Cooking Mama'], label=0),
        InputExample(texts=['fantasia', 'Just Dance 2014'], label=0),
        InputExample(texts=['sombrio', 'Super Smash Bros.'], label=0),
        InputExample(texts=['sombrio', 'Pokémon Sun and Moon'], label=0),
        InputExample(texts=['sombrio', 'Super Meat Boy'], label=0),
        InputExample(texts=['difícil', 'Donut County'], label=0),
        InputExample(texts=['luta', 'Destiny 2'], label=0),
        InputExample(texts=['luta', 'The Legend of Zelda: Breath of the Wild'], label=0),
        InputExample(texts=['luta', 'Pokémon Sun and Moon'], label=0),
    ]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.ContrastiveLoss(model)

    # Treinamento
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=5, warmup_steps=100)

    # Salvar o modelo treinado
    model.save('custom_model')

    # Carregar os dados dos jogos
    df_games = load_games_data()

    # Gerar os embeddings com o modelo treinado
    embeddings = process_embeddings(df_games, model)

    # Salvar os embeddings e metadados
    np.save('embeddings.npy', embeddings)
    joblib.dump(df_games.to_dict(orient='records'), 'metadata.pkl')

# Executar o treinamento
if __name__ == "__main__":
    train_model()
