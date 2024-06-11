import unittest
from DataBusiness import BaseConnection
from LanguageModel import embeddings

class TestGameRecommender(unittest.TestCase):
    def test_database_connection(self):
        connection = BaseConnection()
        self.assertIsNotNone(connection.returning())

    def test_generate_embedding(self):
        descricao = "Descrição do jogo"
        model = embeddings.GamesEmbedding(1000, 100)
        embedding = embeddings.gerar_embedding(descricao, model)
        self.assertIsNotNone(embedding)

if __name__ == '__main__':
    unittest.main()
