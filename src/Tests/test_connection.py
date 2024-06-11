import unittest
from pymongo import MongoClient
import pytest
from unittest.mock import patch
from DataBusiness.index import MongoDBConnectionFactory, load_variables

class TestMongoDBConnection(unittest.TestCase):


    @patch('DataBusiness.index.load_variables')
    @patch('pymongo.MongoClient')
    def test_mongodb_connection(self, mock_mongo_client, mock_load_variables):
        # Defina os valores de ambiente simulados para o teste
        mock_load_variables.return_value = {
            "uri": "mongodb+srv://GuiMcs00:KadidjaEstressada123@clustersv.zriin0z.mongodb.net/?retryWrites=true&w=majority&appName=ClusterSV",
            "database": "test_database"
        }

        # Simule o objeto MongoClient
        mock_client_instance = mock_mongo_client.return_value

        # Chame o método para obter o banco de dados
        db = MongoDBConnectionFactory.get_db()

        # Verifique se MongoClient foi inicializado com a URI correta
        mock_mongo_client.assert_called_once_with("mongodb+srv://GuiMcs00:KadidjaEstressada123@clustersv.zriin0z.mongodb.net/?retryWrites=true&w=majority&appName=ClusterSV")

        # Verifique se o método get_db retornou o banco de dados correto
        self.assertEqual(db, mock_client_instance["test_database"])

    def test_data_retrieval(self):
        """Test retrieving data from a known collection."""
        db = MongoDBConnectionFactory.get_db()
        db['test_collection'].insert_one({'name': 'test_item'})
        count = db['test_collection'].count_documents({})
        assert count == 1, "Should retrieve data from 'test_collection'."

    def test_singleton_client_instance(self):
        """Test that the MongoDBConnectionFactory returns the same client instance."""
        client1 = MongoDBConnectionFactory.get_db().client
        client2 = MongoDBConnectionFactory.get_db().client
        assert client1 is client2, "MongoDBConnectionFactory should return the same MongoClient instance."



if __name__ == '__main__':
    unittest.main()
