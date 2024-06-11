from pymongo import MongoClient
from dotenv import load_dotenv
import os


def load_variables():
    load_dotenv()  # Ensure the environment variables are loaded
    # Retrieve and return MongoDB environment variables
    config = {
        "host": os.getenv("MONGO_HOST"),
        "database": os.getenv("MONGO_DATABASE"),
        "username": os.getenv("MONGO_USERNAME"),
        "password": os.getenv("MONGO_PASSWORD")
    }
    return {
        "uri": f"mongodb+srv://{config['username']}:{config['password']}@{config['host']}/",
        "database": os.getenv("MONGO_DATABASE")
    }
class MongoDBConnectionFactory:
    _client = None

    @staticmethod
    def get_db():
        config = load_variables()
        if MongoDBConnectionFactory._client is None:
            MongoDBConnectionFactory._client = MongoClient(config["uri"])
        return MongoDBConnectionFactory._client[config['database']]
