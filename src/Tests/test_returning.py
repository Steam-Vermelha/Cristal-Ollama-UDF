from DataBusiness import BaseConnection


def test_returning_by_name():
    connection = BaseConnection()
    eldenring = connection.games.find_one({'title': 'Elden Ring'})

    assert eldenring['title'] == 'Elden Ring'

def test_deleting():
    connection = BaseConnection()
    connection.embeddings_llm.delete_many({})

    print("All deleted.")

def test_remove_duplicate():
    connection = BaseConnection()
    games_db = connection.games

    removed_count = 0
    seen_documents = set()
    unique_fields = "title"

    for document in games_db.find():
        unique_key = tuple(document[field] for field in unique_fields)

        if unique_key in seen_documents:
            games_db.delete_one({'_id': document['_id']})
            removed_count += 1
        else:
            seen_documents.add(unique_key)

    return removed_count