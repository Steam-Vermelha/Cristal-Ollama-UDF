import faiss
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib

app = Flask(__name__)


# Função para converter ObjectId em string
def convert_objectid_to_str(metadata):
    for item in metadata:
        if '_id' in item:
            item['_id'] = str(item['_id'])
    return metadata


# Carregar o modelo treinado e os embeddings
model = SentenceTransformer('custom_model')
embeddings = np.load('embeddings.npy')
metadata = joblib.load('metadata.pkl')
metadata = convert_objectid_to_str(metadata)

# Criar o índice FAISS
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(embeddings)


@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'Query is required'}), 400

    query_embedding = model.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding).astype('float32'), k=5)

    results = [metadata[idx] for idx in indices[0]]
    return jsonify(results), 200


if __name__ == '__main__':
    app.run(debug=True)
