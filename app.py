# app.py
from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from google import genai
import json
import os

from dotenv import load_dotenv
load_dotenv()

ES_HOST = "http://localhost:9200"
ES_USER = "elastic"
ES_PASSWORD = "123456"
INDEX_NAME = "products"

# configure key for Google GenAI
GENAI_API_KEY = os.getenv("GENAI_API_KEY", "xyz")
client = genai.Client(api_key=GENAI_API_KEY)

# response = client.models.generate_content(
#     model="gemini-3-pro-preview",
#     contents="Explain how AI works in a few words",
# )

app = Flask(__name__)

# Kết nối Elasticsearch
es = Elasticsearch(
    ES_HOST,
    basic_auth=(ES_USER, ES_PASSWORD),
    verify_certs=False
)

# Model embedding dùng cho semantic search
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def hit_to_dict(hit):
    """Giảm bớt thông tin trả về cho gọn."""
    src = hit["_source"]
    return {
        "score": hit.get("_score", 0),
        "product_id": src.get("product_id"),
        "product_name": src.get("product_name"),
        "discounted_price": src.get("discounted_price"),
        "rating": src.get("rating"),
        "rating_count": src.get("rating_count"),
        "category": src.get("category"),
        "product_link": src.get("product_link"),
        "img_link": src.get("img_link"),
    }

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# 0) TRÍCH XUẤT TỪ KHÓA TỪ NHU CẦU NGƯỜI DÙNG
@app.route("/extract/keywords", methods=["POST"])
def extract_keywords():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Field 'text' is required"}), 400

    prompt = f"""
    Extract the most important shopping-related keywords from the user input.
    Return ONLY a list word include 3 word seperate by COMMA for amazon ecommerce recommend system. No explanation.

    Input: "{text}"

    Example Output:
    gaming computer, gaming mouse, mouse pad
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            # generation_config={
            #     "response_mime_type": "application/json"   # ép trả JSON
            # }
        )

        raw = response.text

        return jsonify({"keywords": raw})

    except Exception as e:
        print("Gemini Error:", e)
        return jsonify({"error": f"Error during keyword extraction: {str(e)}"}), 500

# 1) TÌM KIẾM NGÔN NGỮ TỰ NHIÊN (keyword / BM25)
@app.route("/search/keyword", methods=["GET"])
def search_keyword():
    q = request.args.get("q", "").strip()
    size = int(request.args.get("size", 10))

    if not q:
        return jsonify({"error": "Param 'q' is required"}), 400

    body = {
        "query": {
            "multi_match": {
                "query": q,
                "fields": [
                    "product_name^3",
                    "about_product^2",
                    "review_title^2",
                    "review_content",
                    "category"
                ],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        }
    }

    res = es.search(index=INDEX_NAME, body=body, size=size)
    hits = [hit_to_dict(h) for h in res["hits"]["hits"]]
    return jsonify(hits)


# 2) TÌM KIẾM NGỮ NGHĨA (semantic search bằng vector)
@app.route("/search/semantic", methods=["GET"])
def search_semantic():
    q = request.args.get("q", "").strip()
    size = int(request.args.get("size", 10))

    if not q:
        return jsonify({"error": "Param 'q' is required"}), 400

    query_vec = model.encode([q], normalize_embeddings=True)[0].tolist()

    body = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_vec}
                }
            }
        }
    }

    res = es.search(index=INDEX_NAME, body=body, size=size)
    hits = [hit_to_dict(h) for h in res["hits"]["hits"]]
    return jsonify(hits)


# 3) TÌM KIẾM CONTENT-BASED (similar product)
@app.route("/products/<product_id>/similar", methods=["GET"])
def similar_products(product_id):
    size = int(request.args.get("size", 10))

    try:
        doc = es.get(index=INDEX_NAME, id=product_id)
    except Exception:
        return jsonify({"error": f"Product {product_id} not found"}), 404

    src = doc["_source"]
    query_vec = src.get("embedding")

    if not query_vec:
        return jsonify({"error": "Document has no embedding"}), 500

    body = {
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "must_not": {
                            "term": {"product_id": product_id}
                        }
                    }
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_vec}
                }
            }
        }
    }

    res = es.search(index=INDEX_NAME, body=body, size=size)
    hits = [hit_to_dict(h) for h in res["hits"]["hits"]]
    return jsonify(hits)


if __name__ == "__main__":
    # Flask dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
