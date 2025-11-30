# index_data.py
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

CSV_PATH = "ecommerce_data.csv"  # bạn copy file /mnt/data/ecommerce_data.csv về project hoặc đổi path

ES_HOST = "http://localhost:9200"
INDEX_NAME = "products"
ES_USER = "elastic"
ES_PASSWORD = 123456

def create_es_client():
    es = Elasticsearch(
        ES_HOST,
        # basic_auth=(ES_USER, ES_PASSWORD),
        basic_auth=("elastic", "123456"),
        verify_certs=False,
        # ssl_show_warn=False,
    )
    if not es.ping():
        raise RuntimeError("Cannot connect to Elasticsearch with authentication!")
    return es

def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df.fillna("")  # tránh NaN
    return df

def build_text_fields(df: pd.DataFrame):
    # Gộp text để làm field combined_text cho embedding
    combined = (
        df["product_name"].astype(str) + ". " +
        df["about_product"].astype(str) + ". " +
        df["review_title"].astype(str) + ". " +
        df["review_content"].astype(str)
    )
    return combined

def create_index(es, dim: int):
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)

    body = {
        "mappings": {
            "properties": {
                "product_id": {"type": "keyword"},
                "product_name": {"type": "text"},
                "category": {"type": "keyword"},
                "discounted_price": {"type": "float"},
                "actual_price": {"type": "float"},
                "discount_percentage": {"type": "float"},
                "rating": {"type": "float"},
                "rating_count": {"type": "float"},
                "about_product": {"type": "text"},
                "user_id": {"type": "keyword"},
                "user_name": {"type": "keyword"},
                "review_id": {"type": "keyword"},
                "review_title": {"type": "text"},
                "review_content": {"type": "text"},
                "img_link": {"type": "keyword", "index": False},
                "product_link": {"type": "keyword", "index": False},
                "combined_text": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dim,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }

    es.indices.create(index=INDEX_NAME, body=body)
    print(f"Created index '{INDEX_NAME}' with dim={dim}")

def bulk_index(es, df, combined_texts, embeddings):
    def gen_actions():
        for i, row in df.iterrows():
            src = row.to_dict()
            src["combined_text"] = combined_texts.iloc[i]
            src["embedding"] = embeddings[i].tolist()

            # dùng product_id làm _id để dễ get
            yield {
                "_index": INDEX_NAME,
                "_id": row["product_id"],
                "_source": src
            }

    helpers.bulk(es, gen_actions())
    print("Indexed {} documents".format(len(df)))

def main():
    es = create_es_client()
    df = load_data()

    combined_texts = build_text_fields(df)

    # Model đa ngôn ngữ (Anh + Việt đều ok)
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    texts = combined_texts.tolist()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    dim = len(embeddings[0])
    create_index(es, dim)
    bulk_index(es, df, combined_texts, embeddings)

if __name__ == "__main__":
    main()
