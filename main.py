# For Kibana, open your browser at http://localhost:5601
#   Username: elastic
#   Password: <password in the output of start-local>
# Learn more at https://github.com/elastic/start-local

# References:
# https://huggingface.co/learn/cookbook/rag_with_hugging_face_gemma_elasticsearch
# https://ai.google.dev/gemma/docs/core/huggingface_inference

import csv
import json
from typing import Any

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AnyToAnyPipeline, GenerationConfig

# Create the client instance
client = Elasticsearch(
    hosts = "http://localhost:9200",
    api_key = "<key in the output of start-local>"
)

EMBEDDING_MODEL_ID = "thenlper/gte-small"
# https://huggingface.co/thenlper/gte-small's page shows the dimensions of the model
# If you use the `gte-base` or `gte-large` embedding models, the numDimension
# value in the vector search index must be set to 768 and 1024, respectively.
EMBEDDING_DIMENSIONS = 384

embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)


def get_embedding(text: str) -> list[float]:
    embedding = embedding_model.encode(text)
    return embedding.tolist()


index_name = "wildlife"

index_mapping = {
    "properties": {
        "id": {"type": "text"},
        "name": {"type": "text"},
        "average_weight": {"type": "text"},
        "average_height": {"type": "text"},
        "areas": {"type": "text"},
        "embedding": {
            "type": "dense_vector",
            "dims": EMBEDDING_DIMENSIONS,
            "index": "true",
            "similarity": "cosine",
        }
    }
}

# flag to check if index has to be deleted before creating
should_delete_index = True


def create_index():
    if should_delete_index:
        if client.indices.exists(index = index_name):
            print("Deleting existing index: %s" % index_name)
            client.indices.delete(index = index_name)

    print("Creating index: %s" % index_name)

    client.indices.create(
        index = index_name,
        mappings = index_mapping,
        settings = {}
    )

    print("Ingesting data...")

    actions = []
    with open("wildlife_data.csv", 'r') as data:
        for line_dict in csv.DictReader(data):
            actions.append({
                "_index": index_name,
                "_source": {
                    "id": line_dict["id"],
                    "name": line_dict["name"],
                    "average_weight": line_dict["average_weight"],
                    "average_height": line_dict["average_height"],
                    "areas": line_dict["areas"],
                    "embedding": get_embedding(
                        f"The {line_dict['name']} "
                        f"is {line_dict['average_height']} tall, " +
                        f"weighs {line_dict['average_weight']}, " +
                        f"and is found in {line_dict["areas"]}."
                    )
                }
            })

    helpers.bulk(client, actions)

    client.indices.refresh(index = index_name)


def vector_search(vector_search_question: str):
    print(f"Performing a vector search: {vector_search_question}")

    question_embedding = get_embedding(vector_search_question)
    knn = {
        "field": "embedding",
        "query_vector": question_embedding,
        "k": 20,
        "num_candidates": 30,
    }

    response = client.search(index = index_name, knn = knn, size = 3)
    vector_search_results = []
    for hit in response["hits"]["hits"]:
        vector_search_results.append({
            "id": hit["_id"],
            "_score": hit["_score"],
            "_source": hit["_source"]
        })
    return vector_search_results


def search_results_to_question(question: str, vector_search_results: list[object]):
    return f"""
    {question}
    {
    json.dumps(
        (list(map(lambda x: x["_source"], vector_search_results))),
        indent = 4
    )
    }
    """


MODEL_ID = "google/gemma-4-E2B-it"


def rag_query(system_prompt, query) -> dict[str, Any]:
    pipe: AnyToAnyPipeline = pipeline(
        task = "any-to-any",
        model = MODEL_ID,
        device_map = None,
        dtype = "auto"
    )

    config = GenerationConfig.from_pretrained(MODEL_ID)
    config.max_new_tokens = 512
    gen_kwargs = dict(generation_config = config)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": query}]
        },
    ]

    responses = pipe(
        messages,
        return_full_text = False,
        generate_kwargs = gen_kwargs,
        skip_special_tokens = True
    )

    return responses[0]


#####################################

create_index()

vector_search_response = vector_search("animals")

for entry in vector_search_response:
    print(entry)

rag_system_prompt = "You are a biologist."

rag_question = search_results_to_question(
    "Which of these animals live in the forest?",
    vector_search_response
)

print(rag_question)

print(">>>>>>>>> Response: ")

print(rag_query(rag_system_prompt, rag_question)["generated_text"])
