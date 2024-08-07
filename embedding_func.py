from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_function():
    #embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    embeddings = HuggingFaceEmbeddings(model_name="ai-forever/sbert_large_nlu_ru")
    return embeddings