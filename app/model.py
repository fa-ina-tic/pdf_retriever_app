import os
import sys
import yaml
import threading
import time
import queue

# embeddings
from langchain.embeddings import CohereEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.elasticsearch import ElasticsearchEmbeddings

# vectorstores
from langchain.vectorstores import FAISS, Chroma, Bagel
from langchain.document_loaders import TextLoader

class Retriever():
    def __init__(self):
        pass

    def construct_db(self, store_type, raw_text, embedding_function):
        match store_type:
            case "FAISS":
                return FAISS.from_texts(raw_text, embedding_function)
            case "ChromaDB":
                return Chroma.from_texts(raw_text, embedding_function)
            case "BagelDB":
                return Bagel.from_texts(raw_text, embedding_function, cluster_name="db")
            case "Elasticsearch":
                return None
            case "Pinecone":
                return None

    def get_embedding_function(self, embedding_function, cfg):
        embedding_cfg = cfg['EMBEDDING']
        match embedding_function:
            case "OpenAI":
                return OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key = "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")
            case "SentenceTransformers":
                return SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

    def get_retriever(self, cfg, state):
        embedding_function = self.get_embedding_function(embedding_function=state['embeddings'], cfg=cfg)
        db = self.construct_db(self, store_type=state['vectordb'], embeddings=embedding_function)
        return db.as_retriever(search_type="similarity", search_kwargs={'k':5})
