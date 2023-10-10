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

    def construct_db(self, store_type, embeddings):
        match store_type:
            case "FAISS":
                return FAISS.from_texts()
            case "ChromaDB":
                return Chroma.from_texts()
            case "BagelDB":
                return Bagel.from_texts(cluster_name="db")

    def get_embedding_function(self, embedding_function, cfg):
        match embedding_function:
            case "OpenAI":
                return OpenAIEmbeddings(model=cfg['OPEN_AI_MODEL'])
            case "SentenceTransformers":
                return SentenceTransformerEmbeddings(model_name=cfg['SENTENCE_TRANSFORMER_MODEL'])

    def get_retriever(self, cfg, state, store_type, embedding_function):
        embedding_function = self.get_embedding_function(self, embedding_function=embedding_function, cfg=cfg)
        db = self.construct_db(self, store_type=store_type, embeddings=embedding_function)
        return db.as_retriever(search_type="similarity", search_kwargs={'k':5})