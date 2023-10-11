import os
import sys
import yaml
import threading
import time
import queue
import streamlit as st

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
        if type(store_type) != str:
            TypeError(f'store type should be string, now {type(store_type)}')
        match store_type:
            case "FAISS":
                return FAISS.from_texts(raw_text, embedding_function)
            case "ChromaDB":
                return Chroma.from_texts(raw_text, embedding_function)
            # case "BagelDB":
            #     return Bagel.from_texts(cluster_name="bageldb", texts=raw_text)
            # case "Elasticsearch":
            #     return None
            # case "Pinecone":
            #     return None

    def get_embedding_function(self, embedding_function, cfg):
        embedding_cfg = cfg['EMBEDDING']
        match embedding_function:
            case "OpenAI":
                return OpenAIEmbeddings()
            case "SentenceTransformers":
                return SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
            case "ElasticSearch":
                return ElasticsearchEmbeddings(model_id="maum_retriever",
                                               es_cloud_name=st.secrets["ELASTIC_SEARCH"]["ES_CLOUD_ID"],
                                               es_user = st.secrets["ELASTIC_SEARCH"]["ES_USER"],
                                               es_password = st.secrets["ELASTIC_SEARCH"]["ES_PASSWORD"]
                                               )

    def get_retriever(self, cfg, state, raw_text):
        embedding_function = self.get_embedding_function(embedding_function=state['embeddings'], cfg=cfg)
        db = self.construct_db(store_type=state['vectordb'], embedding_function=embedding_function, raw_text=raw_text)
        return db.as_retriever(search_type="similarity", search_kwargs={'k':5})
