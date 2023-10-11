import os
import sys
import yaml
import threading
import time
import queue

from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI, OpenAIChat
from langchain.llms import HuggingFacePipeline, HuggingFaceHub
from langchain.chains import RetrievalQA#WithSourcesChain
import streamlit as st


from model import Retriever



class Chain(Retriever):
    def __init__(self, cfg, state):
        self.cfg = cfg
        self.rqa = self.init_rqa(state)

    @st.cache_resource
    def init_rqa(_self, state):
        # read pdf file
        pdf_reader = PdfReader(state['pdf'])
        raw_text = ''
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

        # make vector database
        text_splitter = TokenTextSplitter(
            chunk_size=state['chunk_size'],
            chunk_overlap=state['chunk_overlap'],
        )
        texts = text_splitter.split_text(raw_text)

        # db/embedding을 선택할 수 있도록
        retriever = Retriever.get_retriever(
            _self,
            cfg = _self.cfg, 
            state = state, 
            store_type = _self.vectordb, 
            embedding_function = _self.embeddings,
        )

        # db = FAISS.from_texts(texts, _self.embeddings)
        # retriever = db.as_retriever(search_type="similarity", search_kwargs={'k':5})

        MODEL = 'beomi/KoAlpaca'
        llm = OpenAIChat(model=_self.cfg['MODEL'])
        #llm = HuggingFaceHub(
        #    repo_id=MODEL, model_kwargs={"temperature": 0, "max_length": 512}
        #)

        # make retriever gpt
        rqa = RetrievalQA.from_chain_type(llm=llm,
                                          chain_type=_self.cfg['CHAIN_TYPE'],
                                          chain_type_kwargs={
                                              "prompt" : PromptTemplate(
                                                  template=state['template'],
                                                  input_variables=["context", "question"],
                                              ),
                                          },
                                          retriever=retriever,
                                          return_source_documents=True)
        return rqa

    def ask(self, query):
        return self.rqa(query)