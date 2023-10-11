import os
import sys
import yaml
import threading
import time
import queue

import tiktoken
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI, OpenAIChat
from langchain.llms import HuggingFacePipeline, HuggingFaceHub
from langchain.chains import RetrievalQA#WithSourcesChain

# embeddings
from langchain.embeddings import CohereEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.elasticsearch import ElasticsearchEmbeddings

from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter

# vectorstores
from langchain.vectorstores import FAISS, Chroma, Bagel
from langchain.document_loaders import TextLoader

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader

import constants as const
from chain import Chain
from model import Retriever
from utils import Utils

class Renderer():
    def __init__(self, cfg):
        # parse config
        self.chain_cfg = cfg['CHAIN']
        self.ui_cfg = cfg['UI']

        # page setting
        st.set_page_config(page_title=self.ui_cfg['PAGE_TITLE'],
                           page_icon=self.ui_cfg['PAGE_ICON'],
                           layout=self.ui_cfg['PAGE_LAYOUT'])
        self.utils = Utils(self.chain_cfg)
        if 'results' not in st.session_state:
            st.session_state['results'] = []
        # if 'user_embeddings' not in st.session_state:
        #     st.session_state['user_embeddings'] = "OpenAI"
        # if 'user_vectorstore' not in st.session_state:
        #     st.session_state['user_vectorstore'] = "FAISS"
        # if 'api_key' not in st.session_state:
        #     st.session_state['api_key'] = ''
        

    def print_result(self, result):
        st.subheader("답변")
        st.markdown(result['result'])

        st.divider()
        st.subheader("답변의 근거")
        for idx, doc in enumerate(result['source_documents']):
            st.markdown(f"#### 근거 {idx+1}")
            st.markdown(doc.page_content)


    def elem_history(self):
        st.write(f"Total {len(st.session_state['results'])}")
        for result in st.session_state['results']:
            with st.expander(result['query'][:40]):
                st.markdown("#### 질문")
                st.markdown(result['query'])
                st.markdown("#### 답변")
                st.markdown(result['result'])
                for idx, doc in enumerate(result['source_documents']):
                    add_vertical_space(1)
                    st.markdown(f"##### 근거 {idx+1}")
                    st.markdown(doc.page_content)



    def elem_retriever_settings(self):
        with st.expander("Retriever 설정", expanded=True):
            with st.form("form"):
                # settings
                st.slider(label="Chunk 사이즈 (단위 : 토큰)",
                            step=self.ui_cfg['DEFAULT_SLIDER_STEPS'],
                            value=self.ui_cfg['DEFAULT_CHUNK_SIZE'],
                            max_value=self.ui_cfg['MAX_TOKENS'],
                            key="chunk_size")
                st.slider(label="Chunk overlap",
                            step=self.ui_cfg['DEFAULT_SLIDER_STEPS'],
                            value=self.ui_cfg['DEFAULT_CHUNK_OVERLAP'],
                            max_value=self.ui_cfg['MAX_TOKENS'],
                        key="chunk_overlap")
                st.text_area(label="Prompt 템플릿",
                            value=const.DEFAULT_PROMPT_TEMPLATE,
                            height=500,
                            key="prompt_template")
                user_embeddings = st.selectbox(label="Embedding Function",
                             options=["OpenAI", "SentenceTransformers"],
                             key="user_embeddings")
                user_vectorstore = st.selectbox(label="Vectorstore",
                            options=["FAISS", "ChromaDB"],
                            key="user_vectorstore")

                submit_button = st.form_submit_button("변경")

    def elem_word_count_dashboard(self):
        st.markdown("글자 수 (단위: 토큰 / 최대: 4097)")
        # word count dashboard
        context_count = st.session_state.chunk_size
        template_count = self.utils.count_tokens(st.session_state.prompt_template) + context_count * 3
        if 'prompt' not in st.session_state:
            st.session_state['prompt'] = ""
        prompt_count = self.utils.count_tokens(st.session_state.prompt)

        col1, col2, col3 = st.columns(3)
        col1.metric("Template", value=template_count)
        col2.metric("Prompt", value=prompt_count)
        col3.metric("Total", value=prompt_count + template_count)


    def elem_ask(self):
        query = st.text_input(label="## 질문", placeholder="PDF 파일의 내용에 관해 질문하세요",
                            key="prompt")

        if st.button("질문"):
            add_vertical_space(1)
            with st.spinner("ChatGPT에 물어보는 중.."):
                result = self.chain.ask(query)

            if result:
                st.session_state['results'].append(result)
                st.success("ChatGPT 답변 완료")
                self.print_result(result)


    def create_sidebar(self):
        with st.sidebar:
            st.title("PDF Retriever")
            st.markdown("## About"\
                        "PDF chatbot from [maum.ai](https://maum.ai)")

            add_vertical_space(3)


    def create_ask_tab(self):
        add_vertical_space(1)

        # file uploader
        pdf = st.file_uploader("PDF 파일을 업로드하세요", type='pdf', accept_multiple_files=False)

        add_vertical_space(1)

        if pdf:
            self.chain = Chain(self.chain_cfg,
                        state = {'pdf' : pdf,
                        'template' : st.session_state.prompt_template,
                        'chunk_size' : st.session_state.chunk_size,
                        'chunk_overlap' : st.session_state.chunk_overlap,
                        'embeddings' : st.session_state.user_embeddings,
                        'vectordb' : st.session_state.user_vectorstore
                        # 'api_key' : st.session_state.api_key
                        }
                        )
            self.elem_word_count_dashboard()
            self.elem_ask()

        add_vertical_space(2)

    def create_history_tab(self):
        add_vertical_space(1)
        self.elem_history()


    def create_body(self):

        col1, col2, col3 = st.columns([0.3, 0.4, 0.3], gap="large")
        with col1:
            add_vertical_space(12)
            self.elem_retriever_settings()
        with col2:
            st.header("PDF Retrieval GPT")
            add_vertical_space(1)
            tab1, tab2 = st.tabs(["질문", "기록"])
            with tab1:
                self.create_ask_tab()
            with tab2:
                self.create_history_tab()



def main(cfg):
    renderer = Renderer(cfg)
    renderer.create_sidebar()
    renderer.create_body()


def load_config():
    with open("app/config.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


if __name__=="__main__":
    cfg = load_config()
    main(cfg)