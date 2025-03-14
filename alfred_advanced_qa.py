import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import torch
import tempfile
import logging
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import OpenAIGenerator
import time

# Configuração de logging
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)

# Carrega variáveis de ambiente
_ = load_dotenv(find_dotenv())

# Verificar se a chave da API OpenAI está configurada
if not os.environ.get("OPENAI_API_KEY"):
    st.error("""
    ⚠️ **Chave da API OpenAI não encontrada!**
    
    Para usar este aplicativo, você precisa configurar sua chave da API OpenAI.
    
    1. Crie um arquivo `.env` na raiz do projeto
    2. Adicione sua chave da API: `OPENAI_API_KEY=sua-chave-aqui`
    
    Alternativamente, você pode usar o modelo local configurando o arquivo alfred_pdf_qa.py ou alfred_haystack_v2.py.
    """)
    st.stop()

# Configuração da página Streamlit
st.set_page_config(
    page_title="Alfred - Assistente Avançado de Documentos PDF",
    page_icon="🧠",
    layout="wide"
)

# Diretório para armazenar documentos
DOCUMENT_DIR = "documents"
os.makedirs(DOCUMENT_DIR, exist_ok=True)

# Inicialização do Document Store
@st.cache_resource
def initialize_document_store():
    return InMemoryDocumentStore(embedding_similarity_function="dot_product")

# Função para processar documentos PDF
def process_documents(document_store, pdf_files):
    # Converter PDFs para documentos
    converter = PyPDFToDocument()
    splitter = DocumentSplitter(
        split_by="word",
        split_length=200,
        split_overlap=20
    )
    
    embedder = SentenceTransformersDocumentEmbedder(
        model_name_or_path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    
    docs = []
    for pdf_file in pdf_files:
        # Salvar o arquivo temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_path = temp_file.name
        
        # Converter para documentos Haystack
        pdf_docs = converter.run(paths=[temp_path])
        split_docs = splitter.run(documents=pdf_docs["documents"])
        embedded_docs = embedder.run(documents=split_docs["documents"])
        docs.extend(embedded_docs["documents"])
        
        # Remover arquivo temporário
        os.unlink(temp_path)
    
    # Limpar o document store e adicionar os novos documentos
    document_store.delete_documents()
    document_store.write_documents(docs)
    
    return len(docs)

# Função para criar o pipeline de QA
@st.cache_resource
def create_qa_pipeline(_document_store):
    # Inicializar o retriever
    retriever = InMemoryEmbeddingRetriever(document_store=_document_store, top_k=5)
    
    # Inicializar o embedder para consultas
    text_embedder = SentenceTransformersTextEmbedder(
        model_name_or_path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    
    # Inicializar o gerador OpenAI
    generator = OpenAIGenerator(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o",
        streaming_callback=None
    )
    
    # Construtor de prompt
    prompt_builder = PromptBuilder(
        template="""
        Você é um assistente especializado em responder perguntas com base em documentos.
        Responda à pergunta com base apenas no contexto fornecido.
        Se a resposta não estiver no contexto, diga "Não encontrei informações suficientes nos documentos para responder a essa pergunta."
        Não invente informações.
        
        Contexto:
        {documents}
        
        Pergunta: {query}
        
        Resposta:
        """
    )
    
    # Criar o pipeline
    pipe = Pipeline()
    pipe.add_component("text_embedder", text_embedder)
    pipe.add_component("retriever", retriever)
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("generator", generator)
    
    # Conectar os componentes
    pipe.connect("text_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "generator.prompt")
    
    return pipe

# Interface Streamlit
st.title("Alfred - Assistente Avançado de Documentos PDF")
st.markdown("""
Este assistente permite que você faça perguntas sobre seus documentos PDF usando modelos avançados de linguagem.
Faça upload dos seus documentos e comece a fazer perguntas!
""")

# Inicializar o document store
document_store = initialize_document_store()

# Sidebar para upload de documentos e configurações
with st.sidebar:
    st.header("Upload de Documentos")
    uploaded_files = st.file_uploader(
        "Faça upload dos seus documentos PDF",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Processar Documentos"):
            with st.spinner("Processando documentos..."):
                num_docs = process_documents(document_store, uploaded_files)
                st.success(f"{num_docs} fragmentos de documentos processados com sucesso!")
                
                # Criar o pipeline após processar os documentos
                qa_pipeline = create_qa_pipeline(document_store)
                st.session_state.pipeline_ready = True
    
    # Exibir documentos já processados
    if document_store.count_documents() > 0:
        st.success(f"{document_store.count_documents()} fragmentos de documentos já carregados.")
        if st.button("Limpar Documentos"):
            document_store.delete_documents()
            st.experimental_rerun()
    
    # Configurações avançadas
    st.header("Configurações Avançadas")
    model_option = st.selectbox(
        "Modelo de Linguagem",
        options=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0
    )
    
    top_k = st.slider(
        "Número de documentos a recuperar",
        min_value=1,
        max_value=10,
        value=5
    )

# Inicializar histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Área principal para perguntas e respostas
if document_store.count_documents() == 0:
    st.info("Por favor, faça upload de documentos PDF para começar.")
else:
    # Inicializar o pipeline se ainda não estiver pronto
    if not hasattr(st.session_state, "pipeline_ready") or not st.session_state.pipeline_ready:
        with st.spinner("Preparando o sistema de QA..."):
            qa_pipeline = create_qa_pipeline(document_store)
            st.session_state.pipeline_ready = True
    
    # Interface de perguntas e respostas
    query = st.chat_input("Sua pergunta:")
    
    if query:
        # Adicionar pergunta ao histórico
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.spinner("Buscando resposta..."):
            # Atualizar o modelo e top_k se necessário
            qa_pipeline.components["generator"].model = model_option
            qa_pipeline.components["retriever"].top_k = top_k
            
            # Executar a consulta
            start_time = time.time()
            result = qa_pipeline.run(query=query)
            end_time = time.time()
            
            # Obter a resposta
            answer = result["generator"]["replies"][0]
            
            # Adicionar resposta ao histórico
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
                st.caption(f"Tempo de resposta: {round(end_time - start_time, 2)} segundos")
            
            # Exibir documentos recuperados
            with st.expander("Ver documentos recuperados"):
                for i, doc in enumerate(result["retriever"]["documents"]):
                    st.markdown(f"**Documento {i+1}**:")
                    st.markdown(f"*{doc.content[:500]}...*")
                    if hasattr(doc, "meta") and doc.meta:
                        st.markdown(f"**Fonte:** {doc.meta.get('source', 'Desconhecido')}")
                    st.markdown("---")

# Rodapé
st.markdown("---")
st.markdown("**Alfred Agent** - Assistente Avançado de Documentos PDF") 