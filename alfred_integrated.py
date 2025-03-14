import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
import os
import torch
import tempfile
import logging
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import PDFToTextConverter, PreProcessor
from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import convert_files_to_docs

# Configuração de logging
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

# Carrega variáveis de ambiente e chaves de acesso.
_ = load_dotenv(find_dotenv())

# Configuração da página Streamlit
st.set_page_config(
    page_title="Alfred - Assistente Inteligente",
    page_icon="🤖",
    layout="wide"
)

# Configuração do Ollama (do sistema original)
ollama_server_url = "http://192.168.1.5:11434" 
model_local = ChatOllama(model="llama3.1:8b-instruct-q4_K_S")

# Diretório para armazenar documentos
DOCUMENT_DIR = "documents"
os.makedirs(DOCUMENT_DIR, exist_ok=True)

# Inicialização do Document Store para PDFs
@st.cache_resource
def initialize_pdf_document_store():
    return FAISSDocumentStore(
        faiss_index_factory_str="Flat",
        return_embedding=True,
        similarity="dot_product"
    )

# Função para processar documentos PDF
def process_pdf_documents(document_store, pdf_files):
    # Converter PDFs para texto
    converter = PDFToTextConverter(
        remove_numeric_tables=True,
        valid_languages=["pt", "en"]
    )
    
    docs = []
    for pdf_file in pdf_files:
        # Salvar o arquivo temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_path = temp_file.name
        
        # Converter para documentos Haystack
        pdf_docs = convert_files_to_docs(temp_path)
        converted_docs = converter.convert(pdf_docs)
        docs.extend(converted_docs)
        
        # Remover arquivo temporário
        os.unlink(temp_path)
    
    # Pré-processamento dos documentos
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=200,
        split_overlap=20,
        split_respect_sentence_boundary=True,
    )
    
    processed_docs = preprocessor.process(docs)
    
    # Limpar o document store e adicionar os novos documentos
    document_store.delete_documents()
    document_store.write_documents(processed_docs)
    
    return len(processed_docs)

# Função para criar o pipeline de QA para PDFs
@st.cache_resource
def create_pdf_qa_pipeline(_document_store):
    # Inicializar o retriever
    retriever = EmbeddingRetriever(
        document_store=_document_store,
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_format="sentence_transformers",
        top_k=5
    )
    
    # Atualizar embeddings no document store
    _document_store.update_embeddings(retriever)
    
    # Inicializar o reader
    reader = FARMReader(
        model_name_or_path="pierreguillou/bert-base-multilingual-cased-squad",
        use_gpu=torch.cuda.is_available(),
        top_k=3
    )
    
    # Criar o pipeline
    pipe = ExtractiveQAPipeline(reader, retriever)
    
    return pipe

# Carregamento da base de conhecimento CSV (do sistema original)
@st.cache_data
def load_csv_data():    
    # Substituia aqui por sua base de conhecimentos.
    loader = CSVLoader(file_path="knowledge_base.csv")

    # No mesmo servidor, uso também um modelo de Embedding
    embeddings = OllamaEmbeddings(base_url=ollama_server_url,
                                model='nomic-embed-text')
    documents = loader.load()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# Configuração do prompt e do modelo para o sistema original
rag_template = """
Você é um atendente de uma empresa.
Seu trabalho é conversar com os clientes, consultando a base de 
conhecimentos da empresa, e dar 
uma resposta simples e precisa para ele, baseada na 
base de dados da empresa fornecida como 
contexto.

Contexto: {context}

Pergunta do cliente: {question}
"""
human = "{text}"
prompt = ChatPromptTemplate.from_template(rag_template)

# Título principal
st.title("Alfred - Assistente Inteligente")

# Inicializar o document store para PDFs
pdf_document_store = initialize_pdf_document_store()

# Tabs para diferentes funcionalidades
tab1, tab2 = st.tabs(["Assistente Geral", "Assistente de Documentos PDF"])

# Tab 1: Assistente Geral (sistema original)
with tab1:
    st.header("Assistente Geral")
    st.markdown("""
    Este assistente responde perguntas com base na base de conhecimento da empresa.
    """)
    
    # Carregar a base de conhecimento CSV
    csv_retriever = load_csv_data()
    
    # Configurar o chain
    chain = (
        {"context": csv_retriever, "question": RunnablePassthrough()}
        | prompt
        | model_local
    )
    
    # Inicializar histórico de mensagens
    if "messages_general" not in st.session_state:
        st.session_state.messages_general = []
    
    # Exibir mensagens do histórico
    for message in st.session_state.messages_general:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Caixa de entrada para o usuário
    if user_input := st.chat_input("Você (Assistente Geral):"):
        st.session_state.messages_general.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
    
        # Adiciona um container para a resposta do modelo
        response_stream = chain.stream({"text": user_input})    
        full_response = ""
        
        response_container = st.chat_message("assistant")
        response_text = response_container.empty()
        
        for partial_response in response_stream:
            full_response += str(partial_response.content)
            response_text.markdown(full_response + "▌")
    
        # Salva a resposta completa no histórico
        st.session_state.messages_general.append({"role": "assistant", "content": full_response})

# Tab 2: Assistente de Documentos PDF
with tab2:
    st.header("Assistente de Documentos PDF")
    st.markdown("""
    Este assistente permite que você faça perguntas sobre seus documentos PDF.
    Faça upload dos seus documentos e comece a fazer perguntas!
    """)
    
    # Sidebar para upload de documentos
    with st.sidebar:
        st.header("Upload de Documentos")
        uploaded_files = st.file_uploader(
            "Faça upload dos seus documentos PDF",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if uploaded_files:
            if st.button("Processar Documentos"):
                with st.spinner("Processando documentos..."):
                    num_docs = process_pdf_documents(pdf_document_store, uploaded_files)
                    st.success(f"{num_docs} fragmentos de documentos processados com sucesso!")
                    
                    # Criar o pipeline após processar os documentos
                    pdf_qa_pipeline = create_pdf_qa_pipeline(pdf_document_store)
                    st.session_state.pdf_pipeline_ready = True
        
        # Exibir documentos já processados
        if pdf_document_store.get_document_count() > 0:
            st.success(f"{pdf_document_store.get_document_count()} fragmentos de documentos já carregados.")
            if st.button("Limpar Documentos"):
                pdf_document_store.delete_documents()
                st.experimental_rerun()
    
    # Inicializar histórico de mensagens para PDF QA
    if "messages_pdf" not in st.session_state:
        st.session_state.messages_pdf = []
    
    # Exibir mensagens do histórico
    for message in st.session_state.messages_pdf:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Área principal para perguntas e respostas
    if pdf_document_store.get_document_count() == 0:
        st.info("Por favor, faça upload de documentos PDF para começar.")
    else:
        # Inicializar o pipeline se ainda não estiver pronto
        if not hasattr(st.session_state, "pdf_pipeline_ready") or not st.session_state.pdf_pipeline_ready:
            with st.spinner("Preparando o sistema de QA..."):
                pdf_qa_pipeline = create_pdf_qa_pipeline(pdf_document_store)
                st.session_state.pdf_pipeline_ready = True
        
        # Interface de perguntas e respostas
        query = st.text_input("Sua pergunta sobre os documentos PDF:", key="pdf_query")
        
        if query:
            # Adicionar pergunta ao histórico
            st.session_state.messages_pdf.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            
            with st.spinner("Buscando resposta..."):
                # Executar a consulta
                results = pdf_qa_pipeline.run(
                    query=query,
                    params={
                        "Retriever": {"top_k": 5},
                        "Reader": {"top_k": 3}
                    }
                )
                
                # Preparar resposta
                response = ""
                
                if results["answers"]:
                    answer = results["answers"][0]  # Melhor resposta
                    confidence = round(answer.score * 100, 2)
                    response = f"**Resposta:** {answer.answer}\n\n"
                    response += f"*Confiança: {confidence}%*\n\n"
                    response += f"**Contexto:** {answer.context}"
                else:
                    response = "Não foi possível encontrar uma resposta para sua pergunta nos documentos fornecidos."
                
                # Adicionar resposta ao histórico
                st.session_state.messages_pdf.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                # Exibir documentos recuperados
                with st.expander("Ver todos os documentos recuperados"):
                    for i, doc in enumerate(results["documents"]):
                        st.markdown(f"**Documento {i+1}** (Relevância: {round(doc.score * 100, 2)}%):")
                        st.markdown(f"*{doc.content[:500]}...*")
                        st.markdown(f"**Fonte:** {doc.meta.get('name', 'Desconhecido')}")
                        st.markdown("---")

# Rodapé
st.markdown("---")
st.markdown("**Alfred Agent** - Assistente Inteligente Integrado") 