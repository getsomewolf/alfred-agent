import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import torch
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import PDFToTextConverter, PreProcessor
from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import convert_files_to_docs
import tempfile
import logging

# Configuração de logging
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

# Carrega variáveis de ambiente
_ = load_dotenv(find_dotenv())

# Configuração da página Streamlit
st.set_page_config(
    page_title="Alfred - Assistente de Documentos PDF",
    page_icon="📚",
    layout="wide"
)

# Diretório para armazenar documentos
DOCUMENT_DIR = "documents"
os.makedirs(DOCUMENT_DIR, exist_ok=True)

# Inicialização do Document Store
@st.cache_resource
def initialize_document_store():
    return FAISSDocumentStore(
        faiss_index_factory_str="Flat",
        return_embedding=True,
        similarity="dot_product"
    )

# Função para processar documentos PDF
def process_documents(document_store, pdf_files):
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

# Função para criar o pipeline de QA
@st.cache_resource
def create_qa_pipeline(_document_store):
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

# Interface Streamlit
st.title("Alfred - Assistente de Documentos PDF")
st.markdown("""
Este assistente permite que você faça perguntas sobre seus documentos PDF.
Faça upload dos seus documentos e comece a fazer perguntas!
""")

# Inicializar o document store
document_store = initialize_document_store()

# Sidebar para upload de documentos
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
    if document_store.get_document_count() > 0:
        st.success(f"{document_store.get_document_count()} fragmentos de documentos já carregados.")
        if st.button("Limpar Documentos"):
            document_store.delete_documents()
            st.experimental_rerun()

# Área principal para perguntas e respostas
if document_store.get_document_count() == 0:
    st.info("Por favor, faça upload de documentos PDF para começar.")
else:
    # Inicializar o pipeline se ainda não estiver pronto
    if not hasattr(st.session_state, "pipeline_ready") or not st.session_state.pipeline_ready:
        with st.spinner("Preparando o sistema de QA..."):
            qa_pipeline = create_qa_pipeline(document_store)
            st.session_state.pipeline_ready = True
    
    # Interface de perguntas e respostas
    st.header("Faça uma pergunta sobre seus documentos")
    query = st.text_input("Sua pergunta:")
    
    if query:
        with st.spinner("Buscando resposta..."):
            # Executar a consulta
            results = qa_pipeline.run(
                query=query,
                params={
                    "Retriever": {"top_k": 5},
                    "Reader": {"top_k": 3}
                }
            )
            
            # Exibir resultados
            st.subheader("Respostas:")
            
            if results["answers"]:
                for i, answer in enumerate(results["answers"]):
                    confidence = round(answer.score * 100, 2)
                    st.markdown(f"**Resposta {i+1}** (Confiança: {confidence}%):")
                    st.markdown(f"**{answer.answer}**")
                    
                    # Exibir contexto
                    with st.expander("Ver contexto"):
                        st.markdown(f"*{answer.context}*")
                        st.markdown(f"**Fonte:** Documento {answer.meta.get('name', 'Desconhecido')}")
            else:
                st.warning("Não foi possível encontrar uma resposta para sua pergunta nos documentos fornecidos.")
            
            # Exibir documentos recuperados
            with st.expander("Ver todos os documentos recuperados"):
                for i, doc in enumerate(results["documents"]):
                    st.markdown(f"**Documento {i+1}** (Relevância: {round(doc.score * 100, 2)}%):")
                    st.markdown(f"*{doc.content[:500]}...*")
                    st.markdown(f"**Fonte:** {doc.meta.get('name', 'Desconhecido')}")
                    st.markdown("---")

# Rodapé
st.markdown("---")
st.markdown("**Alfred Agent** - Assistente Inteligente para Documentos PDF") 