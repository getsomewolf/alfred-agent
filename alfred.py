import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Carrega variáveis de ambiente e chaves de acesso.
_ = load_dotenv(find_dotenv())

# É necessário ter o Ollama instalado na sua máquina local
# Ou no servidor que for utilizar.

# No meu caso, estou usando o servidor da Asimov.
ollama_server_url = "http://192.168.1.5:11434" 
model_local = ChatOllama(model="llama3.1:8b-instruct-q4_K_S")

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


retriever = load_csv_data()
st.title("Alfred - Smart PDF Agent")


# Configuração do prompt e do modelo
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
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model_local
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Caixa de entrada para o usuário
if user_input := st.chat_input("Você:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
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
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    