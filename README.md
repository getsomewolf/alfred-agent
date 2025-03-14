# Alfred Agent

## Descrição
Alfred Agent é um assistente inteligente projetado para automatizar tarefas e fornecer suporte eficiente. Inspirado no famoso mordomo da série Batman, o Alfred Agent está sempre pronto para ajudar com suas necessidades computacionais.

## Funcionalidades
- Automação de tarefas rotineiras
- Processamento de linguagem natural
- Integração com diversos sistemas e APIs
- Assistência personalizada baseada em preferências do usuário
- **QA (Question Answering) para documentos privados (PDFs)** usando o framework Haystack

## Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/alfred-agent.git
cd alfred-agent

# Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

# Configure as variáveis de ambiente
# Crie um arquivo .env com suas chaves de API (se necessário)
```

## Uso do Sistema de QA para PDFs

O Alfred Agent permite que você faça perguntas sobre seus documentos PDF privados:

1. Coloque seus arquivos PDF na pasta `documents/`
2. Execute o aplicativo:
   ```bash
   streamlit run alfred_pdf_qa.py
   ```
3. Faça upload de novos documentos ou use os já existentes
4. Digite suas perguntas e obtenha respostas baseadas no conteúdo dos documentos

## Tecnologias Utilizadas

- **Haystack**: Framework para construção de pipelines de QA
- **Streamlit**: Interface de usuário interativa
- **LangChain**: Orquestração de modelos de linguagem
- **Sentence Transformers**: Modelos de embeddings para processamento semântico

## Estrutura do Projeto

```
alfred-agent/
├── alfred.py                # Aplicativo Streamlit original
├── alfred_pdf_qa.py         # Aplicativo de QA para PDFs
├── documents/               # Pasta para armazenar documentos PDF
├── knowledge_base.csv       # Base de conhecimento existente
├── requirements.txt         # Dependências do projeto
└── README.md                # Este arquivo
```
