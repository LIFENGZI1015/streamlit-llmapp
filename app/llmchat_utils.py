from openai import OpenAI
import os
import PyPDF2
import codecs
from io import StringIO, BytesIO
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredFileLoader,
    Docx2txtLoader,
)
from langchain.document_loaders.image import UnstructuredImageLoader

# from langchain_community.document_loaders import UnstructuredImageLoader
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.document import Document
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
from chromadb import Settings

load_dotenv()

# Initiate openai client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def select_llm(model_name):
    # Map model names to OpenAI model IDs
    if model_name == "gpt-3.5-turbo":
        llm = ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo",
            temperature=0,
        )
    else:
        llm = ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="gpt-4",
            temperature=0,
        )
    return llm


# Vectordb embedding model
embedding_model = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


# Split into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    separators=["\n", "\n\n"], keep_separator=False, chunk_size=1000, chunk_overlap=100
)


# Loader for different data format
def file_loader(file_extension, file_name):
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_name)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(file_name)
    elif file_extension == ".txt":
        loader = TextLoader(file_name)
    elif file_extension == ".png" or file_extension == ".jpg":
        loader = UnstructuredImageLoader(file_name, mode="single")
    else:
        print("Document format is not supported.")
    return loader


def build_retriver(uploaded_files):
    # Re-build retriever
    chromadb = Chroma(
        collection_name="uploaded_docs",
        embedding_function=embeddings,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=False),
    )

    bytes_data = uploaded_files.file.read()

    try:
        file_path = os.path.join("./uploaded_docs/", uploaded_files.name)
    except:
        file_path = os.path.join("./uploaded_docs/", uploaded_files.filename)
    with open(file_path, "wb") as f:
        f.write(bytes_data)

    _, extension = os.path.splitext(file_path)
    loader_file = file_loader(extension, file_path)
    documents = loader_file.load()
    loaded_text = ""
    for page in documents:
        loaded_text += page.page_content + "\n"

    splits = splitter.split_documents(documents)

    if chromadb:
        chromadb.delete_collection()

    chromadb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="uploaded_docs",
        client_settings=Settings(anonymized_telemetry=False, is_persistent=False),
    )
    retriever = chromadb.as_retriever()
    return retriever


# Write prompt
template = """
You are an AI asistent to teach and answer questions for primary students. Use the following context to answer the question. If you don't know the answer, just say you don't know.
Don't try to make up an answer. Try to answer in simple and clear explanation to kids below 10 years old.
Context: {context}
Question: {question}
Answer:
"""
prompt_template = PromptTemplate(
    template=template, input_variables=["context", "question"]
)


# Generate a response over uploaded data
def generate_response(prompt, data_retriever, llm_model):
    # Set up a RAG chain
    rag_chain = (
        {"context": data_retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm_model
    )
    with get_openai_callback() as openai_call_back:
        response_object = rag_chain.invoke(prompt)
    response = response_object.content
    return response, openai_call_back


# Generate a response over context in text box
def chat_completion(prompt, model_name):
    response_object = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages={"role": "user", "content": prompt},
    )
    response = response_object.choices[0].message.content
    total_tokens = response_object.usage.total_tokens
    prompt_tokens = response_object.usage.prompt_tokens
    completion_tokens = response_object.usage.completion_tokens
    return response, total_tokens, prompt_tokens, completion_tokens


def calculate_cost(model_name, total_tokens, prompt_tokens, completion_tokens):
    # from https://openai.com/pricing#language-models
    if model_name == "gpt-3.5-turbo":
        cost = total_tokens * 0.002 / 1000
    else:
        cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000
    return cost
