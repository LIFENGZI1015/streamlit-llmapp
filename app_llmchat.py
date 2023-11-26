from openai import OpenAI
import streamlit as st
from streamlit_chat import message
import os
import PyPDF2
from io import StringIO
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, UnstructuredFileLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.document import Document
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
load_dotenv()

# Setting page title and header
st.set_page_config(page_title="GenieBot", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center; font-size: 40px;'>GenieBot 🏂</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-size: 28px;'>Unleashing the Learning Power of Kids 😬</h2>", unsafe_allow_html=True)

# Initiate openai client
client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY'),
)

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'data_name' not in st.session_state:
    st.session_state['data_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0
if 'temperature' not in st.session_state:
    st.session_state['temperature'] = []

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", ("gpt-3.5-turbo", "gpt-4"))
data_name = st.sidebar.radio(
    "Select data", ("Provide in Text Box", "Uploaded Files"), index=None)
uploaded_files = st.sidebar.file_uploader("Choose your files", type=["pdf", "png", "jpg"])
llm_temperature = st.sidebar.slider("Creativity of model", min_value=0.0, max_value=1.0, value=None, step=0.1)
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(
    f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Map model names to OpenAI model IDs
if model_name == "gpt-3.5-turbo":
    llm = ChatOpenAI(api_key=os.environ.get('OPENAI_API_KEY'),
                      model_name="gpt-3.5-turbo",
                      temperature=llm_temperature)
else:
    llm = ChatOpenAI(api_key=os.environ.get('OPENAI_API_KEY'),
                      model_name="gpt-4",
                      temperature=llm_temperature)

# Vectordb embedding model
embedding_model = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['model_name'] = []
    st.session_state['data_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    st.session_state['temperature'] = []
    counter_placeholder.write(
        f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
    try:
        Chroma.delect_collection("uploaded_docs")
    except:
        st.write("No Chroma collection is deleted.")

# Split into smaller chunks
splitter = RecursiveCharacterTextSplitter(separators=["\n"],
                                          keep_separator=False,
                                          chunk_size=1000,
                                          chunk_overlap=50)

# Build a retriever over uploaded files
def load_files(file_path):
    loader = PyPDFLoader(file_obj)
    pdf_docs = loader.load()
    return pdf_docs

def load_images(file_path):
    loader = UnstructuredImageLoader(file_path)
    img_data = loader.load()
    return img_data

def load_unstructured_files(file_path):
    loader = UnstructuredFileLoader(file_path, mode="elements")
    unstructured_data = loader.load()
    return unstructured_data

# Re-build retriever
if data_name == "Uploaded Files":
    if uploaded_files is not None:
        pdf_reader = PyPDF2.PdfReader(uploaded_files)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() + "\n"
        st.write(pdf_text)
        pdf_docs = [Document(page_content=x)
                    for x in splitter.split_text(pdf_text)]
        splits = splitter.split_documents(pdf_docs)
        chromadb = Chroma.from_documents(
            documents=splits, embedding=embeddings, collection_name="uploaded_docs")
        retriever = chromadb.as_retriever()
        st.write(
            "Chatbot is reading your files and you can start talking to your data...")
    else:
        st.write("Please upload your files.")
elif data_name == "Provide in Text Box":
    st.write("Please provide your text in the chat box.")
else:
    st.write("Please select data to chat.")

# Write prompt
template = """
You are an AI asistent to teach and answer questions for primary students. Use the following context to answer the question. If you don't know the answer, just say you don't know.
Don't try to make up an answer. Try to answer in simple and clear explanation to kids below 10 years old.
Context: {context}
Question: {question}
Answer:
"""
prompt_template = PromptTemplate(template=template,
                        input_variables=["context", "question"])

# Generate a response over uploaded data
def generate_response(prompt, retriever, llm):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    # Set up a RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} | prompt_template | llm
        )
    with get_openai_callback() as call_back:
        response_object = rag_chain.invoke(prompt)
    response = response_object.content
    st.session_state['messages'].append(
        {"role": "assistant", "content": response})
    return response, call_back

# Generate a response over context in text box
def chat_completion(prompt, temperature):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    # response_object = client.completions.create(
    response_object = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=st.session_state['messages']
    )
    # st.write(response_object)
    response = response_object.choices[0].message.content
    # response = response_object.choices[0].text
    st.session_state["messages"].append(
        {"role": "assistant", "content": response})
    # response_object_json = response_object.model_dump_json(indent=2)
    # total_tokens = dict(response_object).get('usage').total_tokens
    # prompt_tokens = dict(response_object).get('usage').prompt_tokens
    # completion_tokens = dict(response_object).get('usage').completion_tokens
    total_tokens = response_object.usage.total_tokens
    prompt_tokens = response_object.usage.prompt_tokens
    completion_tokens = response_object.usage.completion_tokens
    return response, total_tokens, prompt_tokens, completion_tokens
    # return response_object

# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        if data_name == "Uploaded Files":
            output, call_back = generate_response(user_input, retriever, llm)
            total_tokens = call_back.total_tokens
            prompt_tokens = call_back.prompt_tokens
            completion_tokens = call_back.completion_tokens
        elif data_name == "Provide in Text Box":
            output, total_tokens, prompt_tokens, completion_tokens = chat_completion(
                user_input, llm_temperature)
            # response_object = chat_completion(user_input)
        else:
            st.write("Please select data.")
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)
        st.session_state['data_name'].append(data_name)                                         
        st.session_state['total_tokens'].append(total_tokens)
        st.session_state['temperature'].append(llm_temperature)

        # from https://openai.com/pricing#language-models
        if model_name == "gpt-3.5-turbo":
            cost = total_tokens * 0.002 / 1000
        else:
            cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i],
                    is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            st.write(
                f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}; Data used: {st.session_state['data_name'][i]}; Temperature: {st.session_state['temperature'][i]}")
            counter_placeholder.write(
                f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
