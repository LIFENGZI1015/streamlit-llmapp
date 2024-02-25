# Streamlit-LLMApp
![Alt text](./images/MathGenieBot_UI.png)

GenieBot - Unleashing the Learning Power of Kids 😬. 

A LLM Chat application built using streamlit with features to select LLM models, calculating total cost, chat in box, chat over uploaded data.

### Components
- A Chatbot using LangChain, Streamlit and LLMs like OpenAI GPTs
- It can be run locally
- It can be run using Docker
- It can be deployed to Huggingface Spaces via Docker Space SDK
- It can be developed as API using FastAPI


## Pre-requisites
- OpenAI API Key
- Docker (if using this option)
- Huggingface account (if deploy to Huggingface Spaces)


## Running Locally
1. Clone the repository
```bash
git clone https://github.com/LIFENGZI1015/streamlit-llmapp.git
```
2. Install dependencis
```bash
pip install -r requirements.txt
pip install "unstructured[all-docs]"
pip install "fastapi[all]"
```
3. Install Tesseract
- https://tesseract-ocr.github.io/tessdoc/Installation.html
- For example macOS:
```bash
brew install tesseract
```
4. Run the application with this command
```bash
streamlit run app_llmchat.py
```
5. Ctrl+C to stop the app


## Running app locally using Docker
1. Run the docker container using docker-compose (Recommended)
```bash
docker-compose --env-file .env up --build
```
2. Stop docker-compose
```bash
docker compose stop
```
3. Ctrl+C to stop everything


## Deploy to Huggingface Spaces via Docker Space SDK
![Alt text](./images/huggingface_streamlit_llm_app.png)

1. Create Spaces in your Huggingface account
- https://huggingface.co/docs/hub/spaces-overview

2. Add OPENAI_API_KEY as secret in the Spaces Settings
- https://huggingface.co/docs/hub/spaces-overview#managing-secrets

3. Upload or create following files to Huggingface Spaces
![Alt text](./images/huggingface_space_files.png)

Please note that Dockerfile_hf in this repo is used to deploy streamlit llm app on Huggingface. Change name when you upload it to Huggingface.


## Deploy LLM App as API
1. Run the FastAPI application:
```
uvicorn app.fastapi_app:app --reload --env-file .env
```
Once the server is running, you can access the Swagger documentation by visiting http://127.0.0.1:8000/docs in your web browser. The Swagger UI provides an interactive interface to test your API.
2. Option 1 - Call API in python scripts:
```
python run_fastapi.py
```
3. Option 2 - Call API in command line
```
curl -x POST -F "files=@./docs/PSLE-Challenging-Math-Questions_pg2.pdf" -F "question=You are a math teacher. Please give a step by step explanation to the provided math questions to a 8 years old kid." http://127.0.0.1:8000/uploasfiles/
```
4. You can stop the server using Ctrl+C in the terminal where the app is running.


### References
1. Build LLM App with Docker and Streamlit
- https://www.packtpub.com/article-hub/building-a-containerized-llm-chatbot-application
- https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker
- https://www.docker.com/blog/build-and-deploy-a-langchain-powered-chat-app-with-docker-and-streamlit/
2. Deploy Docker App in Huggingface Spaces
- https://huggingface.co/docs/hub/spaces-sdks-docker-first-demo
- https://www.docker.com/blog/build-machine-learning-apps-with-hugging-faces-docker-spaces/
- https://huggingface.co/blog/HemanthSai7/deploy-applications-on-huggingface-spaces
- https://huggingface.co/docs/hub/spaces-sdks-docker#secret-management
- https://www.docker.com/blog/build-machine-learning-apps-with-hugging-faces-docker-spaces/