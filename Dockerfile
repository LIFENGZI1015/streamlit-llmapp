FROM python:3.8
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN sudo apt install tesseract-ocr

RUN sudo apt install libtesseract-dev

RUN git clone https://github.com/LIFENGZI1015/streamlit-llmapp.git .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app_llmchat.py", "--server.port=8501", "--server.address=0.0.0.0"]