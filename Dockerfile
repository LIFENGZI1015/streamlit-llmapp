FROM python:3.8-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    tesseract-ocr libgl1 \
    # ffmpeg libsm6 libxext6 libgl1 \
    #libleptonica-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# RUN git clone https://github.com/LIFENGZI1015/streamlit-llmapp.git .

COPY . .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN pip install "unstructured[all-docs]" 

RUN 

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app_llmchat.py", "--server.port=8501", "--server.address=0.0.0.0"]