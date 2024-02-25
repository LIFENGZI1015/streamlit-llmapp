from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llmchat_utils import *

# FASTAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
