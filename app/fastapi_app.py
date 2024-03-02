from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from app.llmchat_utils import *


# FASTAPI
app = FastAPI()

model_name = "gpt-3.5-turbo"
llm = select_llm(model_name=model_name)


@app.post("/uploadfiles/")
async def create_upload_files(
    files: List[UploadFile] = File(None),
    text: str = Form(None),
    question: str = Form("What is the answer?"),
):
    content = ""

    for file in files:
        content += (
            f"Uploaded File Content ({file.filename}):\n{file.read()}\n"
        )
    if text is None:
        content += f"User Provided Text:\n{text}\n"

    if content is not None:
        pass
    else:
        print("Please provide data by uploading files or provide as text input.")

    if len(files)!=0 and text is None:
        retriever = build_retriver(files[0])
        output, call_back = generate_response(
            prompt=question, data_retriever=retriever, llm_model=llm
        )
        total_tokens = call_back.total_tokens
        prompt_tokens = call_back.prompt_tokens
        completion_tokens = call_back.completion_tokens
    elif len(files)==0 and text is not None:
        (
            output,
            total_tokens,
            prompt_tokens,
            completion_tokens,
        ) = chat_completion(text, model_name)  

    cost = calculate_cost(model_name, total_tokens, prompt_tokens, completion_tokens)

    return JSONResponse(
        content={
            "message": "Files and text processed successfully",
            "answers": {
                "input": content,
                "question": question,
                "answer": output,
                "total tokens": total_tokens,
                "cost": cost,
            },
        }
    )
