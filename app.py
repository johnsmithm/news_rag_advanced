import os
from fastapi import FastAPI, HTTPException, status, Request, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
import openai
from typing import Optional, List, Dict
import traceback
import logging
from datetime import datetime
import json
import copy
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from rag_service import retrieve_news, generate_rag_response, extract_queries_and_date_filters

load_dotenv()

# Initialize FastAPI app once
app = FastAPI()

# Set up CORS
origins = ["*"]  # Allows all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RetrievalRequest(BaseModel):
    query: str

class CompletionRequest(BaseModel):
    messages: List[Dict[str, str]]

class NewsArticle(BaseModel):
    title: str
    url: str
    date: str

class RetrievalResponse(BaseModel):
    articles: List[NewsArticle]

class CompletionResponse(BaseModel):
    response: str

def get_api_key_header(request: Request):
    return request.headers.get("x-api-key", "")


def get_api_key(
    request: Request,
    api_key: str = Depends(get_api_key_header),
):
    if api_key != os.getenv("APP_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key"
        )
    return api_key

path_to_logs = "logs"


def write_log(question: str, answer: str):
    current_date = datetime.today().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {"time": current_time, "question": question, "answer": answer}

    with open(
        f"{path_to_logs}/{current_date}_chat_log.txt", "a", encoding="UTF-8"
    ) as log_file:
        json.dump(record, log_file)
        print(",", file=log_file)

@app.post("/api/retrieval", response_model=RetrievalResponse)
async def get_retrieval(
    request: RetrievalRequest,
    api_key: str = Depends(get_api_key)
):
    try:
        # Use the retrieve_news function from rag_service

        extracted = extract_queries_and_date_filters([{"role": "user", "content": request.query}])
        articles = retrieve_news(request.query, extracted["date_filter"])
        return RetrievalResponse(articles=articles)
    except Exception as e:
        stack_trace = traceback.format_exc()
        write_log(request.query, f"ERROR: {str(e)}\nStack Trace:\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{str(e)}\nStack Trace:\n{stack_trace}"
        )

@app.post("/api/completion", response_model=CompletionResponse)
async def get_completion(
    request: CompletionRequest,
    api_key: str = Depends(get_api_key)
):
    try:
        # Extract the last user message
        last_message = next((msg["content"] for msg in reversed(request.messages) 
                           if msg["role"] == "user"), "")
        
        # First get relevant articles
        # Extract queries and date filters from the full conversation
        extracted = extract_queries_and_date_filters(request.messages)
        articles = retrieve_news(last_message, extracted["date_filter"])
        
        # Generate response using RAG
        response = generate_rag_response(request.messages, articles)
        
        return CompletionResponse(response=response)
    except Exception as e:
        stack_trace = traceback.format_exc()
        write_log(str(request.messages), f"ERROR: {str(e)}\nStack Trace:\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{str(e)}\nStack Trace:\n{stack_trace}"
        )

# Start logger service
logging.basicConfig(
    filename=f"{path_to_logs}/{datetime.today().strftime('%Y-%m-%d')}_system_log.txt",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

logger.debug(f"API keys loaded")

API_KEY = os.getenv("APP_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

logger.info(f"Starting Bot API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)