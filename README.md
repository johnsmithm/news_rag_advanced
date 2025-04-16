# News RAG API Service

This service provides a REST API for retrieving and analyzing news articles using RAG (Retrieval Augmented Generation) with OpenAI's GPT models.

## Setup

1. Clone the repository
2. Copy `.env.default` to `.env` and fill in your API keys:
```bash
cp .env.default .env
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file with the following variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `APP_API_KEY`: API key for authenticating requests to this service

## Running the Service

```bash
python app.py
```

The service will start on `http://localhost:8000`

## API Endpoints

### 1. Retrieve News Articles

**Endpoint:** `POST /api/retrieval`

Retrieves relevant news articles based on a query.

```bash
# Using curl
curl -X POST "http://localhost:8000/api/retrieval" \
     -H "x-api-key: YOUR_APP_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"query": "latest AI developments"}'

# Using Python
import requests
import json

response = requests.post(
    "http://localhost:8000/api/retrieval",
    headers={
        "x-api-key": "YOUR_APP_API_KEY",
        "Content-Type": "application/json"
    },
    json={"query": "latest AI developments"}
)
print(json.dumps(response.json(), indent=2))
```

### 2. Generate Completion

**Endpoint:** `POST /api/completion`

Generates an AI response based on conversation history and relevant news articles.

```bash
# Using curl
curl -X POST "http://localhost:8000/api/completion" \
     -H "x-api-key: YOUR_APP_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "What are the latest developments in AI?"}]}'

# Using Python
import requests
import json

response = requests.post(
    "http://localhost:8000/api/completion",
    headers={
        "x-api-key": "YOUR_APP_API_KEY",
        "Content-Type": "application/json"
    },
    json={
        "messages": [
            {"role": "user", "content": "What are the latest developments in AI?"}
        ]
    }
)
print(json.dumps(response.json(), indent=2))
```

## Response Formats

### Retrieval Response
```json
{
  "articles": [
    {
      "title": "Article Title",
      "url": "https://example.com/article",
      "date": "2024-03-21"
    }
  ]
}
```

### Completion Response
```json
{
  "response": "Generated response with citations [Source 1] and analysis..."
}
```

## Advanced Usage

### Date Filtering
You can include date-related queries in your questions:
```bash
curl -X POST "http://localhost:8000/api/retrieval" \
     -H "x-api-key: YOUR_APP_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"query": "AI news from last week"}'
```

### Conversation Context
The completion endpoint maintains conversation context:
```python
messages = [
    {"role": "user", "content": "What are the latest AI developments?"},
    {"role": "assistant", "content": "According to recent news..."},
    {"role": "user", "content": "Tell me more about those developments"}
]

response = requests.post(
    "http://localhost:8000/api/completion",
    headers={
        "x-api-key": "YOUR_APP_API_KEY",
        "Content-Type": "application/json"
    },
    json={"messages": messages}
)
```

## Error Handling

The API returns standard HTTP status codes:
- 200: Successful request
- 401: Invalid API key
- 500: Internal server error with stack trace

Error responses include a detail message explaining the error. 