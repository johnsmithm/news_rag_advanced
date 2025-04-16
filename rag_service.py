import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI
from qdrant_client.http import models

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

client = OpenAI()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

client_qdrant = QdrantClient(path="news_rag_db")

vector_store = QdrantVectorStore(
    client=client_qdrant,
    collection_name="news_rag",
    embedding=embeddings,
)

collection_info = vector_store.client.get_collection('news_rag')
total_points = collection_info.points_count
print('total_points', total_points)


def extract_queries_and_date_filters(messages):
    """
    Extract search queries and date filters from chat history using a language model.
    
    Args:
        messages: List of dictionaries with keys 'role' and 'content'
                 representing the chat history.
    
    Returns:
        dict: A dictionary containing:
            - 'queries': List of search query strings
            - 'date_filter': Dictionary with 'gte' and/or 'lte' date strings if detected
    """
    import json
    
    # Create a new system prompt for the extraction
    system_prompt = """Extract search queries and date filters from the conversation.
    Format the response as a JSON object with the following structure:
    {
      "queries": ["search phrase 1", "search phrase 2"],
      "date_filter": {
        "gte": "YYYY-MM-DD", // Optional greater than or equal to date
        "lte": "YYYY-MM-DD"  // Optional less than or equal to date
      }
    }
    
    For date filters:
    - Extract dates mentioned with phrases like "after", "since", "from", "newer than" as "gte"
    - Extract dates mentioned with phrases like "before", "until", "older than" as "lte"
    - Convert all dates to YYYY-MM-DD format
    - Only include the date_filter object if dates are explicitly mentioned
    - If a date filter is not specified, omit that field completely
    
    For queries:
    - Extract main search terms and topics the user is looking for
    - Ignore common words like "find", "search", "show me", etc.
    - split and phase in 2 different queries

    Example:
    Input: news about AI
    Output queries: ['news about AI']
    """

    system_prompt += f"Date now: {datetime.now()}"
    
    # Copy the original messages and add our system prompt
    extraction_messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add the chat history
    for msg in messages:
        extraction_messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Use the chat model to extract information
    completion = client.chat.completions.create(
        model="gpt-4o",  # Choose appropriate model
        messages=extraction_messages,
        response_format={"type": "json_object"}
    )
    
    try:
        # Parse the JSON response
        extracted_data = json.loads(completion.choices[0].message.content)
        
        # Ensure we have the expected structure
        result = {
            "queries": extracted_data.get("queries", []),
            "date_filter": {}
        }
        
        # Add date filters if they exist
        if "date_filter" in extracted_data:
            date_filter = extracted_data["date_filter"]
            if "gte" in date_filter:
                result["date_filter"]["gte"] = date_filter["gte"]
            if "lte" in date_filter:
                result["date_filter"]["lte"] = date_filter["lte"]
        
        return result
        
    except json.JSONDecodeError:
        # Fallback in case of parsing error
        return {"queries": [], "date_filter": {},'error':"JSONDecodeError"}

def generate_rag_response(messages, retrieved_news):
    """
    Generate a response to the last message using RAG with retrieved news articles.
    
    Args:
        messages: List of dictionaries with keys 'role' and 'content' representing the chat history
        retrieved_news: List of dictionaries containing news articles with 'title' and 'url' keys
    
    Returns:
        str: Markdown-formatted response with citations to sources
    """
    import json
    
    # Extract the last user message
    last_message = next((msg["content"] for msg in reversed(messages) 
                         if msg["role"] == "user"), "")
    
    # Format the retrieved news as context
    context = ""
    for i, news in enumerate(retrieved_news):
        context += f"Source {i+1}: {news['title']}\nURL: {news['url']}\n\n"
    
    # Build system prompt
    system_prompt = """You are a helpful assistant that answers questions based on retrieved news articles.
    Always use the provided news sources to inform your answers.
    
    Guidelines:
    1. Answer ONLY based on the provided sources - don't use outside knowledge
    2. If the sources don't contain relevant information, acknowledge this limitation
    3. Cite sources using [Source X] notation inline when referring to specific information
    4. Format your response in Markdown
    5. Include a "Sources" section at the end with numbered references to the original URLs
    6. Be concise but comprehensive
    7. If sources contradict each other, note the discrepancies
    """
    
    # Build user prompt with context and question
    user_prompt = f"""
    ### Retrieved News Sources:
    
    {context}
    
    ### User Question:
    {last_message}
    
    Based on these sources, please provide a detailed answer in Markdown format.
    """
    
    # Create messages for the completion
    rag_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Get completion from the model
    completion = client.chat.completions.create(
        model="gpt-4o",  # Choose appropriate model
        messages=rag_messages
    )
    
    # Return the generated response
    return completion.choices[0].message.content


# Function to retrieve news from Qdrant
def retrieve_news(query, date_filters, max_results=5):
    """
    Retrieve news articles from Qdrant vector store based on query and date filters
    
    Args:
        query: String containing the search query
        date_filters: Dictionary with optional 'gte' and 'lte' date strings
        max_results: Maximum number of results to return
        
    Returns:
        List of dictionaries with title and URL for relevant news articles
    """
    # Initialize filter conditions
    must_conditions = []
    
    # Add date filters if they exist
    if date_filters:
        date_range = models.DatetimeRange()
        
        if "gte" in date_filters:
            try:
                gte_date = datetime.strptime(date_filters["gte"], "%Y-%m-%d")
                date_range.gte = gte_date
            except ValueError:
                print(f"Invalid gte date format: {date_filters['gte']}")
        
        if "lte" in date_filters:
            try:
                lte_date = datetime.strptime(date_filters["lte"], "%Y-%m-%d")
                date_range.lte = lte_date
            except ValueError:
                print(f"Invalid lte date format: {date_filters['lte']}")
                
        # Only add date condition if we have at least one valid date
        if date_range.gte is not None or date_range.lte is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.date",
                    range=date_range
                )
            )
    
    # Create the filter if we have conditions
    filter_condition = None
    if must_conditions:
        filter_condition = models.Filter(must=must_conditions)
    
    # Perform the vector search
    results = vector_store.similarity_search(
        query=query,
        k=max_results,
        filter=filter_condition
    )
    
    # Format results
    news_articles = []
    for doc in results:
        news_articles.append({
            "title": doc.page_content,
            "url": doc.metadata.get("url", "#"),
            "date": doc.metadata.get("date", "Unknown date")
        })
    
    return news_articles


def respond(message, history):
    """Main chat response function for Gradio"""
    # Convert Gradio history format to message format
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    
    # Add the current message
    messages.append({"role": "user", "content": message})
    
    # Extract queries and date filters
    extracted = extract_queries_and_date_filters(messages)
    
    # Default to the full message as query if extraction fails
    query = " ".join(extracted["queries"]) if extracted["queries"] else message
    
    print(extracted["date_filter"])
    # Get relevant news articles
    news_articles = retrieve_news(query, extracted["date_filter"])
    print(news_articles)
    # Generate response using RAG
    response = generate_rag_response(messages, news_articles)
    
    return response


if __name__ == "__main__":
    respond("What is the news about AI?", [])