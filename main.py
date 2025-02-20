from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from bs4 import BeautifulSoup
from config import Config  # Use absolute import
import google.generativeai as genai
from starlette.websockets import WebSocketState

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to the specific origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchQuery(BaseModel):
    query: str

@app.websocket("/ws/search")
async def websocket_search(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        query = SearchQuery(**data)
        page = data.get("page", 1)
        
        status = "Extracting keywords..."
        await websocket.send_json({"status": status})
        keywords = extract_keywords(query.query)
        
        status = "Generating search query..."
        await websocket.send_json({"status": status})
        search_query = generate_search_query(keywords)

        status = "Performing search..."
        await websocket.send_json({"status": status})
        search_results = await perform_search(search_query, page)

        status = "Evaluating and filtering results..."
        await websocket.send_json({"status": status})
        filtered_results = evaluate_and_filter_results(search_results)

        # Send Google search results immediately
        await websocket.send_json({
            "status": "Google results ready",
            "googleResults": [
                {
                    "link": result["link"],
                    "title": result["title"],
                    "snippet": result["snippet"]
                } for result in filtered_results
            ]
        })

        status = "Summarizing results..."
        await websocket.send_json({"status": status})
        summary, sources = await summarize_results(filtered_results)

        status = "Completed"
        await websocket.send_json({
            "status": status,
            "result": summary,
            "sources": sources
        })
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e.response.status_code} - {e.response.text}")
        await websocket.send_json({"status": "Error", "detail": e.response.text})
    except Exception as e:
        print(f"Unexpected error: {e}")
        await websocket.send_json({"status": "Error", "detail": str(e)})
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()

@app.post("/search")
async def search(query: SearchQuery, page: int = 1):
    try:
        keywords = extract_keywords(query.query)
        search_query = generate_search_query(keywords)
        search_results = await perform_search(search_query, page)
        filtered_results = evaluate_and_filter_results(search_results)
        summary, sources = await summarize_results(filtered_results)
        return {
            "status": "Completed",
            "result": summary,
            "sources": sources,
            "googleResults": [
                {
                    "link": result["link"],
                    "title": result["title"],
                    "snippet": result["snippet"]
                } for result in filtered_results
            ]
        }
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_keywords(query):
    # Simple keyword extraction by splitting the query
    return query.split()

def generate_search_query(keywords):
    # Generate a general search query for Google
    return ' '.join(keywords)

async def perform_search(search_query, page):
    # Perform search using Google Custom Search JSON API
    api_key = Config.GOOGLE_API_KEY
    cx = Config.CUSTOM_SEARCH_ENGINE_ID
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cx, "q": search_query, "start": (page - 1) * 10 + 1}

    async with httpx.AsyncClient() as client:
        response = await client.get(search_url, params=params)
        response.raise_for_status()
        search_results = response.json()
        return search_results.get("items", [])

def evaluate_and_filter_results(results):
    # Evaluate and filter results based on some criteria (e.g., relevance, source credibility)
    # Here we just take the top 10 results
    return results[:10]

async def fetch_content(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
        response = await client.get(url)
        if response.status_code == 403 and "cloudflare" in response.text.lower():
            raise Exception("Blocked by Cloudflare")
        response.raise_for_status()
        response.encoding = response.charset_encoding or 'utf-8'  # Ensure proper encoding
        return response.text

def parse_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = soup.find_all('p')
    return ' '.join([para.get_text() for para in paragraphs])

genai.configure(api_key=Config.GOOGLE_GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.0-flash")

async def fetch_wikipedia_content(keywords):
    search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={' '.join(keywords)}&utf8=&format=json"
    async with httpx.AsyncClient() as client:
        response = await client.get(search_url)
        response.raise_for_status()
        search_results = response.json()
        if search_results["query"]["search"]:
            page_id = search_results["query"]["search"][0]["pageid"]
            page_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&format=json&pageids={page_id}"
            page_response = await client.get(page_url)
            page_response.raise_for_status()
            page_content = page_response.json()
            return page_content["query"]["pages"][str(page_id)]["extract"]
    return ""

async def summarize_results(results):
    # Summarize the results using Google Gemini
    content_to_summarize = ""
    sources = [result["link"] for result in results[:3]]  # Use top 3 results for AI summarization
    
    for source in sources:
        html_content = await fetch_content(source)
        content_to_summarize += parse_content(html_content) + "\n\n"
    
    # Fetch Wikipedia content
    wikipedia_content = await fetch_wikipedia_content(extract_keywords(results[0]["title"]))
    content_to_summarize += wikipedia_content + "\n\n"
    
    response = model.generate_content(content_to_summarize)
    summary = response.text[:300]  # Limit AI response size to 300 characters
    summary = render_latex(summary)  # Render LaTeX
    return summary, sources

def render_latex(text):
    # Function to render LaTeX in the text
    return text.replace("$", "\\(").replace("$$", "\\[").replace("$$", "\\]")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
