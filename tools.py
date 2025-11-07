# tools.py
import requests
import wikipedia
import trafilatura
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")


def google_search(query, api_key=GOOGLE_API_KEY, cse_id=SEARCH_ENGINE_ID, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {'q': query, 'key': api_key, 'cx': cse_id}
    r = requests.get(url, params=params)
    results = r.json().get('items', [])[:num_results]
    return [{"title": i.get("title"), "content": i.get("snippet"), "link": i.get("link")} for i in results]

def wiki_search(query, num_results=5):
    results = []
    for title in wikipedia.search(query, results=num_results):
        try:
            page = wikipedia.page(title)
            results.append({"title": title, "content": wikipedia.summary(title, sentences=2), "link": page.url})
        except Exception:
            continue
    return results

def fetch_page_content(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        page_text = trafilatura.extract(response.text, include_comments=False, include_tables=False)
        return page_text[:2000] if page_text else "[No main content extracted]"
    except Exception as e:
        return f"[Error fetching page: {e}]"
