import requests
import urllib.parse
import os
from enum import Enum

class SearchEngine(Enum):
    """Enumeration of supported search engines.
    
    Attributes:
        GOOGLE (int): Google search engine
        BING (int): Bing search engine
        Yandex (int): Yandex search engine
    """
    GOOGLE = 1 
    BING = 2 
    Yandex = 3 

def _make_api_request(url: str, **kwargs):
    """Make an API request to the Bright Data API.
    
    Args:
        url (str): The API endpoint URL.
        **kwargs: Additional arguments to pass to the requests.post() function.
        
    Returns:
        dict | None: JSON response from the API if successful, None otherwise.
        
    Raises:
        requests.exceptions.RequestException: If the API request fails.
        Exception: For other unexpected errors.
    """
    api_key = os.getenv("BRIGHT_DATA_API_KEY")
    
    headers = { 
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:    
        response = requests.post(url, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    
def serp_search(query: str, engine: SearchEngine = SearchEngine.GOOGLE):
    """Perform a search query using specified search engine through Bright Data's SERP API.
    
    Args:
        query (str): The search query to perform.
        engine (SearchEngine, optional): The search engine to use. Defaults to SearchEngine.GOOGLE.
            Can be one of: GOOGLE, BING, or Yandex.
            
    Returns:
        dict | None: Extracted search results containing:
            - knowledge: Knowledge graph data if available
            - organic: Organic search results
            Returns None if the request fails.
            
    Raises:
        ValueError: If an unsupported search engine is specified or if environment variables are not set.
    """
    if engine not in SearchEngine:
        raise ValueError("Unsupported search engine.")
    
    base_url = ""
    if engine == SearchEngine.GOOGLE:
        base_url = "https://www.google.com/search?q="
    if engine == SearchEngine.BING:
        base_url = "https://www.bing.com/search?q="
    elif engine == SearchEngine.Yandex:
        base_url = "https://yandex.com/search?text="
   
    data = {
        "zone": "google_bing_serp_api_ai_agent",
        "url": f"{base_url}{urllib.parse.quote_plus(query)}&brd_json=1",
        "format": "raw" 
    }
    
    brigt_data_api_url = os.getenv("BRIGHT_DATA_API_URL")
    if not brigt_data_api_url:
        raise ValueError("BRIGHT_DATA_API_URL environment variable not set.")
        
    full_response = _make_api_request(brigt_data_api_url, json=data) 
    if not full_response:
        return None
    
    extracted_data  = { 
        "knowledge": full_response.get("knowledge", {}),
        "organic": full_response.get("organic", {}),
    }

    return extracted_data