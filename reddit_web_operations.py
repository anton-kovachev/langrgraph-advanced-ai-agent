from enum import Enum
import os
import time
import requests

REDDIT_DATASET_ID = "gd_lvz8ah06191smkebj4"
REDDIT_DATA_COLLECTION_BASE_URL = f"https://api.brightdata.com/datasets/v3/trigger"
REDDIT_SNAPSHOT_PROGRESS_BASE_URL = "https://api.brightdata.com/datasets/v3/progress"
REDDIT_SNAPSHOT_URL = "https://api.brightdata.com/datasets/v3/snapshot"

MAX_ATTEMPTS = 50


class RedditScrapeOperationType(Enum):
    """Enumeration of Reddit scraping operation types.
    
    Attributes:
        POSTS (int): Operation to scrape Reddit posts
        POST_COMMENTS (int): Operation to scrape comments from Reddit posts
    """
    POSTS = 1
    POST_COMMENTS = 2


def scrap_reddit(
    query: str | list[str],
    scrape_operation_type: RedditScrapeOperationType = RedditScrapeOperationType.POSTS
):
    """Scrape Reddit data using Bright Data's Reddit dataset API.
    
    Args:
        query (str | list[str]): For POSTS operation type: a search query string.
            For POST_COMMENTS operation type: a list of Reddit post URLs.
        scrape_operation_type (RedditScrapeOperationType, optional): The type of scraping operation.
            Defaults to RedditScrapeOperationType.POSTS.
            
    Returns:
        dict | None: For POSTS operation:
            - parsed_posts: List of dictionaries containing post titles and URLs
            - total_found: Number of posts found
            For POST_COMMENTS operation:
            - parsed_comments: List of dictionaries containing comment details
            - total_found: Number of comments found
            Returns None if the operation fails.
    """
    response_data: dict | None = None
    if scrape_operation_type == RedditScrapeOperationType.POSTS and isinstance(query, str):
        response_data = _trigger_reddit_data_collection(query)
    elif scrape_operation_type == RedditScrapeOperationType.POST_COMMENTS and isinstance(query, list):
        response_data = _get_post_comments_by_urls(query)

    if not response_data:
        return None

    snapshot_id = response_data.get("snapshot_id")
    if not snapshot_id:
        print("No snapshot_id found in the response.")
        return None

    for attempt in range(MAX_ATTEMPTS):
        print(f"Attempt {attempt + 1} to check snapshot {snapshot_id} status...")
        snapshot_status_response = _get_snapshot_status_by_snapshot_id(snapshot_id)
        if not snapshot_status_response:
            print("Failed to get snapshot status.")
            return None
        status = snapshot_status_response.get("status")

        if status == "ready":
            snapshot_response_data = _get_snapshot_data_by_id(snapshot_id)
            if not snapshot_response_data:
                print("Failed to get snapshot data.")
                return None
            if scrape_operation_type == RedditScrapeOperationType.POSTS:
                return _parse_reddit_data_collection_response(snapshot_response_data)
            elif scrape_operation_type == RedditScrapeOperationType.POST_COMMENTS:
                return _parse_reddit_post_details_response(snapshot_response_data)
        elif status == "failed":
            print("Data collection failed.")
            return None
        time.sleep(10)

    print("Max attempts reached without completing data collection.")
    return None


def _trigger_reddit_data_collection(
    query: str,
    date="All time",
    sort_by="Hot",
    num_of_posts=100,
    dataset_id: str = REDDIT_DATASET_ID
):
    """Trigger a Reddit data collection job through Bright Data's API.
    
    Args:
        query (str): The search query to find Reddit posts.
        date (str, optional): Time range for posts. Defaults to "All time".
        sort_by (str, optional): Sort order for posts. Defaults to "Hot".
        num_of_posts (int, optional): Maximum number of posts to retrieve. Defaults to 100.
        dataset_id (str, optional): Bright Data dataset ID. Defaults to REDDIT_DATASET_ID.
        
    Returns:
        dict | None: API response containing snapshot_id if successful, None if failed.
    """
    return __make_reddit_post_api_request(
        REDDIT_DATA_COLLECTION_BASE_URL,
        params={
            "dataset_id": dataset_id,
            "include_errors": "true",
            "type": "discover_new",
            "discover_by": "keyword"
        },
        json=[{
            "keyword": query,
            "date": date,
            "sort_by": sort_by,
            "num_of_posts": num_of_posts
        }]
    )

def _get_post_comments_by_urls(post_urls: list[str], days_back = 365, loead_all_replies: bool = False, comment_limit: int = 20):
    """Retrieve comments from specified Reddit posts through Bright Data's API.
    
    Args:
        post_urls (list[str]): List of Reddit post URLs to get comments from.
        days_back (int, optional): How many days back to retrieve comments. Defaults to 365.
        loead_all_replies (bool, optional): Whether to load all comment replies. Defaults to False.
        comment_limit (int, optional): Maximum number of comments to retrieve. Defaults to 20.
        
    Returns:
        dict | None: API response containing snapshot_id if successful, None if failed.
    """
    params = {  "dataset_id": "gd_lvzdpsdlw09j6t702", "include_errors": "true" }
    data = [{ "url": post_url, "days_back": 365, "load_all_replies": loead_all_replies, "comment_limit": comment_limit } for post_url in post_urls ]
    
    return __make_reddit_post_api_request(REDDIT_DATA_COLLECTION_BASE_URL, params = params, json=data)
    
def __make_reddit_post_api_request(url: str, **kwargs):
    """Make a POST request to Bright Data's Reddit API endpoints.
    
    Args:
        url (str): The API endpoint URL.
        **kwargs: Additional arguments to pass to the requests.post() function.
        
    Returns:
        dict | None: JSON response from the API if successful, None if failed.
        
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

def _get_snapshot_data_by_id(snapshot_id: str) -> list[dict] | None:
    """Retrieve snapshot data for a completed Reddit data collection job.
    
    Args:
        snapshot_id (str): The ID of the snapshot to retrieve.
        
    Returns:
        list[dict] | None: List of dictionaries containing Reddit data if successful,
            None if failed.
    """
    snapshot_url = f"{REDDIT_SNAPSHOT_URL}/{snapshot_id}"
    return _make_reddit_get_api_request(snapshot_url)


def _get_snapshot_status_by_snapshot_id(snapshot_id: str):
    """Check the status of a Reddit data collection job.
    
    Args:
        snapshot_id (str): The ID of the snapshot to check.
        
    Returns:
        dict | None: Status response containing 'status' field if successful,
            which can be 'ready' or 'failed'. Returns None if request fails.
    """
    print(f"Fetching snapshot status for snaphot id: {snapshot_id}")
    snapshot_status_url = f"{REDDIT_SNAPSHOT_PROGRESS_BASE_URL}/{snapshot_id}"
    return _make_reddit_get_api_request(snapshot_status_url)


def _make_reddit_get_api_request(url: str, **kwargs):
    """Make a GET request to Bright Data's Reddit API endpoints.
    
    Args:
        url (str): The API endpoint URL.
        **kwargs: Additional arguments to pass to the requests.get() function.
        
    Returns:
        dict | None: JSON response from the API if successful, None if failed.
        
    Raises:
        requests.exceptions.RequestException: If the API request fails.
        Exception: For other unexpected errors.
    """
    api_key = os.getenv("BRIGHT_DATA_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    params = {"format": "json"}

    try:
        response = requests.get(url, params=params, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def _parse_reddit_data_collection_response(response_data: list[dict]):
    """Parse the Reddit posts data collection response.
    
    Args:
        response_data (list[dict]): Raw response data from Bright Data API.
        
    Returns:
        dict: Parsed response containing:
            - parsed_posts: List of dictionaries with post title and URL.
            - total_found: Total number of posts found.
    """
    return {"parsed_posts": [ {"title": post.get("title"), "url": post.get("url")}  for post in response_data], "total_found": len(response_data) }

def _parse_reddit_post_details_response(response_data: list[dict]):
    """Parse the Reddit post comments data response.
    
    Args:
        response_data (list[dict]): Raw response data from Bright Data API.
        
    Returns:
        dict: Parsed response containing:
            - parsed_comments: List of dictionaries with comment details including:
                - comment_id: Unique identifier for the comment
                - content: Comment text content
                - date: Comment posting date
                - parent_comment_id: ID of parent comment if it's a reply
                - post_title: Title of the post
                - url: URL of the post
            - total_found: Total number of comments found.
    """
    return {"parsed_comments": [ {
        "comment_id": post_detail.get("comment_id"),
        "content": post_detail.get("content"),
        "date": post_detail.get("date"),
        "parent_comment_id": post_detail.get("parent_comment_id"),
        "post_title": post_detail.get("post_title"),
        "url": post_detail.get("url")
    } for post_detail in response_data], "total_found": len(response_data) }