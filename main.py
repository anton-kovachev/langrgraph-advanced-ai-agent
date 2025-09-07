"""Advanced AI Agent using LangGraph for multi-source information retrieval and analysis.

This module implements a graph-based AI agent that:
1. Searches multiple sources (Google, Bing, Yandex, Reddit)
2. Analyzes search results
3. Synthesizes information into a final answer
"""

from dotenv import load_dotenv
from typing import Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from serp_web_operations import serp_search, SearchEngine
from reddit_web_operations import RedditScrapeOperationType, scrap_reddit
from prompts import (
    get_google_analysis_messages,
    get_bing_analysis_messages,
    get_yandex_analysis_messages,
    get_reddit_url_analysis_messages,
    get_reddit_analysis_messages,
    get_synthesis_messages
)

load_dotenv()

# llm = init_chat_model("gpt-4o-2024-05-13")
llm = init_chat_model("claude-sonnet-4-20250514", temperature = 0)

class RedditURLAnalysis(BaseModel):
    """Model for storing selected Reddit URLs from search results.
    
    Attributes:
        selected_urls (List[str]): List of selected Reddit post URLs relevant to the user's query.
    """
    selected_urls: List[str] = Field(
        ...,
        description="List of selected Reddit post URLs relevant to the user's query."
    )


class State(TypedDict):
    """State class for managing the agent's workflow.
    
    Attributes:
        messages (Annotated[list, add_messages]): Message history
        user_input (str | None): User's original query
        google_results (str | None): Raw results from Google search
        bing_results (str | None): Raw results from Bing search
        yandex_results (str | None): Raw results from Yandex search
        reddit_results (str | None): Raw results from Reddit search
        selected_reddit_urls (list[str] | None): Selected Reddit URLs for analysis
        reddit_post_data (str | None): Detailed data from selected Reddit posts
        google_analysis (str | None): Analysis of Google search results
        bing_analysis (str | None): Analysis of Bing search results
        yandex_analysis (str | None): Analysis of Yandex search results
        reddit_analysis (str | None): Analysis of Reddit posts and comments
        final_answer (str | None): Synthesized final answer
    """
    messages: Annotated[list, add_messages]
    user_input: str | None
    google_results: str | None
    bing_results: str | None
    yandex_results: str | None
    reddit_results: str | None
    selected_reddit_urls: list[str] | None
    reddit_post_data: str | None
    google_analysis: str | None
    bing_analysis: str | None
    yandex_analysis: str | None
    reddit_analysis: str | None
    final_answer: str | None
    
def google_search(state: State):
    user_input = state.get("user_input")
    
    if not user_input:
        return { "google_results": "" }
    
    google_results = serp_search(user_input, engine=SearchEngine.GOOGLE) 
    print(f"\nGoogle results: {google_results}")

    return { "google_results": google_results } 

def bing_search(state: State):
    user_input = state.get("user_input")
    
    if not user_input:
        return { "bing_results": "" }
    
    bing_results = serp_search(user_input, engine=SearchEngine.BING)
    print(f"\nBing results: {bing_results}")

    return { "bing_results": bing_results }    

def yandex_search(state: State):
    user_input = state.get("user_input") 
    
    if not user_input:
        return {"yandex_results" : ""}
    
    yandex_results = serp_search(user_input, SearchEngine.Yandex)
    print(f"\nYandex results {yandex_results}")
    
    return { "yandex_results": yandex_results }

def reddit_search(state: State):
    user_input = state.get("user_input")

    if not user_input:
        return { "reddit_results": "" }
    
    reddit_results = scrap_reddit(user_input)
    print(f"Reddit results: {reddit_results}")
    
    return { "reddit_results": reddit_results }

def select_reddit_urls(state: State) -> dict:
    """Select relevant Reddit URLs from search results.
    
    Args:
        state (State): Current state containing user input and Reddit results
        
    Returns:
        dict: Updated state with selected Reddit URLs
        
    Raises:
        ValueError: If user input or Reddit results are missing or invalid
    """
    user_input = state.get("user_input", "")
    reddit_results = state.get("reddit_results")
    
    if not isinstance(user_input, str) or not reddit_results:
        return {"selected_reddit_urls": []}
        
    if not isinstance(reddit_results, str):
        reddit_results = str(reddit_results)
     
    structured_llm = llm.with_structured_output(RedditURLAnalysis)  
    
    messages = get_reddit_url_analysis_messages(user_input, reddit_results)
    selected_urls: list[str] = [] 
    
    try:
        analysis = structured_llm.invoke(messages)
        
        # Handle various possible return types from the LLM
        if isinstance(analysis, dict):
            selected_urls = analysis.get("selected_urls", [])
        elif isinstance(analysis, RedditURLAnalysis):
            selected_urls = analysis.selected_urls
        elif hasattr(analysis, 'selected_urls'):
            selected_urls = getattr(analysis, 'selected_urls', [])
        else:
            print("Warning: Unexpected response format from LLM")
            selected_urls = []
            
        # Ensure we have a valid list
        if not isinstance(selected_urls, list):
            selected_urls = list(selected_urls) if selected_urls is not None else []
            
        print(f"Selected Reddit URLs")
        for i, url in enumerate(selected_urls, start=1):
            print(f"{i}. {url}")
            
        return {"selected_reddit_urls": selected_urls}
    except Exception as e:
        print(f"Error during Reddit URL selection: {e}")
        
    return { "selected_reddit_urls": selected_urls }

def retrieve_reddit_posts(state: State) -> dict:
    """Retrieve and process Reddit posts from selected URLs.
    
    Args:
        state (State): The current state containing user input and selected Reddit URLs
        
    Returns:
        dict: Updated state with processed Reddit post data
    """
    user_input = state.get("user_input", "")
    reddit_results = state.get("reddit_results", "")
    reddit_post_data: list = []
    
    # Ensure input types are correct
    if not isinstance(user_input, str):
        user_input = str(user_input) if user_input is not None else ""
    if not isinstance(reddit_results, str):
        reddit_results = str(reddit_results) if reddit_results is not None else ""
    
    selected_reddit_urls = state.get("selected_reddit_urls")
    if selected_reddit_urls and isinstance(selected_reddit_urls, list) and len(selected_reddit_urls) > 0:
        # Process Reddit posts
        try:
            response_data = scrap_reddit(selected_reddit_urls, scrape_operation_type=RedditScrapeOperationType.POST_COMMENTS)
            if response_data:
                reddit_post_data = response_data.get("parsed_comments", [])
                print(f"Retrieved {len(reddit_post_data)} Reddit comments from selected URLs.")
                
                if reddit_post_data:
                    messages = get_reddit_analysis_messages(user_input, reddit_results, reddit_post_data)
                    reply = llm.invoke(messages)
                    content = str(reply.content) if hasattr(reply, 'content') else str(reply)
                    return {"reddit_post_data": content}
        except Exception as e:
            print(f"Error retrieving Reddit posts: {e}")
    
    return {"reddit_post_data": ""}
def analyze_google_results(state: State):
    user_input = state.get("user_input", "")
    google_results = state.get("google_results")
    
    if not user_input or not google_results:
        return { "google_analysis": "" }
    
    messages = get_google_analysis_messages(user_input, google_results)
    reply = llm.invoke(messages)
    return { "google_analysis": reply.content }

def analyze_bing_results(state: State) -> dict:
    """Analyze Bing search results using LLM.
    
    This function processes Bing search results to extract relevant insights
    and generate an analysis.
    
    Args:
        state (State): The current state containing:
            - user_input (str | None): The original user query
            - bing_results (str | None): Raw Bing search results
            
    Returns:
        dict: Updated state with Bing analysis results
            - bing_analysis (str): Analysis of Bing search results or empty string
              if no results are available
    """
    user_input = state.get("user_input", "")
    bing_results = state.get("bing_results", "")
    
    # Type validation
    if not isinstance(user_input, str):
        user_input = str(user_input) if user_input is not None else ""
    if not isinstance(bing_results, str):
        bing_results = str(bing_results) if bing_results is not None else ""
    
    if not bing_results:
        return {"bing_analysis": ""}
    
    messages = get_bing_analysis_messages(user_input, bing_results)
    reply = llm.invoke(messages)
    
    return {"bing_analysis": str(reply.content) if hasattr(reply, 'content') else str(reply)}

def analyze_yandex_results(state: State) -> dict:
    """Analyze Yandex search results using LLM.
    
    This function processes Yandex search results to extract relevant insights
    and generate an analysis.
    
    Args:
        state (State): The current state containing:
            - user_input (str | None): The original user query
            - yandex_results (str | None): Raw Yandex search results
            
    Returns:
        dict: Updated state with Yandex analysis results
            - yandex_analysis (str): Analysis of Yandex search results or empty string
              if no results are available
    """
    user_input = state.get("user_input", "")
    yandex_results = state.get("yandex_results", "")
    
    # Type validation
    if not isinstance(user_input, str):
        user_input = str(user_input) if user_input is not None else ""
    if not isinstance(yandex_results, str):
        yandex_results = str(yandex_results) if yandex_results is not None else ""
    
    if not yandex_results:
        return {"yandex_analysis": ""}
    
    messages = get_yandex_analysis_messages(user_input, yandex_results)
    reply = llm.invoke(messages)
    
    return {"yandex_analysis": str(reply.content) if hasattr(reply, 'content') else str(reply)}

def analyze_reddit_results(state: State) -> dict:
    """Analyze Reddit results using LLM.
    
    Args:
        state (State): Current state containing user input and Reddit data
        
    Returns:
        dict: Updated state with Reddit analysis results
        
    Note:
        All inputs are validated and converted to required types before processing
    """
    user_input = state.get("user_input")
    reddit_results = state.get("reddit_results")
    reddit_post_data = state.get("reddit_post_data")

    if not reddit_post_data:
        return {"reddit_analysis": ""}
        
    # Convert potential None values to empty strings/lists
    user_input_str = str(user_input if user_input is not None else "")
    reddit_results_str = str(reddit_results if reddit_results is not None else "")
    
    # Handle reddit_post_data type conversion
    if isinstance(reddit_post_data, str):
        reddit_post_data_list = [reddit_post_data]
    else:
        reddit_post_data_list = list(reddit_post_data) if reddit_post_data is not None else []
        
    messages = get_reddit_analysis_messages(
        user_input_str,
        reddit_results_str,
        reddit_post_data_list
    )
    reply = llm.invoke(messages)
    
    return { "reddit_analysis": reply.content }

def synthesize_final_answer(state: State) -> dict:
    """Synthesize all analysis results into a final answer.
    
    This function combines the analysis results from multiple sources (Google, Bing,
    Yandex, and Reddit) to generate a comprehensive final answer to the user's query.
    
    Args:
        state (State): The current state containing all analysis results
            - user_input: The original user query
            - google_analysis: Analysis of Google search results
            - bing_analysis: Analysis of Bing search results
            - yandex_analysis: Analysis of Yandex search results
            - reddit_analysis: Analysis of Reddit content
            
    Returns:
        dict: A dictionary containing the final synthesized answer
            - final_answer (str): The comprehensive response to the user's query
            
    Note:
        All inputs are validated and converted to strings before processing.
        If no analysis results are available, returns an empty string as the answer.
    """
    user_input = state.get("user_input", "")
    google_analysis = state.get("google_analysis", "")
    bing_analysis = state.get("bing_analysis", "")
    yandex_analysis = state.get("yandex_analysis", "")
    reddit_analysis = state.get("reddit_analysis", "")

    # Type validation and conversion
    if not isinstance(user_input, str):
        user_input = str(user_input) if user_input is not None else ""
    
    # Convert all analysis inputs to strings
    analyses = {
        'google': google_analysis,
        'bing': bing_analysis,
        'yandex': yandex_analysis,
        'reddit': reddit_analysis
    }
    
    analyses = {k: str(v) if v is not None else "" for k, v in analyses.items()}
    
    if not any(analyses.values()):
        return {"final_answer": ""}

    messages = get_synthesis_messages(
        user_input,
        analyses['google'],
        analyses['bing'],
        analyses['yandex'],
        analyses['reddit']
    )
    reply = llm.invoke(messages)

    print(f"Final answer: {reply}")
    return {"final_answer": str(reply.content) if hasattr(reply, 'content') else str(reply)}

graph_builder = StateGraph(State)

graph_builder.add_node("google_search", google_search)
graph_builder.add_node("bing_search", bing_search)
graph_builder.add_node("yandex_search", yandex_search)
graph_builder.add_node("reddit_search", reddit_search)

graph_builder.add_node("select_reddit_urls", select_reddit_urls)
graph_builder.add_node("retrieve_reddit_posts", retrieve_reddit_posts)

graph_builder.add_node("analyze_google_results", analyze_google_results)
graph_builder.add_node("analyze_bing_results", analyze_bing_results)
graph_builder.add_node("analyze_yandex_results", analyze_yandex_results)
graph_builder.add_node("analyze_reddit_results", analyze_reddit_results)
graph_builder.add_node("synthesize_final_answer", synthesize_final_answer)

graph_builder.add_edge(START, "google_search")
graph_builder.add_edge(START, "bing_search")
graph_builder.add_edge(START, "yandex_search")
graph_builder.add_edge(START, "reddit_search")

graph_builder.add_edge("google_search", "select_reddit_urls")
graph_builder.add_edge("bing_search", "select_reddit_urls")
graph_builder.add_edge("yandex_search", "select_reddit_urls")
graph_builder.add_edge("reddit_search", "select_reddit_urls")
graph_builder.add_edge("select_reddit_urls", "retrieve_reddit_posts")

graph_builder.add_edge("retrieve_reddit_posts", "analyze_google_results")
graph_builder.add_edge("retrieve_reddit_posts", "analyze_bing_results")
graph_builder.add_edge("retrieve_reddit_posts", "analyze_yandex_results")
graph_builder.add_edge("retrieve_reddit_posts", "analyze_reddit_results")

graph_builder.add_edge("analyze_google_results", "synthesize_final_answer")
graph_builder.add_edge("analyze_bing_results", "synthesize_final_answer")
graph_builder.add_edge("analyze_yandex_results", "synthesize_final_answer")
graph_builder.add_edge("analyze_reddit_results", "synthesize_final_answer")

graph_builder.add_edge("synthesize_final_answer", END)

graph = graph_builder.compile()

def start_chatbot():
    while True:
        user_input = input("Enter your question (or 'exit' to quit): ")
        if user_input.lower() in { "exit", "quit" }:
            break
        
        initial_state = State(
            messages=[],
            user_input=user_input,
            google_results=None,
            bing_results=None,
            yandex_results=None,
            reddit_results=None,
            selected_reddit_urls=None,
            reddit_post_data=None,
            google_analysis=None,
            bing_analysis=None,
            yandex_analysis=None,
            reddit_analysis=None,
            final_answer=None
        )
        
        final_state = graph.invoke(initial_state)
        print(f"My final conclusion is {final_state.get("final_answer", "")}")

if __name__ == "__main__":
    start_chatbot()

