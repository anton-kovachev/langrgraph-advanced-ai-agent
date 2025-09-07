"""Tests for Reddit-specific operations."""

import pytest
from unittest.mock import patch, Mock
from main import select_reddit_urls, retrieve_reddit_posts, RedditURLAnalysis
from reddit_web_operations import RedditScrapeOperationType

def test_select_reddit_urls_success(mock_state, mock_llm):
    """Test successful Reddit URL selection."""
    expected_urls = ["url1", "url2"]
    mock_analysis = RedditURLAnalysis(selected_urls=expected_urls)
    
    llm = mock_llm
    llm.with_structured_output.return_value.complete.return_value = mock_analysis
    with patch('main.llm', llm):
        result = select_reddit_urls(mock_state)
        
        assert result["selected_reddit_urls"] == expected_urls

def test_select_reddit_urls_empty_results(mock_state):
    """Test Reddit URL selection with empty results."""
    mock_state["reddit_results"] = ""
    
    result = select_reddit_urls(mock_state)
    assert result["selected_reddit_urls"] == []

def test_select_reddit_urls_invalid_input(mock_state):
    """Test Reddit URL selection with invalid input."""
    mock_state["user_input"] = None
    mock_state["reddit_results"] = None
    
    result = select_reddit_urls(mock_state)
    assert result["selected_reddit_urls"] == []

def test_retrieve_reddit_posts_success(mock_state, mock_reddit_posts, mock_llm):
    """Test successful Reddit post retrieval."""
    mock_state["selected_reddit_urls"] = ["url1", "url2"]
    
    with patch('main.scrap_reddit', return_value=mock_reddit_posts):
        result = retrieve_reddit_posts(mock_state)
        assert "reddit_post_data" in result
        assert isinstance(result["reddit_post_data"], str)

def test_retrieve_reddit_posts_no_urls(mock_state):
    """Test Reddit post retrieval with no URLs."""
    mock_state["selected_reddit_urls"] = []
    
    result = retrieve_reddit_posts(mock_state)
    assert result["reddit_post_data"] == ""

def test_retrieve_reddit_posts_error(mock_state):
    """Test Reddit post retrieval error handling."""
    mock_state["selected_reddit_urls"] = ["url1"]
    
    with patch('main.scrap_reddit', side_effect=Exception("Network error")):
        result = retrieve_reddit_posts(mock_state)
        assert result["reddit_post_data"] == ""

@pytest.mark.parametrize("urls,expected_type", [
    (["url1", "url2"], list),
    (None, type(None)),
    ([], list)
])
def test_reddit_url_types(mock_state, urls, expected_type):
    """Test handling of different URL types in Reddit operations."""
    mock_state["selected_reddit_urls"] = urls
    result = retrieve_reddit_posts(mock_state)
    assert result["reddit_post_data"] == ""
