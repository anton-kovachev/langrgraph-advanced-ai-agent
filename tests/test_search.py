"""Tests for search operations across different search engines."""

import pytest
from unittest.mock import patch, Mock
from main import google_search, bing_search, yandex_search, reddit_search
from serp_web_operations import SearchEngine

@pytest.mark.parametrize("user_input,expected", [
    ("test query", "mock results"),
    ("", ""),
    (None, ""),
])
def test_google_search(mock_state, user_input, expected):
    """Test Google search with various inputs."""
    mock_state["user_input"] = user_input
    
    with patch('main.serp_search', return_value=expected) as mock_search:
        result = google_search(mock_state)
        
        if user_input:
            mock_search.assert_called_once_with(user_input, engine=SearchEngine.GOOGLE)
        else:
            mock_search.assert_not_called()
        
        assert result["google_results"] == expected

@pytest.mark.parametrize("user_input,expected", [
    ("test query", "mock results"),
    ("", ""),
    (None, ""),
])
def test_bing_search(mock_state, user_input, expected):
    """Test Bing search with various inputs."""
    mock_state["user_input"] = user_input
    
    with patch('main.serp_search', return_value=expected) as mock_search:
        result = bing_search(mock_state)
        
        if user_input:
            mock_search.assert_called_once_with(user_input, engine=SearchEngine.BING)
        else:
            mock_search.assert_not_called()
        
        assert result["bing_results"] == expected

@pytest.mark.parametrize("user_input,expected", [
    ("test query", "mock results"),
    ("", ""),
    (None, ""),
])
def test_yandex_search(mock_state, user_input, expected):
    """Test Yandex search with various inputs."""
    mock_state["user_input"] = user_input
    
    with patch('main.serp_search', return_value=expected) as mock_search:
        result = yandex_search(mock_state)
        
        if user_input:
            mock_search.assert_called_once_with(user_input, SearchEngine.Yandex)
        else:
            mock_search.assert_not_called()
        
        assert result["yandex_results"] == expected

def test_reddit_search(mock_state):
    """Test Reddit search functionality."""
    with patch('main.scrap_reddit', return_value="mock reddit results") as mock_reddit:
        result = reddit_search(mock_state)
        
        mock_reddit.assert_called_once_with("test query")
        assert result["reddit_results"] == "mock reddit results"

def test_reddit_search_empty_input(mock_state):
    """Test Reddit search with empty input."""
    mock_state["user_input"] = ""
    
    with patch('main.scrap_reddit') as mock_reddit:
        result = reddit_search(mock_state)
        
        mock_reddit.assert_not_called()
        assert result["reddit_results"] == ""

@pytest.mark.parametrize("error", [
    Exception("Network error"),
    ValueError("Invalid input"),
    TimeoutError("Request timeout")
])
def test_search_error_handling(mock_state, error):
    """Test error handling in search operations."""
    with patch('main.serp_search', side_effect=error):
        with pytest.raises(Exception):
            google_search(mock_state)
