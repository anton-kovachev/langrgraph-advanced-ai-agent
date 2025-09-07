"""Shared test fixtures and configurations for the AI agent test suite."""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List
from pydantic import BaseModel

# Mock response classes
class MockLLMResponse:
    """Mock LLM response object."""
    def __init__(self, content: str):
        self.content = content

class MockRedditURLAnalysis(BaseModel):
    """Mock Reddit URL analysis response."""
    selected_urls: List[str]

@pytest.fixture
def mock_state() -> Dict[str, Any]:
    """Fixture providing a clean initial state for tests."""
    return {
        "messages": [],
        "user_input": "test query",
        "google_results": None,
        "bing_results": None,
        "yandex_results": None,
        "reddit_results": None,
        "selected_reddit_urls": None,
        "reddit_post_data": None,
        "google_analysis": None,
        "bing_analysis": None,
        "yandex_analysis": None,
        "reddit_analysis": None,
        "final_answer": None
    }

@pytest.fixture
def mock_llm():
    """Fixture providing a mocked LLM instance."""
    mock = Mock()
    mock.invoke = Mock(return_value=Mock(content="mocked response"))
    mock.with_structured_output = Mock(return_value=mock)
    return mock

@pytest.fixture
def mock_search_results():
    """Fixture providing mock search results."""
    return {
        "google_results": "Mocked Google search results about the query",
        "bing_results": "Mocked Bing search results about the query",
        "yandex_results": "Mocked Yandex search results about the query",
        "reddit_results": "Mocked Reddit search results about the query"
    }

@pytest.fixture
def mock_reddit_posts():
    """Fixture providing mock Reddit post data."""
    return {
        "parsed_comments": [
            "Useful comment 1 about the topic",
            "Relevant discussion point about the query",
            "Expert opinion on the matter"
        ]
    }

@pytest.fixture
def mock_analysis_results():
    """Fixture providing mock analysis results."""
    return {
        "google_analysis": "Analysis of Google results",
        "bing_analysis": "Analysis of Bing results",
        "yandex_analysis": "Analysis of Yandex results",
        "reddit_analysis": "Analysis of Reddit discussions"
    }
