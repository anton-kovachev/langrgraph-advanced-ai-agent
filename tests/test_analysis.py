"""Tests for analysis operations across different sources."""

import pytest
from unittest.mock import patch, Mock
from main import (
    analyze_google_results,
    analyze_bing_results,
    analyze_yandex_results,
    analyze_reddit_results
)

def test_analyze_google_results_success(mock_state, mock_llm):
    """Test successful Google results analysis."""
    mock_state["google_results"] = "test results"
    
    with patch('main.get_google_analysis_messages') as mock_get_messages:
        llm = mock_llm
        llm.complete.return_value = Mock(content="analysis result")
        with patch('main.llm', llm):
            result = analyze_google_results(mock_state)
            
            mock_get_messages.assert_called_once()
            assert result["google_analysis"] == "analysis result"

def test_analyze_google_results_empty(mock_state):
    """Test Google analysis with empty results."""
    mock_state["google_results"] = ""
    
    result = analyze_google_results(mock_state)
    assert result["google_analysis"] == ""

def test_analyze_bing_results_success(mock_state, mock_llm):
    """Test successful Bing results analysis."""
    mock_state["bing_results"] = "test results"
    
    with patch('main.get_bing_analysis_messages') as mock_get_messages:
        llm = mock_llm
        llm.complete.return_value = Mock(content="analysis result")
        with patch('main.llm', llm):
            result = analyze_bing_results(mock_state)
            
            mock_get_messages.assert_called_once()
            assert result["bing_analysis"] == "analysis result"

def test_analyze_yandex_results_success(mock_state, mock_llm):
    """Test successful Yandex results analysis."""
    mock_state["yandex_results"] = "test results"
    
    with patch('main.get_yandex_analysis_messages') as mock_get_messages:
        llm = mock_llm
        llm.complete.return_value = Mock(content="analysis result")
        with patch('main.llm', llm):
            result = analyze_yandex_results(mock_state)
            
            mock_get_messages.assert_called_once()
            assert result["yandex_analysis"] == "analysis result"

def test_analyze_reddit_results_success(mock_state, mock_llm):
    """Test successful Reddit results analysis."""
    mock_state.update({
        "reddit_results": "test results",
        "reddit_post_data": ["post1", "post2"]
    })
    
    with patch('main.get_reddit_analysis_messages') as mock_get_messages:
        llm = mock_llm
        llm.complete.return_value = Mock(content="analysis result")
        with patch('main.llm', llm):
            result = analyze_reddit_results(mock_state)
            
            mock_get_messages.assert_called_once()
            assert result["reddit_analysis"] == "analysis result"

@pytest.mark.parametrize("analysis_func,state_key", [
    (analyze_google_results, "google_results"),
    (analyze_bing_results, "bing_results"),
    (analyze_yandex_results, "yandex_results")
])
def test_analysis_invalid_inputs(mock_state, analysis_func, state_key):
    """Test analysis functions with invalid inputs."""
    mock_state[state_key] = None
    result = analysis_func(mock_state)
    assert result[f"{state_key.split('_')[0]}_analysis"] == ""

@pytest.mark.parametrize("analysis_func", [
    analyze_google_results,
    analyze_bing_results,
    analyze_yandex_results,
    analyze_reddit_results
])
def test_analysis_error_handling(mock_state, analysis_func, mock_llm):
    """Test error handling in analysis functions."""
    llm = mock_llm
    llm.complete.side_effect = Exception("Analysis error")
    with patch('main.llm', llm):
        result = analysis_func(mock_state)
        assert result.get(next(iter(result.keys()))) == ""
